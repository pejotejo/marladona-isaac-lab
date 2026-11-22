"""Isaac soccer adapter. Importing this module does not launch Kit."""

from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from importlib.metadata import version

from .scenarios import SCENARIOS, Diagnostic

REASONS = Diagnostic.RULES.reasons
TASK = "Isaac-Soccer-Play-v0"


def exit_code(x, y, length, width, goal_width):
    """Same strict goal mouth and inclusive boundary as SoccerBall.check_ball_location."""
    mouth = abs(y) < goal_width / 2
    blue = (x < -length / 2) & mouth
    red = (x > length / 2) & mouth
    outside = (abs(x) >= length / 2) | (abs(y) >= width / 2)
    return 3 * outside - 2 * blue - 1 * red


@contextmanager
def open_backend(suite, launcher_args):
    from isaaclab.app import AppLauncher

    parser = ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    launch = parser.parse_args([])
    launch.device = launcher_args.device
    launch.headless = launcher_args.headless
    launcher = AppLauncher(launch)
    try:
        import gymnasium
        import isaaclab_marl.tasks  # noqa: F401 - registers the soccer environment
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_marl.utils.custom_vecenv_wrapper import CustomVecEnvWrapper

        cfg = parse_env_cfg(TASK, device=launcher_args.device, num_envs=suite.num_envs)
        team_size = SCENARIOS[suite.scenario].TEAM_SIZE
        cfg.soccer_game.num_agents_per_team = team_size
        cfg.soccer_game.num_agents_range = [team_size, team_size]
        cfg.scene.agents.rigid_objects.clear()
        cfg.create_agent()
        cfg.seed = suite.seed
        cfg.curriculum.terrain_levels = None
        cfg.events.physics_material_ball = None
        cfg.soccer_game.ball.roll_friction_range = (0.00005, 0.00005)
        cfg.soccer_game.default_control = [1, 1]
        cfg.soccer_game.agent_id_control = [
            [1] * cfg.soccer_game.num_agents_per_team for _ in range(2)
        ]
        for group in ("neighbor", "policy", "bots"):
            getattr(cfg.observations, group).enable_corruption = False
        gym_env = gymnasium.make(TASK, cfg=cfg)
        try:
            with torch.inference_mode():
                yield IsaacBackend(CustomVecEnvWrapper(gym_env), suite)
        finally:
            gym_env.close()
    finally:
        launcher.app.close()


class IsaacBackend:
    rules = Diagnostic.RULES

    def __init__(self, wrapped, suite):
        import torch
        from rsl_marl.modules.bots import Bots

        self.torch = torch
        self.rules = SCENARIOS[suite.scenario].RULES
        self.env = env = wrapped.unwrapped
        self.wrapped = wrapped
        data = env.env_data
        n, players = env.num_envs, data.num_agents_per_env
        self.actor_ids = torch.arange(n * players, device=env.device).reshape(
            n, players
        )
        self.team = torch.arange(2, device=env.device).repeat_interleave(players // 2)
        self.bots = Bots(wrapped, env.device)
        self.previous = torch.zeros(n * players, wrapped.num_actions, device=env.device)
        self.policies = [
            self._policy(path) if path else None
            for path in (suite.candidate_checkpoint, suite.opponent_checkpoint)
        ]
        self.live = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.ticks = torch.zeros(n, dtype=torch.long, device=env.device)
        self.limit = torch.zeros_like(self.ticks)
        self.provenance = {
            "name": "isaac",
            "version": 2,
            "physics_backend": type(env.cfg.sim.physics).__name__,
            "observation_corruption": {
                k: getattr(env.cfg.observations, k).enable_corruption
                for k in ("policy", "neighbor", "bots")
            },
            "bot_roles": ["keeper"] + ["player"] * (players // 2 - 1),
            "controller_observations": ["policy", "neighbor"],
            "ball_roll_friction_range": list(
                env.cfg.soccer_game.ball.roll_friction_range
            ),
            "task": TASK,
            "isaaclab": version("isaaclab"),
            "physics_dt_s": env.physics_dt,
            "decimation": env.cfg.decimation,
            "field_length_m": float(data.field_length[0]),
            "field_width_m": float(data.field_width),
            "goal_width_m": float(data.goal_width[0]),
            "device": str(env.device),
        }

    def _policy(self, path):
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_marl.runners import OnPolicyRunner

        cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        cfg.policy_replay.dynamic_generate_replay_level = False
        runner = OnPolicyRunner(
            self.wrapped,
            cfg.to_dict(),
            device=self.env.device,
            command_args=Namespace(),
        )
        runner.load(path, load_optimizer=False)
        return runner.get_inference_policy(device=self.env.device)

    def reset(self, assignments):
        from isaaclab.utils.math import quat_from_euler_xyz

        env, torch = self.env, self.torch
        slots = torch.tensor(list(assignments), device=env.device, dtype=torch.long)

        def values(name, dtype):
            return torch.tensor(
                [t[name] for t in assignments.values()], device=env.device, dtype=dtype
            )

        data = env.env_data
        origin = data.env_origins[slots]
        active = values("active", torch.bool).reshape(-1, 2, data.num_agents_per_team)
        data.active[slots], data.inactive[slots] = active, ~active
        data.active_agent_per_team[slots] = active.sum(-1).to(
            data.active_agent_per_team.dtype
        )
        self.ticks[slots] = 0
        self.live[slots] = True
        self.limit[slots] = values("max_steps", torch.long) * env.cfg.decimation
        actors = self.actor_ids[slots].flatten()
        self.previous[actors] = 0
        env.action_manager.reset(slots)
        env.observation_manager.reset_for_matches(slots)
        data.reset(slots)
        env.scene.reset(slots)
        pose = values("player_pose", torch.float32)
        state = env.agents.asset.data.default_body_state.torch[slots].clone()
        state[..., :2] = origin[:, None, :2] + pose[..., :2]
        yaw = pose[..., 2]
        state[..., 3:7] = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        vel = values("player_velocity", torch.float32)
        state[..., 7:] = 0
        state[..., 7:9], state[..., 12] = vel[..., :2], vel[..., 2]
        env.agents.asset.write_body_pose_to_sim_index(
            body_poses=state[..., :7], env_ids=slots
        )
        env.agents.asset.write_body_com_velocity_to_sim_index(
            body_velocities=state[..., 7:], env_ids=slots
        )
        ball = env.ball.asset.data.default_root_state.torch[slots].clone()
        ball[:, :3] = origin + values("ball_position", torch.float32)
        ball[:, 7:] = 0
        ball[:, 7:10] = values("ball_velocity", torch.float32)
        env.ball.asset.write_root_pose_to_sim_index(
            root_pose=ball[:, :7], env_ids=slots
        )
        env.ball.asset.write_root_com_velocity_to_sim_index(
            root_velocity=ball[:, 7:], env_ids=slots
        )
        env.scene.write_data_to_sim()
        env.sim.forward()
        return self._frame(*self._positions(), *self._motion(), self.ticks)

    def _positions(self):
        env = self.env
        origin = env.env_data.env_origins[:, :2]
        return (
            env.ball.asset.data.root_pos_w.torch[:, :2] - origin,
            env.agents.asset.data.body_state_w.torch[..., :2] - origin[:, None, :],
        )

    def _motion(self):
        from isaaclab.utils.math import euler_xyz_from_quat

        quat = self.env.agents.asset.data.body_state_w.torch[..., 3:7]
        yaw = euler_xyz_from_quat(quat.reshape(-1, 4))[2].reshape(quat.shape[:-1])
        return yaw, self.env.ball.asset.data.root_lin_vel_w.torch

    def _actions(self):
        env, torch = self.env, self.torch
        env.agents.refresh_state()
        env.ball.refresh_state()
        env.ball.compute_ball2agent_states(
            env.agents.base_pose_w, inactive=env.env_data.inactive
        )
        env.agents.compute_ego_ball_state(env.ball.base_pos_w)
        obs = env.observation_manager.compute(update_history=True)
        actions = torch.zeros_like(self.previous)
        for side, policy in enumerate(self.policies):
            ids = self.actor_ids[:, self.team == side].flatten()
            if policy is None:
                roles = ["keeper"] + ["player"] * (env.env_data.num_agents_per_team - 1)
                actions[ids] = self.bots.act(
                    obs["bots"], self.previous, ids, configs=roles
                )
            else:
                actions[ids] = policy(
                    torch.cat([obs["policy"][ids], obs["neighbor"][ids]], dim=1)
                )
        mask = env.env_data.active & self.live[:, None, None]
        self.previous = actions.masked_fill(~mask.reshape(-1, 1), 0)
        return self.previous

    def _frame(self, ball, players, heading, velocity, ticks):
        env = self.env
        return {
            "time_s": (ticks.double() * env.physics_dt),
            "ball_xy": ball,
            "player_heading": heading,
            "ball_velocity": velocity,
            "player_xy": players,
            "active": env.env_data.active.flatten(1),
            "team": self.team,
        }

    def step(self):
        env, torch = self.env, self.torch
        data = env.env_data
        codes = torch.zeros_like(self.ticks)
        ball, players = self._positions()
        heading, velocity = self._motion()
        velocity = velocity.clone()

        def sample(_substep):
            nonlocal ball, players, codes, heading, velocity
            tracking = self.live & (codes == 0)
            current_ball, current_players = self._positions()
            current_heading, current_velocity = self._motion()
            heading = torch.where(tracking[:, None], current_heading, heading)
            velocity = torch.where(tracking[:, None], current_velocity, velocity)
            self.ticks += tracking
            ball = torch.where(tracking[:, None], current_ball, ball)
            players = torch.where(tracking[:, None, None], current_players, players)
            event = exit_code(
                ball[:, 0],
                ball[:, 1],
                data.field_length,
                data.field_width,
                data.goal_width,
            )
            frame = self._frame(ball, players, heading, velocity, self.ticks)
            event = self.rules.classify(frame, event)
            event = torch.where(
                (event == 0) & (self.ticks >= self.limit), self.rules.timeout, event
            )
            codes = torch.where(tracking, event, codes)

        env.step_physics(self._actions(), after_substep=sample)
        self.live &= codes == 0
        return self._frame(ball, players, heading, velocity, self.ticks), [
            self.rules.reasons[i] for i in codes.tolist()
        ]
