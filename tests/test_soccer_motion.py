"""Single-player control and motion checks without running the physics simulator."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab")

from isaaclab.managers import ActionTermCfg
from isaaclab_marl.assets.env_data import EnvData, SoccerGameCfg
from isaaclab_marl.managers.ma_action_manager import MultiAgentActionManager, SoccerMotionAction


@pytest.fixture
def motion_scene():
    cfg = SoccerGameCfg(eval_env_ratio=0.25)
    data = EnvData(cfg, 4, "cpu")
    data.field_length[:] = 4.32
    state = torch.zeros(4, 13)
    state[:, :3] = torch.tensor([0.1, 0.0, 0.05])
    writes = []
    composer = SimpleNamespace(set_forces_and_torques_index=lambda **kwargs: None)
    ball_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_state_w=SimpleNamespace(torch=state),
            root_pos_w=SimpleNamespace(torch=state[:, :3]),
            root_ang_vel_b=SimpleNamespace(torch=state[:, 10:]),
        ),
        permanent_wrench_composer=composer,
        write_root_com_velocity_to_sim_index=lambda **kwargs: writes.append(kwargs),
    )
    agents = SimpleNamespace(
        agent_cfg=cfg.agents,
        base_pose_w=torch.zeros(4, 2, 3, 3),
        base_lin_vel_b=torch.zeros(4, 6, 2),
        base_ang_vel_w=torch.zeros(4, 6),
        ball_pos_b=torch.tensor([0.1, 0.0]).expand(24, 2).clone(),
    )
    ball = SimpleNamespace(
        ball_cfg=cfg.ball,
        asset=ball_asset,
        base_pos_w=state[:, :2],
        rolling_friction=torch.full((4, 1), 0.00008),
    )
    env = SimpleNamespace(
        num_envs=4, device="cpu", physics_dt=0.02, env_data=data, agents=agents, ball=ball,
        scene={"agents": SimpleNamespace(permanent_wrench_composer=composer), "ball": ball_asset},
    )
    term = SoccerMotionAction(ActionTermCfg(class_type=SoccerMotionAction, asset_name="agents"), env)
    return env, term, writes


def test_one_learner_and_five_bots_per_field(motion_scene):
    env, _, _ = motion_scene
    data = env.env_data
    assert data.num_actions == 7
    assert data.num_training_agents_per_env == 1
    assert data.num_policy_replay_agents_per_env == 0
    assert data.active.all()
    controlled = data.training_actor_index | data.eval_actor_index
    assert controlled.view(4, 6).sum(dim=1).tolist() == [1] * 4
    assert data.bots_actor_index.view(4, 6).sum(dim=1).tolist() == [5] * 4
    assert not data.policy_replay_actor_index.any()


def test_walk_scales_velocity_and_never_kicks_without_request(motion_scene):
    _, term, writes = motion_scene
    actions = torch.zeros(24, 7)
    actions[:, 0] = 1
    actions[:, 2] = -1
    term.process_actions(actions)
    term.apply_actions()
    torch.testing.assert_close(term.target_velocity_cmd[0, 0], torch.tensor([0.35, 0.0, -2.5]))
    actions[:, 0] = -1
    term.process_actions(actions)
    term.apply_actions()
    torch.testing.assert_close(term.target_velocity_cmd[0, 0], torch.tensor([-0.25, 0.0, -2.5]))
    assert writes == []


@pytest.mark.parametrize("strength", [0.0, 0.25, 1.0, 2.0, -1.0])
def test_kick_speed_uses_clamped_strength(motion_scene, strength):
    env, term, writes = motion_scene
    actions = torch.zeros(4, 6, 7)
    actions[:, 0, 3:] = torch.tensor([1.0, 1.0, 0.0, strength])
    term.process_actions(actions.flatten(0, 1))
    term.apply_actions()
    if strength <= 0:
        assert writes == []
        return
    assert len(writes) == 1
    expected = torch.zeros(4, 6)
    expected[:, 0] = min(strength, 1.0) * env.ball.ball_cfg.max_kick_speed
    torch.testing.assert_close(writes[0]["root_velocity"], expected)
    term.apply_actions()
    assert len(writes) == 1


@pytest.mark.parametrize("player", [0, 3])
def test_target_frame_and_angled_destination(motion_scene, player):
    env, term, writes = motion_scene
    target = torch.tensor([-0.8, 0.6])
    destination = target * torch.tensor([4.32 / 2, 3.6 / 2]) * (1 if player == 0 else -1)
    direction = torch.nn.functional.normalize(destination - env.ball.base_pos_w[0], dim=0)
    poses = env.agents.base_pose_w.view(4, 6, 3)
    poses[:, player, :2] = env.ball.base_pos_w - direction * 0.16
    poses[:, player, 2] = torch.atan2(direction[1], direction[0])
    env.agents.ball_pos_b.view(4, 6, 2)[:, player] = torch.tensor([0.16, 0.0])
    actions = torch.zeros(4, 6, 7)
    actions[:, player, 3:] = torch.tensor([1.0, *target, 0.5])
    term.process_actions(actions.flatten(0, 1))
    term.apply_actions()
    velocity = writes[0]["root_velocity"][:, :2]
    expected = direction.expand(4, 2) * env.ball.ball_cfg.max_kick_speed * 0.5
    torch.testing.assert_close(velocity, expected)


def test_approach_latches_target_and_strength_until_aligned(motion_scene):
    env, term, writes = motion_scene
    env.ball.base_pos_w[:, 0] = 0.8
    env.agents.ball_pos_b[:, 0] = 0.8
    actions = torch.zeros(4, 6, 7)
    actions[:, 0, 3:] = torch.tensor([1.0, 1.0, 0.0, 0.4])
    term.process_actions(actions.flatten(0, 1))
    term.apply_actions()
    assert (term.target_velocity_cmd[:, 0, 0] > 0).all()
    assert not writes
    term.process_actions(torch.zeros(24, 7))
    torch.testing.assert_close(term.kick_state[::6, 2:5], torch.tensor([1.0, 0.0, 0.4]).expand(4, 3))
    env.agents.base_pose_w.view(4, 6, 3)[:, 0, 0] = 0.64
    env.agents.ball_pos_b.view(4, 6, 2)[:, 0, 0] = 0.16
    term.apply_actions()
    assert len(writes) == 1
    torch.testing.assert_close(
        writes[0]["root_velocity"][:, 0], torch.full((4,), 0.4 * env.ball.ball_cfg.max_kick_speed)
    )
    assert not term._kick_active[:, 0].any()


@pytest.mark.parametrize("reason", ["angle", "side", "height", "inactive", "target_at_ball"])
def test_unready_player_cannot_kick(motion_scene, reason):
    env, term, writes = motion_scene
    actions = torch.zeros(4, 6, 7)
    actions[:, 0, 3:] = torch.tensor([1.0, 1.0, 0.0, 1.0])
    if reason == "angle":
        env.agents.base_pose_w.view(4, 6, 3)[:, 0, 2] = 0.3
    elif reason == "side":
        env.agents.ball_pos_b.view(4, 6, 2)[:, 0, 1] = 0.06
    elif reason == "height":
        env.ball.asset.data.root_pos_w.torch[:, 2] = 0.15
    elif reason == "inactive":
        env.env_data.inactive[:, 0, 0] = True
    else:
        actions[:, 0, 4] = 0.1 / (4.32 / 2)
    term.process_actions(actions.flatten(0, 1))
    term.apply_actions()
    assert writes == []


def test_timeout_and_selected_field_reset_clear_all_player_state(motion_scene):
    env, term, _ = motion_scene
    manager = MultiAgentActionManager(SimpleNamespace(motion=term.cfg), env)
    actions = torch.ones(24, 7)
    manager.process_action(actions)
    manager.process_action(actions)
    manager.reset(torch.tensor([1]))
    assert not manager.action.view(4, 6, 7)[1].any()
    assert not manager.prev_action.view(4, 6, 7)[1].any()
    assert manager.action.view(4, 6, 7)[0].all()
    assert not manager.get_term("motion").kick_state.view(4, 6, 7)[1].any()
    term.process_actions(actions)
    env.physics_dt = env.ball.ball_cfg.kick_timeout_s + 1
    term.compute()
    assert not term._kick_active.any()


def test_invalid_motion_commands_are_rejected(motion_scene):
    env, term, _ = motion_scene
    with pytest.raises(ValueError, match="finite"):
        term.process_actions(torch.full((24, 7), float("nan")))
    manager = MultiAgentActionManager(SimpleNamespace(motion=term.cfg), env)
    with pytest.raises(ValueError, match="shape"):
        manager.process_action(torch.zeros(4, 7))


@pytest.mark.parametrize("initial_pose", [(-0.8, -0.7, 1.2), (0.8, 0.4, -1.0), (0.5, 0.0, 3.14)])
def test_automatic_approach_finishes_from_different_sides(motion_scene, initial_pose):
    env, term, writes = motion_scene
    poses = env.agents.base_pose_w.view(4, 6, 3)
    poses[:, 0] = torch.tensor(initial_pose)
    actions = torch.zeros(4, 6, 7)
    actions[:, 0, 3:] = torch.tensor([1.0, 1.0, 0.0, 0.7])
    term.process_actions(actions.flatten(0, 1))
    # Ideal velocity tracking isolates approach geometry from the simulator's force controller.
    for _ in range(int(env.ball.ball_cfg.kick_timeout_s / env.physics_dt)):
        yaw = poses[..., 2]
        delta = env.ball.base_pos_w.unsqueeze(1) - poses[..., :2]
        ball_b = env.agents.ball_pos_b.view(4, 6, 2)
        ball_b[..., 0] = yaw.cos() * delta[..., 0] + yaw.sin() * delta[..., 1]
        ball_b[..., 1] = -yaw.sin() * delta[..., 0] + yaw.cos() * delta[..., 1]
        term.compute()
        if writes:
            break
        vx, vy, yaw_rate = term.target_velocity_cmd.unbind(dim=-1)
        poses[..., 0] += (yaw.cos() * vx - yaw.sin() * vy) * env.physics_dt
        poses[..., 1] += (yaw.sin() * vx + yaw.cos() * vy) * env.physics_dt
        poses[..., 2] += yaw_rate * env.physics_dt
    assert len(writes) == 1
    torch.testing.assert_close(
        writes[0]["root_velocity"][:, 0], torch.full((4,), 0.7 * env.ball.ball_cfg.max_kick_speed)
    )


def test_simultaneous_requests_choose_closest_eligible_player(motion_scene):
    env, term, writes = motion_scene
    env.agents.base_pose_w.view(4, 6, 3)[:, 1, 0] = 0.05
    env.agents.ball_pos_b.view(4, 6, 2)[:, 1, 0] = 0.05
    actions = torch.zeros(4, 6, 7)
    actions[:, :2, 3] = 1
    actions[:, :2, 4] = 1
    actions[:, 0, 6] = 0.25
    actions[:, 1, 6] = 0.75
    term.process_actions(actions.flatten(0, 1))
    term.compute()
    assert len(writes) == 1
    torch.testing.assert_close(
        writes[0]["root_velocity"][:, 0], torch.full((4,), 0.75 * env.ball.ball_cfg.max_kick_speed)
    )


def test_scripted_players_emit_targets_and_rearm_requests(motion_scene):
    from rsl_marl.modules.bots import Bots

    env, _, _ = motion_scene
    env.unwrapped = env
    bots = Bots(env, "cpu")
    obs = {
        "base_own_pose_w": torch.zeros(24, 3),
        "ball_pos_b": torch.tensor([0.1, 0.0]).expand(24, 2),
        "ball_pos_w": torch.tensor([0.1, 0.1, 0.0, 0.0]).expand(24, 4),
        "closest_agent_to_ball": torch.ones(24, dtype=torch.bool),
        "player_id": torch.arange(3).repeat(8),
        "field_infos": torch.tensor([0.5, -0.5, 2.16, -2.16, 1.8, 3, 3]).expand(24, 7),
    }
    previous = torch.zeros(24, 7)
    selected = env.env_data.bots_actor_index
    commands = bots.act(obs, previous, selected)
    assert commands.shape == (20, 7)
    torch.testing.assert_close(commands[:, 3:], torch.tensor([1.0, -1.0, 0.0, 1.0]).expand(20, 4))
    previous[selected] = commands
    assert not bots.act(obs, previous, selected)[:, 3].any()
