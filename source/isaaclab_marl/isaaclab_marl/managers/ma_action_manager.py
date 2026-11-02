# Copyright 2025 Zichong Li, ETH Zurich

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_base import ManagerBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab_marl.assets.env_data import SoccerAgentsCfg
    from isaaclab_marl.tasks.soccer.soccer_marl_env import SoccerMARLEnv
    from isaaclab_marl.assets.agents import Agents

from isaaclab.managers.action_manager import ActionTermCfg

from isaaclab_marl.utils.math_utils import rotate2d


class MultiAgentAction:
    @property
    def num_agents_per_team(self) -> int:
        """Number of environments."""
        return self._env.env_data.num_agents_per_team

    @property
    def num_agents_per_env(self) -> int:
        """Number of environments."""
        return self._env.env_data.num_agents_per_team * 2


class BaseMovementAction(ActionTerm, MultiAgentAction):

    cfg: ActionTermCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    agents: Agents
    agent_cfg: SoccerAgentsCfg

    def __init__(self, cfg: ActionTermCfg, env: SoccerMARLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.agents = env.agents
        self.agent_cfg = env.agents.agent_cfg
        # # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.num_agents_per_env, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        self.target_velocity_cmd = torch.zeros_like(self._processed_actions[..., :3])
        # self.target_force_cmd = torch.zeros_like(self.target_velocity_cmd)

        self._force = torch.zeros(self.num_envs, self.num_agents_per_env, 3, device=self.device)
        self._torque = torch.zeros(self.num_envs, self.num_agents_per_env, 3, device=self.device)

        self._scale = 1.0
        self._offset = 0.0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.view(self.num_envs, self.num_agents_per_env, -1)
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def apply_actions(self):
        self.compute()

        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._force, torques=self._torque
        )

    def compute(self):
        actions = self._processed_actions
        original_target_vel = torch.clip(actions[..., :2], -1, 1)
        target_vel = original_target_vel.clone()

        target_vel[..., 0] = original_target_vel[..., 0] * torch.sqrt(1 - original_target_vel[..., 1] ** 2 / 2)
        target_vel[..., 1] = original_target_vel[..., 1] * torch.sqrt(1 - original_target_vel[..., 0] ** 2 / 2)

        target_rot_vel = torch.clip(actions[..., 2], -1, 1)

        speed_limit = self.agent_cfg.cmd_speed_limit

        target_vel[..., 0][target_vel[..., 0] > 0] = target_vel[..., 0][target_vel[..., 0] > 0] * speed_limit[0][0]
        target_vel[..., 0][target_vel[..., 0] < 0] = target_vel[..., 0][target_vel[..., 0] < 0] * (-speed_limit[0][1])
        target_vel[..., 1] = target_vel[..., 1] * speed_limit[1][0]
        target_rot_vel = target_rot_vel * speed_limit[2][0]

        self.target_velocity_cmd[..., :2] = target_vel
        self.target_velocity_cmd[..., 2] = target_rot_vel

        self._force[..., :2] = (target_vel - self.agents.base_lin_vel_b) * self.agent_cfg.translation_force_d_gain
        self._torque[..., 2] = (target_rot_vel - self.agents.base_ang_vel_w) * self.agent_cfg.rotation_torque_d_gain

        inactive_agents = self._env.env_data.inactive.view(self.num_envs, -1)
        self._force[inactive_agents] = 0
        self._torque[inactive_agents] = 0


class SoccerMotionAction(BaseMovementAction):
    """Walk or execute a latched approach, alignment and kick request for each player."""

    def __init__(self, cfg: ActionTermCfg, env: SoccerMARLEnv) -> None:
        super().__init__(cfg, env)
        self.ball = env.ball
        self.ball_cfg = self.ball.ball_cfg
        self._kick_active = torch.zeros(self.num_envs, self.num_agents_per_env, dtype=torch.bool, device=self.device)
        self._kick_requested = torch.zeros_like(self._kick_active)
        self._target = torch.zeros_like(self._raw_actions[..., 4:6])
        self._strength = torch.zeros_like(self._raw_actions[..., 6])
        self._remaining = torch.zeros_like(self._strength)
        self._cooldown = torch.zeros_like(self._strength)
        self._ball_force = torch.zeros(self.num_envs, 1, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def kick_state(self) -> torch.Tensor:
        return torch.cat(
            [
                self._kick_active.unsqueeze(-1),
                self._kick_requested.unsqueeze(-1),
                self._target,
                self._strength.unsqueeze(-1),
                (self._remaining / self.ball_cfg.kick_timeout_s).unsqueeze(-1),
                (self._cooldown / self.ball_cfg.kick_cooldown_s).unsqueeze(-1),
            ],
            dim=-1,
        ).flatten(0, 1)

    def process_actions(self, actions: torch.Tensor):
        if not torch.isfinite(actions).all():
            raise ValueError("Motion commands must be finite.")
        super().process_actions(actions)
        self._processed_actions.clamp_(-1.0, 1.0)
        self._processed_actions[..., 3].clamp_(0.0, 1.0)
        self._processed_actions[..., 6].clamp_(0.0, 1.0)
        requested = self._processed_actions[..., 3] > 0.5
        start = requested & ~self._kick_requested & ~self._kick_active & (self._cooldown <= 0)
        start &= ~self._env.env_data.inactive.view(self.num_envs, -1)
        self._target[start] = self._processed_actions[..., 4:6][start]
        self._strength[start] = self._processed_actions[..., 6][start]
        self._remaining[start] = self.ball_cfg.kick_timeout_s
        self._kick_active |= start
        self._kick_requested[:] = requested

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._processed_actions[env_ids] = 0
        for state in (
            self._kick_active,
            self._kick_requested,
            self._target,
            self._strength,
            self._remaining,
            self._cooldown,
        ):
            state[env_ids] = 0

    def compute(self):
        dt = self._env.physics_dt
        self._cooldown.sub_(dt).clamp_(min=0)
        self._remaining.sub_(dt).clamp_(min=0)
        self._kick_active &= (self._remaining > 0) & ~self._env.env_data.inactive.view(self.num_envs, -1)

        pose = self.agents.base_pose_w.view(self.num_envs, self.num_agents_per_env, 3)
        ball_pos = self.ball.base_pos_w.unsqueeze(1)
        target = self._target.clone()
        target[..., 0] *= self._env.env_data.field_length.unsqueeze(-1) / 2
        target[..., 1] *= self._env.env_data.field_width / 2
        target[:, self.num_agents_per_team :] *= -1  # Both teams attack toward negative x in their own frame.
        ball_to_target = target - ball_pos
        direction = torch.nn.functional.normalize(ball_to_target, dim=-1)
        sideways = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
        relative = pose[..., :2] - ball_pos
        forward_distance = (relative * direction).sum(dim=-1)
        lateral_distance = (relative * sideways).sum(dim=-1)
        # Route around the ball when a direct approach from the front would push it away.
        route_around = forward_distance > -0.5 * relative.norm(dim=-1)
        side = torch.where(lateral_distance >= 0, 1.0, -1.0)
        behind_ball = ball_pos - direction * (self.ball_cfg.kick_margin * 0.8)
        waypoint = ball_pos - direction * 0.35 + sideways * side.unsqueeze(-1) * 0.35
        approach = torch.where(route_around.unsqueeze(-1), waypoint, behind_ball)
        delta = approach - pose[..., :2]
        delta_b = rotate2d(delta[..., 0], delta[..., 1], pose[..., 2])
        yaw_error = torch.atan2(direction[..., 1], direction[..., 0]) - pose[..., 2]
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))

        limits = self.agent_cfg.cmd_speed_limit
        auto_walk = torch.zeros_like(self._processed_actions[..., :3])
        x_limit = torch.where(delta_b[..., 0] >= 0, limits[0][0], -limits[0][1])
        auto_walk[..., 0] = 2.0 * delta_b[..., 0] / x_limit
        auto_walk[..., 1] = 2.0 * delta_b[..., 1] / limits[1][0]
        auto_walk[..., 2] = 3.0 * yaw_error / limits[2][0]
        self._processed_actions[..., :3] = torch.where(
            self._kick_active.unsqueeze(-1), auto_walk, self._raw_actions[..., :3]
        )
        super().compute()

        ball_b = self.agents.ball_pos_b.view(self.num_envs, self.num_agents_per_env, 2)
        ready = self._kick_active & (self._strength > 0) & (ball_to_target.norm(dim=-1) > 1e-4)
        ready &= (ball_b[..., 0] > 0) & (ball_b.norm(dim=-1) < self.ball_cfg.kick_margin)
        ready &= ball_b[..., 1].abs() < self.ball_cfg.kick_lateral_tolerance
        ready &= yaw_error.abs() < self.ball_cfg.kick_alignment_tolerance
        ready &= (self.ball.asset.data.root_pos_w.torch[:, 2] < 0.08).unsqueeze(-1)
        # Resolve simultaneous requests to the closest ready player, once per ball.
        distances = torch.where(ready, ball_b.norm(dim=-1), float("inf"))
        chosen = distances.argmin(dim=1)
        env_ids = ready.any(dim=1).nonzero(as_tuple=False).flatten()
        players = chosen[env_ids]
        velocity = self.ball.asset.data.root_state_w.torch[env_ids, 7:].clone()
        velocity[:, :2] = (
            direction[env_ids, players]
            * self._strength[env_ids, players].unsqueeze(-1)
            * self.ball_cfg.max_kick_speed
        )
        velocity[:, 2:] = 0
        # shortcut: proxy kicks set ball velocity; replace when adding articulated humanoid motions.
        if env_ids.numel():
            self.ball.asset.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        self._kick_active[env_ids, players] = False
        self._remaining[env_ids, players] = 0
        self._cooldown[env_ids, players] = self.ball_cfg.kick_cooldown_s

    def apply_actions(self):
        super().apply_actions()
        self.ball.asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._ball_force,
            torques=(-self.ball.asset.data.root_ang_vel_b.torch * self.ball.rolling_friction).unsqueeze(1),
        )


class MultiAgentActionManager(ManagerBase, MultiAgentAction):
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the action manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
        """
        # check if config is None
        if cfg is None:
            raise ValueError("Action manager configuration is None. Please provide a valid configuration.")

        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)
        # create buffers to store actions
        self._action = torch.zeros((self.num_envs * self.num_agents_per_env, self.total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)

        # check if any term has debug visualization implemented
        self.cfg.debug_vis = False
        for term in self._terms.values():
            self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.action_dim])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def total_action_dim(self) -> int:
        """Total dimension of actions."""
        return sum(self.action_term_dim)

    @property
    def active_terms(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [term.action_dim for term in self._terms.values()]

    @property
    def action(self) -> torch.Tensor:
        """The actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        """The previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._prev_action

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    """
    Operations.
    """

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        idx = 0
        for name, term in self._terms.items():
            term_actions = self._action[env_idx, idx : idx + term.action_dim].cpu()
            terms.append((name, term_actions.tolist()))
            idx += term.action_dim
        return terms

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the action data.
        Args:
            debug_vis: Whether to visualize the action data.
        Returns:
            Whether the debug visualization was successfully set. False if the action
            does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the action history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the action history
        self._prev_action.view(self.num_envs, self.num_agents_per_env, -1)[env_ids] = 0.0
        self._action.view(self.num_envs, self.num_agents_per_env, -1)[env_ids] = 0.0
        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        # check if action dimension is valid
        if action.shape != self._action.shape:
            raise ValueError(f"Invalid action shape, expected: {self._action.shape}, received: {action.shape}.")
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            term.apply_actions()

    def get_term(self, name: str) -> ActionTerm:
        """Returns the action term with the specified name.

        Args:
            name: The name of the action term.

        Returns:
            The action term with the specified name.
        """
        return self._terms[name]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._terms: dict[str, ActionTerm] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # parse action terms from the config
        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term
