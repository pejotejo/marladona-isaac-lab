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

from dataclasses import dataclass, field

import torch
from torch import Tensor

from mjlab_marl.config import (
    APPLY_RANDOMIZATION,
    FIELD_LEVEL_SETTINGS,
    GOAL_LEVEL_SETTINGS,
    KICK_MODEL,
)


@dataclass
class SoccerBallCfg:
    ownership_radius: float = 0.25
    reset_type: str = "no_speed"
    randomize_friction: bool = True
    roll_friction_range: tuple[float, float] = (0.00006, 0.0001) if APPLY_RANDOMIZATION else (0.00005, 0.00005)
    friction_range: tuple[float, float] = (0.0, 0.5) if APPLY_RANDOMIZATION else (0.25, 0.25)
    kick_manipulation_type: str = "force" if KICK_MODEL == "default" else "velocity"
    kick_force_gain: float = 0.5
    kick_area: str = "default"
    kick_margin: float = 0.2


@dataclass
class SoccerAgentsCfg:
    friction_range: tuple[float, float] = (0.0, 0.2) if APPLY_RANDOMIZATION else (0.1, 0.1)
    translation_shape: str = "ellipse"
    position_control_pid_gain: list[float] = field(default_factory=lambda: [20.0, 0.2])
    velocity_control_pid_gain: list[float] = field(default_factory=lambda: [20.0, 0.2])
    velocity_p_gain: float = 3.0
    translation_force_d_gain: float = 10.0
    rotation_torque_d_gain: float = 0.2
    cmd_speed_limit: list[list[float]] = field(default_factory=lambda: [[0.35, -0.25], [0.25, -0.25], [2.5, -2.5]])


@dataclass
class SoccerGameCfg:
    num_teams: int = 2
    num_agents_per_team: int = 3
    num_actions: int = 5
    default_control: list[int] = field(default_factory=lambda: [1, 2])
    agent_id_control: list[list[int]] | None = None
    eval_env_ratio: float = 0.15
    num_agents_range: list[int] | None = None
    level_settings: list[list[int]] = field(default_factory=lambda: [[2, 2], [3, 2], [3, 3], [3, 4]])
    ball: SoccerBallCfg = field(default_factory=SoccerBallCfg)
    agents: SoccerAgentsCfg = field(default_factory=SoccerAgentsCfg)
    create_field_visuals: bool = False

    def __post_init__(self):
        if self.num_teams != 2:
            raise ValueError("Soccer requires exactly two teams.")
        if not 1 <= self.num_agents_per_team <= 11:
            raise ValueError("num_agents_per_team must be between 1 and 11.")
        if self.num_actions != 5:
            raise ValueError("Soccer agents require five actions.")
        if not 0 <= self.eval_env_ratio <= 1:
            raise ValueError("eval_env_ratio must be in [0, 1].")
        if self.num_agents_range is None:
            self.num_agents_range = [1, self.num_agents_per_team]
        if (
            len(self.num_agents_range) != 2
            or not 1 <= self.num_agents_range[0] <= self.num_agents_range[1] <= self.num_agents_per_team
        ):
            raise ValueError("num_agents_range must fit within num_agents_per_team.")
        if len(self.default_control) != 2 or any(control not in (0, 1, 2, 3) for control in self.default_control):
            raise ValueError("default_control must contain a control ID in [0, 3] for each team.")
        if self.agent_id_control is None:
            self.agent_id_control = [[control] * self.num_agents_per_team for control in self.default_control]
        if len(self.agent_id_control) != 2 or any(
            len(team) != self.num_agents_per_team or any(control not in (0, 1, 2, 3) for control in team)
            for team in self.agent_id_control
        ):
            raise ValueError("agent_id_control must contain one control ID in [0, 3] per player.")
        if not self.level_settings or any(
            len(level) != 2
            or not 0 <= level[0] < len(FIELD_LEVEL_SETTINGS)
            or not 0 <= level[1] < len(GOAL_LEVEL_SETTINGS)
            for level in self.level_settings
        ):
            raise ValueError("level_settings must contain valid field and goal level indices.")


class EnvCoreData:
    def __init__(self, cfg: SoccerGameCfg, num_envs, device):
        self.cfg = cfg
        self.num_envs = num_envs  # Denote als N
        self.num_teams = cfg.num_teams  # Always 2
        self.num_agents_per_team = cfg.num_agents_per_team  # Denote als N_T
        self.num_actions = cfg.num_actions

        self.num_agents_per_env = self.num_teams * self.num_agents_per_team
        self.num_agents = self.num_envs * self.num_agents_per_env

        self.device = device

        # common base shapes MARL data tensors.
        self.env_base_shape = (self.num_envs,)
        self.team_base_shape = (self.num_envs, self.num_teams)
        self.agent_base_shape = self.team_base_shape + (self.num_agents_per_team,)
        self.agent2agent_base_shape = self.agent_base_shape + (self.num_agents_per_team,)

        self.all_agent2agent_base_shape = (
            self.num_envs,
            self.num_agents_per_env,
            self.num_agents_per_env,
        )

        self.agent_pos_shape = self.agent_base_shape + (2,)
        self.agent_pose_shape = self.agent_base_shape + (3,)
        self.agent_quat_shape = self.agent_base_shape + (4,)

        self.env_flatten_base_shape = (self.num_envs * self.num_agents_per_env,)
        self.team_flatten_base_shape = (self.num_envs, self.num_agents_per_env)

        self.agent2agent_pose_shape = self.agent2agent_base_shape + (3,)

        self.env_flatten_pos_shape = self.env_flatten_base_shape + (2,)
        self.env_flatten_pose_shape = self.env_flatten_base_shape + (3,)
        self.env_flatten_quat_shape = self.env_flatten_base_shape + (4,)

        self.team_flatten_pos_shape = self.team_flatten_base_shape + (2,)
        self.team_flatten_pose_shape = self.team_flatten_base_shape + (3,)
        self.team_flatten_quat_shape = self.team_flatten_base_shape + (4,)

    def expand_to_env(self, tensor: Tensor, flatten_type: str | None = None):
        def expand(tensor: Tensor):
            if tensor.shape == self.env_base_shape:
                return (
                    tensor.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(
                        self.env_base_shape
                        + (
                            2,
                            self.num_agents_per_team,
                        )
                    )
                )
            elif tensor.shape == self.team_base_shape:
                return tensor.unsqueeze(-1).expand(self.team_base_shape + (self.num_agents_per_team,))
            else:
                raise ValueError(f"Cannot expand tensor shape {tuple(tensor.shape)} into agent dimensions.")

        if flatten_type == "env":
            return expand(tensor).flatten()
        elif flatten_type == "team":
            return expand(tensor).reshape(self.team_flatten_base_shape)
        elif flatten_type == None:
            return expand(tensor)

    def expand_vector_to_env(self, tensor: Tensor, flatten_type: str | None = None):
        def expand_vector(tensor: Tensor):
            if tensor.shape[:-1] == self.env_base_shape:
                return (
                    tensor.unsqueeze(-2)
                    .unsqueeze(-2)
                    .expand(
                        self.env_base_shape
                        + (
                            2,
                            self.num_agents_per_team,
                            tensor.shape[-1],
                        )
                    )
                )
            elif tensor.shape[:-1] == self.team_base_shape:
                return tensor.unsqueeze(-2).expand(
                    self.team_base_shape
                    + (
                        self.num_agents_per_team,
                        tensor.shape[-1],
                    )
                )
            else:
                raise ValueError(f"Cannot expand tensor shape {tuple(tensor.shape)} into agent dimensions.")

        if flatten_type == "env":
            return expand_vector(tensor).reshape(self.env_flatten_base_shape + (tensor.shape[-1],))
        elif flatten_type == "team":
            return expand_vector(tensor).reshape(self.team_flatten_base_shape + (tensor.shape[-1],))
        elif flatten_type == None:
            return expand_vector(tensor)

    def combine_index(self, env_ids=None, flag=None, return_index=True):
        if env_ids is None:
            env_ids = flag
        else:
            env_ids_expand = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            env_ids_expand[env_ids] = True
            env_ids = torch.logical_and(flag, env_ids_expand)
        if return_index:
            return env_ids.clone().nonzero(as_tuple=False).flatten()
        else:
            return env_ids.clone()


class EnvData(EnvCoreData):
    field_curriculum_level: Tensor = None
    env_origins: Tensor = None

    def __init__(self, cfg: SoccerGameCfg, num_envs, device):
        super().__init__(cfg, num_envs, device)

        self.num_eval_envs = int(self.num_envs * self.cfg.eval_env_ratio)
        self.team_size_curriculum_level = 0
        self.team_size_range = [
            [
                [self.cfg.num_agents_range[0], self.cfg.num_agents_range[1]],
                [self.cfg.num_agents_range[0], self.cfg.num_agents_range[1]],
            ]
        ]

        self.agent_ids_control = torch.tensor(self.cfg.agent_id_control, device=self.device, dtype=torch.int32)
        self.num_training_agents_per_env = torch.sum(self.agent_ids_control.flatten() == 1).item()
        self.num_policy_replay_agents_per_env = torch.sum(self.agent_ids_control.flatten() == 2).item()
        self.num_bots_per_env = torch.sum(self.agent_ids_control.flatten() == 3).item()

        self.field_width = 3.6
        self.init_tensors()

    def init_tensors(self):
        # Env base shaped tensors (N,)
        self.is_training_env = torch.zeros(self.env_base_shape, device=self.device, dtype=bool)

        self.goal_status = torch.zeros_like(self.is_training_env, dtype=torch.int32)

        self.field_length = torch.zeros_like(self.is_training_env, dtype=torch.float32)
        self.goal_width = torch.zeros_like(self.is_training_env, dtype=torch.float32)

        # (N, 2,)
        self.goal_post_pos = torch.zeros(self.env_base_shape + (2,), device=self.device, dtype=torch.float32)

        # Team base shaped tensors (N, 2)
        self.score = torch.zeros(self.team_base_shape, device=self.device, dtype=torch.int32)
        self.accumulated_score = torch.zeros_like(self.score)
        self.active_agent_per_team = torch.zeros_like(self.score)

        # Agent base shaped tensors (N, 2, N_T,)
        self.active = torch.zeros(self.agent_base_shape, device=self.device, dtype=torch.bool)
        self.inactive = torch.zeros_like(self.active)
        self.agent_control_type = torch.zeros_like(self.active, dtype=torch.int32)

        # Env flattened base shape (N * 2 * N_T,)
        self.training_actor_index = torch.zeros(self.env_flatten_base_shape, device=self.device, dtype=torch.int32)
        self.eval_actor_index = torch.zeros_like(self.training_actor_index)

        # The tensors are filled with initial value
        shuffled_env_index = torch.randperm(self.num_envs, device=self.device, dtype=torch.int32)
        if self.num_eval_envs == 0:
            self.is_training_env[:] = True
        else:
            self.is_training_env[shuffled_env_index[: -self.num_eval_envs]] = True
        self.is_eval_env = torch.logical_not(self.is_training_env)

        self.agent_control_type[:] = self.agent_ids_control

        self.regenerate_level_config(self.is_training_env)
        self.active_agent_per_team[self.is_eval_env] = torch.randint(
            low=self.cfg.num_agents_range[0],
            high=self.cfg.num_agents_range[1] + 1,
            size=(len(self.is_eval_env.nonzero()), 2),
            device=self.device,
            dtype=torch.int32,
        )
        self.regenerate_mask_index()

    def set_env_origin(self, env_origins):
        self.env_origins = env_origins

    def step(self, goal_status):
        self.goal_status[:] = goal_status
        self.score[goal_status == 1, 0] += 1
        self.score[goal_status == -1, 1] += 1

    def set_field_curriculum_level(self, field_curriculum_level):
        self.field_curriculum_level = field_curriculum_level

    def update_field(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        settings = torch.tensor(self.cfg.level_settings, device=self.device, dtype=torch.long)
        levels = settings[self.field_curriculum_level[env_ids]]
        field_lengths = torch.tensor(FIELD_LEVEL_SETTINGS, device=self.device, dtype=torch.float32)
        goal_widths = torch.tensor(GOAL_LEVEL_SETTINGS, device=self.device, dtype=torch.float32)
        self.field_length[env_ids] = field_lengths[levels[:, 0]]
        self.goal_width[env_ids] = goal_widths[levels[:, 1]]
        self.goal_post_pos[env_ids, 0] = self.field_length[env_ids] / 2
        self.goal_post_pos[env_ids, 1] = self.goal_width[env_ids] / 2

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = ...

        self.update_field(env_ids)
        self.goal_status[env_ids] = 0
        self.accumulated_score[env_ids, :] += self.score[env_ids, :]
        self.score[env_ids, :] = 0

    def regenerate_level_config(self, env_ids: Tensor = None):
        num_regenerate_env = self.num_envs
        if env_ids is None:
            env_ids = ...
        else:
            num_regenerate_env = int(env_ids.sum().item()) if env_ids.dtype == torch.bool else len(env_ids)

        current_settings = self.team_size_range[min(self.team_size_curriculum_level, len(self.team_size_range) - 1)]
        for i in range(self.num_teams):
            self.active_agent_per_team[env_ids, i] = torch.randint(
                low=current_settings[i][0],
                high=current_settings[i][1] + 1,
                size=(num_regenerate_env,),
                device=self.device,
                dtype=torch.int32,
            )

    def regenerate_mask_index(self):
        for i in range(self.num_agents_per_team):
            self.active[..., i] = self.active_agent_per_team > i
        self.inactive = torch.logical_not(self.active)
        active_train = torch.logical_and(
            self.active.flatten(),
            self.expand_to_env(self.is_training_env, flatten_type="env"),
        )
        active_eval = torch.logical_and(
            self.active.flatten(),
            self.expand_to_env(self.is_eval_env, flatten_type="env"),
        )

        # Env flattened base shape (N * 2 * N_T,)
        self.training_actor_index = torch.logical_and(self.agent_control_type.flatten() == 1, active_train)
        self.eval_actor_index = torch.logical_and(self.agent_control_type.flatten() == 1, active_eval)
        self.policy_replay_actor_index = torch.logical_and(self.agent_control_type.flatten() == 2, active_train)
        self.bots_actor_index = torch.logical_or(
            torch.logical_and(self.agent_control_type.flatten() == 3, active_train),
            torch.logical_and(self.agent_control_type.flatten() != 1, active_eval),
        )
