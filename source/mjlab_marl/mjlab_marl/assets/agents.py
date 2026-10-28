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

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api import math as math_utils
from torch import Tensor

from ..utils.math_utils import transform2dPos
from .env_data import EnvCoreData, EnvData

if TYPE_CHECKING:
    from mjlab.entity import Entity

    from .env_data import SoccerGameCfg


INACTIVE_AGENT_VALUE = float("nan")


class Agents(EnvCoreData):
    base_pose_w: Tensor = None

    base_lin_vel_w: Tensor = None
    base_lin_vel_b: Tensor = None

    base_ang_vel_w: Tensor = None

    asset: list[Entity]

    def __init__(self, cfg: SoccerGameCfg, env_data: EnvData, num_envs, device, asset):
        super().__init__(cfg, num_envs, device)
        self.agent_cfg = cfg.agents
        self.asset = asset
        self.env_data = env_data
        self.init_tensors()

    def init_tensors(self):
        # Agent base shaped tensors (N, 2, N_T,)
        # pose: base shape + (x, y, rot_z)
        self.base_pose_w = torch.zeros(self.agent_pose_shape, device=self.device, dtype=torch.float32)
        self.base_default_pose_w = torch.zeros_like(self.base_pose_w)

        self.constraint_state_indexes = torch.tensor([2, 4, 5, 9, 10, 11], device=self.device)
        self.constraint_state_values = torch.tensor([0.08, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)

        self.ball_pos_b = torch.zeros(self.agent_pos_shape, device=self.device, dtype=torch.float32)
        # Team base shaped tensors (N, 2 * N_T,)
        self.base_lin_vel_b = torch.zeros(self.team_flatten_pos_shape, device=self.device)
        self.base_lin_vel_w = torch.zeros_like(self.base_lin_vel_b)
        self.base_ang_vel_w = torch.zeros(self.team_flatten_base_shape, device=self.device)

        # Agent2agent base shaped tensors (N, 2, N_T, N_T,)
        self.closest_teammate_indexes = torch.zeros(self.agent2agent_base_shape, device=self.device, dtype=torch.int32)
        self.closest_opponent_indexes = torch.zeros_like(self.closest_teammate_indexes)

        self.closest_teammate_pose_w = torch.zeros(self.agent2agent_pose_shape, device=self.device, dtype=torch.float32)
        self.closest_opponent_pose_w = torch.zeros_like(self.closest_teammate_pose_w)

    def set_agents_default_pos(self, blue_default_pose, red_default_pose):
        self.base_default_pose_w[:, 0, ...] = blue_default_pose
        self.base_default_pose_w[:, 1, ...] = red_default_pose

    def step(self, enforce_constraints=True):
        agents_state = torch.stack(
            [torch.cat((asset.data.root_link_pose_w, asset.data.root_link_vel_w), dim=-1) for asset in self.asset],
            dim=1,
        )
        if enforce_constraints:
            agents_state[..., self.constraint_state_indexes] = self.constraint_state_values
            for i, asset in enumerate(self.asset):
                asset.write_root_state_to_sim(agents_state[:, i])
        self.base_quat_w = agents_state[..., 3:7]

        self.base_pose_w[..., :2] = (
            agents_state[..., :2].view(self.agent_pos_shape)
            - self.expand_vector_to_env(self.env_data.env_origins)[..., :2]
        )
        self.base_pose_w[..., 2] = math_utils.euler_xyz_from_quat(
            agents_state[..., 3:7].view(self.env_flatten_quat_shape)
        )[2].view(self.agent_base_shape)

        self.base_lin_vel_w = agents_state[..., 7:9]
        self.base_lin_vel_b = math_utils.quat_apply_inverse(
            agents_state[..., 3:7].reshape(-1, 4), agents_state[..., 7:10].reshape(-1, 3)
        ).reshape(self.num_envs, self.num_agents_per_env, 3)[..., :2]
        self.base_ang_vel_w = agents_state[..., 12]

        self.compute_closest_agents()

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)
            num_reset_env = self.num_envs
        else:
            num_reset_env = len(env_ids)
        base_default_pose = self.base_default_pose_w.reshape(self.team_flatten_pose_shape)[env_ids].clone()
        base_default_pose[..., 0] *= self.expand_to_env(self.env_data.field_length, flatten_type="team")[env_ids] / 2
        base_default_pose[..., 1] *= self.env_data.field_width / 2
        base_default_pose[..., 2] += torch.pi
        # poses
        inactive = self.env_data.inactive[env_ids]
        if len(inactive.nonzero(as_tuple=False)) > 0:
            inactive = inactive.view(num_reset_env, -1)
            inactive_base_default_pose = base_default_pose.clone()
            inactive_base_default_pose[..., 2] = torch.pi / 2
            inactive_base_default_pose[..., 1] = -2.7
            inactive_base_default_pose[..., 0] = (
                torch.cat(
                    [
                        torch.arange(self.num_agents_per_team, device=self.device) * 0.5,
                        -torch.arange(self.num_agents_per_team, device=self.device) * 0.5,
                    ]
                )
                .unsqueeze(0)
                .repeat(num_reset_env, 1)
            )
            base_default_pose[inactive] = inactive_base_default_pose[inactive]

        positions = self.expand_vector_to_env(self.env_data.env_origins, flatten_type="team")[env_ids].clone()
        positions[..., :2] += base_default_pose[..., :2]
        positions[..., 2] = 0.08
        orientations = math_utils.quat_from_euler_xyz(
            torch.zeros_like(base_default_pose[..., 2]),
            torch.zeros_like(base_default_pose[..., 2]),
            base_default_pose[..., 2],
        )

        # set into the physics simulation
        for i, asset in enumerate(self.asset):
            asset.write_root_link_pose_to_sim(torch.cat([positions[:, i], orientations[:, i]], dim=-1), env_ids=env_ids)
            asset.write_root_link_velocity_to_sim(torch.zeros(num_reset_env, 6, device=self.device), env_ids=env_ids)

    def compute_ego_ball_state(self, ball_pos_w):
        self.ball_pos_b = transform2dPos(
            self.base_pose_w.view(self.env_flatten_pose_shape),
            self.expand_vector_to_env(ball_pos_w, flatten_type="env"),
        )

    def compute_closest_agents(self):
        augmented_base_pose_w = self.base_pose_w.clone()
        augmented_base_pose_w[self.env_data.inactive] = INACTIVE_AGENT_VALUE
        self.agent2agent_distance = torch.cdist(
            augmented_base_pose_w[..., :2].view(self.team_flatten_pos_shape),
            augmented_base_pose_w[..., :2].view(self.team_flatten_pos_shape),
            p=2,
        )
        self.agent2agent_distance.diagonal(dim1=-2, dim2=-1).fill_(INACTIVE_AGENT_VALUE)
        self.agent2agent_distance.nan_to_num_(nan=float("inf"))

        self.closest_teammate_indexes[:, 0, ...] = torch.argsort(
            self.agent2agent_distance[:, : self.num_agents_per_team, : self.num_agents_per_team],
            dim=-1,
        )
        self.closest_teammate_indexes[:, 1, ...] = torch.argsort(
            self.agent2agent_distance[:, self.num_agents_per_team :, self.num_agents_per_team :],
            dim=-1,
        )
        self.closest_opponent_indexes[:, 0, ...] = torch.argsort(
            self.agent2agent_distance[:, : self.num_agents_per_team, self.num_agents_per_team :],
            dim=-1,
        )
        self.closest_opponent_indexes[:, 1, ...] = torch.argsort(
            self.agent2agent_distance[:, self.num_agents_per_team :, : self.num_agents_per_team],
            dim=-1,
        )

        teammate_pose_w = augmented_base_pose_w.unsqueeze(-3).expand(self.agent2agent_pose_shape)
        opponent_pose_w = torch.zeros_like(teammate_pose_w)
        opponent_pose_w[:, 0, ...] = teammate_pose_w[:, 1, ...]
        opponent_pose_w[:, 1, ...] = teammate_pose_w[:, 0, ...]

        self.closest_teammate_pose_w = teammate_pose_w[
            torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            torch.arange(self.num_teams).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            torch.arange(self.num_agents_per_team).unsqueeze(0).unsqueeze(0).unsqueeze(-1),
            self.closest_teammate_indexes,
        ]
        self.closest_opponent_pose_w = opponent_pose_w[
            torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            torch.arange(self.num_teams).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            torch.arange(self.num_agents_per_team).unsqueeze(0).unsqueeze(0).unsqueeze(-1),
            self.closest_opponent_indexes,
        ]

        self.closest_teammate_pose_w[self.env_data.inactive] = INACTIVE_AGENT_VALUE
        self.closest_opponent_pose_w[self.env_data.inactive] = INACTIVE_AGENT_VALUE
