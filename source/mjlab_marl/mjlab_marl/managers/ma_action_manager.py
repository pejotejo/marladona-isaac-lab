# Copyright 2025 Zichong Li, ETH Zurich
# Licensed under the Apache License, Version 2.0.
"""The original five MARLadona controls, applied as native mjlab body wrenches."""

from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionManager, ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply

from mjlab_marl.utils.math_utils import rotate2d


@dataclass(kw_only=True)
class BaseMovementActionCfg(ActionTermCfg):
    def build(self, env):
        return BaseMovementAction(self, env)


@dataclass(kw_only=True)
class BallControlActionCfg(ActionTermCfg):
    def build(self, env):
        return BallControlAction(self, env)


class SoccerActionTerm(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(
            env.num_envs, env.env_data.num_agents_per_env, self.action_dim, device=env.device
        )

    @property
    def raw_action(self):
        return self._raw_actions

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._raw_actions

    def process_actions(self, actions):
        self._raw_actions.copy_(actions.reshape_as(self._raw_actions))

    def reset(self, env_ids=None):
        self._raw_actions[slice(None) if env_ids is None else env_ids] = 0


class BaseMovementAction(SoccerActionTerm):
    action_dim = 3

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.agents = env.agents
        self.agent_cfg = env.cfg.soccer_game.agents
        self.target_velocity_cmd = torch.zeros_like(self.raw_action)
        self._force = torch.zeros_like(self.raw_action)
        self._torque = torch.zeros_like(self.raw_action)

    def compute(self):
        original = self.raw_action[..., :2].clamp(-1, 1)
        target = original * torch.sqrt(1 - original.flip(-1).square() / 2)
        limits = self.agent_cfg.cmd_speed_limit
        target[..., 0] *= torch.where(target[..., 0] > 0, limits[0][0], -limits[0][1])
        target[..., 1] *= limits[1][0]
        rotation = self.raw_action[..., 2].clamp(-1, 1) * limits[2][0]
        self.target_velocity_cmd[..., :2] = target
        self.target_velocity_cmd[..., 2] = rotation
        self._force[..., :2] = (target - self.agents.base_lin_vel_b) * self.agent_cfg.translation_force_d_gain
        self._torque[..., 2] = (rotation - self.agents.base_ang_vel_w) * self.agent_cfg.rotation_torque_d_gain
        inactive = self._env.env_data.inactive.reshape(self.num_envs, -1)
        self._force[inactive] = 0
        self._torque[inactive] = 0

    def apply_actions(self):
        self.compute()
        for i, asset in enumerate(self.agents.asset):
            force = quat_apply(self.agents.base_quat_w[:, i], self._force[:, i])
            asset.write_external_wrench_to_sim(force.unsqueeze(1), self._torque[:, i].unsqueeze(1))


class BallControlAction(SoccerActionTerm):
    action_dim = 2

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ball = env.ball
        self.ball_cfg = env.cfg.soccer_game.ball
        self.agents = env.agents
        self.total_target_cmd_w = torch.zeros(env.num_envs, 3, device=env.device)
        self._force = torch.zeros_like(self.total_target_cmd_w)
        self._torque = torch.zeros_like(self.total_target_cmd_w)

    def compute(self):
        data = self._env.env_data
        actions = self.raw_action
        target = actions * torch.sqrt(1 - actions.flip(-1).square() / 2)
        target[..., 0] = (target[..., 0] + 0.01) / 2
        target_w = rotate2d(
            target[..., 0].flatten(), target[..., 1].flatten(), -self.agents.base_pose_w[..., 2].flatten()
        )
        pos_b = self.agents.ball_pos_b
        controllable = (
            (pos_b.norm(dim=-1) < self.ball_cfg.kick_margin)
            & (pos_b[:, 0] > 0)
            & (pos_b[:, 1].abs() < 0.1)
            & (data.expand_to_env(self._entity.data.root_link_pos_w[:, 2], flatten_type="env") < 0.08)
            & data.active.flatten()
        )
        force = (
            (target_w - data.expand_vector_to_env(self.ball.base_lin_vel_w, flatten_type="env"))
            * self.ball_cfg.kick_force_gain
            * controllable.unsqueeze(-1)
        )
        self.total_target_cmd_w[:, :2] = force.reshape(data.team_flatten_pos_shape).sum(dim=1)
        self._force.copy_(self.total_target_cmd_w)
        self._torque.copy_(-self._entity.data.root_link_ang_vel_w * self.ball.rolling_friction)

    def apply_actions(self):
        self.compute()
        self._entity.write_external_wrench_to_sim(self._force.unsqueeze(1), self._torque.unsqueeze(1))


class MultiAgentActionManager(ActionManager):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._action = torch.zeros(env.env_data.num_agents, self.total_action_dim, device=env.device)
        self._prev_action = torch.zeros_like(self._action)
        self._prev_prev_action = torch.zeros_like(self._action)

    def process_action(self, action):
        if action.shape != self._action.shape:
            raise ValueError(f"Expected actions of shape {tuple(self._action.shape)}, got {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            raise ValueError("Actions must be finite")
        super().process_action(action.clamp(-1, 1))

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = self._env.env_data.num_agents_per_env
        rows = (env_ids[:, None] * n + torch.arange(n, device=self.device)).flatten()
        self._action[rows] = 0
        self._prev_action[rows] = 0
        self._prev_prev_action[rows] = 0
        for term in self._terms.values():
            term.reset(env_ids)
        return {}
