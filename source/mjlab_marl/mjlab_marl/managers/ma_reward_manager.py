# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) 2025 Zichong Li, ETH Zurich

"""mjlab reward aggregation with one reward per player."""

import torch
from mjlab.managers.reward_manager import RewardManager


class MultiAgentRewardManager(RewardManager):
    def __init__(self, cfg, env, *, scale_by_dt=True):
        super().__init__(cfg if isinstance(cfg, dict) else vars(cfg), env, scale_by_dt=scale_by_dt)
        self.performance_metrics = {"average_score": 0.0}
        self._episode_sums = {
            name: torch.zeros(self.num_envs * self.num_agents_per_env, device=self.device) for name in self._term_names
        }
        self._reward_buf = torch.zeros(self.num_envs * self.num_agents_per_env, device=self.device)
        self._step_reward = torch.zeros(
            self.num_envs * self.num_agents_per_env,
            len(self._term_names),
            device=self.device,
        )

    @property
    def num_agents_per_env(self):
        return self._env.env_data.num_agents_per_env

    def _check_term_shape(self, term_name, value):
        expected = (self.num_envs * self.num_agents_per_env,)
        if value.shape != expected:
            raise ValueError(f"Reward term {term_name!r} returned {tuple(value.shape)}, expected {expected}.")

    def compute(self, dt):
        for term_cfg in self._term_cfgs:
            threshold = getattr(term_cfg, "decay_threshhold", None)
            if threshold is not None and threshold < self.performance_metrics["average_score"]:
                term_cfg.weight = 0.0
        return super().compute(dt)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is not None:
            ids = ids[env_ids]
        agent_ids = (
            ids[:, None] * self.num_agents_per_env + torch.arange(self.num_agents_per_env, device=self.device)
        ).flatten()
        extras = {}
        if agent_ids.numel():
            for name, values in self._episode_sums.items():
                extras[f"Episode_Reward/{name}"] = values[agent_ids].mean() / self._env.max_episode_length_s
                values[agent_ids] = 0.0
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return extras
