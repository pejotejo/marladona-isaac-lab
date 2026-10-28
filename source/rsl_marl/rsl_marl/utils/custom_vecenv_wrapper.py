# Copyright 2025 Zichong Li, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

import torch


class CustomVecEnvWrapper:
    """Keep MARLadona's flattened actor interface around an mjlab environment."""

    def __init__(self, env, on_step=None):
        self.env = env
        self.on_step = on_step
        self.num_envs = self.unwrapped.num_envs
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self.device = self.unwrapped.device
        self.cfg = self.unwrapped.cfg
        self.max_episode_length = self.unwrapped.max_episode_length
        self.env.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf[:] = value

    def get_observations(self):
        return self.unwrapped.obs_buf

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        extras["observations"] = obs_dict
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        if self.on_step is not None:
            self.on_step()
        return obs_dict, rew, dones, extras

    def close(self):
        self.env.close()
