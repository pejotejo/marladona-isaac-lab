# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) 2025 Zichong Li, ETH Zurich

"""mjlab observations with MARLadona's per-player batching and feature-major history."""

from math import prod

import torch
from mjlab.managers.observation_manager import ObservationManager
from mjlab.utils.buffers import CircularBuffer, DelayBuffer
from mjlab.utils.noise import NoiseCfg, NoiseModelCfg


class MultiAgentObservationManager(ObservationManager):
    def __init__(self, cfg, env):
        super().__init__(cfg if isinstance(cfg, dict) else vars(cfg), env)

    @property
    def num_agents_per_env(self):
        return self._env.env_data.num_agents_per_env

    def _batch_ids(self, env_ids, batch_size):
        if env_ids is None:
            return None
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if batch_size == self.num_envs:
            return ids
        if batch_size != self.num_envs * self.num_agents_per_env:
            raise ValueError(f"Unexpected observation batch size {batch_size}.")
        return (
            ids[:, None] * self.num_agents_per_env + torch.arange(self.num_agents_per_env, device=self.device)
        ).flatten()

    def _prepare_terms(self):
        super()._prepare_terms()
        self._group_symmetry_index = {}
        self._group_obs_slice_pos = {}
        self._group_term_batch_sizes = {}
        for group_name, names in self._group_obs_term_names.items():
            self._group_symmetry_index[group_name] = []
            self._group_obs_slice_pos[group_name] = {}
            self._group_term_batch_sizes[group_name] = {}
            offset = 0
            for term_name, term_cfg in zip(names, self._group_obs_term_cfgs[group_name]):
                obs = term_cfg.func(self._env, **term_cfg.params)
                batch_size = obs.shape[0]
                self._group_term_batch_sizes[group_name][term_name] = batch_size
                hist = term_cfg.history_length or 1
                dim = prod(obs.shape[1:])
                if group_name in (
                    "policy",
                    "critic",
                    "neighbor",
                    "neighbor_critic",
                ) and any(word in term_name for word in ("pose", "vel", "pos")):
                    unit = 2 if "ball" in term_name else (4 if term_cfg.params.get("expand_rot") else 3)
                    for i in range(dim // unit):
                        for j in range(hist):
                            self._group_symmetry_index[group_name].append(offset + i * unit * hist + j + hist)
                            if "ball" not in term_name:
                                self._group_symmetry_index[group_name].append(offset + i * unit * hist + j + 2 * hist)
                self._group_obs_slice_pos[group_name][term_name] = [
                    offset,
                    offset + dim * hist,
                ]
                offset += dim * hist
                if term_cfg.history_length:
                    self._group_obs_term_history_buffer[group_name][term_name] = CircularBuffer(
                        max_len=term_cfg.history_length,
                        batch_size=batch_size,
                        device=self.device,
                    )
                if term_cfg.delay_max_lag:
                    self._group_obs_term_delay_buffer[group_name][term_name] = DelayBuffer(
                        min_lag=term_cfg.delay_min_lag,
                        max_lag=term_cfg.delay_max_lag,
                        batch_size=batch_size,
                        device=self.device,
                        per_env=term_cfg.delay_per_env,
                        hold_prob=term_cfg.delay_hold_prob,
                        update_period=term_cfg.delay_update_period,
                        per_env_phase=term_cfg.delay_per_env_phase,
                    )
                if isinstance(term_cfg.noise, NoiseModelCfg):
                    self._group_obs_class_instances[group_name][term_name] = term_cfg.noise.class_type(
                        term_cfg.noise, num_envs=batch_size, device=self.device
                    )

    def compute_group(self, group_name, update_history=False, env_ids=None):
        group_cfg = self.cfg[group_name]
        values = {}
        for term_name, term_cfg in zip(
            self._group_obs_term_names[group_name],
            self._group_obs_term_cfgs[group_name],
        ):
            obs = term_cfg.func(self._env, **term_cfg.params).clone()
            batch_ids = self._batch_ids(env_ids, obs.shape[0])
            if isinstance(term_cfg.noise, NoiseCfg):
                obs = term_cfg.noise.apply(obs)
            elif isinstance(term_cfg.noise, NoiseModelCfg):
                obs = self._group_obs_class_instances[group_name][term_name](obs)
            if term_cfg.clip:
                obs = obs.clamp_(*term_cfg.clip)
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)
            if group_cfg.nan_check_per_term:
                obs = self._check_and_handle_nans(obs, f"{group_name}/{term_name}", group_cfg.nan_policy)
            if term_cfg.delay_max_lag:
                buffer = self._group_obs_term_delay_buffer[group_name][term_name]
                if batch_ids is None or not buffer.is_initialized:
                    buffer.append(obs)
                    obs = buffer.compute()
                else:
                    buffer.backfill(obs, batch_ids)
                    obs = buffer.peek()
            if term_cfg.history_length:
                buffer = self._group_obs_term_history_buffer[group_name][term_name]
                if batch_ids is None or not buffer.is_initialized:
                    if update_history or not buffer.is_initialized:
                        buffer.append(obs)
                else:
                    buffer.backfill(obs, batch_ids)
                history = buffer.buffer
                # Pretrained MARLadona weights expect [x_old, x_new, y_old, y_new, ...].
                values[term_name] = (
                    history.transpose(-1, -2).reshape(obs.shape[0], -1) if term_cfg.flatten_history_dim else history
                )
            else:
                values[term_name] = obs
        if self._group_obs_concatenate[group_name]:
            result = torch.cat(list(values.values()), dim=self._group_obs_concatenate_dim[group_name])
            return (
                self._check_and_handle_nans(result, group_name, group_cfg.nan_policy)
                if not group_cfg.nan_check_per_term
                else result
            )
        if not group_cfg.nan_check_per_term:
            values = {
                name: self._check_and_handle_nans(value, f"{group_name}/{name}", group_cfg.nan_policy)
                for name, value in values.items()
            }
        return values

    def reset(self, env_ids=None):
        self._obs_buffer = None
        for group_name, term_cfgs in self._group_obs_class_term_cfgs.items():
            for term_cfg in term_cfgs:
                term_cfg.func.reset(env_ids=env_ids)
        for group_name, names in self._group_obs_term_names.items():
            for term_name in names:
                batch_ids = self._batch_ids(env_ids, self._group_term_batch_sizes[group_name][term_name])
                for buffers in (
                    self._group_obs_term_history_buffer,
                    self._group_obs_term_delay_buffer,
                ):
                    if term_name in buffers[group_name]:
                        buffers[group_name][term_name].reset(batch_ids=batch_ids)
                if term_name in self._group_obs_class_instances[group_name]:
                    self._group_obs_class_instances[group_name][term_name].reset(env_ids=batch_ids)
        return {}
