# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) 2025 Zichong Li, ETH Zurich
"""Vectorized MARLadona soccer using mjlab's native scene and MuJoCo-Warp simulation."""

from functools import partial

import gymnasium as gym
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import NullCommandManager
from mjlab.managers.curriculum_manager import CurriculumManager
from mjlab.managers.event_manager import EventManager
from mjlab.managers.recorder_manager import NullRecorderManager
from mjlab.managers.termination_manager import TerminationManager

from mjlab_marl.assets.agents import Agents
from mjlab_marl.assets.env_data import EnvData
from mjlab_marl.assets.scene import FieldCurriculum, build_entities, field_spec
from mjlab_marl.assets.soccer_ball import SoccerBall
from mjlab_marl.managers.init_pos_manager import InitPosManager
from mjlab_marl.managers.ma_action_manager import MultiAgentActionManager
from mjlab_marl.managers.ma_observation_manager import MultiAgentObservationManager
from mjlab_marl.managers.ma_reward_manager import MultiAgentRewardManager
from mjlab_marl.tasks.soccer.soccer_marl_env_cfg import SoccerMARLEnvCfg


class SoccerMARLEnv(ManagerBasedRlEnv, gym.Env):
    def __init__(self, cfg=None, device="cuda:0", render_mode=None, **kwargs):
        cfg_type = kwargs.pop("env_cfg_entry_point", SoccerMARLEnvCfg)
        if cfg is None:
            cfg = cfg_type()
        cfg.soccer_game.__post_init__()
        if cfg.scene.num_envs < 1 or cfg.decimation < 1 or cfg.episode_length_s <= 0 or cfg.sim.mujoco.timestep <= 0:
            raise ValueError("Environment count, decimation, episode length and physics timestep must be positive")
        if cfg.recorders or cfg.metrics or cfg.commands:
            raise ValueError(
                "Soccer uses its custom multi-agent logger and does not support additional mjlab recorders, metrics or commands"
            )
        cfg.scene.entities = build_entities(cfg.soccer_game)
        cfg.scene.spec_fn = partial(
            field_spec,
            level_settings=cfg.soccer_game.level_settings,
            create_visuals=cfg.soccer_game.create_field_visuals,
        )
        super().__init__(cfg=cfg, device=device, render_mode=render_mode, **kwargs)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def load_managers(self):
        self.field = FieldCurriculum(self)
        self.env_data = EnvData(self.cfg.soccer_game, self.num_envs, self.device)
        self.env_data.set_field_curriculum_level(self.field.terrain_levels)
        self.env_data.set_env_origin(self.scene.env_origins)
        self.env_data.reset()
        assets = [
            self.scene[f"agent_{team}_{i}"]
            for team in ("blue", "red")
            for i in range(self.cfg.soccer_game.num_agents_per_team)
        ]
        self.agents = Agents(self.cfg.soccer_game, self.env_data, self.num_envs, self.device, assets)
        self.ball = SoccerBall(self.cfg.soccer_game, self.env_data, self.num_envs, self.device, self.scene["ball"])
        self.init_pos_manager = InitPosManager(self.num_envs, self.env_data.num_agents_per_team, self.device)
        self._refresh_states(check_location=True)
        self.event_manager = EventManager(self.cfg.events, self)
        self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)
        self.command_manager = NullCommandManager()
        self.recorder_manager = NullRecorderManager()
        self.action_manager = MultiAgentActionManager(self.cfg.actions, self)
        self.observation_manager = MultiAgentObservationManager(self.cfg.observations, self)
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        self.reward_manager = MultiAgentRewardManager(self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        self._configure_gym_env_spaces()
        self._randomize_ball_friction()
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def _randomize_ball_friction(self):
        # PhysX's multiply rule with ground friction 1 is the ball's sampled coefficient.
        self.sim.expand_model_fields(("geom_friction",))
        buckets = torch.empty(64, device=self.device).uniform_(0.1, 0.5)
        ids = torch.randint(64, (self.num_envs,), device=self.device)
        geom_id = self.sim.mj_model.geom("ball/collision").id
        self.sim.model.geom_friction[:, geom_id, 0] = buckets[ids]

    def _refresh_states(self, check_location=False):
        self.agents.step(enforce_constraints=False)
        self.ball.step(check_location=check_location)
        self.ball.compute_ball2agent_states(self.agents.base_pose_w)
        self.agents.compute_ego_ball_state(self.ball.base_pos_w)

    def reset(self, *, seed=None, env_ids=None, options=None):
        del options
        if seed is not None:
            self.seed(seed)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.extras["log"] = {}
        self._reset_idx(env_ids)
        self._manual_reset_pending[env_ids] = False
        self.scene.write_data_to_sim()
        self.sim.forward()
        self._refresh_states(check_location=True)
        self.obs_buf = self.observation_manager.compute(update_history=True, env_ids=env_ids)
        return self.obs_buf, self.extras

    def step(self, action):
        if not self.cfg.auto_reset and self._manual_reset_pending.any():
            raise RuntimeError("Reset terminated matches with reset(env_ids=...) before stepping again")
        self.extras["log"] = {}
        self.action_manager.process_action(action.to(self.device))
        # Latch events across all physics substeps, then count each goal once.
        goals = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        outside = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.agents.step()
            self.ball.step()
            goals = torch.where(goals != 0, goals, self.ball.goal_status.int())
            outside |= self.ball.ball_reset
            self.ball.compute_ball2agent_states(self.agents.base_pose_w)
            self.agents.compute_ego_ball_state(self.ball.base_pos_w)
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.sim.forward()
            self.scene.update(dt=self.physics_dt)
            self._refresh_states(check_location=True)
            goals = torch.where(goals != 0, goals, self.ball.goal_status.int())
            outside |= self.ball.ball_reset
        self.ball.goal_status = goals
        self.ball.ball_inside_field &= ~outside
        self.env_data.step(goals)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        terminated = self.env_data.expand_to_env(self.termination_manager.terminated, flatten_type="env").clone()
        truncated = self.env_data.expand_to_env(self.termination_manager.time_outs, flatten_type="env").clone()
        reward = self.reward_manager.compute(dt=self.step_dt)
        reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.cfg.auto_reset and reset_ids.numel():
            self._reset_idx(reset_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self._refresh_states(check_location=True)
        elif not self.cfg.auto_reset:
            self._manual_reset_pending.copy_(self.reset_buf)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.reset_terminated, self.reset_time_outs, self.reward_buf = terminated, truncated, reward
        return self.obs_buf, reward, terminated, truncated, self.extras

    def _reset_idx(self, env_ids):
        training = self.env_data.combine_index(env_ids, self.env_data.is_training_env)
        self.curriculum_manager.compute(env_ids=training)
        self.init_pos_manager.re_sample_init_pos(env_ids=training)
        scores = self.env_data.score.clone()
        self.env_data.reset(env_ids)
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        self.agents.set_agents_default_pos(
            self.init_pos_manager.blue_default_pos, self.init_pos_manager.red_default_pos
        )
        self.ball.set_ball_default_pos(self.init_pos_manager.ball_default_pos)
        self.agents.reset(env_ids)
        self.ball.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=self._sim_step_counter // self.cfg.decimation
            )
        for manager in (
            self.observation_manager,
            self.action_manager,
            self.reward_manager,
            self.curriculum_manager,
            self.command_manager,
            self.event_manager,
            self.termination_manager,
        ):
            self.extras["log"].update(manager.reset(env_ids))
        self.extras["score"] = scores
        self.episode_length_buf[env_ids] = 0

    def _configure_gym_env_spaces(self):
        self.single_observation_space = gym.spaces.Dict()
        for group, terms in self.observation_manager.active_terms.items():
            dims = self.observation_manager.group_obs_dim[group]
            if self.observation_manager.group_obs_concatenate[group]:
                space = gym.spaces.Box(-np.inf, np.inf, shape=dims)
            else:
                space = gym.spaces.Dict(
                    {name: gym.spaces.Box(-np.inf, np.inf, shape=dim) for name, dim in zip(terms, dims)}
                )
            self.single_observation_space[group] = space
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.env_data.num_agents)
        # World observations have one row per match; player groups have one row per player.
        self.observation_space = gym.spaces.Dict(
            {
                name: gym.vector.utils.batch_space(
                    space, self.num_envs if name == "world_state" else self.env_data.num_agents
                )
                for name, space in self.single_observation_space.spaces.items()
            }
        )

    def sync_mj_data(self, env_idx=0):
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(env_idx)
        model, data = self.sim.mj_model, self.sim.mj_data
        for name in ("qpos", "qvel", "mocap_pos", "mocap_quat", "ctrl"):
            target = getattr(data, name)
            if target.size:
                target[:] = getattr(self.sim.data, name)[env_idx].detach().cpu().numpy()
        model.geom_friction[:] = self.sim.model.geom_friction[env_idx].detach().cpu().numpy()
        mujoco.mj_forward(model, data)
        return data
