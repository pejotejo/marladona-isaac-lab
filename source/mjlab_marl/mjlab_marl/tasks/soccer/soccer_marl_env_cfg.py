# Copyright 2025 Zichong Li, ETH Zurich
# Licensed under the Apache License, Version 2.0.
"""MARLadona's task configuration on mjlab 1.6."""

from copy import deepcopy
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers import CurriculumTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from mjlab_marl.assets.env_data import SoccerGameCfg
from mjlab_marl.config import WORLD_POS_NORMALIZATION
from mjlab_marl.managers.ma_action_manager import BallControlActionCfg, BaseMovementActionCfg
from mjlab_marl.tasks.soccer.mdp import curriculums as C
from mjlab_marl.tasks.soccer.mdp import observations as O
from mjlab_marl.tasks.soccer.mdp import rewards as R
from mjlab_marl.tasks.soccer.mdp import terminations as T


def observations_cfg():
    def term(func, noise=None, **params):
        return ObservationTermCfg(
            func=func, params=params, noise=UniformNoiseCfg(n_min=-noise, n_max=noise) if noise else None
        )

    policy = ObservationGroupCfg(
        terms={
            "base_own_vel": term(O.base_own_vel, 0.005),
            "base_own_pose_w": term(O.base_own_pose_w, 0.002, normalize=True, expand_rot=True),
            "ball_vel_w": term(O.ball_vel_w, 0.005),
            "ball_pos_w": term(O.ball_pos_w, 0.002, normalize=True),
            "field_infos": term(O.field_infos, normalize=True),
        },
        enable_corruption=True,
    )
    critic = deepcopy(policy)
    critic.enable_corruption = False
    neighbor = ObservationGroupCfg(
        terms={
            "teammate_pose": ObservationTermCfg(
                func=O.closest_teammate_pose,
                history_length=2,
                params={"normalize": WORLD_POS_NORMALIZATION, "expand_rot": True, "max_num_neighbor": 5},
            ),
            "opponent_pose": ObservationTermCfg(
                func=O.closest_opponent_pose,
                history_length=2,
                params={"normalize": WORLD_POS_NORMALIZATION, "expand_rot": True, "max_num_neighbor": 5},
            ),
        }
    )
    return {
        "policy": policy,
        "critic": critic,
        "neighbor": neighbor,
        "neighbor_critic": deepcopy(neighbor),
        "world_state": ObservationGroupCfg(terms={"world_state": term(O.world_state)}, concatenate_terms=False),
        "bots": ObservationGroupCfg(
            terms={
                "base_own_pose_w": term(O.base_own_pose_w, normalize=False, expand_rot=False),
                "ball_pos_b": term(O.ball_pos_b),
                "ball_pos_w": ObservationTermCfg(func=O.ball_pos_w, history_length=2, params={"normalize": False}),
                "closest_agent_to_ball": term(O.closest_agent_to_ball),
                "player_id": term(O.player_id),
                "field_infos": term(O.field_infos, normalize=False),
            },
            concatenate_terms=False,
        ),
    }


@dataclass(kw_only=True)
class ShapingRewardCfg(RewardTermCfg):
    decay_condition_type: str = "score"
    decay_threshhold: float = 1.5


def rewards_cfg():
    return {
        "score": RewardTermCfg(func=R.score, weight=100.0),
        "ball_outside_field_penalty": RewardTermCfg(func=R.ball_outside_field_penalty, weight=1.0),
        "collision_penalty": RewardTermCfg(
            func=R.collision_penalty,
            weight=1.0,
            params={"collision_radius_agent": 0.2, "collision_radius_goal_post": 0.17},
        ),
        "vel_2_ball": ShapingRewardCfg(func=R.vel_2_ball, weight=0.5, params={"mode": "until_close"}),
        "vel_ball_2_goal": ShapingRewardCfg(func=R.vel_ball_2_goal, weight=2.0),
        "ball_direction": ShapingRewardCfg(func=R.ball_direction, weight=0.5, params={"std": 0.4}),
    }


@dataclass(kw_only=True)
class SoccerMARLEnvCfg(ManagerBasedRlEnvCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(num_envs=1024, env_spacing=0.0))
    decimation: int = 4
    episode_length_s: float = 30.0
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            mujoco=MujocoCfg(timestep=0.02, iterations=20, ls_iterations=10), nconmax=256, njmax=512
        )
    )
    viewer: ViewerConfig = field(default_factory=lambda: ViewerConfig(distance=7.0, elevation=-65, azimuth=90))
    soccer_game: SoccerGameCfg = field(default_factory=SoccerGameCfg)
    observations: dict = field(default_factory=observations_cfg)
    actions: dict = field(
        default_factory=lambda: {
            "base_movement": BaseMovementActionCfg(entity_name="agent_blue_0"),
            "ball_control": BallControlActionCfg(entity_name="ball"),
        }
    )
    rewards: dict = field(default_factory=rewards_cfg)
    terminations: dict = field(
        default_factory=lambda: {
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "goal_scored": TerminationTermCfg(func=T.goal_scored),
        }
    )
    curriculum: dict = field(default_factory=lambda: {"terrain_levels": CurriculumTermCfg(func=C.field_levels)})
    events: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class SoccerMARLEnvPlayCfg(SoccerMARLEnvCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(num_envs=1, env_spacing=0.0))
    soccer_game: SoccerGameCfg = field(
        default_factory=lambda: SoccerGameCfg(
            num_agents_per_team=5,
            num_agents_range=[5, 5],
            eval_env_ratio=0.0,
            level_settings=[[3, 3]],
            create_field_visuals=True,
        )
    )
    # Preserve the original effective duration (the Isaac base config overwrote 120 s).
    episode_length_s: float = 30.0
