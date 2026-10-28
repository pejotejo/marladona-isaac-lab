"""Regression checks for the policy interface carried over from MARLadona."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from mjlab_marl.assets.env_data import EnvData, SoccerGameCfg
from mjlab_marl.managers.ma_observation_manager import MultiAgentObservationManager
from mjlab_marl.managers.ma_reward_manager import MultiAgentRewardManager
from mjlab_marl.tasks.soccer.mdp import observations, rewards


def test_game_config_defaults_follow_team_size_and_do_not_share_mutable_state():
    cfg = SoccerGameCfg(num_agents_per_team=5)
    other = SoccerGameCfg()
    assert cfg.num_agents_range == [1, 5]
    assert cfg.agent_id_control == [[1] * 5, [2] * 5]
    cfg.agents.cmd_speed_limit[0][0] = 1.0
    assert other.agents.cmd_speed_limit[0][0] == 0.35
    assert asdict(cfg)["num_agents_per_team"] == 5
    with pytest.raises(ValueError, match="num_agents_range"):
        SoccerGameCfg(num_agents_per_team=2, num_agents_range=[1, 3])
    with pytest.raises(ValueError, match="agent_id_control"):
        SoccerGameCfg(agent_id_control=[[1], [2]])


def test_score_and_per_player_control_masks():
    cfg = SoccerGameCfg(
        num_agents_per_team=2,
        num_agents_range=[2, 2],
        eval_env_ratio=0.0,
        agent_id_control=[[1, 3], [2, 0]],
    )
    data = EnvData(cfg, 2, "cpu")
    data.set_field_curriculum_level(torch.tensor([0, 2]))
    data.set_env_origin(torch.zeros(2, 3))
    data.reset()
    assert data.training_actor_index.tolist() == [True, False, False, False] * 2
    assert data.policy_replay_actor_index.tolist() == [False, False, True, False] * 2
    data.step(torch.tensor([1, -1]))
    env = SimpleNamespace(env_data=data)
    assert rewards.score(env).tolist() == [1, 1, -1, -1, -1, -1, 1, 1]
    assert data.score.tolist() == [[1, 0], [0, 1]]
    data.reset(torch.tensor([1]))
    assert data.score.tolist() == [[1, 0], [0, 0]]
    assert data.accumulated_score.tolist() == [[0, 0], [0, 1]]


def test_feature_major_history_and_partial_environment_reset():
    env = SimpleNamespace(num_envs=2, device="cpu", env_data=SimpleNamespace(num_agents_per_env=4))
    env.player_state = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    env.world_state = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    cfg = {
        "neighbor": ObservationGroupCfg(
            terms={"teammate_pose": ObservationTermCfg(func=lambda e: e.player_state, params={}, history_length=2)}
        ),
        "world_state": ObservationGroupCfg(
            terms={"world_state": ObservationTermCfg(func=lambda e: e.world_state, history_length=2)},
            concatenate_terms=False,
        ),
    }
    manager = MultiAgentObservationManager(cfg, env)
    first = manager.compute(update_history=True)
    torch.testing.assert_close(first["neighbor"], env.player_state.repeat_interleave(2, dim=-1))
    old = env.player_state.clone()
    env.player_state += 100
    env.world_state += 100
    second = manager.compute(update_history=True)
    expected = torch.stack([old, env.player_state], dim=-1).reshape(8, -1)
    torch.testing.assert_close(second["neighbor"], expected)
    assert manager.compute() is second
    assert manager._group_symmetry_index["neighbor"] == [2, 4, 3, 5]
    assert manager._group_obs_slice_pos["neighbor"]["teammate_pose"] == [0, 8]
    manager.reset(torch.tensor([1]))
    env.player_state[4:] = 999
    env.world_state[1] = 999
    reset = manager.compute(update_history=True, env_ids=torch.tensor([1]))
    torch.testing.assert_close(reset["neighbor"][:4], second["neighbor"][:4])
    torch.testing.assert_close(reset["neighbor"][4:], torch.full((4, 8), 999.0))
    torch.testing.assert_close(reset["world_state"]["world_state"][0], second["world_state"]["world_state"][0])
    torch.testing.assert_close(reset["world_state"]["world_state"][1], torch.full((6,), 999.0))


def test_reward_partial_reset_uses_all_players_of_selected_environment():
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        max_episode_length_s=30.0,
        env_data=SimpleNamespace(num_agents_per_env=4),
    )
    manager = MultiAgentRewardManager(
        {"score": RewardTermCfg(func=lambda e: torch.arange(8, dtype=torch.float32), weight=2.0)},
        env,
    )
    reward = manager.compute(0.08).clone()
    torch.testing.assert_close(reward, torch.arange(8, dtype=torch.float32) * 0.16)
    info = manager.reset(torch.tensor([1]))
    torch.testing.assert_close(manager._episode_sums["score"][:4], reward[:4])
    assert manager._episode_sums["score"][4:].count_nonzero() == 0
    torch.testing.assert_close(info["Episode_Reward/score"], reward[4:].mean() / 30.0)


def test_team_perspectives_preserve_policy_features():
    data = EnvData(SoccerGameCfg(num_agents_per_team=1, eval_env_ratio=0.0), 1, "cpu")
    data.set_field_curriculum_level(torch.zeros(1, dtype=torch.long))
    data.reset()
    data.active_agent_per_team[:] = 1
    env = SimpleNamespace(
        env_data=data,
        agents=SimpleNamespace(base_pose_w=torch.tensor([[[[1.0, 0.5, 0.0]], [[-1.0, -0.5, torch.pi]]]])),
        ball=SimpleNamespace(base_pos_w=torch.tensor([[0.2, -0.3]])),
        device="cpu",
        num_envs=1,
    )
    pose = observations.base_own_pose_w(env, expand_rot=True, normalize=False)
    torch.testing.assert_close(pose[0], pose[1], atol=1e-6, rtol=1e-6)
    ball = observations.ball_pos_w(env, normalize=False)
    torch.testing.assert_close(ball[0], -ball[1])
