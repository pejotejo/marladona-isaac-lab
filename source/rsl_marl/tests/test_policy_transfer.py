"""Policy transfer invariants across the supported team sizes."""

from pathlib import Path

import pytest
import torch
from rsl_marl.modules import ActorCriticBeta
from rsl_marl.modules.policy_replay_manager import PolicyReplayManager


@pytest.mark.parametrize("team_size", [1, 3, 5])
def test_bundled_checkpoint_retains_finite_actions_for_each_team_size(team_size):
    policy = ActorCriticBeta(
        18,
        18,
        8,
        team_size - 1,
        team_size,
        2 * team_size - 1,
        5,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
    )
    root = Path(__file__).resolve().parents[3]
    checkpoint = (
        root / "wks_logs/00_example_policies/24_09_28_11_56_41_3v3/model_20000.pt"
    )
    policy.load_state_dict(
        torch.load(checkpoint, weights_only=True, map_location="cpu")[
            "model_state_dict"
        ]
    )
    obs = torch.zeros(2, 18 + 8 * (2 * team_size - 1))
    # A one-player team has no active teammates. Other slots use the same NaN mask.
    obs[:, 18 : 18 + 8 * (team_size - 1)] = torch.nan
    with torch.no_grad():
        actions = policy.act_inference(obs)
        values = policy.evaluate(obs)
    assert actions.shape == (2, 5)
    assert torch.isfinite(actions).all()
    assert ((actions >= -1) & (actions <= 1)).all()
    assert torch.isfinite(values).all()


def test_replay_policy_buffers_belong_to_their_runner():
    policy = ActorCriticBeta(18, 18, 8, 2, 3, 5, 5)
    cfg = {
        "max_num_policy_replay_level": 2,
        "dynamic_generate_replay_level": True,
        "load_path": "",
        "shuffle": False,
    }
    first = PolicyReplayManager(cfg, policy, 2, 1, "cpu")
    second = PolicyReplayManager(cfg, policy, 0, 1, "cpu")
    assert len(first.policy_buffer) == 2
    assert not second.policy_buffer
    assert first.policy_buffer is not second.policy_buffer


def test_self_play_snapshot_after_a_policy_update():
    policy = ActorCriticBeta(18, 18, 8, 2, 3, 5, 5)
    cfg = {
        "max_num_policy_replay_level": 2,
        "dynamic_generate_replay_level": True,
        "load_path": "",
        "shuffle": False,
    }
    manager = PolicyReplayManager(cfg, policy, 2, 1, "cpu")
    policy.act(torch.zeros(2, 58))
    policy.actor[0].weight.data.fill_(0.1)
    manager.append_policy(policy)
    snapshot = list(manager.policy_buffer.values())[-1]
    torch.testing.assert_close(snapshot.actor[0].weight, policy.actor[0].weight)
    assert snapshot.distribution is None
