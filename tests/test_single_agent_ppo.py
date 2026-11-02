"""Policy output contract and a short CPU PPO update for one learner per match."""

import pytest
import torch

from rsl_marl.modules import ActorCriticBeta


def test_beta_commands_preserve_kick_and_strength_ranges():
    samples = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0, 0.25, 0.8]])
    commands = ActorCriticBeta.to_motion_actions(samples)
    torch.testing.assert_close(commands, torch.tensor([[-1.0, -0.5, 0.0, 0.75, 1.0, -0.5, 0.8]]))
    torch.testing.assert_close(samples, torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0, 0.25, 0.8]]))


def test_seven_action_rollout_and_ppo_update():
    pytest.importorskip("rsl_rl")
    from rsl_marl.algorithms import PPO

    torch.manual_seed(42)
    policy = ActorCriticBeta(25, 25, 8, 2, 3, 5, 7, actor_hidden_dims=[16], critic_hidden_dims=[16])
    ppo = PPO(policy, num_learning_epochs=1, num_mini_batches=1,
              symmetry_cfg={"use_symmetry": True, "use_augmentation": True})
    ppo.obs_mirror_index = torch.tensor([1, 2, 21, 26])
    assert ppo.action_mirror_index.tolist() == [1, 2, 5]
    ppo.init_storage(4, 2, [65], [65], [7])
    for _ in range(2):
        obs = torch.randn(4, 65)
        with torch.no_grad():
            commands = ppo.act(obs, obs)
            torch.testing.assert_close(commands, policy.to_motion_actions(ppo.transition.actions))
            assert ((commands[:, [3, 6]] >= 0) & (commands[:, [3, 6]] <= 1)).all()
            inference = policy.act_inference(obs)
            assert inference.shape == (4, 7)
            assert ((inference[:, [3, 6]] >= 0) & (inference[:, [3, 6]] <= 1)).all()
            ppo.process_env_step(torch.randn(4), torch.zeros(4), {"time_outs": torch.zeros(4)})
    ppo.compute_returns(torch.randn(4, 65))
    before = next(policy.actor.parameters()).detach().clone()
    losses = ppo.update()
    assert torch.isfinite(torch.tensor(losses)).all()
    assert not torch.equal(before, next(policy.actor.parameters()).detach())
