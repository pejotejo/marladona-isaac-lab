"""Behavior checks against the actual mjlab / MuJoCo-Warp CPU backend."""

import math

import pytest
import torch
from mjlab_marl import SoccerMARLEnv, SoccerMARLEnvCfg
from mjlab_marl.assets.scene import FIELD_SPACING

pytestmark = pytest.mark.simulation


@pytest.fixture(scope="module")
def simulator():
    cfg = SoccerMARLEnvCfg(seed=123)
    cfg.scene.num_envs = 2
    cfg.soccer_game.eval_env_ratio = 0.5
    cfg.soccer_game.num_agents_range = [3, 3]
    env = SoccerMARLEnv(cfg, device="cpu")
    yield env
    env.close()


@pytest.fixture
def env(simulator):
    simulator.env_data.active_agent_per_team[:] = 3
    simulator.env_data.regenerate_mask_index()
    simulator.field.terrain_levels[:] = 1
    simulator.field._set_origins(torch.arange(simulator.num_envs))
    simulator.reset()
    return simulator


def set_player(env, index, xy=(0.0, 0.0), yaw=0.0, env_id=0):
    asset = env.agents.asset[index]
    state = asset.data.default_root_state[[env_id]].clone()
    state[0, :3] = env.scene.env_origins[env_id] + torch.tensor([*xy, 0.08])
    state[0, 3:7] = torch.tensor([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    state[0, 7:] = 0
    asset.write_root_state_to_sim(state, env_ids=torch.tensor([env_id]))


def set_ball(env, xy, env_id=0, height=0.05):
    state = env.ball.asset.data.default_root_state[[env_id]].clone()
    state[0, :3] = env.scene.env_origins[env_id] + torch.tensor([*xy, height])
    state[0, 7:] = 0
    env.ball.asset.write_root_state_to_sim(state, env_ids=torch.tensor([env_id]))


def refresh(env):
    env.sim.forward()
    env._refresh_states()


def test_native_assets_and_step_contract(env):
    model = env.sim.mj_model
    assert model.body("agent_blue_0/base").mass == pytest.approx(5.480100154876709)
    assert model.body("ball/ball").mass == pytest.approx(0.044)
    assert model.geom("ball/collision").size[0] == pytest.approx(0.05)
    assert env.step_dt == pytest.approx(0.08)
    for _ in range(5):
        obs, rewards, terminated, truncated, _ = env.step(torch.zeros(12, 5))
        assert obs["policy"].shape == (12, 18)
        assert obs["neighbor"].shape == (12, 40)
        assert obs["world_state"]["world_state"].shape == (2, 20)
        assert rewards.shape == terminated.shape == truncated.shape == (12,)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(rewards).all()
    assert env.action_space.shape == (12, 5)
    assert env.sync_mj_data().qpos.shape == (model.nq,)


def test_body_frame_movement_and_inactive_controls(env):
    set_player(env, 0, yaw=math.pi / 2)
    refresh(env)
    actions = torch.zeros(12, 5)
    actions[0, 0] = 1
    env.action_manager.process_action(actions)
    movement = env.action_manager.get_term("base_movement")
    movement.apply_actions()
    wrench = env.sim.data.xfrc_applied[0, env.agents.asset[0].indexing.root_body_id]
    assert wrench[0].item() == pytest.approx(0, abs=1e-5)
    assert wrench[1].item() == pytest.approx(3.5)
    env.env_data.active[0, 0, 0] = False
    env.env_data.inactive = ~env.env_data.active
    movement.apply_actions()
    assert torch.equal(env.sim.data.xfrc_applied[0, env.agents.asset[0].indexing.root_body_id], torch.zeros(6))


def test_kick_force_and_eligibility(env):
    for index in range(6):
        set_player(env, index, xy=(1.0, -1.5 + index * 0.4))
    set_player(env, 0)
    set_ball(env, (0.1, 0))
    refresh(env)
    actions = torch.zeros(12, 5)
    actions[0, 3] = 1
    env.action_manager.process_action(actions)
    kick = env.action_manager.get_term("ball_control")
    kick.compute()
    assert kick._force[0, 0].item() == pytest.approx(0.2525, abs=1e-6)
    assert kick._force[0, 1].item() == pytest.approx(0, abs=1e-6)
    set_ball(env, (-0.1, 0))
    refresh(env)
    kick.compute()
    assert torch.equal(kick._force[0], torch.zeros(3))
    set_ball(env, (0.1, 0), height=0.2)
    refresh(env)
    kick.compute()
    assert torch.equal(kick._force[0], torch.zeros(3))


def test_goal_scored_once_rewards_and_advances_training_field(env):
    env_id = env.env_data.is_training_env.nonzero().flatten().item()
    previous_score = env.env_data.accumulated_score[env_id, 0].item()
    set_ball(env, (-env.env_data.field_length[env_id].item() / 2 - 0.06, 0), env_id)
    refresh(env)
    obs, reward, terminated, truncated, info = env.step(torch.zeros(12, 5))
    rows = slice(env_id * 6, (env_id + 1) * 6)
    assert terminated[rows].all() and not truncated[rows].any()
    assert info["score"][env_id, 0] == 1
    assert env.env_data.accumulated_score[env_id, 0].item() == previous_score + 1
    assert (reward[env_id * 6 : env_id * 6 + 3] > 7).all()
    assert (reward[env_id * 6 + 3 : env_id * 6 + 6] < -7).all()
    assert env.field.terrain_levels[env_id] == 2
    assert env.scene.env_origins[env_id, 0] == pytest.approx(2 * FIELD_SPACING)
    assert env.episode_length_buf[env_id] == 0
    assert torch.isfinite(obs["policy"]).all()


def test_ball_border_reset_and_partial_reset_isolation(env):
    set_ball(env, (0, 1.85))
    refresh(env)
    _, reward, terminated, _, _ = env.step(torch.zeros(12, 5))
    assert not terminated[:6].any()
    assert abs(env.ball.base_pos_w[0, 1]) < env.env_data.field_width / 2
    assert (reward[:6] < 0).all()
    other_state = env.sim.data.qpos[1].clone()
    other_history = (
        env.observation_manager._group_obs_term_history_buffer["neighbor"]["teammate_pose"].buffer[6:].clone()
    )
    env.reset(env_ids=torch.tensor([0]))
    assert torch.equal(env.sim.data.qpos[1], other_state)
    actual = env.observation_manager._group_obs_term_history_buffer["neighbor"]["teammate_pose"].buffer[6:]
    torch.testing.assert_close(actual, other_history, equal_nan=True)


def test_timeout_expands_to_every_player(env):
    env.episode_length_buf[:] = env.max_episode_length - 1
    _, _, terminated, truncated, _ = env.step(torch.zeros(12, 5))
    assert truncated.all() and not terminated.any()
    assert torch.equal(env.episode_length_buf, torch.zeros(2, dtype=torch.long))


def test_manual_reset_requires_every_terminated_match_to_be_reset(env):
    env.cfg.auto_reset = False
    try:
        env.episode_length_buf[:] = env.max_episode_length - 1
        _, _, _, truncated, _ = env.step(torch.zeros(12, 5))
        assert truncated.all()
        with pytest.raises(RuntimeError, match="Reset terminated matches"):
            env.step(torch.zeros(12, 5))
        env.reset(env_ids=torch.tensor([0]))
        with pytest.raises(RuntimeError, match="Reset terminated matches"):
            env.step(torch.zeros(12, 5))
        env.reset(env_ids=torch.tensor([1]))
        env.step(torch.zeros(12, 5))
    finally:
        env.cfg.auto_reset = True
