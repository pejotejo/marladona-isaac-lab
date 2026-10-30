"""Policy-facing compatibility checks using the installed Isaac Lab 3.0 core."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab")

from pxr import Gf, Usd, UsdGeom, UsdShade

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from isaaclab.sim.utils import use_stage
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_marl.assets.agent_cfg import AgentCfg
from isaaclab_marl.assets.from_files import spawn_from_usd
from isaaclab_marl.managers.ma_observation_manager import MultiAgentObservationManager
from isaaclab_marl.managers.ma_reward_manager import MultiAgentRewardManager
from isaaclab_marl.tasks.soccer.soccer_marl_env import SoccerMARLEnv


@pytest.mark.parametrize("task", ["Isaac-Soccer-v0", "Isaac-Soccer-Play-v0"])
def test_registered_configs_keep_physx_and_training_defaults(task):
    cfg, runner = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    cfg.validate()
    runner.validate()
    assert isinstance(cfg.sim.physics, PhysxCfg)
    assert cfg.sim.physics.gpu_max_rigid_patch_count == 10 * 2**15
    assert cfg.sim.dt == 0.02
    assert cfg.decimation == 4
    assert runner.num_steps_per_env == 48
    assert runner.policy.actor_hidden_dims == [128, 128, 128]
    assert runner.policy.critic_hidden_dims == [128, 128, 128]
    assert runner.algorithm.learning_rate == 1e-3


@pytest.mark.parametrize("backend, writer_class", [("wandb", "WandbLogWriter"), ("neptune", "NeptuneLogWriter")])
def test_external_loggers_use_rsl_rl_writer_api(monkeypatch, tmp_path, backend, writer_class):
    import rsl_rl.utils

    from rsl_marl.utils.log_manager import LogManager

    calls = []

    class Writer:
        def __init__(self, log_dir, project_name):
            calls.append((log_dir, project_name))

        def store_config(self, env_cfg, train_cfg):
            calls.append((env_cfg, train_cfg))

    monkeypatch.setattr(rsl_rl.utils, writer_class, Writer)
    manager = LogManager.__new__(LogManager)
    manager.log_dir = str(tmp_path)
    manager.writer = None
    manager.cfg = {"logger": backend, "wandb_project": "wandb-test", "neptune_project": "neptune-test"}
    manager.env = SimpleNamespace(cfg=object())
    manager.init_writer()
    manager.init_writer()
    assert calls == [(str(tmp_path), f"{backend}-test"), (manager.env.cfg, manager.cfg)]


def signal(env):
    return env.signal


@pytest.fixture
def env():
    return SimpleNamespace(
        num_envs=2,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        scene={},
        env_data=SimpleNamespace(num_agents_per_team=1),
        max_episode_length_s=10,
        signal=torch.tensor([[1.0, 10.0]]).repeat(4, 1),
    )


def test_history_keeps_feature_order_and_only_advances_on_steps(env):
    group = ObservationGroupCfg(concatenate_terms=True, enable_corruption=True)
    group.signal = ObservationTermCfg(
        func=signal, history_length=3, noise=UniformNoiseCfg(n_min=0.0, n_max=0.0)
    )
    manager = MultiAgentObservationManager({"policy": group}, env)
    first = manager.compute(update_history=True)["policy"].clone()
    torch.testing.assert_close(first, torch.tensor([[1., 1., 1., 10., 10., 10.]]).repeat(4, 1))
    env.signal *= 2
    torch.testing.assert_close(manager.compute()["policy"], first)
    torch.testing.assert_close(
        manager.compute(update_history=True)["policy"],
        torch.tensor([[1., 1., 2., 10., 10., 20.]]).repeat(4, 1),
    )
    manager.reset()
    torch.testing.assert_close(
        manager.compute(update_history=True)["policy"],
        torch.tensor([[2., 2., 2., 20., 20., 20.]]).repeat(4, 1),
    )


def reward(env):
    return torch.arange(1., 5.)


def test_rewards_accumulate_for_every_player_with_dt_scaling(env):
    manager = MultiAgentRewardManager({"reward": RewardTermCfg(func=reward, weight=2)}, env)
    expected = torch.tensor([0.2, 0.4, 0.6, 0.8])
    torch.testing.assert_close(manager.compute(0.1), expected)
    torch.testing.assert_close(manager.compute(0.1), expected)
    torch.testing.assert_close(manager._episode_sums["reward"], 2 * expected)
    torch.testing.assert_close(manager.reset()["Episode_Reward/reward"], torch.tensor(0.1))
    assert torch.count_nonzero(manager._episode_sum_buf) == 0


def test_native_shutdown_releases_managers_and_flushes_video():
    calls = []
    env = SoccerMARLEnv.__new__(SoccerMARLEnv)
    env._is_closed = False
    env._window = None
    for name in (
        "command_manager", "reward_manager", "termination_manager", "curriculum_manager",
        "action_manager", "observation_manager", "event_manager", "recorder_manager", "scene",
    ):
        setattr(env, name, object())
    env.obs_buf = {}
    env.video_recorders = [SimpleNamespace(close=lambda: calls.append("video"))]
    env.sim = SimpleNamespace(stop=lambda: calls.append("stop"), clear_instance=lambda: calls.append("clear"))
    env.close()
    env.close()
    assert env._is_closed
    assert calls == ["stop", "video", "clear"]


def test_usd_spawner_keeps_heading_material_and_xyzw_transform(tmp_path):
    # A local USD isolates the custom material binding from remote NVIDIA assets.
    path = tmp_path / "agent.usda"
    asset_stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(asset_stage, "/Agent").GetPrim()
    asset_stage.SetDefaultPrim(root)
    for name in ("soccer_agent_circle", "heading"):
        UsdGeom.Cube.Define(asset_stage, f"/Agent/base/visuals/{name}")
    heading_material = UsdShade.Material.Define(asset_stage, "/Agent/HeadingMaterial")
    UsdShade.MaterialBindingAPI.Apply(asset_stage.GetPrimAtPath("/Agent/base/visuals/heading")).Bind(heading_material)
    asset_stage.GetRootLayer().Save()
    cfg = AgentCfg().spawn
    cfg.usd_path = str(path)
    stage = Usd.Stage.CreateInMemory()
    with use_stage(stage):
        prim = spawn_from_usd("/World/Agent", cfg, translation=(1., 2., 3.), orientation=(0., 0., 1., 0.))
        transform = UsdGeom.Xformable(prim).GetLocalTransformation()
        assert Gf.IsClose(transform.ExtractTranslation(), Gf.Vec3d(1., 2., 3.), 1e-6)
        assert Gf.IsClose(transform.TransformDir(Gf.Vec3d(1., 0., 0.)), Gf.Vec3d(-1., 0., 0.), 1e-6)
        circle = stage.GetPrimAtPath("/World/Agent/base/visuals/soccer_agent_circle")
        heading = stage.GetPrimAtPath("/World/Agent/base/visuals/heading")
        assert str(UsdShade.MaterialBindingAPI(circle).ComputeBoundMaterial()[0].GetPath()) == "/World/Agent/material"
        assert str(UsdShade.MaterialBindingAPI(heading).ComputeBoundMaterial()[0].GetPath()) == "/World/Agent/HeadingMaterial"
