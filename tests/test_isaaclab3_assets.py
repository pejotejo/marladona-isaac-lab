"""Asset migration invariants using Isaac Lab 3 math, ProxyArray and wrench kernels."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

pytest.importorskip("isaaclab")

from isaaclab.managers import ActionTermCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.wrench_composer import WrenchComposer
from isaaclab_marl.assets.agents import Agents
from isaaclab_marl.assets.env_data import EnvCoreData, SoccerGameCfg
from isaaclab_marl.assets.soccer_ball import SoccerBall
from isaaclab_marl.managers.ma_action_manager import (
    BallControlAction,
    BaseMovementAction,
)


def proxy(value):
    return ProxyArray(wp.from_torch(value.contiguous()))


class Collection:
    def __init__(self, state):
        self.num_instances, self.num_bodies = state.shape[:2]
        self.device = "cpu"
        zeros = torch.zeros(*state.shape[:2], 3)
        self.data = SimpleNamespace(
            body_state_w=proxy(state),
            default_body_state=proxy(state.clone()),
            body_com_lin_vel_b=proxy(zeros),
            body_com_lin_vel_w=proxy(zeros),
            body_com_ang_vel_w=proxy(zeros),
        )
        self.permanent_wrench_composer = WrenchComposer(self)

    def write_body_pose_to_sim_index(self, *, body_poses, env_ids=None):
        self.pose, self.pose_ids = body_poses.clone(), env_ids

    def write_body_com_velocity_to_sim_index(self, *, body_velocities, env_ids=None):
        self.velocity, self.velocity_ids = body_velocities.clone(), env_ids


class BallAsset:
    def __init__(self, num_envs):
        self.num_instances, self.num_bodies, self.device = num_envs, 1, "cpu"
        state = torch.zeros(num_envs, 13)
        state[:, 2] = 0.05
        state[:, 6] = 1
        self.data = SimpleNamespace(
            default_root_state=proxy(state.clone()),
            root_state_w=proxy(state),
            root_pos_w=proxy(state[:, :3]),
            root_quat_w=proxy(state[:, 3:7]),
            root_lin_vel_w=proxy(state[:, 7:10]),
            root_ang_vel_b=proxy(torch.zeros(num_envs, 3)),
        )
        self.permanent_wrench_composer = WrenchComposer(self)

    def write_root_pose_to_sim_index(self, *, root_pose, env_ids=None):
        self.pose, self.pose_ids = root_pose.clone(), env_ids

    def write_root_com_velocity_to_sim_index(self, *, root_velocity, env_ids=None):
        self.velocity, self.velocity_ids = root_velocity.clone(), env_ids


@pytest.fixture
def scene():
    cfg = SoccerGameCfg(num_agents_per_team=2)
    data = EnvCoreData(cfg, 3, "cpu")
    data.env_origins = torch.zeros(3, 3)
    data.field_length = torch.full((3,), 4.32)
    data.field_width = 3.6
    data.inactive = torch.zeros(data.agent_base_shape, dtype=torch.bool)
    state = torch.zeros(3, 4, 13)
    state[..., 2] = 0.3
    state[..., 3:7] = quat_from_euler_xyz(
        torch.zeros(12), torch.zeros(12), torch.full((12,), -torch.pi / 2)
    ).view(3, 4, 4)
    state[..., 7:] = 0.2
    collection = Collection(state)
    agents = Agents(cfg, data, 3, "cpu", collection)
    ball_asset = BallAsset(3)
    ball = SoccerBall(cfg, data, 3, "cpu", ball_asset)
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        env_data=data,
        agents=agents,
        ball=ball,
        scene={"agents": collection, "ball": ball_asset},
    )
    return env


def test_planar_constraint_keeps_xyzw_yaw_and_wrapped_pose(scene):
    scene.agents.step()
    asset = scene.agents.asset
    torch.testing.assert_close(asset.pose[..., 3:5], torch.zeros(3, 4, 2))
    torch.testing.assert_close(asset.pose[..., 5], torch.full((3, 4), -(2**-0.5)))
    torch.testing.assert_close(asset.pose[..., 6], torch.full((3, 4), 2**-0.5))
    torch.testing.assert_close(asset.pose[..., 2], torch.full((3, 4), 0.08))
    torch.testing.assert_close(asset.velocity[..., 2:5], torch.zeros(3, 4, 3))
    torch.testing.assert_close(
        scene.agents.base_pose_w[..., 2], torch.full((3, 2, 2), 3 * torch.pi / 2)
    )


def test_sliced_resets_write_xyzw_pose_and_com_velocity(scene):
    selected = slice(1, 3)
    scene.agents.reset(selected)
    assert scene.agents.asset.pose.shape == (2, 4, 7)
    expected = torch.tensor([0.0, 0.0, 1.0, 0.0]).expand(2, 4, 4)
    torch.testing.assert_close(
        scene.agents.asset.pose[..., 3:7], expected, atol=1e-6, rtol=0
    )
    assert torch.count_nonzero(scene.agents.asset.velocity) == 0
    pose = scene.agents.asset.pose.clone()
    scene.agents.reset(selected)
    torch.testing.assert_close(scene.agents.asset.pose, pose)
    assert torch.count_nonzero(scene.agents.base_default_pose_w) == 0
    assert torch.count_nonzero(scene.env_data.env_origins) == 0
    scene.ball.reset(selected)
    assert scene.ball.asset.pose.shape == (2, 7)
    torch.testing.assert_close(
        scene.ball.asset.pose[:, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(2, 4)
    )
    assert torch.count_nonzero(scene.ball.asset.velocity) == 0


def test_native_wrench_composer_replaces_local_player_forces(scene):
    term = BaseMovementAction(
        ActionTermCfg(class_type=BaseMovementAction, asset_name="agents"), scene
    )
    commands = torch.zeros(12, 3)
    commands[:, 0] = 1
    term.process_actions(commands)
    term.apply_actions()
    term.apply_actions()
    force, torque, is_global = (
        scene.agents.asset.permanent_wrench_composer.get_forces_and_torques()
    )
    assert is_global is False
    torch.testing.assert_close(wp.to_torch(force)[..., 0], torch.full((3, 4), 3.5))
    torch.testing.assert_close(wp.to_torch(force)[..., 1:], torch.zeros(3, 4, 2))
    assert torch.count_nonzero(wp.to_torch(torque)) == 0


def test_ball_world_force_is_converted_to_local_xyzw_frame(scene):
    quat = quat_from_euler_xyz(
        torch.zeros(3), torch.zeros(3), torch.full((3,), torch.pi / 2)
    )
    scene.ball.asset.data.root_quat_w = proxy(quat)
    scene.ball.rolling_friction[:] = 0.00008
    scene.ball.asset.data.root_ang_vel_b = proxy(
        torch.tensor([0.0, 0.0, 1.25]).expand(3, 3)
    )
    scene.agents.ball_pos_b = torch.ones(12, 2)
    scene.agents.ball_pos_b[::4] = torch.tensor([0.1, 0.0])
    term = BallControlAction(
        ActionTermCfg(class_type=BallControlAction, asset_name="ball"), scene
    )
    commands = torch.zeros(12, 2)
    commands[::4, 0] = 1
    term.process_actions(commands)
    term.apply_actions()
    force, torque, is_global = (
        scene.ball.asset.permanent_wrench_composer.get_forces_and_torques()
    )
    assert is_global is False
    force_w = quat_apply(quat, wp.to_torch(force)[:, 0])
    torch.testing.assert_close(force_w[:, 0], torch.full((3,), 0.2525))
    torch.testing.assert_close(force_w[:, 1:], torch.zeros(3, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(wp.to_torch(torque)[:, 0, 2], torch.full((3,), -0.0001))
