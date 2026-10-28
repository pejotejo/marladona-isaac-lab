# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line configuration shared by the mjlab train and play scripts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("rsl_rl")
    group.add_argument("--experiment_name", default=None)
    group.add_argument("--run_name", default=None)
    group.add_argument("--resume", action="store_true", default=None)
    group.add_argument(
        "--load_run", default=None, help="Run directory name or regular expression."
    )
    group.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path or filename pattern in --load_run.",
    )
    group.add_argument(
        "--logger", choices=("tensorboard", "wandb", "neptune"), default=None
    )
    group.add_argument("--log_project_name", default=None)


def add_sim_args(parser: argparse.ArgumentParser, *, play=False):
    parser.add_argument(
        "--task", default="Mjlab-Soccer-Play-v0" if play else "Mjlab-Soccer-v0"
    )
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument(
        "--num_agents",
        type=int,
        default=None,
        help="Maximum number of players on each team.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda:0", help="mjlab simulation and policy device."
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without the native MuJoCo viewer."
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record offscreen video of environment zero.",
    )
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--video_interval", type=int, default=2000)
    add_rsl_rl_args(parser)


def parse_rsl_rl_cfg(task_name, args_cli):
    from mjlab_marl.tasks.soccer.agents.soccer_marl_ppo_runner_cfg import (
        SoccerMARLPPORunnerCfg,
    )

    if task_name not in ("Mjlab-Soccer-v0", "Mjlab-Soccer-Play-v0"):
        raise ValueError(f"Unknown task: {task_name}")
    return update_rsl_rl_cfg(SoccerMARLPPORunnerCfg(), args_cli)


def update_rsl_rl_cfg(agent_cfg, args_cli):
    for name in (
        "seed",
        "resume",
        "load_run",
        "run_name",
        "logger",
        "experiment_name",
        "device",
    ):
        value = getattr(args_cli, name, None)
        if value is not None:
            setattr(agent_cfg, name, value)
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    return agent_cfg


def make_env_cfg(args_cli, seed):
    from mjlab_marl.tasks.soccer.soccer_marl_env_cfg import (
        SoccerMARLEnvCfg,
        SoccerMARLEnvPlayCfg,
    )

    configs = {
        "Mjlab-Soccer-v0": SoccerMARLEnvCfg,
        "Mjlab-Soccer-Play-v0": SoccerMARLEnvPlayCfg,
    }
    if args_cli.task not in configs:
        raise ValueError(f"Unknown task: {args_cli.task}")
    cfg = configs[args_cli.task]()
    cfg.seed = seed
    if args_cli.num_envs is not None:
        if args_cli.num_envs < 1:
            raise ValueError("--num_envs must be positive")
        cfg.scene.num_envs = args_cli.num_envs
    if args_cli.num_agents is not None:
        if not 1 <= args_cli.num_agents <= 5:
            raise ValueError("--num_agents must be between 1 and 5")
        cfg.soccer_game.num_agents_per_team = args_cli.num_agents
        cfg.soccer_game.num_agents_range = (
            [args_cli.num_agents, args_cli.num_agents]
            if "Play" in args_cli.task
            else [1, args_cli.num_agents]
        )
        cfg.soccer_game.agent_id_control = [
            [control] * args_cli.num_agents
            for control in cfg.soccer_game.default_control
        ]
    return cfg


def get_checkpoint_path(log_root, run_pattern, checkpoint_pattern):
    checkpoint = Path(checkpoint_pattern).expanduser()
    if checkpoint.is_file():
        return checkpoint.resolve()
    root = Path(log_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {root}")
    runs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and re.fullmatch(run_pattern, path.name)
    )
    if not runs:
        raise FileNotFoundError(f"No run matching {run_pattern!r} in {root}")
    checkpoints = [
        path
        for path in runs[-1].iterdir()
        if path.is_file() and re.fullmatch(checkpoint_pattern, path.name)
    ]
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint matching {checkpoint_pattern!r} in {runs[-1]}"
        )
    return max(
        checkpoints,
        key=lambda path: [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", path.name)
        ],
    )


def save_config(log_dir, name, cfg):
    # JSON is valid YAML and avoids serializing Python objects into configuration files.
    params = Path(log_dir) / "params"
    params.mkdir(parents=True, exist_ok=True)
    with (params / f"{name}.yaml").open("w") as stream:
        json.dump(asdict(cfg), stream, indent=2, default=str)


class NativeViewer:
    def __init__(self, env):
        import mujoco.viewer

        self.env = env.unwrapped
        self.viewer = mujoco.viewer.launch_passive(
            self.env.sim.mj_model, self.env.sync_mj_data()
        )
        origin = self.env.env_data.env_origins[0].cpu().numpy()
        self.origin = origin.copy()
        self.viewer.cam.lookat[:] = origin
        self.viewer.cam.distance = 8.0
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -80

    def sync(self):
        if not self.viewer.is_running():
            raise KeyboardInterrupt
        self.env.sync_mj_data()
        origin = self.env.env_data.env_origins[0].cpu().numpy()
        self.viewer.cam.lookat[:] += origin - self.origin
        self.origin = origin.copy()
        self.viewer.sync()

    def close(self):
        self.viewer.close()


class VideoRecorder:
    def __init__(self, env, folder, length, interval):
        if length < 1 or interval < 1:
            raise ValueError("Video length and interval must be positive")
        self.env = env
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.length = length
        self.interval = interval
        self.step = 0
        self.writer = None

    def capture(self):
        import imageio.v2 as imageio

        if self.step % self.interval == 0:
            self.close()
            self.writer = imageio.get_writer(
                self.folder / f"step_{self.step}.mp4",
                fps=1 / self.env.unwrapped.step_dt,
            )
        if self.writer is not None:
            self.writer.append_data(self.env.render())
            if self.step % self.interval + 1 >= self.length:
                self.close()
        self.step += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
