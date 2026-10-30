# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Modifications copyright (c) 2025 Zichong Li, ETH Zurich

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import sys
import torch

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{normal_repr(self)} \n {self.shape}, {self.min()}, {self.max()}"
np.set_printoptions(edgeitems=3, linewidth=1000, threshold=100)
torch.set_printoptions(edgeitems=3, linewidth=1000, threshold=100)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--headless", action="store_true", help="Run without a Kit window.")
parser.add_argument("--enable_cameras", action="store_true", help="Enable offscreen camera rendering.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pickle
import torch
from datetime import datetime

# from rsl_rl.runners import OnPolicyRunner
from rsl_marl.runners import OnPolicyRunner
from rsl_marl.utils.custom_vecenv_wrapper import CustomVecEnvWrapper

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    VideoRecorderCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config
from isaaclab_visualizers.kit import KitVisualizerCfg

# Import extensions to set up environment tasks
import isaaclab_marl.tasks  # noqa: F401
from isaaclab_marl.config import WKS_LOGS_DIR
from isaaclab_marl.tasks.soccer.agents.soccer_marl_ppo_runner_cfg import SoccerMARLPPORunnerCfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def dump_pickle(filename, data):
    """Preserve the existing config artifacts after Isaac Lab removed its pickle helper."""
    with open(filename, "wb") as stream:
        pickle.dump(data, stream)


def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: SoccerMARLPPORunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join(WKS_LOGS_DIR, agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # configure Isaac Lab's internal video recorders before creating the environment
    if args_cli.video:
        env_cfg.sim.visualizer_cfgs.append(KitVisualizerCfg(headless=args_cli.headless))
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:kit",
                output_dir=os.path.join(log_dir, "videos", "train"),
                video_interval=args_cli.video_interval,
                video_length=args_cli.video_length,
            )
        ]
        print("[INFO] Recording videos during training.")
        print_dict(env_cfg.video_recorders[0].to_dict(), nesting=4)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = CustomVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, command_args=args_cli)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point", overrides=hydra_args)
    main(env_cfg, agent_cfg)
    # close sim app
    simulation_app.close()
