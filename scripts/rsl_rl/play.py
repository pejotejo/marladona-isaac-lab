# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) 2025 Zichong Li, ETH Zurich

"""Play a MARLadona checkpoint in mjlab with MuJoCo's native viewer."""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import cli_args
import numpy as np
import torch

PLOT_VALUE_FUNC_LIST = ["base_own_pose_w", "ball_pos_w"]


def compute_all_value_fun(env, critic, x, y, num_players):
    values = []
    obs_dict = env.get_observations()
    critic_obs = torch.cat([obs_dict["critic"], obs_dict["neighbor_critic"]], dim=-1)[
        :num_players
    ]
    for term_name in PLOT_VALUE_FUNC_LIST:
        obs_slice = env.unwrapped.observation_manager._group_obs_slice_pos["critic"][
            term_name
        ]
        for obs in critic_obs:
            obs_grid = obs.unsqueeze(0).repeat(x.numel(), 1)
            obs_grid[:, obs_slice[0]] = x
            obs_grid[:, obs_slice[0] + 1] = y
            values.append(critic(obs_grid))
    return torch.cat(values, dim=1)


def prepare_value_fun_plot(values, resolution):
    scaled = []
    for column in values.T:
        z = column.detach().view(resolution, resolution).cpu().numpy()
        scaled.append((z - z.min()) / max(float(np.ptp(z)), 1e-8))
    return scaled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cli_args.add_sim_args(parser, play=True)
    parser.add_argument(
        "--max_steps",
        "--steps",
        type=int,
        default=0,
        help="Stop after this many steps (0: unlimited GUI, 1000 headless).",
    )
    parser.add_argument(
        "--visualize_value_fun",
        action="store_true",
        help="Open the optional matplotlib value function plots.",
    )
    args = parser.parse_args()

    from mjlab_marl.config import WKS_LOGS_DIR
    from mjlab_marl.tasks.soccer.soccer_marl_env import SoccerMARLEnv
    from rsl_marl.runners import OnPolicyRunner
    from rsl_marl.utils.custom_vecenv_wrapper import CustomVecEnvWrapper

    if args.max_steps < 0:
        parser.error("--max_steps cannot be negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for a small smoke test")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    agent_cfg.policy_replay.dynamic_generate_replay_level = False
    env_cfg = cli_args.make_env_cfg(args, agent_cfg.seed)
    checkpoint = cli_args.get_checkpoint_path(
        Path(WKS_LOGS_DIR) / agent_cfg.experiment_name,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
    )
    print(f"[INFO] Loading model checkpoint: {checkpoint}")
    env = SoccerMARLEnv(
        env_cfg, device=args.device, render_mode="rgb_array" if args.video else None
    )
    viewer = None
    video = None
    plot_process = None
    queue = None
    try:
        wrapped = CustomVecEnvWrapper(env)
        runner = OnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=None,
            device=args.device,
            command_args=args,
        )
        runner.load(checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        if not args.headless:
            viewer = cli_args.NativeViewer(env)
        if args.video:
            video = cli_args.VideoRecorder(
                env,
                checkpoint.parent / "videos" / "play",
                args.video_length,
                args.video_length + 1,
            )
        if args.visualize_value_fun:
            from mjlab_marl.utils.value_plotter import RESOLUTION, ValuePlotter

            num_players = min(3, env.env_data.num_agents_per_team)
            value_plotter = ValuePlotter(
                env.env_data.num_agents_per_team, num_players, PLOT_VALUE_FUNC_LIST
            )
            context = mp.get_context("spawn")
            queue = context.Queue(maxsize=1)
            plot_process = context.Process(
                target=value_plotter, args=(queue,), daemon=True
            )
            plot_process.start()
            x = (
                torch.as_tensor(value_plotter.x, device=env.device).flatten()
                / value_plotter.field_length
            )
            y = (
                torch.as_tensor(value_plotter.y, device=env.device).flatten()
                / value_plotter.field_width
            )
            critic = runner.get_inference_critic(device=env.device)

        obs_dict = wrapped.get_observations()
        max_steps = args.max_steps or (1000 if args.headless else 0)
        if args.video:
            max_steps = min(max_steps or args.video_length, args.video_length)
        step = 0
        while not max_steps or step < max_steps:
            start = time.monotonic()
            with torch.inference_mode():
                actions = policy(
                    torch.cat([obs_dict["policy"], obs_dict["neighbor"]], dim=1)
                )
                obs_dict, _, _, _ = wrapped.step(actions)
                if args.visualize_value_fun and step % 2 == 0 and queue.empty():
                    values = compute_all_value_fun(wrapped, critic, x, y, num_players)
                    queue.put(
                        (
                            obs_dict["world_state"]["world_state"][0].cpu(),
                            prepare_value_fun_plot(values, RESOLUTION),
                        )
                    )
            if video is not None:
                video.capture()
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, env.step_dt - (time.monotonic() - start)))
            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        if video is not None:
            video.close()
        if viewer is not None:
            viewer.close()
        if plot_process is not None:
            plot_process.terminate()
            plot_process.join(timeout=2)
        if queue is not None:
            queue.close()
        env.close()


if __name__ == "__main__":
    main()
