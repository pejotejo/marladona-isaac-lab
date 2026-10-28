# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modifications copyright (c) 2025 Zichong Li, ETH Zurich

"""Train MARLadona with mjlab and the original multi-agent PPO implementation."""

import argparse
from datetime import datetime
from pathlib import Path

import cli_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cli_args.add_sim_args(parser)
    parser.add_argument("--max_iterations", type=int, default=None)
    args = parser.parse_args()

    import torch
    from mjlab_marl.config import WKS_LOGS_DIR
    from mjlab_marl.tasks.soccer.soccer_marl_env import SoccerMARLEnv
    from rsl_marl.runners import OnPolicyRunner
    from rsl_marl.utils.custom_vecenv_wrapper import CustomVecEnvWrapper

    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    env_cfg = cli_args.make_env_cfg(args, agent_cfg.seed)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if agent_cfg.max_iterations < 1:
        parser.error("--max_iterations must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for a small smoke test")

    log_root = Path(WKS_LOGS_DIR) / agent_cfg.experiment_name
    resume_path = None
    if agent_cfg.resume or args.checkpoint is not None:
        resume_path = cli_args.get_checkpoint_path(
            log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging experiment in: {log_dir}")

    env = SoccerMARLEnv(
        env_cfg, device=args.device, render_mode="rgb_array" if args.video else None
    )
    viewer = None
    video = None
    runner = None
    try:
        if not args.headless:
            viewer = cli_args.NativeViewer(env)
        if args.video:
            video = cli_args.VideoRecorder(
                env,
                log_dir / "videos" / "train",
                args.video_length,
                args.video_interval,
            )

        def on_step():
            if viewer is not None:
                viewer.sync()
            if video is not None:
                video.capture()

        wrapped = CustomVecEnvWrapper(env, on_step=on_step)
        runner = OnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=str(log_dir),
            device=args.device,
            command_args=args,
        )
        if resume_path is not None:
            print(f"[INFO] Loading model checkpoint: {resume_path}")
            runner.load(resume_path)
        cli_args.save_config(log_dir, "env", env_cfg)
        cli_args.save_config(log_dir, "agent", agent_cfg)
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
        )
    except KeyboardInterrupt:
        if runner is not None:
            runner.exit_handler()
    finally:
        if video is not None:
            video.close()
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
