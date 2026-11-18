import argparse
import os
import sys

from isaaclab.app import AppLauncher


def main(env_cfg, agent_cfg):
    raw_env = None

    try:
        import gymnasium as gym

        from rsl_marl.algorithms.fast_sac.fast_sac_agent import FastSACAgent
        from rsl_marl.algorithms.fast_sac.fast_sac_env import FastSACEnv

        env_cfg.seed = args_cli.seed
        raw_env = gym.make(args_cli.task, cfg=env_cfg)
        env = FastSACEnv(raw_env)

        agent = FastSACAgent(
            env,
            agent_cfg,
            device=env.device,
            log_dir=os.path.dirname(os.path.abspath(args_cli.checkpoint)),
        )
        agent.setup()
        agent.load(args_cli.checkpoint)

        policy = agent.get_inference_policy()
        obs = env.reset()

        for _ in range(args_cli.max_steps):
            if not simulation_app.is_running():
                break
            obs, _, _, _ = env.step(policy(obs).float())

    finally:
        if raw_env is not None:
            raw_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a FastSAC soccer policy.")
    parser.add_argument("--task", default="Isaac-Soccer-FastSAC-v0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])

    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import torch

        import isaaclab_marl.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_marl.algorithms.fast_sac.config import FastSACConfig

        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            overrides=hydra_args,
        )

        checkpoint = torch.load(
            args_cli.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        agent_cfg = FastSACConfig(**checkpoint["args"])
        del checkpoint

        main(env_cfg, agent_cfg)
    finally:
        simulation_app.close()
