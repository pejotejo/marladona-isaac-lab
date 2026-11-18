import argparse
import logging
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime

from isaaclab.app import AppLauncher


def main(env_cfg, agent_cfg):
    raw_env = None
    wandb_run = None
    log_metrics = None

    try:
        # Simulator-dependent imports must follow AppLauncher.
        import gymnasium as gym
        from isaaclab.utils.io import dump_yaml
        from isaaclab_marl.config import WKS_LOGS_DIR
        from rsl_marl.algorithms.fast_sac.fast_sac_agent import FastSACAgent
        from rsl_marl.algorithms.fast_sac.fast_sac_env import FastSACEnv

        logging.getLogger("rsl_marl.algorithms.fast_sac").setLevel(logging.INFO)

        env_cfg.seed = args_cli.seed

        config_overrides = {}
        if args_cli.max_iterations is not None:
            config_overrides["num_learning_iterations"] = args_cli.max_iterations
        if args_cli.no_compile:
            config_overrides["compile"] = False
        if args_cli.no_amp:
            config_overrides["amp"] = False

        agent_cfg = replace(agent_cfg, **config_overrides)

        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args_cli.run_name:
            run_name += f"_{args_cli.run_name}"

        log_dir = os.path.abspath(
            os.path.join(WKS_LOGS_DIR, args_cli.experiment_name, run_name)
        )
        os.makedirs(log_dir, exist_ok=False)
        print(f"[INFO] Logging run in: {log_dir}")

        dump_yaml(
            os.path.join(log_dir, "params", "env.yaml"),
            env_cfg,
        )
        dump_yaml(
            os.path.join(log_dir, "params", "agent.yaml"),
            asdict(agent_cfg),
        )
        dump_yaml(
            os.path.join(log_dir, "params", "run.yaml"),
            {
                "task": args_cli.task,
                "seed": env_cfg.seed,
                "device": env_cfg.sim.device,
            },
        )

        if args_cli.logger == "wandb":
            import wandb

            wandb_run = wandb.init(
                project=args_cli.log_project_name or args_cli.experiment_name,
                name=args_cli.run_name or run_name,
                dir=log_dir,
                config={
                    "algorithm": "fast_sac",
                    "agent": asdict(agent_cfg),
                    "task": args_cli.task,
                    "seed": env_cfg.seed,
                    "num_envs": env_cfg.scene.num_envs,
                    "device": env_cfg.sim.device,
                },
            )
            log_metrics = lambda metrics, step: wandb_run.log(
                metrics, step=step, commit=True
            )

        raw_env = gym.make(args_cli.task, cfg=env_cfg)
        env = FastSACEnv(raw_env)

        agent = FastSACAgent(
            env,
            agent_cfg,
            device=env.device,
            log_dir=log_dir,
            log_metrics=log_metrics,
        )

        agent.setup()
        agent.learn()

    except Exception:
        logging.exception("FastSAC setup or training failed")
        raise
    finally:
        try:
            if raw_env is not None:
                raw_env.close()
        finally:
            if wandb_run is not None:
                wandb_run.finish(exit_code=1 if sys.exc_info()[0] is not None else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a FastSAC soccer policy.")
    parser.add_argument("--task", default="Isaac-Soccer-FastSAC-v0")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel matches, with one learner per match.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--experiment_name", default="soccer_fast_sac")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--logger", choices=("console", "wandb"), default="console")
    parser.add_argument("--log_project_name", default=None)

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])

    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import isaaclab_marl.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_marl.algorithms.fast_sac.config import FastSACConfig

        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            overrides=hydra_args,
        )
        agent_cfg = FastSACConfig()

        main(env_cfg, agent_cfg)
    finally:
        simulation_app.close()
