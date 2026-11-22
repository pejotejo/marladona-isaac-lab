"""View a MARLadona checkpoint in mjlab's browser viewer on a headless machine."""

import argparse
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="Use CPU to leave the training GPUs free.")
    parser.add_argument("--num_agents", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--opponent", choices=("policy", "bot"), default="policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Stop after this many simulation steps; 0 runs until Ctrl+C."
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.max_steps < 0:
        parser.error("--max_steps must be nonnegative")

    import torch
    import viser
    from mjlab.viewer.viser import ViserPlayViewer
    from mjlab_marl import SoccerMARLEnv, SoccerMARLEnvPlayCfg
    from mjlab_marl.tasks.soccer.agents.soccer_marl_ppo_runner_cfg import SoccerMARLPPORunnerCfg
    from rsl_marl.runners import OnPolicyRunner
    from rsl_marl.utils.custom_vecenv_wrapper import CustomVecEnvWrapper

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    cfg = SoccerMARLEnvPlayCfg(seed=args.seed)
    cfg.soccer_game = replace(
        cfg.soccer_game,
        num_agents_per_team=args.num_agents,
        num_agents_range=[args.num_agents, args.num_agents],
        default_control=[1, 3 if args.opponent == "bot" else 1],
        agent_id_control=None,
    )
    env = SoccerMARLEnv(cfg, device=args.device)
    server = None
    try:
        wrapped = CustomVecEnvWrapper(env)
        runner_cfg = SoccerMARLPPORunnerCfg(device=args.device)
        runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=args.device, command_args=args)
        runner.load(checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=args.device)

        def actions(obs):
            with torch.inference_mode():
                combined = torch.cat((obs["policy"], obs["neighbor"]), dim=-1)
                commands = torch.zeros(env.env_data.num_agents, 5, device=env.device)
                policy_actors = env.env_data.training_actor_index | env.env_data.eval_actor_index
                commands[policy_actors] = policy(combined[policy_actors])
                if args.opponent == "bot":
                    bots = env.env_data.bots_actor_index
                    commands[bots] = runner.bot_policy.act(
                        obs["bots"], commands, bots, configs=["keeper"] + ["player"] * (args.num_agents - 1)
                    )
                return commands

        server = viser.ViserServer(host=args.host, port=args.port, label="MARLadona / mjlab")
        server.gui.add_markdown(
            f"**Checkpoint:** `{checkpoint.name}`\n\n"
            f"**Match:** {args.num_agents}v{args.num_agents}, opponent: {args.opponent}, seed: {args.seed}\n\n"
            "The Rewards tab shows the first blue player's reward terms."
        )
        viewer = ViserPlayViewer(wrapped, actions, viser_server=server, frame_rate=30)
        print(f"[INFO] Viewer: http://{args.host}:{server.get_port()} (SSH-forward this port)", flush=True)
        print(f"[INFO] Checkpoint: {checkpoint}", flush=True)
        viewer.run(num_steps=args.max_steps or None)
        if viewer.get_status().last_error:
            raise RuntimeError(viewer.get_status().last_error)
        print(f"[INFO] Viewer stopped after {viewer.get_status().step_count} steps", flush=True)
    finally:
        if server is not None:
            server.stop()
        env.close()


if __name__ == "__main__":
    main()
