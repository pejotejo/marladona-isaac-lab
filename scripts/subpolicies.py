"""Train or evaluate one candidate on an Isaac soccer drill."""

import argparse
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.isaac import open_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--drill", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--baseline", choices=["bot", "teacher", "idle"])
    parser.add_argument("--checkpoint", "--resume", type=Path)
    parser.add_argument(
        "--teacher-steps",
        type=int,
        default=0,
        help="Initialize the FastSAC actor from scripted demonstrations before RL",
    )
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument(
        "--recovery-steps",
        type=int,
        default=0,
        help="Extra demonstration steps from states visited by the learned actor",
    )
    parser.add_argument(
        "--demo-weight",
        type=float,
        default=0,
        help="Weight of demonstration imitation in the FastSAC actor loss",
    )
    parser.add_argument("--gamma", type=float, default=0.97)
    for name, default in (
        ("num-envs", 32),
        ("seed", 1),
        ("episodes", 64),
        ("updates", 100),
        ("learning-starts", 32),
        ("batch-size", 512),
        ("num-updates", 2),
        ("checkpoint-interval", 10),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if args.mode == "eval" and bool(args.checkpoint) == bool(args.baseline):
        parser.error("eval requires exactly one of --checkpoint or --baseline")
    if args.mode == "train" and args.baseline:
        parser.error("--baseline applies to eval only")
    if not 0 < args.gamma < 1:
        parser.error("gamma must be between zero and one")
    if min(args.recovery_steps, args.demo_weight) < 0 or (
        (args.demo_weight or args.recovery_steps) and not args.teacher_steps
    ):
        parser.error("a positive demo weight requires teacher steps")
    if (
        args.teacher_steps < 0
        or args.actor_lr <= 0
        or (args.teacher_steps and args.checkpoint)
    ):
        parser.error(
            "teacher steps require a fresh training run and actor-lr must be positive"
        )
    if (
        min(
            args.episodes,
            args.num_envs,
            args.updates,
            args.batch_size,
            args.num_updates,
            args.checkpoint_interval,
        )
        < 1
    ):
        parser.error("counts must be positive")
    suite = SimpleNamespace(
        scenario="diagnostic",
        num_envs=args.num_envs,
        seed=args.seed,
        candidate_checkpoint=None,
        opponent_checkpoint=None,
    )
    with open_backend(suite, args, training=args.mode == "train") as backend:
        from evaluation.subpolicy_training import run

        try:
            run(args, backend)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
