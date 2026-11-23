"""Train or evaluate one candidate on an Isaac soccer drill."""
import argparse
import sys
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
    parser.add_argument("--baseline", choices=["bot", "script"])
    parser.add_argument("--checkpoint", "--resume", type=Path)
    for name, default in (("num-envs", 32), ("seed", 1), ("episodes", 64),
                          ("updates", 100), ("learning-starts", 32), ("batch-size", 512), ("num-updates", 2), ("checkpoint-interval", 10)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if args.mode == "eval" and bool(args.checkpoint) == bool(args.baseline):
        parser.error("eval requires exactly one of --checkpoint or --baseline")
    if args.mode == "train" and args.baseline:
        parser.error("--baseline applies to eval only")
    if min(args.episodes, args.num_envs, args.updates, args.batch_size, args.num_updates, args.checkpoint_interval) < 1:
        parser.error("counts must be positive")
    args.headless = True
    suite = SimpleNamespace(scenario="diagnostic", num_envs=args.num_envs, seed=args.seed,
                           candidate_checkpoint=None, opponent_checkpoint=None)
    with open_backend(suite, args, training=args.mode == "train") as backend:
        from evaluation.subpolicy_training import run
        run(args, backend)


if __name__ == "__main__":
    main()
