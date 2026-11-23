"""Evaluate each named specialist on the same trials in each requested drill."""

import argparse
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.isaac import open_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", nargs=2, action="append", default=[], metavar=("NAME", "CHECKPOINT")
    )
    parser.add_argument("--bot", action="store_true")
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--idle", action="store_true")
    parser.add_argument(
        "--drills",
        nargs="+",
        choices=("midfield", "duel", "positioning", "defense"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    candidates = [(name, Path(path)) for name, path in args.policy]
    if any(not path.is_file() for _, path in candidates):
        parser.error("A checkpoint file is missing")
    for baseline in ("bot", "teacher", "idle"):
        if getattr(args, baseline):
            candidates.append((baseline, None))
    labels = [name for name, _ in candidates]
    if (
        not labels
        or len(set(labels)) != len(labels)
        or min(args.num_envs, args.episodes) <= 0
    ):
        parser.error("Supply unique candidate names and positive counts")
    if any(not label.replace("-", "").replace("_", "").isalnum() for label in labels):
        parser.error("Use simple candidate names without directory separators")
    args.headless = True
    suite = SimpleNamespace(
        scenario="diagnostic",
        num_envs=args.num_envs,
        seed=args.seed,
        candidate_checkpoint=None,
        opponent_checkpoint=None,
    )
    with open_backend(suite, args) as backend:
        try:
            from evaluation.subpolicy_training import run

            for drill in args.drills:
                for name, checkpoint in candidates:
                    run(
                        SimpleNamespace(
                            mode="eval",
                            drill=drill,
                            num_envs=args.num_envs,
                            seed=args.seed,
                            episodes=args.episodes,
                            device=args.device,
                            checkpoint=checkpoint,
                            baseline=None if checkpoint else name,
                            output=args.output / drill / name,
                        ),
                        backend,
                    )
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
