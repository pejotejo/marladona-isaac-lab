"""Evaluate a frozen suite with the Isaac soccer environment."""

import argparse
import hashlib
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.archive import Archive
from evaluation.config import resolve_suite
from evaluation.metrics import METRICS
from evaluation.runner import run_trials
from evaluation.scenarios import make_trials


def weight_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision():
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--trials", type=Path, help="Reuse a frozen trials.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    for key in ("episodes", "num-envs", "max-steps", "seed", "replay-count"):
        parser.add_argument("--" + key, type=int)
    for team in ("candidate", "opponent"):
        parser.add_argument("--" + team, choices=["bot", "policy"])
        parser.add_argument("--" + team + "-checkpoint")
    args = parser.parse_args(argv)
    options = vars(args).copy()
    for key in ("suite", "trials", "output", "device", "headless"):
        del options[key]
    try:
        suite = resolve_suite(
            args.suite, {k: v for k, v in options.items() if v is not None}
        )
        trials = make_trials(suite, args.trials)
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    from evaluation.isaac import open_backend

    with open_backend(suite, args) as backend:
        provenance = {
            "backend": backend.provenance,
            "source": source_revision(),
            "launcher": {"device": args.device, "headless": args.headless},
        }
        provenance["metrics"] = {
            m.name: {"version": m.version, "units": m.unit, "requires": m.requires}
            for m in METRICS
            if m.name in suite.metrics
        }
        provenance["weights_sha256"] = {
            team: weight_digest(path)
            for team in ("candidate", "opponent")
            if (path := getattr(suite, team + "_checkpoint"))
        }
        archive = Archive(
            args.output,
            suite=asdict(suite),
            trials=trials,
            provenance=provenance,
            replay_limit=suite.replay_count,
            seed=suite.seed,
        )
        run_trials(
            backend,
            trials,
            suite.num_envs,
            archive,
            metrics=suite.metrics,
            proximity_radius_m=suite.proximity_radius_m,
        )
        archive.finish()


if __name__ == "__main__":
    main()
