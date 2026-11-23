"""Compare saved drill episodes on identical starts with paired bootstrap intervals."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_run(directory):
    summary = json.loads((directory / "summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (directory / "episodes.jsonl").read_text().splitlines()
    ]
    rows.sort(key=lambda row: row["episode"])
    if [row["episode"] for row in rows] != list(range(summary["episodes"])):
        raise ValueError(f"{directory} has missing or duplicated episodes")
    return summary, rows


def compare(reference, competitor):
    ref, left = read_run(reference)
    other, right = read_run(competitor)
    if (ref["drill"], ref["seed"]) != (other["drill"], other["seed"]):
        raise ValueError("The drill and evaluation seed must match")
    if [row["start"] for row in left] != [row["start"] for row in right]:
        raise ValueError("The evaluated starts differ")
    rng = np.random.default_rng(12)
    samples = rng.integers(len(left), size=(5000, len(left)))
    result = {}
    for name in left[0]["metrics"]:
        delta = np.array(
            [a["metrics"][name] - b["metrics"][name] for a, b in zip(left, right)]
        )
        interval = np.quantile(delta[samples].mean(axis=1), (0.025, 0.975))
        result[name] = {
            "mean_difference": float(delta.mean()),
            "ci95": interval.tolist(),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("competitor", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.competitor), indent=2))
