"""Record matched Isaac rollouts and build a standalone pitch replay viewer."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.isaac import open_backend

NAMES = ("midfield", "duel", "positioning", "defense")


def record(backend, drill, args):
    import torch
    from evaluation.subpolicy_training import DrillEnv, inference_policy

    args.drill = drill
    env = DrillEnv(args, backend)
    copies = args.starts
    env.trials = env.trials[:copies] * len(NAMES)
    env.frame = backend.reset(dict(enumerate(env.trials)))
    env.obs = backend.candidate_observation()
    policies = [inference_policy(args.weights / f"{name}.pt", env) for name in NAMES]
    clips = [
        {
            "policy": name,
            "drill": drill,
            "start": i + 1,
            "frames": [],
            "active": [j for j, active in enumerate(env.trials[i]["active"]) if active],
        }
        for name in NAMES
        for i in range(copies)
    ]

    def capture():
        frame = env.frame
        pose = torch.cat(
            (frame["player_xy"], frame["player_heading"].unsqueeze(-1)), dim=-1
        )
        poses = pose.cpu().numpy().round(3).tolist()
        balls = frame["ball_xy"].cpu().tolist()
        times = frame["time_s"].tolist()
        for slot, clip in enumerate(clips):
            if "end" in clip:
                continue
            xyh = [round(v, 3) for j in clip["active"] for v in poses[slot][j]]
            clip["frames"].append(
                [round(times[slot], 3), *map(lambda v: round(v, 3), balls[slot]), *xyh]
            )

    capture()
    for _ in range(max(t["max_steps"] for t in env.trials)):
        actions = torch.cat(
            [policy(obs) for policy, obs in zip(policies, env.obs.split(copies))]
        )
        _, _, reasons = env.advance(actions)
        capture()
        for clip, reason in zip(clips, reasons):
            if reason and "end" not in clip:
                clip["end"] = reason
        if all("end" in clip for clip in clips):
            break
        env.obs = backend.candidate_observation()
    if any("end" not in clip for clip in clips):
        raise RuntimeError("Incomplete rollout")
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "artifacts/subpolicies1/weights"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=73104)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.starts < 1:
        parser.error("--starts must be positive")
    args.num_envs = args.starts * len(NAMES)
    args.headless = True
    suite = SimpleNamespace(
        scenario="diagnostic",
        num_envs=args.num_envs,
        seed=args.seed,
        candidate_checkpoint=None,
        opponent_checkpoint=None,
    )
    template = (ROOT / "evaluation/subpolicy_replay.html").read_text()
    with open_backend(suite, args) as backend:
        clips = [clip for drill in NAMES for clip in record(backend, drill, args)]
        data = {"goalWidth": backend.provenance["goal_width_m"], "clips": clips}
        args.output.write_text(
            template.replace(
                "/*RECORDINGS*/null", json.dumps(data, separators=(",", ":"))
            )
        )


if __name__ == "__main__":
    main()
