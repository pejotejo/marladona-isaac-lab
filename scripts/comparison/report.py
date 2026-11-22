"""Build a local comparison page from experiment scalars; no tracking account required."""

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

COLORS = ("#2563eb", "#e17825", "#15803d", "#9333ea")


def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def load_run(path):
    metadata = read_json(path / "metadata.json")
    series = defaultdict(list)
    scalar_path = path / "scalars.csv"
    if scalar_path.exists():
        with scalar_path.open() as stream:
            for row in csv.DictReader(stream):
                try:
                    elapsed, value = float(row["elapsed_s"]), float(row["value"])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(elapsed) and math.isfinite(value):
                    series[row["tag"]].append((elapsed, value))
    timing_columns = {
        "iteration_wall_s": "Timing/iteration_wall_s",
        "rollout_wall_s": "Timing/rollout_wall_s",
        "learning_wall_s": "Timing/learning_wall_s",
        "environment_transitions_per_s": "Throughput/environment_transitions_per_s",
        "actor_transitions_per_s": "Throughput/actor_transitions_per_s",
        "sampled_step_ms_mean": "Timing/sampled_step_ms_mean",
        "sampled_step_ms_p50": "Timing/sampled_step_ms_p50",
        "sampled_step_ms_p95": "Timing/sampled_step_ms_p95",
    }
    gpu_columns = {
        "utilization.gpu": "GPU/utilization_percent",
        "memory.used": "GPU/memory_used_MiB",
        "temperature.gpu": "GPU/temperature_C",
        "power.draw": "GPU/power_W",
        "clocks.sm": "GPU/clock_MHz",
    }
    for filename, columns in (("iterations.csv", timing_columns), ("gpu.csv", gpu_columns)):
        source = path / filename
        if not source.exists():
            continue
        with source.open() as stream:
            for row in csv.DictReader(stream):
                if filename == "iterations.csv" and row.get("phase") != "steady":
                    continue
                for column, tag in columns.items():
                    try:
                        elapsed, value = float(row["elapsed_s"]), float(row[column])
                    except (KeyError, ValueError):
                        continue
                    if math.isfinite(elapsed) and math.isfinite(value):
                        series[tag].append((elapsed, value))
    return {
        "path": str(path.resolve()),
        "engine": metadata.get("engine", path.name),
        "metadata": metadata,
        "progress": read_json(path / "progress.json"),
        "summary": read_json(path / "summary.json"),
        "series": dict(series),
    }


def chart(tag, runs):
    available = [(run, run["series"].get(tag, [])) for run in runs]
    available = [(run, sorted(points)) for run, points in available if points]
    if not available:
        return ""
    all_points = [point for _, points in available for point in points]
    max_x = max(max(p[0] for p in all_points), 1)
    min_y, max_y = min(p[1] for p in all_points), max(p[1] for p in all_points)
    padding = (max_y - min_y) * 0.05 or max(abs(max_y) * 0.05, 0.01)
    min_y, max_y = min_y - padding, max_y + padding

    def xy(point):
        return 80 + point[0] / max_x * 800, 250 - (point[1] - min_y) / (max_y - min_y) * 220

    parts = [
        (
            f'<section><h2>{html.escape(tag)}</h2><svg viewBox="0 0 920 300" role="img" '
            f'aria-label="{html.escape(tag, quote=True)} over elapsed wall time">'
        )
    ]
    for i in range(5):
        y = 250 - i * 55
        value = min_y + i / 4 * (max_y - min_y)
        parts.append(
            f'<path d="M80 {y} H880" stroke="#e2e8f0"/><text x="72" y="{y + 4}" text-anchor="end">{value:.4g}</text>'
        )
        x = 80 + i * 200
        parts.append(f'<text x="{x}" y="274" text-anchor="middle">{max_x * i / 4 / 60:.1f}</text>')
    parts.append('<text x="480" y="296" text-anchor="middle">Elapsed wall time (minutes)</text>')
    for index, (run, points) in enumerate(available):
        color = COLORS[index % len(COLORS)]
        stride = max(1, math.ceil(len(points) / 800))
        sampled = points[::stride]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(xy, sampled))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>')
        for point in sampled:
            x, y = xy(point)
            title = html.escape(f"{run['engine']}: {point[1]:.6g} at {point[0]:.1f}s")
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" opacity="0.1"><title>{title}</title></circle>'
            )
    parts.append(
        "</svg><p>"
        + " · ".join(
            f'<span style="color:{COLORS[i % len(COLORS)]}">{html.escape(str(run["engine"]))}</span>'
            for i, (run, _) in enumerate(available)
        )
        + "</p></section>"
    )
    return "".join(parts)


def build_report(paths, output):
    runs = [load_run(path) for path in paths]
    tags = sorted(set().union(*(run["series"] for run in runs)))
    preferred = (
        "Timing/sampled_step_ms_mean",
        "Timing/sampled_step_ms_p95",
        "Throughput/environment_transitions_per_s",
        "Timing/iteration_wall_s",
        "Train/mean_reward",
        "Eval/mean_reward",
        "Train/mean_episode_length",
        "Eval/mean_episode_length",
        "Eval/mean_score_blue_all",
        "Eval/mean_score_red_all",
        "Loss/value_function",
        "Loss/surrogate",
        "Loss/entropy",
        "Loss/learning_rate",
    )
    ordered = [tag for tag in preferred if tag in tags] + [tag for tag in tags if tag not in preferred]
    summary_rows = []
    for run in runs:
        for tag, points in sorted(run["series"].items()):
            values = [point[1] for point in points]
            summary_rows.append(
                {
                    "engine": run["engine"],
                    "tag": tag,
                    "samples": len(values),
                    "last": values[-1],
                    "last_100_mean": statistics.mean(values[-100:]),
                    "mean": statistics.mean(values),
                    "elapsed_s": points[-1][0],
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("engine", "tag", "samples", "last", "last_100_mean", "mean", "elapsed_s")
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    state = [{key: value for key, value in run.items() if key != "series"} for run in runs]
    (output / "runs.json").write_text(json.dumps(state, indent=2))
    details = "".join(
        f"<details><summary>{html.escape(str(run['engine']))} — run metadata</summary><pre>"
        f"{html.escape(json.dumps(item, indent=2))}</pre></details>"
        for run, item in zip(runs, state)
    )
    plots = "".join(chart(tag, runs) for tag in ordered)
    if not plots:
        plots = "<p>No training scalar samples are available yet. Regenerate this page after training starts.</p>"
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MARLadona training comparison</title><style>
body{font:15px system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033;background:#f4f7fb}
section,details{background:white;border:1px solid #dbe3ec;border-radius:10px;margin:18px 0;padding:18px}
h1{font-size:28px}h2{font-size:17px}svg{width:100%;height:auto}svg text{font:12px system-ui;fill:#475569}
pre{overflow:auto;max-height:360px;font-size:12px}summary{cursor:pointer}p{line-height:1.5}
</style><h1>MARLadona training comparison</h1>
<p>Local training metrics, plotted against measured wall time. Hover over points for values.
Warmup samples and known implementation differences are recorded in each run's metadata.
These are single-seed experiments; the curves describe these runs.</p>"""
    (output / "comparison.html").write_text(page + details + plots + "</html>")
    return output / "comparison.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in args.runs:
        if not path.is_dir():
            parser.error(f"Run directory does not exist: {path}")
    print(build_report(args.runs, args.output))


if __name__ == "__main__":
    main()
