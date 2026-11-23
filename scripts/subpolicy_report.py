"""Build an offline HTML report from subpolicy_benchmark.py output."""

import argparse
import html
from pathlib import Path

from compare_subpolicies import compare, read_run

PRIMARY = {
    "midfield": "goals_for",
    "duel": "control_s",
    "positioning": "receptions",
    "defense": "clearances",
}


def table(headers, rows):
    def cells(values, tag):
        return (
            "<tr>"
            + "".join(f"<{tag}>{html.escape(str(x))}</{tag}>" for x in values)
            + "</tr>"
        )

    return (
        "<table>"
        + cells(headers, "th")
        + "".join(cells(row, "td") for row in rows)
        + "</table>"
    )


def report(root):
    runs = {}
    for summary in sorted(root.glob("*/*/summary.json")):
        directory = summary.parent
        drill, name = directory.relative_to(root).parts
        runs[drill, name] = read_run(directory)[0]
    if not runs:
        raise ValueError(f"No finished benchmark evaluations in {root}")
    drills = [drill for drill in PRIMARY if any(d == drill for d, _ in runs)]
    names = sorted({name for _, name in runs})
    matrix, outcomes, differences = [], [], []
    for name in names:
        row = [name]
        for drill in drills:
            run = runs.get((drill, name))
            if run is None:
                row.append("—")
                continue
            metrics = run["metrics"]
            row.append(f"{metrics['return']:.2f}")
            outcomes.append(
                [
                    drill,
                    name,
                    run["episodes"],
                    run["seed"],
                    PRIMARY[drill],
                    f"{metrics[PRIMARY[drill]]:.3f}",
                    run["reasons"].get("goal_a", 0),
                    run["reasons"].get("goal_b", 0),
                    run["reasons"].get("boundary", 0),
                ]
            )
            if name != drill and (drill, drill) in runs:
                result = compare(root / drill / drill, root / drill / name)["return"]
                low, high = result["ci95"]
                differences.append(
                    [
                        drill,
                        name,
                        f"{result['mean_difference']:+.2f}",
                        f"{low:+.2f} to {high:+.2f}",
                    ]
                )
        matrix.append(row)
    return (
        '<!doctype html><meta charset="utf-8"><title>Soccer subpolicy benchmark</title>'
        "<style>body{font:16px system-ui;max-width:64rem;margin:3rem auto;padding:0 1rem;color:#172a25}"
        "table{border-collapse:collapse;margin:1rem 0 2rem}td,th{padding:.45rem .8rem;text-align:right;"
        "border-bottom:1px solid #cbd7d4}td:first-child,th:first-child{text-align:left}th{background:#edf4f1}</style>"
        "<h1>Soccer subpolicy benchmark</h1><h2>Mean episode return</h2>"
        "<p>Compare policies within a drill. Each drill has its own reward scale.</p>"
        + table(["Policy", *drills], matrix)
        + "<h2>Paired comparisons</h2><p>Home specialist minus competitor on the same starts. "
        "Intervals are pointwise 95% percentile bootstrap intervals over episodes.</p>"
        + table(["Home drill", "Competitor", "Difference", "95% interval"], differences)
        + "<h2>Physical outcomes</h2><p>Control is measured in seconds of exclusive ball proximity. "
        "Receptions and clearances are verified events. The reported metric is the per-episode mean.</p>"
        + table(
            [
                "Drill",
                "Policy",
                "Episodes",
                "Seed",
                "Metric",
                "Mean",
                "Goals",
                "Conceded",
                "Out",
            ],
            outcomes,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(report(args.root))
