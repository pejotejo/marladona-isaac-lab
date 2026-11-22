"""Render a portable HTML report using committed archive data only."""

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .archive import read_run


def _table(headings, rows):
    def cell(value):
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        return html.escape(str(value))

    header = "".join(f"<th>{cell(h)}</th>" for h in headings)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _comparison(run, other):
    lhs, rhs = run.manifest["provenance"], other.manifest["provenance"]
    if (
        run.suite.get("opponent") != other.suite.get("opponent")
        or run.suite.get("proximity_radius_m", 0.25)
        != other.suite.get("proximity_radius_m", 0.25)
        or lhs.get("weights_sha256", {}).get("opponent")
        != rhs.get("weights_sha256", {}).get("opponent")
        or any(lhs.get(k) != rhs.get(k) for k in ("backend", "metrics"))
    ):
        return "<h2>Paired trials</h2><p>These archives have incompatible opponent, backend or metric settings.</p>"
    trials = {t["trial_id"]: t for t in run.trials}
    compatible = {t["trial_id"] for t in other.trials if t == trials.get(t["trial_id"])}
    baseline = {r["trial_id"]: r for r in other.results}
    rows = []
    for r in run.results:
        key = r["trial_id"]
        if key in compatible and key in baseline:
            b = baseline[key]
            rows.append(
                (
                    key,
                    r["outcome"],
                    b["outcome"],
                    round(r["elapsed_s"] - b["elapsed_s"], 6),
                )
            )
    return (
        f"<h2>Paired trials</h2><p>{len(rows)} completed pairs with identical frozen specifications.</p>"
        + _table(
            ["Trial", "Run outcome", "Comparison outcome", "Duration difference (s)"],
            rows,
        )
    )


def _replay(path, run):
    if not run.replays:
        return ""
    series = []
    for trial_id, filename in run.replays.items():
        with np.load(path / filename, allow_pickle=False) as data:
            series.append(
                dict(
                    id=trial_id,
                    **{
                        k: data[k].tolist()
                        for k in (
                            "time_s",
                            "ball_xy",
                            "player_xy",
                            "active",
                            "team",
                            "player_heading",
                        )
                        if k in data
                    },
                )
            )
    provenance = run.manifest["provenance"]
    backend = provenance.get("backend", {})
    field = backend if isinstance(backend, dict) else {}
    payload = {
        "series": series,
        "length": field.get("field_length_m"),
        "width": field.get("field_width_m"),
    }
    safe = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    return (
        """
<h2>Retained replays</h2><label>Trial <select id="trial"></select></label>
<svg id="pitch" role="img" aria-label="Players and ball in field coordinates"></svg>
<button id="play">Play</button> <label>Frame <input type="range" min="0" value="0" id="frame"></label>
<output id="clock"></output><p>Team A in blue, team B in orange, ball in white. Centre origin, metres.</p>
<script id="replay-data" type="application/json">"""
        + safe
        + """</script>
<script>
const data=JSON.parse(document.getElementById("replay-data").textContent);
const el=id=>document.getElementById(id), select=el("trial"), frame=el("frame"), pitch=el("pitch");
let timer;
function stop(){clearTimeout(timer);el("play").textContent="Play"}
function draw(){
  const r=data.series[select.selectedIndex], i=Number(frame.value);
  const ns="http://www.w3.org/2000/svg";
  function add(tag,attrs){const n=document.createElementNS(ns,tag);
    for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);pitch.append(n)}
  let length=data.length,width=data.width;
  if(!length || !width){
    length=2;width=2;
    for(let t=0;t<r.time_s.length;t++)for(const p of [r.ball_xy[t],...r.player_xy[t]]){
      length=Math.max(length,2.2*Math.abs(p[0])); width=Math.max(width,2.2*Math.abs(p[1]))}
  }
  pitch.setAttribute("viewBox",[-length*.55,-width*.55,length*1.1,width*1.1].join(" "));
  pitch.replaceChildren();const stroke=length*.004;
  add("rect",{x:-length/2,y:-width/2,width:length,height:width,fill:"#245238",stroke:"white","stroke-width":stroke});
  add("line",{x1:0,x2:0,y1:-width/2,y2:width/2,stroke:"white","stroke-width":stroke});
  function dot(p,color,radius){add("circle",{cx:p[0],cy:-p[1],r:radius,fill:color,stroke:"#132329","stroke-width":stroke})}
  r.player_xy[i].forEach((p,j)=>{if(!r.active[i][j])return;dot(p,r.team[j]===0?"#70c4ff":"#ff9a45",length*.016);
    if(r.player_heading){const h=r.player_heading[i][j];add("line",{x1:p[0],y1:-p[1],x2:p[0]+length*.025*Math.cos(h),y2:-p[1]-length*.025*Math.sin(h),stroke:"white","stroke-width":stroke})}});
  dot(r.ball_xy[i],"white",length*.01);el("clock").textContent=r.time_s[i].toFixed(3)+" s";
}
function change(){stop();frame.value=0;frame.max=data.series[select.selectedIndex].time_s.length-1;draw()}
for(const r of data.series) select.add(new Option(r.id));
select.onchange=change;frame.oninput=()=>{stop();draw()};
el("play").onclick=()=>{
 if(el("play").textContent==="Pause"){stop();return}
 if(+frame.value===+frame.max)frame.value=0;
 el("play").textContent="Pause";
 function next(){draw();let i=+frame.value;const t=data.series[select.selectedIndex].time_s;
   if(i+1===t.length){stop();return}timer=setTimeout(()=>{frame.value=i+1;next()},1000*(t[i+1]-t[i]))}
 next()
};change()
</script>"""
    )


def render_report(path, compare=None):
    path = Path(path)
    run = read_run(path)
    counts = Counter(row["outcome"] for row in run.results)
    chart = "".join(
        f'<p>{html.escape(label)} <meter min="0" max="{len(run.results)}" value="{n}"></meter> {n}</p>'
        for label, n in sorted(counts.items())
    )
    metrics = sorted(
        {m for r in run.results for m in r["metrics"]} - {"elapsed_s", "outcome"}
    )
    rows = [
        (
            r["trial_id"],
            r["outcome"],
            r.get("termination", ""),
            r["elapsed_s"],
            *(r["metrics"].get(m, "") for m in metrics),
        )
        for r in run.results
    ]
    parts = [
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>Evaluation report</title>
<style>body{font:1rem system-ui;margin:2rem auto;max-width:64rem;padding:0 1rem;color:#182620}
table{border-collapse:collapse}td,th{overflow-wrap:anywhere;padding:.45rem;border-bottom:1px solid #ccd6d0;text-align:left}
svg{display:block;max-width:45rem;width:100%;height:20rem;background:#e8eee9;margin:.5rem 0}</style>
<h1>Evaluation report</h1>""",
        f"<p>{len(rows)} of {len(run.trials)} trials committed · {html.escape(run.manifest['status'])}</p>",
        "<details><summary>Suite and provenance</summary>",
        _table(["Setting", "Value"], sorted(run.suite.items())),
        _table(["Provenance", "Value"], sorted(run.manifest["provenance"].items())),
        "</details><h2>Outcomes</h2>",
        chart,
        "<h2>Episodes</h2>",
        _table(["Trial", "Outcome", "Terminal event", "Elapsed (s)", *metrics], rows),
    ]
    if compare:
        parts.append(_comparison(run, read_run(compare)))
    parts.extend(
        [
            _replay(path, run),
            "<details><summary>Events</summary>",
            _table(
                ["Trial", "Time (s)", "Event", "State"],
                [
                    (e["trial_id"], e["time_s"], e["kind"], e.get("state", ""))
                    for e in run.events
                ],
            ),
            "</details></html>",
        ]
    )
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = render_report(args.run, args.compare)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(report)
