"""Version 1 evaluation archives. Each results.jsonl line commits one episode."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=False)


def _save(path, contents):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append(path, contents):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def _rows(path):
    with path.open("rb") as stream:
        return [json.loads(line) for line in stream if line.endswith(b"\n")]


def _replay_file(trial_id):
    return f"replays/{hashlib.sha256(trial_id.encode()).hexdigest()}.npz"


def _arrays(data, elapsed):
    arrays = {key: np.asarray(value) for key, value in data.items()}
    if any(a.dtype.kind not in "biufUS" for a in arrays.values()):
        raise ValueError("replays require numeric or fixed-width string arrays")
    times, team = arrays["time_s"], arrays["team"]
    if times.ndim != 1 or not len(times) or times[0] != 0 or times[-1] != elapsed:
        raise ValueError("replay must include initial and exact terminal time")
    if (
        not np.all(np.diff(times) > 0)
        or team.ndim != 1
        or not np.isin(team, [0, 1]).all()
    ):
        raise ValueError("invalid timestamps or teams")
    shape = (len(times), len(team))
    for key, expected in (
        ("active", shape),
        ("player_heading", shape),
        ("ball_velocity", (len(times), 3)),
        ("player_xy", (*shape, 2)),
        ("ball_xy", (len(times), 2)),
    ):
        if key not in arrays or arrays[key].shape != expected or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid replay {key}")
    return arrays


class Archive:
    def __init__(self, path, *, suite, trials, provenance, replay_limit, seed):
        self.path = Path(path)
        ids = [trial["trial_id"] for trial in trials]
        self.pending = set(ids)
        if len(self.pending) != len(ids) or any(
            not isinstance(i, str) or not i for i in ids
        ):
            raise ValueError("trial IDs must be unique nonempty strings")
        if type(replay_limit) is not int or replay_limit < 0:
            raise ValueError("invalid replay limit")
        ranked = sorted(
            ids, key=lambda i: hashlib.sha256(_json([seed, i]).encode()).digest()
        )
        self.replay_ids = frozenset(ranked[:replay_limit])
        self.manifest = {
            "schema_version": 1,
            "replay_version": 2,
            "status": "partial",
            "provenance": provenance,
            "replay_ids": sorted(self.replay_ids),
        }
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "replays").mkdir()
        _save(self.path / "suite.json", _json(suite))
        _save(self.path / "trials.jsonl", "".join(_json(t) + "\n" for t in trials))
        for name in ("results.jsonl", "events.jsonl"):
            (self.path / name).touch()
        self._manifest()

    def _manifest(self):
        _save(self.path / "manifest.json", _json(self.manifest))

    def complete(self, result, events, replay=None):
        required = {"trial_id", "termination", "outcome", "elapsed_s", "metrics"}
        if missing := required - result.keys():
            raise ValueError(f"missing episode fields: {sorted(missing)}")
        trial_id = result["trial_id"]
        if trial_id not in self.pending or any(
            e["trial_id"] != trial_id for e in events
        ):
            raise ValueError("unknown, duplicated or inconsistent trial ID")
        if (trial_id in self.replay_ids) != (replay is not None):
            raise ValueError("replay does not match selected trial")
        result_line = _json(result) + "\n"
        event_lines = "".join(_json(e) + "\n" for e in events)
        if replay is not None:
            arrays = _arrays(replay, result["elapsed_s"])
            path = self.path / _replay_file(trial_id)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        _append(self.path / "events.jsonl", event_lines)
        _append(self.path / "results.jsonl", result_line)
        self.pending.remove(trial_id)

    def finish(self):
        if self.pending:
            raise ValueError(f"{len(self.pending)} trial(s) still pending")
        self.manifest["status"] = "complete"
        self._manifest()


@dataclass
class Run:
    manifest: dict
    suite: dict
    trials: list
    results: list
    events: list
    replays: dict


def read_run(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported evaluation archive schema")
    results = _rows(path / "results.jsonl")
    committed = {row["trial_id"] for row in results}
    return Run(
        manifest,
        json.loads((path / "suite.json").read_text(encoding="utf-8")),
        _rows(path / "trials.jsonl"),
        results,
        [e for e in _rows(path / "events.jsonl") if e["trial_id"] in committed],
        {i: _replay_file(i) for i in manifest["replay_ids"] if i in committed},
    )
