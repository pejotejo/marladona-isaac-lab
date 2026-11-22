"""Explicit scenario registry and portable, content-addressed trial format."""

import hashlib
import json
import math
from pathlib import Path

from .diagnostic import Diagnostic

SCENARIOS = {Diagnostic.NAME: Diagnostic}
TRIAL_SCHEMA = 1
STATE_FIELDS = {
    "player_pose",
    "player_velocity",
    "active",
    "ball_position",
    "ball_velocity",
}


def _metadata(suite, scenario, index):
    return {
        "schema_version": TRIAL_SCHEMA,
        "scenario": suite.scenario,
        "scenario_version": scenario.VERSION,
        "parameter_version": scenario.PARAMETER_VERSION,
        "params": suite.params,
        "index": index,
        "seed": (suite.seed + index) % 2**31,
        "max_steps": suite.max_steps,
    }


def _trial_id(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _vector(value, length):
    return (
        type(value) is list
        and len(value) == length
        and all(type(x) in (float, int) and math.isfinite(x) for x in value)
    )


def _validate(record, suite, scenario, index):
    meta = _metadata(suite, scenario, index)
    if not isinstance(record, dict) or record.keys() != meta.keys() | STATE_FIELDS | {
        "trial_id"
    }:
        raise ValueError("invalid trial fields")
    stored_meta = {key: record[key] for key in meta}
    if _trial_id(meta) != _trial_id(stored_meta):
        raise ValueError(
            "frozen trial metadata disagrees with resolved suite or trial index"
        )
    count = 2 * scenario.TEAM_SIZE
    for key in ("player_pose", "player_velocity"):
        rows = record[key]
        if (
            not isinstance(rows, list)
            or len(rows) != count
            or not all(_vector(row, 3) for row in rows)
        ):
            raise ValueError(f"invalid trial {key}")
    active = record["active"]
    if (
        type(active) is not list
        or len(active) != count
        or any(type(flag) is not bool for flag in active)
    ):
        raise ValueError("invalid trial active mask")
    if not all(_vector(record[key], 3) for key in ("ball_position", "ball_velocity")):
        raise ValueError("invalid trial ball state")
    scenario.validate_state(record)
    if record["trial_id"] != _trial_id(
        {k: v for k, v in record.items() if k != "trial_id"}
    ):
        raise ValueError("trial_id does not match the frozen trial content")


def make_trials(suite, path=None):
    """One fixed-side episode per record. Controller identities live in the suite manifest."""
    scenario = SCENARIOS[suite.scenario]
    if path is not None:
        with Path(path).open() as stream:
            records = []
            for index, line in enumerate(stream):
                if index >= suite.episodes:
                    raise ValueError("frozen trial count exceeds suite episodes")
                try:
                    record = json.loads(line)
                    _validate(record, suite, scenario, index)
                except (ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"invalid frozen trial on line {index + 1}: {exc}"
                    ) from exc
                records.append(record)
        if len(records) != suite.episodes:
            raise ValueError("frozen trial count differs from suite episodes")
        return records
    records = []
    for index in range(suite.episodes):
        meta = _metadata(suite, scenario, index)
        record = meta | scenario.initial_state(meta)
        record["trial_id"] = _trial_id(record)
        _validate(record, suite, scenario, index)
        records.append(record)
    return records
