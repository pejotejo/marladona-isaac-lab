"""Resolve suite defaults, scenario defaults, YAML, then explicit CLI values."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .metrics import METRICS
from .scenarios import SCENARIOS


@dataclass(frozen=True)
class Suite:
    scenario: str = "diagnostic"
    episodes: int = 4
    num_envs: int = 2
    max_steps: int = 120
    seed: int = 0
    replay_count: int = 2
    candidate: str = "bot"
    opponent: str = "bot"
    candidate_checkpoint: str | None = None
    opponent_checkpoint: str | None = None
    proximity_radius_m: float = 0.25
    metrics: tuple[str, ...] = tuple(m.name for m in METRICS)
    params: dict = field(default_factory=dict)


class _UniqueKeys(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        values = super().construct_mapping(node, deep=deep)
        if len(node.value) != len(values):
            raise ValueError("duplicate YAML field")
        return values


def _mapping(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")  # noqa: TRY004 - uniform external validation error
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {list(unknown)}")
    return value


def resolve_suite(path=None, overrides=None):
    defaults = asdict(Suite())
    source = {}
    if path is not None:
        try:
            source = yaml.load(Path(path).read_text(), Loader=_UniqueKeys)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid suite YAML: {exc}") from exc
    source = _mapping(source, defaults.keys(), "suite")
    cli = _mapping(overrides if overrides is not None else {}, defaults.keys(), "CLI")
    name = cli.get("scenario", source.get("scenario", defaults["scenario"]))
    if not isinstance(name, str) or name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {name!r}")
    scenario = SCENARIOS[name]
    values = defaults | scenario.DEFAULTS | source | cli
    radius = values["proximity_radius_m"]
    if type(radius) not in (int, float) or not 0 < radius < math.inf:
        raise ValueError("proximity_radius_m must be a positive finite distance")
    names = values["metrics"]
    if (
        not isinstance(names, (list, tuple))
        or any(not isinstance(n, str) or n not in defaults["metrics"] for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("metrics must list unique registered names")
    values["metrics"] = tuple(names)
    params = scenario.PARAMS.copy()
    for layer in (source, cli):
        params.update(
            _mapping(layer.get("params", {}), scenario.PARAMS.keys(), "params")
        )
    values["params"] = scenario.validate_params(params)
    for key, minimum in (
        ("episodes", 1),
        ("num_envs", 1),
        ("max_steps", 1),
        ("seed", 0),
        ("replay_count", 0),
    ):
        if type(values[key]) is not int or not minimum <= values[key] <= 2**31 - 1:
            raise ValueError(
                f"{key} must be an integer between {minimum} and {2**31 - 1}"
            )
    for team in ("candidate", "opponent"):
        controller = values[team]
        checkpoint = values[team + "_checkpoint"]
        if controller not in ("bot", "policy"):
            raise ValueError(f"{team} must be bot or policy")
        if controller == "policy":
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                raise ValueError(f"{team} policy requires a checkpoint path")
            directory = (
                Path(path).resolve().parent
                if path and team + "_checkpoint" not in cli
                else Path.cwd()
            )
            checkpoint = directory / Path(checkpoint).expanduser()
            if not checkpoint.is_file():
                raise ValueError(f"missing {team} checkpoint: {checkpoint}")
            values[team + "_checkpoint"] = str(checkpoint.resolve())
        elif checkpoint is not None:
            raise ValueError(f"{team} bot cannot use a checkpoint")
    return Suite(**values)
