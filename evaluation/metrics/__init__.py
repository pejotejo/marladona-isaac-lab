"""Metric definitions consume episode truth in field coordinates."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    name: str
    unit: str
    requires: tuple[str, ...]
    measure: Callable
    version: int = 1


METRICS = (
    Metric("elapsed_s", "s", ("time_s",), lambda t, reason, signals: t),
    Metric(
        "goals_a",
        "goals",
        ("terminal_event",),
        lambda t, reason, signals: int(reason == "goal_a"),
    ),
    Metric(
        "goals_b",
        "goals",
        ("terminal_event",),
        lambda t, reason, signals: int(reason == "goal_b"),
    ),
    Metric(
        "proximity_turnovers",
        "transfers",
        ("possession",),
        lambda t, reason, signals: signals["possession"].turnovers,
    ),
)


class Possession:
    """Proximity proxy sampled at controller frames. Both teams in range means contested."""

    def __init__(self):
        self.last_owner = None
        self.state = None
        self.turnovers = 0

    requires = ("ball_xy", "player_xy", "active", "team")

    @staticmethod
    def classify(frame, radius_m=0.25):
        delta = frame["player_xy"] - frame["ball_xy"][:, None, :]
        near = frame["active"] & ((delta * delta).sum(axis=-1) <= radius_m * radius_m)
        a = (near & (frame["team"] == 0)).any(axis=-1)
        b = (near & (frame["team"] == 1)).any(axis=-1)
        return (a + 2 * b).tolist()

    def observe(self, code):
        self.state = ("unowned", "team_a", "team_b", "contested")[code]
        if self.state in ("team_a", "team_b"):
            self.turnovers += int(
                self.last_owner is not None and self.last_owner != self.state
            )
            self.last_owner = self.state
        return self.state


# Observers receive the initial, decision and exact terminal frames of a single
# trial. Returning a changed categorical state emits a timestamped event.
OBSERVERS = {"possession": Possession}
