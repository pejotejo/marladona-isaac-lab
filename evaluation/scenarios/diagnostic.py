"""Fixed 5v5 fixture on the playback field, with blue attacking negative x."""

import math
from typing import ClassVar

from .rules import SoccerRules


class Diagnostic:
    RULES = SoccerRules()
    NAME = "diagnostic"
    VERSION = 1
    PARAMETER_VERSION = 1
    DEFAULTS: ClassVar = {"max_steps": 60}
    PARAMS: ClassVar = {"ball_y": 0.0}
    TEAM_SIZE = 5

    @staticmethod
    def validate_params(values):
        ball_y = values["ball_y"]
        if type(ball_y) not in (int, float) or not -1.5 <= ball_y <= 1.5:
            raise ValueError("params.ball_y must be a number within [-1.5, 1.5] metres")
        return {"ball_y": float(ball_y)}

    @classmethod
    def initial_state(cls, trial):
        formation = [(2.25, 0), (1.5, -0.85), (1.5, 0.85), (0.65, -0.48), (0.65, 0.48)]
        return {
            "player_pose": [[x, y, math.pi] for x, y in formation]
            + [[-x, -y, 0.0] for x, y in formation],
            "player_velocity": [[0.0, 0.0, 0.0] for _ in range(2 * cls.TEAM_SIZE)],
            "active": [True] * (2 * cls.TEAM_SIZE),
            "ball_position": [0.0, trial["params"]["ball_y"], 0.3],
            "ball_velocity": [0.0, 0.0, 0.0],
        }

    @staticmethod
    def validate_state(state):
        """Park stationary inactive bodies clear of the pitch to avoid phantom collisions."""
        for active, pose, velocity in zip(
            state["active"], state["player_pose"], state["player_velocity"]
        ):
            if not active and abs(pose[1]) < 2.5:
                raise ValueError("inactive players must be parked at |y| >= 2.5 metres")
            if not active and any(velocity):
                raise ValueError("inactive players must be stationary")
        x, y, _ = state["ball_position"]
        if abs(x) >= 2.7 or abs(y) >= 1.8:
            raise ValueError("diagnostic ball must start strictly inside the field")
