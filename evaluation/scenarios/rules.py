"""Batched scenario stopping rules. Zero continues the trial."""


class SoccerRules:
    reasons = (None, "goal_a", "goal_b", "boundary", "timeout")
    timeout = 4

    def classify(self, frame, physics_event):
        return physics_event

    def outcome(self, reason):
        return {"goal_a": "win", "goal_b": "loss"}.get(reason, "draw")
