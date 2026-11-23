"""Paired seeded 1v1 drills on the diagnostic field. Blue attacks negative x."""

import math

import torch


class Drill:
    """Reset samples form one reproducible CPU RNG stream across batch sizes."""

    def __init__(self, name, num_envs, device, seed):
        if name not in ("midfield", "duel"):
            raise ValueError(f"unknown drill {name!r}")
        self.name = name
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def starts(self, count):
        """Sample both initial geometry and the defender's behavior."""
        samples = torch.rand((count, 10), generator=self.rng, device="cpu")
        trials = []
        for (
            bx,
            by,
            gap,
            lateral,
            phase,
            angle,
            speed,
            mode,
            rival_gap,
            heading,
        ) in samples.tolist():
            bx, by = bx - 0.5, (by - 0.5) * 1.8
            gap = 0.19 + gap * 0.24
            blue_angle = (lateral - 0.5) * (2.4 if self.name == "duel" else 1.3)
            poses = [
                [(i % 5 - 2) * 0.7, 2.7 if i < 5 else -2.7, 0.0] for i in range(10)
            ]
            poses[1] = [
                bx + gap * math.cos(blue_angle),
                by + gap * math.sin(blue_angle),
                blue_angle + math.pi + (heading - 0.5) * 1.2,
            ]
            if self.name == "duel":
                red_angle = blue_angle + math.pi + (angle - 0.5) * 2.8
                defense_gap = 0.20 + 0.38 * rival_gap
                behavior = ("challenge", "block", "approach")[int(mode * 3)]
                pace = 0.45 + 0.55 * speed
            else:
                red_angle = math.pi + (angle - 0.5) * 1.1
                defense_gap = 1.20 + 0.62 * rival_gap
                behavior = "hold" if mode < 0.55 else "block"
                pace = 0.10 + 0.25 * speed
            poses[6] = [
                bx + defense_gap * math.cos(red_angle),
                by + defense_gap * math.sin(red_angle),
                red_angle + math.pi,
            ]
            trials.append(
                {
                    "player_pose": poses,
                    "player_velocity": [[0.0, 0.0, 0.0] for _ in poses],
                    "active": [i in (1, 6) for i in range(10)],
                    "ball_position": [bx, by, 0.055],
                    "ball_velocity": [0.0, 0.0, 0.0],
                    "max_steps": 160,
                    "kick_phase": phase,
                    "behavior": behavior,
                    "pace": pace,
                }
            )
        return trials

    def script(self, frame, trials):
        """Hold position, screen a shooting lane, close down, or challenge and kick."""
        ball = frame["ball_xy"]
        red_xy = frame["player_xy"][:, 6]
        lane = torch.stack(
            ((ball[:, 0] - 0.55).clamp_min(-2.40), 0.75 * ball[:, 1]), dim=-1
        )
        blocking = ball.new_tensor(
            [t["behavior"] == "block" for t in trials], dtype=torch.bool
        )
        delta = torch.where(blocking[:, None], lane, ball) - red_xy
        heading = frame["player_heading"][:, 6]
        c, s = heading.cos(), heading.sin()
        x, y = delta.unbind(-1)
        forward = c * x + s * y
        sideways = -s * x + c * y
        actions = ball.new_zeros((len(ball), 10, 7))
        red = actions[:, 6]
        red[:, 0] = (2 * forward).clamp(-1, 1)
        red[:, 1] = (2 * sideways).clamp(-1, 1)
        red[:, 2] = torch.atan2(sideways, forward).clamp(-1, 1)
        pace = ball.new_tensor(
            [0 if t["behavior"] == "hold" else t["pace"] for t in trials]
        )
        red[:, :3] *= pace[:, None]
        phase = ball.new_tensor([t["kick_phase"] for t in trials])
        challenge = ball.new_tensor(
            [t["behavior"] == "challenge" for t in trials], dtype=torch.bool
        )
        red[:, 3] = (
            challenge
            & ((frame["time_s"] + phase) % 1 < 0.4)
            & (delta.norm(dim=-1) < 0.45)
        )
        red[:, 4] = -1
        red[:, 6] = 1
        return actions

    def measure(self, previous, current, reasons, trials):
        """Sum metrics over live steps. Control seconds proxy exclusive proximity
        (within 0.3 m and closer than red). Compare seconds / playing seconds,
        progress metres / trial and goal counts / trial on identical seeded resets.
        Reward combines net forward progress, approach and signed goal outcome.
        """
        ball = current["ball_xy"]
        old_ball = previous["ball_xy"]
        distance = (ball - current["player_xy"][:, 1]).norm(dim=-1)
        old_distance = (old_ball - previous["player_xy"][:, 1]).norm(dim=-1)
        rival = (ball - current["player_xy"][:, 6]).norm(dim=-1)
        dt = (current["time_s"] - previous["time_s"]).to(ball)
        goal = ball.new_tensor([-2.7, 0.0])
        progress = (old_ball - goal).norm(dim=-1) - (ball - goal).norm(dim=-1)
        scored = ball.new_tensor([r == "goal_a" for r in reasons])
        conceded = ball.new_tensor([r == "goal_b" for r in reasons])
        outside = ball.new_tensor([r == "boundary" for r in reasons])
        control = (distance < 0.30) & (distance + 0.02 < rival)
        red_control = (rival < 0.30) & (rival + 0.02 < distance)
        old_rival = (old_ball - previous["player_xy"][:, 6]).norm(dim=-1)
        old_control = (old_distance < 0.30) & (old_distance + 0.02 < old_rival)
        loss = old_control & ~control
        # Avoid farming repeated acquisition transitions. This term telescopes.
        control_delta = control.float() - old_control.float()
        reward = (
            1.7 * progress
            + 5 * scored
            - 5 * conceded
            - 2.5 * outside
            + 0.35 * (old_distance - distance)
            - dt * (0.02 + 0.35 * red_control)
        )
        if self.name == "duel":
            reward += 0.50 * control_delta + dt * (0.24 * control - 0.40 * red_control)
        return reward, {
            "playing_s": dt,
            "control_s": dt * control,
            "opponent_control_s": dt * red_control,
            "possession_exits": loss.float(),
            "gains": (~old_control & control).float(),
            "boundary_exits": outside,
            "progress_m": progress,
            "goals_for": scored,
            "goals_against": conceded,
        }
