"""Loose-ball defense drill. Blue 1 defends the positive-x goal against red 6."""

import math

import torch


class Defense:
    def __init__(self, num_envs, device, seed):
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.recovered = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.cleared = torch.zeros_like(self.recovered)
        self.pending = torch.zeros_like(self.recovered)
        self.kick_x = torch.zeros(num_envs, device=device)
        self.control_time = torch.zeros_like(self.kick_x)

    def reset(self, slots):
        for tensor in (
            self.recovered,
            self.cleared,
            self.pending,
            self.kick_x,
            self.control_time,
        ):
            tensor[slots] = 0

    def starts(self, count):
        trials = []
        for x, y, speed, gap, offset, pace, lead in torch.rand(
            count, 7, generator=self.rng, device="cpu"
        ).tolist():
            x, y = -0.35 + 1.45 * x, (y - 0.5) * 1.9
            angle = math.atan2(-y, 2.7 - x)
            c, s = math.cos(angle), math.sin(angle)
            gap = 0.38 + 0.9 * gap
            lead = 0.48 + 0.65 * lead
            poses = [
                [0.65 * (i % 5 - 2), 2.7 if i < 5 else -2.7, 0.0] for i in range(10)
            ]
            poses[6] = [x - gap * c, y - gap * s, angle]
            poses[1] = [
                x + lead * c,
                y + lead * s + 0.6 * (offset - 0.5),
                angle + math.pi,
            ]
            speed = 0.18 + 0.57 * speed
            trials.append(
                {
                    "player_pose": poses,
                    "player_velocity": [[0.0, 0.0, 0.0] for _ in poses],
                    "active": [i in (1, 6) for i in range(10)],
                    "ball_position": [x, y, 0.055],
                    "ball_velocity": [speed * c, speed * s, 0.0],
                    "pace": 0.55 + 0.45 * pace,
                    "max_steps": 220,
                }
            )
        return trials

    @staticmethod
    def _move(frame, player, target, facing):
        xy = frame["player_xy"][:, player]
        yaw = frame["player_heading"][:, player]
        dx, dy = (target - xy).unbind(-1)
        c, s = yaw.cos(), yaw.sin()
        action = xy.new_zeros((len(xy), 7))
        action[:, 0] = (3 * (c * dx + s * dy)).clamp(-1, 1)
        action[:, 1] = (3 * (c * dy - s * dx)).clamp(-1, 1)
        angle = torch.atan2(facing[:, 1], facing[:, 0]) - yaw
        action[:, 2] = torch.atan2(angle.sin(), angle.cos()).clamp(-1, 1)
        return action

    @staticmethod
    def _pulse(frame):
        # Kick requests are edge-triggered. Pulse again if alignment timed out.
        return (frame["time_s"] % 0.6) < 0.3

    def script(self, frame, trials):
        ball = frame["ball_xy"]
        shot = ball.new_tensor([2.7, 0.0]) - ball
        pursuit = ball + 0.15 * frame["ball_velocity"][:, :2]
        pursuit -= 0.11 * torch.nn.functional.normalize(shot, dim=-1)
        red = self._move(frame, 6, pursuit, shot)
        red[:, :2] *= ball.new_tensor([t["pace"] for t in trials])[:, None]
        red[:, 3] = self._pulse(frame) & (
            (ball - frame["player_xy"][:, 6]).norm(dim=-1) < 0.55
        )
        red[:, 4] = -1  # red's team-relative target is physical +2.7
        red[:, 6] = 1
        actions = red.new_zeros((len(red), 10, 7))
        actions[:, 6] = red
        return actions

    def expert(self, frame, trials):
        ball = frame["ball_xy"]
        blue = frame["player_xy"][:, 1]
        velocity = frame["ball_velocity"][:, :2]
        goal = ball.new_tensor([2.7, 0.0])
        intercept = ball + 0.25 * velocity
        screen = intercept + 0.24 * torch.nn.functional.normalize(
            goal - intercept, dim=-1
        )
        screen[:, 0].clamp_(-2.54, 2.54)
        clear = ball.new_tensor([-2.7, 0.0]) - ball
        action = self._move(frame, 1, screen, clear)
        action[:, 3] = frame["kick_ready"][:, 1] & ((ball - blue).norm(dim=-1) < 0.90)
        action[:, 4] = -1
        red = frame["player_xy"][:, 6]
        action[:, 5] = torch.where(red[:, 1] >= ball[:, 1], -0.7, 0.7)
        action[:, 6] = 1
        return action

    def measure(self, prev, current, reasons, trials):
        ball, old = current["ball_xy"], prev["ball_xy"]
        dt = (current["time_s"] - prev["time_s"]).to(ball).clamp_min(0)
        live = dt > 0
        blue = (ball - current["player_xy"][:, 1]).norm(dim=-1)
        red = (ball - current["player_xy"][:, 6]).norm(dim=-1)
        control = (blue < 0.25) & (blue + 0.06 < red)
        kick = live & (current["kicker"] == 1)
        slow = current["ball_velocity"][:, :2].norm(dim=-1) < 0.4
        self.control_time = torch.where(control & slow, self.control_time + dt, 0.0)
        recovery = live & (kick | (self.control_time >= 0.24)) & ~self.recovered
        self.recovered |= recovery
        self.pending &= ~(
            live & ((current["kicker"] == 6) | (red + 0.06 < blue) & (red < 0.25))
        )
        self.pending |= kick
        self.kick_x.copy_(torch.where(kick, current["kick_xy"][:, 0], self.kick_x))
        outside = ball.new_tensor([r == "boundary" for r in reasons]) * live
        conceded = ball.new_tensor([r == "goal_b" for r in reasons]) * live
        scored = ball.new_tensor([r == "goal_a" for r in reasons]) * live
        clearance = (
            live
            & self.pending
            & ~self.cleared
            & (self.kick_x - ball[:, 0] >= 0.75)
            & ~outside.bool()
            & ~conceded.bool()
        )
        self.cleared |= clearance
        progress = (old[:, 0] - ball[:, 0]) * live
        reward = (
            0.5 * progress
            + recovery.float()
            + 3 * clearance.float()
            - 6 * conceded
            - outside
            - 0.02 * dt
        )
        return reward, {
            "playing_s": dt,
            "control_s": dt * control,
            "recoveries": recovery.float(),
            "clearances": clearance.float(),
            "progress_m": progress,
            "goals_for": scored,
            "goals_against": conceded,
            "boundary_exits": outside,
        }
