"""Seeded support-play drill. Blue attacks the negative-x goal."""

import math

import torch


def drive(action, xy, heading, target, facing):
    delta = target - xy
    c, s = heading.cos(), heading.sin()
    x, y = delta.unbind(-1)
    action[..., 0] = (2 * (x * c + y * s)).clamp(-1, 1)
    action[..., 1] = (2 * (y * c - x * s)).clamp(-1, 1)
    direction = facing - xy
    error = torch.atan2(direction[..., 1], direction[..., 0]) - heading
    action[..., 2] = torch.atan2(error.sin(), error.cos()).clamp(-1, 1)


def clearance(start, end, opponents):
    line = end - start
    rel = opponents - start.unsqueeze(-2)
    t = (
        (rel * line.unsqueeze(-2)).sum(-1)
        / line.square().sum(-1, keepdim=True).clamp_min(1e-6)
    ).clamp(0, 1)
    return (rel - t.unsqueeze(-1) * line.unsqueeze(-2)).norm(dim=-1).amin(-1)


def quality(carrier, receiver, defenders):
    goal = torch.zeros_like(receiver)
    goal[..., 0] = -2.7
    spacing = (1 - ((carrier - receiver).norm(dim=-1) - 1.35).abs() / 1.35).clamp(0, 1)
    lane = (clearance(carrier, receiver, defenders) / 0.55).clamp(0, 1)
    shot = (clearance(receiver, goal, defenders) / 0.5).clamp(0, 1)
    shot *= (1 - (receiver - goal).norm(dim=-1) / 4).clamp(0, 1)
    return spacing, lane, shot


class Positioning:
    def __init__(self, num_envs, device, seed):
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.age = torch.full((num_envs,), -1.0, device=device)
        self.origin = torch.zeros(num_envs, 2, device=device)
        self.free = torch.zeros_like(self.age, dtype=torch.bool)
        self.used = self.free.clone()
        self.dwell = torch.zeros_like(self.age)
        self.completed = self.free.clone()

    def reset(self, slots):
        self.age[slots] = -1
        self.origin[slots] = 0
        self.free[slots] = False
        self.used[slots] = False
        self.dwell[slots] = 0
        self.completed[slots] = False

    def starts(self, count):
        trials = []
        for u in torch.rand(count, 16, generator=self.rng, device="cpu").tolist():
            sign = 1 if u[0] < 0.5 else -1
            x, y = 0.35 + u[1], sign * (0.35 + 0.6 * u[2])
            angle = math.pi + sign * (u[3] - 0.25)
            poses = [
                [x, y, angle],
                [-0.7 + 0.8 * u[4], -sign * (0.35 + 0.8 * u[5]), 6 * u[6] - 3],
                [2.4, 0, math.pi],
                [1.8, 1.45, math.pi],
                [1.8, -1.45, math.pi],
                [-2.45, 0, 0],
                [-0.35 - 0.4 * u[7], sign * (0.2 + 0.55 * u[8]), 0],
                [-1.25 - 0.3 * u[9], -sign * (0.4 + 0.5 * u[10]), 0],
                [-2, 1.4, 0],
                [-2, -1.4, 0],
            ]
            trials.append(
                {
                    "player_pose": poses,
                    "active": [True] * 10,
                    "player_velocity": [[0.0, 0.0, 0.0] for _ in poses],
                    "ball_position": [
                        x + 0.16 * math.cos(angle),
                        y + 0.16 * math.sin(angle),
                        0.055,
                    ],
                    "ball_velocity": [0.0, 0.0, 0.0],
                    "max_steps": 200,
                    "route": [x - 0.65, y - sign * (0.2 + 0.4 * u[11])],
                    "delay": 0.55 + 0.8 * u[12],
                    "pressure": [u[13] > 0.45, u[14] > 0.45],
                    "pace": 0.25 + 0.3 * u[15],
                }
            )
        return trials

    def script(self, frame, trials):
        p, b, h = frame["player_xy"], frame["ball_xy"], frame["player_heading"]
        a = p.new_zeros((len(p), 10, 7))
        route = p.new_tensor([t["route"] for t in trials])
        delay = p.new_tensor([t["delay"] for t in trials])
        time = frame["time_s"].to(b)
        gap = (p[:, 1] - b).norm(dim=-1)
        ready = (
            (time >= delay)
            & (gap > 0.65)
            & (gap < 2.65)
            & (p[:, 1, 0] < p[:, 0, 0] + 0.15)
            & (clearance(b, p[:, 1], p[:, 5:]) > 0.18)
        )
        aim = torch.where(ready[:, None], p[:, 1], route)
        direction = torch.nn.functional.normalize(aim - b, dim=-1)
        target = b - 0.11 * direction
        close = (p[:, 0] - target).norm(dim=-1) < 0.17
        target = torch.where((close & ~ready)[:, None], b + 0.20 * direction, target)
        drive(a[:, 0], p[:, 0], h[:, 0], target, aim)
        a[:, 0, 3] = (
            ready & ((b - p[:, 0]).norm(dim=-1) < 0.31) & ((time - delay) % 0.5 < 0.25)
        )
        a[:, 0, 4:6] = (p[:, 1] / b.new_tensor([2.7, 1.8])).clamp(-1, 1)
        a[:, 0, 6] = (0.62 + 0.22 * gap).clamp(0.76, 1.0)
        a[:, 0] *= (~self.used)[:, None]
        targets = torch.stack(
            (
                0.55 * b + 0.45 * p[:, 1],
                0.60 * p[:, 1] + 0.40 * b.new_tensor([-2.7, 0]),
            ),
            1,
        )
        drive(a[:, 6:8], p[:, 6:8], h[:, 6:8], targets, b[:, None])
        pace = b.new_tensor([t["pace"] for t in trials])
        a[:, 6:8, :3] *= (
            pace[:, None] * b.new_tensor([t["pressure"] for t in trials])
        )[:, :, None]
        return a

    def expert(self, frame, trials):
        p, b = frame["player_xy"], frame["ball_xy"]
        # Search support points on both sides, favouring nearby reachable openings.
        x = b.new_tensor([-0.65, -1.15, -1.65])
        y = b.new_tensor([-1.25, -0.75, 0, 0.75, 1.25])
        offsets = torch.cartesian_prod(x, y)
        options = (p[:, 0, None] + offsets).clamp(
            b.new_tensor([-2.2, -1.45]), b.new_tensor([1.9, 1.45])
        )
        space, lane, shot = quality(p[:, 0, None], options, p[:, None, 5:])
        score = (
            space + 1.3 * lane + shot - 0.28 * (options - p[:, 1, None]).norm(dim=-1)
        )
        target = options[torch.arange(len(p), device=p.device), score.argmax(1)]
        intercept = b + 0.10 * frame["ball_velocity"][..., :2]
        target = torch.where(self.used[:, None], intercept, target)
        a = b.new_zeros(len(b), 7)
        goal = b.new_tensor([-2.7, 0]).expand_as(b)
        drive(
            a,
            p[:, 1],
            frame["player_heading"][:, 1],
            target,
            torch.where(self.used[:, None], goal, b),
        )
        a[:, 3] = (
            self.used & frame["kick_ready"][:, 1] & ((b - p[:, 1]).norm(dim=-1) < 0.28)
        )
        a[:, 4] = -1
        a[:, 6] = 0.9
        return a

    def measure(self, previous, current, reasons, trials):
        b, p = current["ball_xy"], current["player_xy"]
        dt = (current["time_s"] - previous["time_s"]).to(b)
        kicker = current["kicker"]
        release = (kicker == 0) & ~self.used
        self.used |= release
        self.origin[release] = current["kick_xy"][release]
        self.age = torch.where(release, 0.0, self.age)
        live = self.age >= 0
        self.age += dt * live
        distances = (p - b[:, None]).norm(dim=-1)
        other = torch.cat((distances[:, :1], distances[:, 2:]), 1).amin(1)
        first = (
            (kicker == 1)
            & self.free
            & ((current["kick_xy"] - self.origin).norm(dim=-1) > 0.45)
        )
        travelled = (b - self.origin).norm(dim=-1) > 0.45
        control = (
            (distances[:, 1] < 0.23)
            & (other > 0.32)
            & (current["ball_velocity"][..., :2].norm(dim=-1) < 0.45)
        )
        self.dwell = torch.where(
            live & self.free & travelled & control, self.dwell + dt, 0.0
        )
        intercepted = (kicker >= 0) & ~release & ~first
        intercepted |= self.free & (other < 0.22) & ~first
        intercepted |= distances[:, 2:].amin(1) < 0.20
        valid = live & ~intercepted & (self.age < 4.0)
        received = valid & (first | (self.dwell >= 0.12))
        self.completed |= received
        self.free |= (
            live & (distances.amin(1) > 0.25) & ((b - self.origin).norm(dim=-1) > 0.22)
        )
        done = b.new_tensor([r is not None for r in reasons], dtype=torch.bool)
        self.age.masked_fill_(~valid | received | done, -1.0)
        q = quality(p[:, 0], p[:, 1], p[:, 5:])
        old = previous["player_xy"]
        prior = quality(old[:, 0], old[:, 1], old[:, 5:])
        gain = sum(q) - sum(prior)
        scored = b.new_tensor([r == "goal_a" for r in reasons])
        failed = b.new_tensor([r in ("goal_b", "boundary") for r in reasons])
        shaping = gain * ~self.used + release * q[0] * q[1] * (1 + q[2])
        old_gap = (previous["ball_xy"] - old[:, 1]).norm(dim=-1)
        shaping += 0.3 * valid * (old_gap - distances[:, 1])
        return (
            shaping
            + 5 * received
            + 2 * scored * self.completed
            - 2 * failed
            - 0.02 * dt,
            {
                "playing_s": dt,
                "spacing_s": dt * q[0],
                "open_lane_s": dt * q[1],
                "shot_opportunity_s": dt * q[2],
                "releases": release.float(),
                "receptions": received.float(),
                "goals_for": scored,
            },
        )
