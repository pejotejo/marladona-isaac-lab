"""Seeded support-play drill. Blue attacks the negative-x goal."""
import math

import torch


def drive(action, xy, heading, target, facing):
    delta = target - xy
    c, s = heading.cos(), heading.sin()
    x, y = delta.unbind(-1)
    action[..., 0] = (2 * (x*c + y*s)).clamp(-1, 1)
    action[..., 1] = (2 * (y*c - x*s)).clamp(-1, 1)
    direction = facing - xy
    error = torch.atan2(direction[..., 1], direction[..., 0]) - heading
    action[..., 2] = torch.atan2(error.sin(), error.cos()).clamp(-1, 1)


def clearance(start, end, opponents):
    line = end - start
    rel = opponents - start.unsqueeze(-2)
    t = ((rel * line.unsqueeze(-2)).sum(-1)
         / line.square().sum(-1, keepdim=True).clamp_min(1e-6)).clamp(0, 1)
    return (rel - t.unsqueeze(-1)*line.unsqueeze(-2)).norm(dim=-1).amin(-1)


def quality(carrier, receiver, defenders):
    goal = torch.zeros_like(receiver)
    goal[..., 0] = -2.7
    spacing = (1 - ((carrier-receiver).norm(dim=-1)-1.35).abs()/1.35).clamp(0, 1)
    lane = (clearance(carrier, receiver, defenders)/0.55).clamp(0, 1)
    shot = (clearance(receiver, goal, defenders)/0.5).clamp(0, 1)
    shot *= (1-(receiver-goal).norm(dim=-1)/4).clamp(0, 1)
    return spacing, lane, shot


class Positioning:
    def __init__(self, num_envs, device, seed):
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.age = torch.full((num_envs,), -1., device=device)
        self.origin = torch.zeros(num_envs, 2, device=device)
        self.free = torch.zeros_like(self.age, dtype=torch.bool)
        self.used = self.free.clone()
        self.dwell = torch.zeros_like(self.age)

    def reset(self, slots):
        self.age[slots] = -1
        self.origin[slots] = 0
        self.free[slots] = False
        self.used[slots] = False
        self.dwell[slots] = 0

    def starts(self, count):
        trials = []
        for u in torch.rand(count, 16, generator=self.rng, device="cpu").tolist():
            sign = 1 if u[0] < .5 else -1
            x, y = .35 + u[1], sign*(.35+.6*u[2])
            angle = math.pi + sign*(u[3]-.25)
            poses = [[x,y,angle], [-.7+.8*u[4], -sign*(.35+.8*u[5]), 6*u[6]-3],
                     [2.4,0,math.pi], [1.8,1.45,math.pi], [1.8,-1.45,math.pi],
                     [-2.45,0,0], [-.35-.4*u[7],sign*(.2+.55*u[8]),0],
                     [-1.25-.3*u[9],-sign*(.4+.5*u[10]),0], [-2,1.4,0], [-2,-1.4,0]]
            trials.append(dict(player_pose=poses, active=[True]*10,
                               player_velocity=[[0.,0.,0.] for _ in poses],
                               ball_position=[x+.16*math.cos(angle), y+.16*math.sin(angle), .055],
                               ball_velocity=[0.,0.,0.], max_steps=200,
                               route=[x-.65, y-sign*(.2+.4*u[11])],
                               delay=.55+.8*u[12], pressure=[u[13]>.45,u[14]>.45],
                               pace=.25+.3*u[15]))
        return trials

    def script(self, frame, trials):
        p, b, h = frame["player_xy"], frame["ball_xy"], frame["player_heading"]
        a = p.new_zeros((len(p),10,7))
        route = p.new_tensor([t["route"] for t in trials])
        delay = p.new_tensor([t["delay"] for t in trials])
        time = frame["time_s"].to(b)
        gap = (p[:,1]-b).norm(dim=-1)
        ready = ((time >= delay) & (gap > .65) & (gap < 2.65)
                 & (p[:,1,0] < p[:,0,0]+.15)
                 & (clearance(b,p[:,1],p[:,5:]) > .18))
        aim = torch.where(ready[:,None], p[:,1], route)
        direction = torch.nn.functional.normalize(aim-b, dim=-1)
        target = b-.11*direction
        close = (p[:,0]-target).norm(dim=-1)<.17
        target = torch.where((close & ~ready)[:,None], b+.20*direction,target)
        drive(a[:,0],p[:,0],h[:,0],target,aim)
        a[:,0,3] = ready & ((b-p[:,0]).norm(dim=-1)<.31) & ((time-delay)%0.5<.25)
        a[:,0,4:6] = (p[:,1]/b.new_tensor([2.7,1.8])).clamp(-1,1)
        a[:,0,6] = (.28+.13*gap).clamp(.35,.62)
        a[:,0] *= (~self.used)[:,None]
        targets = torch.stack((.55*b+.45*p[:,1], .60*p[:,1]+.40*b.new_tensor([-2.7,0])),1)
        drive(a[:,6:8],p[:,6:8],h[:,6:8],targets,b[:,None])
        pace = b.new_tensor([t["pace"] for t in trials])
        a[:,6:8,:3] *= (pace[:,None]*b.new_tensor([t["pressure"] for t in trials]))[:,:,None]
        return a

    def expert(self, frame, trials):
        p, b = frame["player_xy"], frame["ball_xy"]
        # Search support points on both sides, favouring nearby reachable openings.
        x = b.new_tensor([-.65,-1.15,-1.65])
        y = b.new_tensor([-1.25,-.75,0,.75,1.25])
        offsets = torch.cartesian_prod(x,y)
        options = (p[:,0,None]+offsets).clamp(b.new_tensor([-2.2,-1.45]),b.new_tensor([1.9,1.45]))
        space,lane,shot = quality(p[:,0,None],options,p[:,None,5:])
        score = space+1.3*lane+shot-.28*(options-p[:,1,None]).norm(dim=-1)
        target = options[torch.arange(len(p),device=p.device),score.argmax(1)]
        intercept = b+.10*frame["ball_velocity"][...,:2]
        target = torch.where(self.used[:,None],intercept,target)
        a = b.new_zeros(len(b),7)
        goal = b.new_tensor([-2.7,0]).expand_as(b)
        drive(a,p[:,1],frame["player_heading"][:,1],target,torch.where(self.used[:,None],goal,b))
        a[:,3] = self.used & ((b-p[:,1]).norm(dim=-1)<.28)
        a[:,4] = -1
        a[:,6] = .9
        return a

    def measure(self, previous, current, reasons, trials):
        b,p = current["ball_xy"],current["player_xy"]
        dt = (current["time_s"]-previous["time_s"]).to(b)
        kicker = current["kicker"]
        release = (kicker==0) & ~self.used
        self.used |= release
        self.origin[release] = current["kick_xy"][release]
        self.age = torch.where(release,0.,self.age)
        live = self.age>=0
        self.age += dt*live
        distances = (p-b[:,None]).norm(dim=-1)
        other = torch.cat((distances[:,:1],distances[:,2:]),1).amin(1)
        first = (kicker==1) & self.free & ((current["kick_xy"]-self.origin).norm(dim=-1)>.45)
        travelled = (b-self.origin).norm(dim=-1)>.45
        control = (distances[:,1]<.23) & (other>.32) & (current["ball_velocity"][...,:2].norm(dim=-1)<1.25)
        self.dwell = torch.where(live & self.free & travelled & control,self.dwell+dt,0.)
        intercepted = (kicker>=0) & ~release & ~first
        intercepted |= self.free & (other<.22) & ~first
        intercepted |= distances[:,2:].amin(1)<.20
        valid = live & ~intercepted & (self.age<2.5)
        received = valid & (first | (self.dwell>=.12))
        self.free |= live & (distances.amin(1)>.25) & ((b-self.origin).norm(dim=-1)>.22)
        done = b.new_tensor([r is not None for r in reasons],dtype=torch.bool)
        self.age.masked_fill_(~valid | received | done,-1.)
        q = quality(p[:,0],p[:,1],p[:,5:])
        old = previous["player_xy"]
        prior = quality(old[:,0],old[:,1],old[:,5:])
        gain = sum(q)-sum(prior)
        scored = b.new_tensor([r=="goal_a" for r in reasons])
        failed = b.new_tensor([r in ("goal_b","boundary") for r in reasons])
        shaping = (gain+.15*dt*q[0]*q[1]*(1+q[2]))*~self.used
        return shaping+3*received+4*scored-2*failed, dict(
            playing_s=dt, spacing_s=dt*q[0], open_lane_s=dt*q[1],
            shot_opportunity_s=dt*q[2], releases=release.float(),
            receptions=received.float(), goals_for=scored)
