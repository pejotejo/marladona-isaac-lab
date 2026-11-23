"""FastSAC transitions and paired-start evaluation on real Isaac drill matches."""

import json
import subprocess

import torch
from rsl_marl.algorithms.fast_sac.config import FastSACConfig
from rsl_marl.algorithms.fast_sac.fast_sac import Actor
from rsl_marl.algorithms.fast_sac.fast_sac_agent import FastSACAgent
from rsl_marl.algorithms.fast_sac.fast_sac_utils import EmpiricalNormalization

from .subpolicies import Drill


def record(path, data):
    with path.open("a") as stream:
        stream.write(json.dumps(data, allow_nan=False) + "\n")


def warm_start(agent, env, steps):
    """Use Isaac demonstrations for initialization; FastSAC then trains on its own replay."""
    states, targets = [], []
    for _ in range(steps):
        with torch.no_grad():
            target = env.teacher()
            states.append(env.obs)
            targets.append(target)
            env.step(target)
    states, targets = torch.cat(states), torch.cat(targets)
    agent.obs_normalizer.update(states)
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=4e-4)

    def imitation_loss():
        ids = torch.randint(len(states), (512,), device=states.device)
        predicted = agent.actor(agent.obs_normalizer(states[ids], update=False))[0]
        differences = (predicted - targets[ids]).square()
        differences[:, 3] *= 1 + 8 * (targets[ids, 3] > 0.5)
        return differences.mean()

    for _ in range(800):
        optimizer.zero_grad()
        imitation_loss().backward()
        optimizer.step()
    return imitation_loss


class DrillEnv:
    num_actions = 7

    def __init__(self, args, backend):
        self.backend = backend
        self.num_envs = args.num_envs
        if args.drill == "positioning":
            from .positioning import Positioning

            self.drill = Positioning(args.num_envs, args.device, args.seed)
        elif args.drill == "defense":
            from .defense import Defense

            self.drill = Defense(args.num_envs, args.device, args.seed)
        else:
            self.drill = Drill(args.drill, args.num_envs, args.device, args.seed)
        self.trials = self.drill.starts(args.num_envs)
        self.frame = backend.reset(dict(enumerate(self.trials)))
        self.obs = backend.candidate_observation()
        width = self.obs.shape[1]
        span = {"start": 0, "end": width, "size": width}
        self.actor_obs_indices = {"actor_obs": span}
        self.critic_obs_indices = {"critic_obs": span}
        self.action_scale = self.obs.new_tensor([1, 1, 1, 0.5, 1, 1, 0.5])
        self.action_bias = 1 - self.action_scale

    def reset_with_critic_obs(self):
        return self.obs, self.obs

    def bot(self):
        return self.backend.bots.act(
            self.backend.current_bots_obs,
            self.backend.previous,
            self.backend.actor_ids[:, 1],
            configs=["keeper", "player"],
        )

    def teacher(self):
        if hasattr(self.drill, "expert"):
            return self.drill.expert(self.frame, self.trials)
        action = self.bot()
        ball = self.frame["ball_xy"]
        players = self.frame["player_xy"]
        targets = ball.new_tensor([[-2.7, 0.0], [-2.7, -0.48], [-2.7, 0.48]])
        lanes = targets - ball[:, None]
        relative = players[:, 6:7] - ball[:, None]
        fraction = (
            (relative * lanes).sum(-1) / lanes.square().sum(-1).clamp_min(1e-5)
        ).clamp(0, 1)
        gaps = (relative - fraction[..., None] * lanes).norm(dim=-1)
        aim = targets[(gaps.clamp_max(0.38) - 0.04 * lanes.norm(dim=-1)).argmax(1)]
        kick_state = self.backend.current_bots_obs["kick_state"][
            self.backend.actor_ids[:, 1]
        ]
        distance = (ball - players[:, 1]).norm(dim=-1)
        action[:, 3] = (
            (kick_state[:, 0] < 0.5) & (kick_state[:, 1] < 0.5) & (distance < 0.70)
        )
        action[:, 4:6] = aim / ball.new_tensor([2.7, 1.8])
        return action

    def advance(self, candidate):
        actions = self.drill.script(self.frame, self.trials)
        if candidate is not None:
            actions[:, 1] = candidate
        current, reasons = self.backend.step(actions)
        reward, metrics = self.drill.measure(self.frame, current, reasons, self.trials)
        metrics["learner_kicks"] = (current["kicker"] == 1).float()
        self.frame = current
        return reward, metrics, reasons

    def restart(self, slots):
        if hasattr(self.drill, "reset"):
            self.drill.reset(slots)
        assignments = dict(zip(slots, self.drill.starts(len(slots))))
        for i, trial in assignments.items():
            self.trials[i] = trial
        self.frame = self.backend.reset(assignments)

    @torch.no_grad()
    def step(self, candidate):
        reward, _, reasons = self.advance(candidate)
        slots = [i for i, reason in enumerate(reasons) if reason]
        final = self.backend.candidate_observation(snapshot=bool(slots))
        self.obs = final
        if slots:
            self.restart(slots)
            self.obs = self.backend.candidate_observation()
        dones = self.obs.new_tensor([bool(r) for r in reasons])
        return (
            self.obs,
            reward,
            dones,
            {
                "time_outs": self.obs.new_tensor([r == "timeout" for r in reasons]),
                "observations": {
                    "critic": self.obs,
                    "final": {"actor_obs": final, "critic_obs": final},
                },
            },
        )


def inference_policy(path, env):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = FastSACConfig(**checkpoint["args"])
    if (
        checkpoint["algorithm"] != "fast_sac"
        or checkpoint["actor_obs_indices"] != env.actor_obs_indices
    ):
        raise ValueError("Incompatible FastSAC checkpoint or observation dimensions")
    device = env.obs.device
    actor = Actor(
        env.actor_obs_indices,
        config.actor_obs_keys,
        env.num_actions,
        env.num_envs,
        config.actor_hidden_dim,
        config.log_std_max,
        config.log_std_min,
        config.use_tanh,
        config.use_layer_norm,
        device=device,
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    normalizer = EmpiricalNormalization(env.obs.shape[1], device)
    normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
    actor.eval()
    normalizer.eval()
    return lambda obs: actor(normalizer(obs, update=False))[0]


def run(args, backend):
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (args.output / "config.json").write_text(
        json.dumps(
            {"revision": revision, "backend": backend.provenance, "args": vars(args)},
            default=str,
            indent=2,
        )
    )
    env = DrillEnv(args, backend)
    if args.mode == "train":
        config = FastSACConfig(
            num_learning_iterations=args.updates,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            num_updates=args.num_updates,
            compile=False,
            amp=False,
            gamma=args.gamma,
            actor_learning_rate=args.actor_lr,
            save_interval=args.checkpoint_interval,
            logging_interval=10,
        )

        def log(metrics, step):
            record(args.output / "training.jsonl", dict(step=step, **metrics))

        agent = FastSACAgent(env, config, args.device, str(args.output), log)
        agent.setup()
        if args.checkpoint:
            agent.load(str(args.checkpoint))
        if args.teacher_steps:
            imitation_loss = warm_start(agent, env, args.teacher_steps)
            if args.demo_weight:
                agent.auxiliary_actor_loss = lambda: args.demo_weight * imitation_loss()
            agent.save(str(args.output / "initial_teacher.pt"))
        agent.learn()
        return
    policy = inference_policy(args.checkpoint, env) if args.checkpoint else None

    log_path = args.output / "episodes.jsonl"
    log_path.touch(exist_ok=False)
    indices = list(range(env.num_envs))
    next_index = env.num_envs
    finished = 0
    totals = {}
    sums = {}
    counts = {}
    while finished < args.episodes:
        candidate = None
        if policy:
            candidate = policy(env.obs)
        elif args.baseline == "teacher":
            candidate = env.teacher()
        elif args.baseline == "bot":
            candidate = env.bot()
        reward, metrics, reasons = env.advance(candidate)
        for key, value in {**metrics, "return": reward}.items():
            if key not in totals:
                totals[key] = torch.zeros_like(value)
            totals[key] += value
        resets = []
        for i, reason in enumerate(reasons):
            if reason is None or indices[i] >= args.episodes:
                continue
            results = {key: float(value[i]) for key, value in totals.items()}
            record(
                log_path,
                {
                    "episode": indices[i],
                    "start": env.trials[i],
                    "reason": reason,
                    "metrics": results,
                },
            )
            counts[reason] = counts.get(reason, 0) + 1
            for key, value in results.items():
                sums[key] = sums.get(key, 0.0) + value
                totals[key][i] = 0
            finished += 1
            if next_index < args.episodes:
                indices[i] = next_index
                next_index += 1
                resets.append(i)
        if resets:
            env.restart(resets)
        env.obs = backend.candidate_observation()
    summary = {
        "episodes": finished,
        "seed": args.seed,
        "drill": args.drill,
        "reasons": counts,
        "baseline": args.baseline,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "metrics": {key: value / finished for key, value in sums.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
