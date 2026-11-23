"""FastSAC transitions and paired-start evaluation on real Isaac drill matches."""
import json

import torch
from rsl_marl.algorithms.fast_sac.config import FastSACConfig
from rsl_marl.algorithms.fast_sac.fast_sac_agent import FastSACAgent

from .subpolicies import Drill


def record(path, data):
    with path.open("a") as stream:
        stream.write(json.dumps(data, allow_nan=False) + "\n")


class DrillEnv:
    num_actions = 7

    def __init__(self, args, backend):
        self.backend = backend
        self.num_envs = args.num_envs
        self.drill = Drill(args.drill, args.num_envs, args.device, args.seed)
        self.trials = self.drill.starts(args.num_envs)
        self.frame = backend.reset(dict(enumerate(self.trials)))
        self.obs = backend.candidate_observation()
        width = self.obs.shape[1]
        span = dict(start=0, end=width, size=width)
        self.actor_obs_indices = {"actor_obs": span}
        self.critic_obs_indices = {"critic_obs": span}
        self.action_scale = self.obs.new_tensor([1, 1, 1, .5, 1, 1, .5])
        self.action_bias = 1 - self.action_scale

    def reset_with_critic_obs(self):
        return self.obs, self.obs

    def advance(self, candidate):
        actions = self.drill.script(self.frame, self.trials)
        if candidate is not None:
            actions[:, 1] = candidate
        current, reasons = self.backend.step(actions)
        reward, metrics = self.drill.measure(self.frame, current, reasons, self.trials)
        self.frame = current
        return reward, metrics, reasons

    def restart(self, slots):
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
        return self.obs, reward, dones, {
            "time_outs": self.obs.new_tensor([r == "timeout" for r in reasons]),
            "observations": {"critic": self.obs, "final": {"actor_obs": final, "critic_obs": final}},
        }


def run(args, backend):
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    env = DrillEnv(args, backend)
    if args.mode == "train" or args.checkpoint:
        config = FastSACConfig(num_learning_iterations=args.updates, batch_size=args.batch_size,
                               learning_starts=args.learning_starts, num_updates=args.num_updates,
                               compile=False, amp=False, save_interval=args.checkpoint_interval,
                               logging_interval=10)
        def log(metrics, step):
            record(args.output / "training.jsonl", dict(step=step, **metrics))
        agent = FastSACAgent(env, config, args.device, str(args.output), log)
        agent.setup()
        if args.checkpoint:
            agent.load(str(args.checkpoint))
        if args.mode == "train":
            agent.learn()
            return
        policy = agent.get_inference_policy()
    else:
        policy = None

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
        elif args.baseline == "bot":
            candidate = backend.bots.act(backend.current_bots_obs, backend.previous,
                                         backend.actor_ids[:, 1], configs=["keeper", "player"])
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
            record(log_path, dict(episode=indices[i], start=env.trials[i], reason=reason, metrics=results))
            counts[reason] = counts.get(reason, 0) + 1
            for key, value in results.items():
                sums[key] = sums.get(key, 0.) + value
                totals[key][i] = 0
            finished += 1
            if next_index < args.episodes:
                indices[i] = next_index
                next_index += 1
                resets.append(i)
        if resets:
            env.restart(resets)
        env.obs = backend.candidate_observation()
    summary = dict(episodes=finished, seed=args.seed, drill=args.drill, reasons=counts,
                   baseline=args.baseline, checkpoint=str(args.checkpoint) if args.checkpoint else None,
                   metrics={key: value / finished for key, value in sums.items()})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
