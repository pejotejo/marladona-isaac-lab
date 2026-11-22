"""Run either existing trainer with the same local metrics and timing capture.

Example (run from the checkout being measured)::

    python /path/to/scripts/comparison/train.py --engine mjlab --output /runs/mjlab -- \
        --device cuda:0 --num_envs 1024 --max_iterations 10000 --headless

This file can be copied unchanged to an Isaac Lab checkout. Its import hook waits
until the trainer imports RSL, after Isaac Sim has launched. Game/reward logic is
unchanged; policy initialization and TF32 are matched and recorded in metadata.
Both runs pay for the same instrumentation, whose diagnostics do not consume RNG.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import math
import os
import platform
import runpy
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def json_value(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=json_value) + "\n")
    temporary.replace(path)


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return (
            result.stdout.strip()
            if result.returncode == 0
            else (result.stderr.strip() or result.stdout.strip() or f"Command exited with code {result.returncode}")
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def mean(values):
    return statistics.fmean(values) if values else None


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    return values[lower] + (values[math.ceil(position)] - values[lower]) * (position - lower)


class Metrics:
    iteration_columns = (
        "iteration",
        "phase",
        "elapsed_s",
        "iteration_wall_s",
        "rollout_wall_s",
        "learning_wall_s",
        "env_step_window_wall_s",
        "ppo_update_wall_s",
        "save_wall_s_since_previous_log",
        "env_step_calls",
        "environment_transitions",
        "actor_transitions",
        "training_actor_transitions",
        "environment_transitions_per_s",
        "actor_transitions_per_s",
        "sampled_step_ms_mean",
        "sampled_step_ms_p50",
        "sampled_step_ms_p95",
        "sampled_steps",
        "value_loss",
        "surrogate_loss",
        "entropy",
        "learning_rate",
        "train_episode_return_mean",
        "eval_episode_return_mean",
        "train_episode_length_mean",
        "eval_episode_length_mean",
    )

    def __init__(self, args, train_arguments):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "metadata.json").exists():
            raise FileExistsError(f"Refusing to mix runs: {self.output}/metadata.json already exists")
        self.started = time.perf_counter()
        self.steps = 0
        self.rows = []
        self.step_samples = []
        self.iteration_samples = []
        self.scalars = {}
        self.stat_tensors = {}
        self.closed = False
        self.runner = None
        self.stop_gpu = threading.Event()
        self.iteration_file, self.iteration_csv = self.csv_file("iterations.csv", self.iteration_columns)
        self.scalar_file, self.scalar_csv = self.csv_file("scalars.csv", ("iteration", "elapsed_s", "tag", "value"))
        self.step_file, self.step_csv = self.csv_file(
            "step_timing.csv",
            ("step", "iteration", "phase", "latency_ms", "environment_transitions", "actor_transitions"),
        )
        self.records = (self.output / "metrics.jsonl").open("w", buffering=1)
        self.metadata = {
            "engine": args.engine,
            "status": "starting",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "checkout": str(Path.cwd()),
            "train_script": str(args.train_script.resolve()),
            "train_arguments": train_arguments,
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_branch": command_output(["git", "branch", "--show-current"]),
            "git_status": command_output(["git", "status", "--short"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_id": args.gpu_id,
            "nvidia_smi": command_output(["nvidia-smi", "-i", args.gpu_id, "-q"]),
            "timing": {
                "step_sample_every": args.sample_every,
                "step_measurement": "CUDA synchronize before/after one env.step call (all matches), otherwise perf_counter",
                "iteration_measurement": "synchronize at iteration boundaries; includes policy, rollout, PPO and instrumentation",
                "warmup_iterations": args.warmup_iterations,
                "gpu_sample_interval_s": args.gpu_interval,
                "warning": "Instrumentation overhead is included equally; first initialization/compilation is separate from steady state",
            },
            "comparison_notes": args.note,
            "known_migration_differences": [
                "Original Isaac EnvData.step does not publish goal_status; mjlab score reward fixes that bug.",
                "mjlab latches goals once across substeps, resets all player history/action rows, and uses current critic bootstrap.",
                "PhysX/MuJoCo integration, contacts, field-origin sampling and random-number consumption differ.",
                "Raw goals/wins are recorded separately from reward; equal seed does not imply equal trajectories.",
            ],
            "packages": {},
        }
        for package in ("torch", "mjlab", "mujoco", "mujoco-warp", "isaacsim", "isaaclab", "rsl-rl-lib"):
            try:
                self.metadata["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        write_json(self.output / "metadata.json", self.metadata)
        self.gpu_thread = threading.Thread(target=self.gpu_sampler, daemon=True)
        self.gpu_thread.start()

    def csv_file(self, filename, columns):
        stream = (self.output / filename).open("w", newline="", buffering=1)
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        return stream, writer

    def gpu_sampler(self):
        fields = "timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm"
        with (self.output / "gpu.csv").open("w", newline="", buffering=1) as stream:
            writer = csv.writer(stream)
            writer.writerow(["elapsed_s", *fields.split(",")])
            while True:
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "-i",
                            self.args.gpu_id,
                            f"--query-gpu={fields}",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    if result.returncode != 0:
                        (self.output / "gpu_sampler_error.txt").write_text(
                            result.stderr or result.stdout or f"nvidia-smi exited with code {result.returncode}"
                        )
                        return
                    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                        writer.writerow([time.perf_counter() - self.started, *row])
                except (OSError, subprocess.TimeoutExpired) as exc:
                    (self.output / "gpu_sampler_error.txt").write_text(str(exc))
                    return
                if self.stop_gpu.wait(self.args.gpu_interval):
                    return

    def sync(self):
        if self.runner is not None and str(self.runner.device).startswith("cuda"):
            self.torch.cuda.synchronize(self.runner.device)

    def add_stat(self, name, value):
        self.stat_tensors.setdefault(name, []).append(value.detach().clone() if hasattr(value, "detach") else value)

    def attach(self, runner):
        import torch

        self.torch, self.runner = torch, runner
        self.env = runner.env.unwrapped
        self.save_seconds = 0.0
        self.last_step_boundary = None
        self.iteration_started = time.perf_counter()
        self.metadata.update(
            {
                "status": "training",
                "runner_log_dir": runner.log_dir,
                "runner_config": runner.cfg,
                "environment_config": runner.env.cfg,
                "device": str(runner.device),
                "num_envs": self.env.num_envs,
                "num_agents_per_env": self.env.env_data.num_agents_per_env,
                "step_dt": self.env.step_dt,
                "physics_dt": self.env.physics_dt,
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "comparison_protocol": {
                    "policy_seed": runner._comparison_policy_seed,
                    "policy_initialization": "Torch/CUDA reseeded immediately before runner construction",
                    "tf32": "matmul and cudnn TF32 enabled in both runs before runner construction",
                    "goal_statistics": "positive accumulated_score deltas count once per completed match; raw increments also retained",
                },
            }
        )
        self.metadata["runner_config"] = json.loads(json.dumps(runner.cfg, default=json_value))
        self.metadata["environment_config"] = json.loads(json.dumps(runner.env.cfg, default=json_value))
        state = {name: value.detach().cpu().clone() for name, value in runner.alg.actor_critic.state_dict().items()}
        digest = hashlib.sha256()
        for name, value in sorted(state.items()):
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        self.metadata["initial_policy_sha256"] = digest.hexdigest()
        torch.save(
            {
                "model_state_dict": state,
                "policy_seed": runner._comparison_policy_seed,
                "initial_policy_sha256": digest.hexdigest(),
            },
            self.output / "initial_policy.pt",
        )
        write_json(self.output / "metadata.json", self.metadata)
        self.original_step = runner.env.step
        runner.env.step = self.step
        self.original_log = runner.logger.log
        runner.logger.log = self.log
        original_writer_init = runner.logger.init_writer

        def init_writer(*args, **kwargs):
            result = original_writer_init(*args, **kwargs)
            if runner.logger.writer is not None:
                runner.logger.writer = ScalarWriter(runner.logger.writer, self)
            return result

        runner.logger.init_writer = init_writer
        for method in ("save_model", "save_eval_trajectory"):
            original = getattr(runner.logger, method)

            def save(*args, _original=original, **kwargs):
                self.sync()
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.sync()
                    self.save_seconds += time.perf_counter() - started

            setattr(runner.logger, method, save)
        self.original_update = runner.alg.update
        runner.alg.update = self.update

    def step(self, actions):
        iteration = self.runner.epoch_counter
        if self.steps % self.runner.num_steps_per_env == 0:
            self.sync()
            self.rollout_started = time.perf_counter()
            if self.last_step_boundary is None:
                self.iteration_started = self.rollout_started
        sampled = self.steps % self.args.sample_every == 0
        if sampled:
            self.sync()
            started = time.perf_counter()
        accumulated_before = self.env.env_data.accumulated_score.clone()
        result = self.original_step(actions)
        if sampled:
            self.sync()
            elapsed = (time.perf_counter() - started) * 1000
            self.iteration_samples.append(elapsed)
            phase = "warmup" if iteration < self.args.warmup_iterations else "steady"
            self.step_samples.append((phase, elapsed))
            self.step_csv.writerow(
                {
                    "step": self.steps,
                    "iteration": iteration,
                    "phase": phase,
                    "latency_ms": elapsed,
                    "environment_transitions": self.env.num_envs,
                    "actor_transitions": self.env.num_envs * self.env.env_data.num_agents_per_env,
                }
            )
        _, reward, dones, info = result
        data = self.env.env_data
        mask = data.training_actor_index
        train_actions = actions[mask]
        self.add_stat("Policy/action_mean", train_actions.mean())
        self.add_stat("Policy/action_squared_mean", train_actions.square().mean())
        self.add_stat("Policy/action_saturation_fraction", (train_actions.abs() >= 0.99).float().mean())
        self.add_stat("Reward/train_step_mean", reward[mask].mean())
        terms = getattr(self.env.reward_manager, "_step_reward", None)
        if terms is not None:
            values = terms[mask].mean(dim=0) * self.env.step_dt
            for name, value in zip(self.env.reward_manager._term_names, values):
                self.add_stat(f"Reward/components/{name}", value)
        ended = dones.reshape(data.team_flatten_base_shape)[:, 0].bool()
        score_delta = (data.accumulated_score - accumulated_before).clamp_min(0)
        scored_blue, scored_red = score_delta[:, 0] > 0, score_delta[:, 1] > 0
        for group, env_mask in (("Train", data.is_training_env), ("Eval", data.is_eval_env)):
            selected = ended & env_mask
            self.add_stat(f"{group}/counts/completed_matches", selected.sum().float())
            self.add_stat(f"{group}/counts/blue_wins", (selected & scored_blue & ~scored_red).sum().float())
            self.add_stat(f"{group}/counts/red_wins", (selected & scored_red & ~scored_blue).sum().float())
            self.add_stat(f"{group}/counts/draws", (selected & (scored_blue == scored_red)).sum().float())
            self.add_stat(f"{group}/counts/goals_blue", (selected & scored_blue).sum().float())
            self.add_stat(f"{group}/counts/goals_red", (selected & scored_red).sum().float())
            self.add_stat(f"{group}/counts/raw_score_increments_blue", score_delta[selected, 0].sum().float())
            self.add_stat(f"{group}/counts/raw_score_increments_red", score_delta[selected, 1].sum().float())
        for name, value in info.get("log", {}).items():
            self.add_stat(f"Episode/{name}", value)
        self.steps += 1
        if self.steps % self.runner.num_steps_per_env == 0:
            self.sync()
            self.last_step_boundary = time.perf_counter()
            self.rollout_wall = self.last_step_boundary - self.rollout_started
        return result

    def update(self):
        torch = self.torch
        self.sync()
        started = time.perf_counter()
        original_clip = torch.nn.utils.clip_grad_norm_

        def clip(*args, **kwargs):
            value = original_clip(*args, **kwargs)
            self.add_stat("Policy/gradient_norm_before_clip", value)
            return value

        torch.nn.utils.clip_grad_norm_ = clip
        try:
            result = self.original_update()
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
        storage = self.runner.alg.storage
        with torch.no_grad():
            residual = storage.returns - storage.values
            self.add_stat("Policy/explained_variance", 1 - residual.var() / storage.returns.var().clamp_min(1e-8))
            self.add_stat("Policy/value_prediction_mean", storage.values.mean())
            self.add_stat("Policy/value_target_mean", storage.returns.mean())
            # Evaluate the updated policy once on a fixed small rollout sample.
            count = min(2048, storage.actions.numel() // storage.actions.shape[-1])
            obs = storage.observations.flatten(0, 1)[:count]
            actions = storage.actions.flatten(0, 1)[:count]
            policy = self.runner.alg.actor_critic
            ego_obs = obs[:, : -policy.total_num_neighbor_obs]
            neighbor_obs = obs[:, -policy.total_num_neighbor_obs :]
            features = policy.preprocess_net(ego_obs, neighbor_obs)
            parameters = policy.actor(torch.cat([ego_obs, features], dim=1))
            alpha, beta = policy.get_beta_parameters(parameters)
            new_log_prob = torch.distributions.Beta(alpha, beta).log_prob(actions).sum(dim=-1)
            old_log_prob = storage.actions_log_prob.flatten()[:count]
            log_ratio = new_log_prob - old_log_prob
            ratio = log_ratio.exp()
            self.add_stat("Policy/approximate_kl_after_update_sample", ((ratio - 1) - log_ratio).mean())
            self.add_stat(
                "Policy/clip_fraction_after_update_sample",
                ((ratio - 1).abs() > self.runner.alg.clip_param).float().mean(),
            )
        self.sync()
        self.learning_wall = time.perf_counter() - started
        return result

    def log(self, locs, *args, **kwargs):
        self.sync()
        stopped = time.perf_counter()
        result = self.original_log(locs, *args, **kwargs)
        native_scalar_tags = set(self.scalars)
        iteration = int(locs["it"])
        row = {
            "iteration": iteration,
            "phase": "warmup" if iteration < self.args.warmup_iterations else "steady",
            "elapsed_s": stopped - self.started,
            "iteration_wall_s": stopped - self.iteration_started,
            "rollout_wall_s": float(locs["collection_time"]),
            "learning_wall_s": float(locs["learn_time"]),
            "env_step_window_wall_s": self.rollout_wall,
            "ppo_update_wall_s": self.learning_wall,
            "save_wall_s_since_previous_log": self.save_seconds,
            "env_step_calls": self.runner.num_steps_per_env,
            "environment_transitions": self.runner.num_steps_per_env * self.env.num_envs,
            "actor_transitions": self.runner.num_steps_per_env
            * self.env.num_envs
            * self.env.env_data.num_agents_per_env,
            "training_actor_transitions": self.runner.num_steps_per_env * self.runner.total_num_training_actors,
            "sampled_step_ms_mean": mean(self.iteration_samples),
            "sampled_step_ms_p50": percentile(self.iteration_samples, 0.5),
            "sampled_step_ms_p95": percentile(self.iteration_samples, 0.95),
            "sampled_steps": len(self.iteration_samples),
            "value_loss": locs["mean_value_loss"],
            "surrogate_loss": locs["mean_surrogate_loss"],
            "entropy": locs["entropy_loss"],
            "learning_rate": self.runner.alg.learning_rate,
        }
        row["environment_transitions_per_s"] = row["environment_transitions"] / row["iteration_wall_s"]
        row["actor_transitions_per_s"] = row["actor_transitions"] / row["iteration_wall_s"]
        for group, source in (("train", "metric_buffer_dict"), ("eval", "metric_buffer_dict_eval")):
            row[f"{group}_episode_return_mean"] = mean(locs[source]["reward"])
            row[f"{group}_episode_length_mean"] = mean(locs[source]["episode_length"])
        for name, values in self.stat_tensors.items():
            if values:
                tensor = self.torch.stack([self.torch.as_tensor(v, device=self.runner.device) for v in values]).float()
                tensor = tensor[self.torch.isfinite(tensor)]
                if tensor.numel():
                    self.scalars[name] = float(tensor.sum() if "/counts/" in name else tensor.mean())
        for group in ("Train", "Eval"):
            completed = self.scalars.get(f"{group}/counts/completed_matches", 0)
            if completed:
                for outcome in ("blue_wins", "red_wins", "draws"):
                    self.scalars[f"{group}/{outcome}_fraction"] = self.scalars[f"{group}/counts/{outcome}"] / completed
                for team in ("blue", "red"):
                    self.scalars[f"{group}/goals_{team}_per_episode"] = (
                        self.scalars[f"{group}/counts/goals_{team}"] / completed
                    )
        if "Policy/action_mean" in self.scalars:
            self.scalars["Policy/action_std"] = math.sqrt(
                max(0, self.scalars["Policy/action_squared_mean"] - self.scalars["Policy/action_mean"] ** 2)
            )
        for tag, value in self.scalars.items():
            if tag not in native_scalar_tags and self.runner.logger.writer is not None:
                self.runner.logger.writer.add_scalar(tag, value, iteration)
            self.scalar_csv.writerow(
                {"iteration": iteration, "elapsed_s": row["elapsed_s"], "tag": tag, "value": value}
            )
        self.records.write(json.dumps({**row, "scalars": self.scalars}, default=json_value) + "\n")
        self.iteration_csv.writerow(row)
        self.rows.append(row)
        self.stat_tensors, self.scalars, self.iteration_samples = {}, {}, []
        self.save_seconds = 0
        self.iteration_started = stopped
        write_json(self.output / "progress.json", row)
        return result

    def close(self, status, error=None):
        if self.closed:
            return
        self.closed = True
        if status == "completed" and self.runner is not None:
            expected = self.runner.cfg.get("max_iterations", 0)
            if len(self.rows) < expected:
                status = "incomplete"
                error = f"Trainer returned after {len(self.rows)} of {expected} requested iterations"
        self.stop_gpu.set()
        self.gpu_thread.join(timeout=12)
        self.metadata.update(
            {
                "status": status,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "total_wall_s": time.perf_counter() - self.started,
                "error": error,
                "gpu_telemetry_status": "unavailable"
                if (self.output / "gpu_sampler_error.txt").exists()
                else "recorded",
            }
        )
        write_json(self.output / "metadata.json", self.metadata)
        steady = [row for row in self.rows if row["phase"] == "steady"]
        samples = [value for phase, value in self.step_samples if phase == "steady"]
        write_json(
            self.output / "summary.json",
            {
                "status": status,
                "iterations": len(self.rows),
                "steady_iterations": len(steady),
                "total_wall_s": time.perf_counter() - self.started,
                "steady_iteration_wall_s_mean": mean([row["iteration_wall_s"] for row in steady]),
                "steady_environment_transitions_per_s_mean": mean(
                    [row["environment_transitions_per_s"] for row in steady]
                ),
                "steady_actor_transitions_per_s_mean": mean([row["actor_transitions_per_s"] for row in steady]),
                "steady_sampled_step_ms_mean": mean(samples),
                "steady_sampled_step_ms_p50": percentile(samples, 0.5),
                "steady_sampled_step_ms_p95": percentile(samples, 0.95),
                "steady_sampled_steps": len(samples),
                "env_step_calls": self.steps,
                "last_iteration": self.rows[-1] if self.rows else None,
            },
        )
        for stream in (self.iteration_file, self.scalar_file, self.step_file, self.records):
            stream.close()


class ScalarWriter:
    def __init__(self, writer, metrics):
        self.writer, self.metrics = writer, metrics

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        value = float(scalar_value)
        if math.isfinite(value):
            self.metrics.scalars[tag] = value
        return self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.writer, name)


class RunnerHook(importlib.abc.MetaPathFinder):
    def __init__(self, metrics):
        self.metrics = metrics

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "rsl_marl.runners.on_policy_runner":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        original_loader = spec.loader
        metrics = self.metrics

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return original_loader.create_module(spec)

            def exec_module(self, module):
                original_loader.exec_module(module)
                original_init = module.OnPolicyRunner.__init__

                def init(runner, *args, **kwargs):
                    import torch

                    train_cfg = kwargs.get("train_cfg", args[1] if len(args) > 1 else {})
                    seed = metrics.args.policy_seed
                    if seed is None:
                        seed = train_cfg.get("seed", 42)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    runner._comparison_policy_seed = seed
                    original_init(runner, *args, **kwargs)
                    metrics.attach(runner)

                module.OnPolicyRunner.__init__ = init

        spec.loader = Loader()
        return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("mjlab", "isaac"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, default=Path("scripts/rsl_rl/train.py"))
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--gpu-interval", type=float, default=2.0)
    parser.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    parser.add_argument("--policy-seed", type=int, default=None, help="Defaults to the training configuration seed")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("train_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.sample_every < 1 or args.warmup_iterations < 0 or args.gpu_interval <= 0:
        parser.error("Sampling periods must be positive and warmup nonnegative")
    arguments = args.train_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not any(arg == "--logger" or arg.startswith("--logger=") for arg in arguments):
        arguments += ["--logger", "tensorboard"]
    os.environ["WANDB_MODE"] = "offline"
    os.environ["NEPTUNE_MODE"] = "offline"
    metrics = Metrics(args, arguments)
    hook = RunnerHook(metrics)
    sys.meta_path.insert(0, hook)
    sys.path.insert(0, str(args.train_script.resolve().parent))
    sys.argv = [str(args.train_script.resolve()), *arguments]
    try:
        runpy.run_path(str(args.train_script.resolve()), run_name="__main__")
    except BaseException:
        metrics.close("failed", traceback.format_exc())
        raise
    else:
        metrics.close("completed")
    finally:
        sys.meta_path.remove(hook)


if __name__ == "__main__":
    main()
