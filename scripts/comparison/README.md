# Training comparison tools

## Matched full training protocol

Run both backends with seed 42, 1,024 matches, up to three players per team (the existing 1–3 team-size distribution), 15% evaluation matches, 48 rollout steps per PPO iteration, and 10,000 iterations. This is 491,520,000 match transitions per backend. Run headless without video during training. Use a different otherwise idle RTX 3090 for each process and pin it by GPU UUID, exposing logical `cuda:0` inside each process.

Use each checkout's own working Python environment. The standalone `train.py` below can be copied unchanged into the original Isaac Lab checkout. From the mjlab checkout:

```bash
CUDA_VISIBLE_DEVICES=GPU-MJLAB-UUID python scripts/comparison/train.py \
  --engine mjlab --gpu-id GPU-MJLAB-UUID --output /absolute/path/to/comparison/mjlab -- \
  --task Mjlab-Soccer-v0 --device cuda:0 --num_envs 1024 --max_iterations 10000 \
  --seed 42 --headless --logger tensorboard \
  --experiment_name 01_backend_comparison --run_name mjlab_seed42
```

From the original Isaac Lab checkout, using the installed Isaac Lab / Isaac Sim Python environment:

```bash
CUDA_VISIBLE_DEVICES=GPU-ISAAC-UUID python scripts/comparison/train.py \
  --engine isaac --gpu-id GPU-ISAAC-UUID --output /absolute/path/to/comparison/isaac -- \
  --task Isaac-Soccer-v0 --device cuda:0 --num_envs 1024 --max_iterations 10000 \
  --seed 42 --headless --logger tensorboard \
  --experiment_name 01_backend_comparison --run_name isaac_seed42
```

Replace both GPU UUID placeholders with the selected devices from `nvidia-smi -L`. The output directory must be new. The wrapper enables the same TF32 settings in both runs, reseeds Torch immediately before policy construction, and saves/hashes the initial policy weights. Check that these hashes match before interpreting the learning comparison.

Each output directory contains `metadata.json`, `initial_policy.pt`, `iterations.csv`, `scalars.csv`, `metrics.jsonl`, `step_timing.csv`, `gpu.csv`, live `progress.json`, and final `summary.json`. Existing trainer checkpoints, evaluation trajectories and TensorBoard logs remain in its normal log directory, recorded as `runner_log_dir` in metadata. Sampled simulator-step latency synchronizes CUDA before and after one vectorized control step every 100 calls. The first five iterations are labeled warmup; summaries of steady timings exclude them. End-to-end wall time includes initialization, compilation and logging. GPU telemetry is sampled every two seconds. Instrumentation overhead is included in both runs.

The captured metrics include episode returns/lengths, reward components, goals/wins/draws, policy actions and saturation, value/surrogate losses, entropy, learning rate, approximate post-update KL and clipping fraction, gradient norm, explained variance, simulation/learning/save timing, throughput, GPU memory/utilization/power/temperature and hardware/software revisions. No W&B login or external upload is used; select the TensorBoard logger as above.

Generate the local comparison page and CSV after either run starts, or again once both finish:

```bash
python scripts/comparison/report.py /absolute/path/to/comparison/mjlab \
  /absolute/path/to/comparison/isaac --output /absolute/path/to/comparison/report
```

Open `report/comparison.html` as a local file. The page plots metrics against actual wall time, needs no server, and includes run metadata and `comparison.csv`. The timing charts omit warmup. `progress.json` and the console log can be read while training runs. These commands describe the intended protocol; they do not mean that a remote run has started.

### Interpretation of the original Isaac baseline

The original code is kept as the legacy baseline. Its `EnvData.goal_status` is never published, so the score reward reads zero. It also clears scores before returning logging info; consequently score-driven replay snapshots and shaping decay in the runner do not behave like the corrected mjlab port. The port also fixes substep goal counting, per-player partial resets and stale critic bootstrapping. The comparison records these differences instead of silently changing the original task. Goal/win telemetry normalizes positive accumulated-score changes to one event per completed match, while retaining raw score increments.

The learning curves therefore compare these two implementations, including their documented fixes. They do not isolate the effect of the physics engine. A further experiment with identical corrected task/runner semantics would be needed for that claim. No runtime compatibility patch should be applied without recording its diff in the run metadata.

## mjlab browser viewer

Run this from the mjlab checkout on Jarvis, using its Python environment:

```bash
python scripts/comparison/visualize.py \
  --checkpoint /absolute/path/to/model_10000.pt \
  --device cpu --num_agents 3 --opponent bot --seed 42 \
  --host 127.0.0.1 --port 8080
```

The viewer uses mjlab's native Viser frontend and requires no display server, OpenGL context, or W&B account. CPU playback leaves both training GPUs free. It supports camera navigation, pause/resume, single stepping, reset, playback speed, collision/geometry overlays, and live reward plots. Reward plots show the first blue player's terms. Use `--opponent policy` for the existing play script's policy-versus-policy behavior; `--opponent bot` uses the original heuristic keeper and players for reproducible evaluation.

From your workstation, forward the port using the same SSH destination as your Jarvis connection (forward on the outer SSH connection when entering a distrobox):

```bash
ssh -N -L 8080:127.0.0.1:8080 USER@JARVIS_HOST
```

Then open <http://127.0.0.1:8080>. The default bind address is localhost. If the distrobox has a separate network namespace, forward to its reachable address instead; standard distrobox containers share the host network.

Use the actual saved checkpoint filename from the training directory. To compare checkpoints, stop and restart the viewer with another `--checkpoint`, keeping `--num_agents`, `--opponent`, and `--seed` identical. `--max_steps 5` runs a bounded startup/physics smoke test and then stops the server. This viewer reads a checkpoint once at startup; it does not load a checkpoint while training is writing it.

## Offline video

The existing play script can render an MP4 for a checkpoint (requires a working EGL/OpenGL implementation):

```bash
MUJOCO_GL=egl python scripts/rsl_rl/play.py \
  --checkpoint /absolute/path/to/model_10000.pt \
  --device cpu --num_agents 3 --seed 42 \
  --headless --video --video_length 200 --max_steps 200
```

The video is written beside the checkpoint under `videos/play/step_0.mp4`. The existing play script uses the loaded policy for both teams.
