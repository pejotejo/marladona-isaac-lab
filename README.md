# MARLadona on mjlab
This repository contains the multi-agent training environment for the [MARLadona - Towards Cooperative Team Play Using Multi-Agent
Reinforcement Learning](https://arxiv.org/pdf/2409.20326) Paper.

The simulator now runs on [mjlab](https://github.com/mujocolab/mjlab) and MuJoCo-Warp. Isaac Sim, Omniverse, and Isaac Lab are not required.

The multi-agent soccer environment is in `source/mjlab_marl`; the original modified PPO/self-play pipeline remains in `source/rsl_marl`. The previous Isaac Lab implementation and USD assets remain in `source/isaaclab_marl` as reference material and are not installed by the main package. The paper results were obtained in Isaac Gym.

The port preserves the five player actions, movement and kick equations, reward weights, observations and history order, field/team configuration, curriculum, bots, policy replay, and checkpoint parameter shapes. Player and ball meshes are converted from the original USD files, with the original collision shape, mass, and inertia. PhysX and MuJoCo contact solvers differ: this is a task/asset/interface port, and numerical trajectory or trained-policy performance equivalence is not established. See [migration details](docs/mjlab-migration.md).

**Maintainer: Zichong Li, zichong1230@gmail.com**

<figure>
  <figcaption><em>Reference gameplay from the original Isaac Lab version</em></figcaption>
  <img src="docs/gifs/isaaclab.gif" alt="Isaaclab Play" title="Typical Gameplay in Isaac Lab">
</figure>


<figure>
  <figcaption><em>This Single Policy Adapts to Any Team Setup! (Isaac Gym)</em></figcaption>
  <img src="docs/gifs/isaacgym.gif" alt="Isaaclab Play" title="Higher Agent Number (Isaac Gym)">
</figure>


## Installation

Use Python 3.10+ (3.12 recommended). For training at scale, install a CUDA-enabled PyTorch build and use an NVIDIA GPU supported by mjlab. CPU simulation is available for small smoke tests.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]'
```

For a CPU-only installation, install PyTorch first:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[dev]'
```

Train with the original multi-agent PPO/self-play runner:

```bash
python scripts/rsl_rl/train.py --task Mjlab-Soccer-v0 --num_envs 1024 --headless
```

Run one training iteration on CPU to check the complete pipeline:

```bash
python scripts/rsl_rl/train.py --device cpu --num_envs 2 --max_iterations 1 --headless
```

Logs and checkpoints are saved under `wks_logs`. Play the bundled example policy in the native MuJoCo viewer:

```bash
python scripts/rsl_rl/play.py --task Mjlab-Soccer-Play-v0 --num_agents 3 \
  --experiment_name 00_example_policies --load_run 24_09_28_11_56_41_3v3
```

Add `--device cpu --headless --max_steps 10` for a playback smoke test without a GPU/display. Use `--checkpoint /path/to/model.pt` for a specific checkpoint. `--num_agents` selects one to five players per team; `--num_envs`, `--seed`, `--video`, and `--visualize_value_fun` are also available. Video needs a working OpenGL/EGL context (`MUJOCO_GL=egl` on a headless GPU host).

`SoccerMARLEnvCfg` and `SoccerMARLEnvPlayCfg` expose the full configuration, including per-player control assignments and team size ranges:

```python
from mjlab_marl import SoccerMARLEnv, SoccerMARLEnvPlayCfg

env = SoccerMARLEnv(SoccerMARLEnvPlayCfg(), device="cuda:0")
obs, info = env.reset()
# Actions: (num_envs * 2 * num_agents_per_team, 5), values in [-1, 1].
# step returns observations, per-player rewards, terminated, truncated, info.
env.close()
```

Run regression checks with `pytest`. The simulation tests use actual mjlab / MuJoCo-Warp on CPU; the first run compiles Warp kernels and may take a few minutes.

## Visualization Tools ## 

### Trajectory Analyser ###
![Trajectory Analyser](docs/gifs/3v3_traj.gif) 

The framework provides a convenient GUI to visualize and compare policy behavior across many experiments. The trajectories are collected periodically during training on the evaluation environments, which is about 15% of the total environment. In these environments, the adversaries are configured to use a simple heuristic bot as a controller to increase reproducibility and also provide a standardized resistance to our trainees. Furthermore, all randomizations regarding the team size and initial position are fixed. This makes qualitative comparisons of behavior between different checkpoints and experiments much easier. 

To start the trajectory analyser, simply run the following command: 
```bash
python scripts/traj_analyser.py 
```

<img src="docs/pngs/traj_analyser_gui.png" alt="GUI"/>

You can select the experiment_folder and run name from the dropdown box on the left. This will automatically update the sliders in the middle. The sliders allow you to filter the trajectories according to the team configuration, and you can easily iterate over all checkpoints and all environments with the given team configuration.  

Furthermore, the GUI also supports storing highlights, which can be managed via the Add and Delete buttons on the right side.      

The GUI assumes all logs are stored inside the `wks_logs` folder. It selects only the experiment folder prefix with digits, e.g., `00_example_policies`. Make sure all runs contain a non-empty `eval_traj` folder. Evaluation trajectories require enough environments for `eval_env_ratio` to allocate at least one evaluation match.

Install the optional GUI dependencies with `uv pip install -e '.[analysis]'`.

### Value Function Visualizer ### 

![Value Functions](docs/gifs/value_function.gif)

The value function visualizer provides additional insight into the agent's intention. Enable it with `--visualize_value_fun` in the play script; it requires a desktop matplotlib backend.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
