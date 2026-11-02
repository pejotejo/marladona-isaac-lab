# MARLadona 
This repository contains the multi-agent training environment for the [MARLadona - Towards Cooperative Team Play Using Multi-Agent
Reinforcement Learning](https://arxiv.org/pdf/2409.20326) Paper.

The open-source version of the MARL soccer environment is built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab) and based on the [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate.git) 

This repository contains the multi-agent soccer environment `isaaclab_marl` and a heavily modified [rsl_rl](https://github.com/leggedrobotics/rsl_rl) training pipeline implemented in `rsl_marl`. The original implementation and paper results are based on Isaac Gym. This migration effort was made due to Isaac Gym's deprecation.         

**Maintainer: Zichong Li, zichong1230@gmail.com**

<figure>
  <figcaption><em>Cooperative Gameplay Like Never Before! (Isaaclab)</em></figcaption>
  <img src="docs/gifs/isaaclab.gif" alt="Isaaclab Play" title="Typical Gameplay in Isaac Lab">
</figure>


<figure>
  <figcaption><em>This Single Policy Adapts to Any Team Setup! (Isaac Gym)</em></figcaption>
  <img src="docs/gifs/isaacgym.gif" alt="Isaaclab Play" title="Higher Agent Number (Isaac Gym)">
</figure>


## Installation

- Install **Isaac Lab 3.0.0**, **Isaac Sim 6.2.0.0**, and **Python 3.12** using the [3.0 installation guide](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/installation.html). With `uv` installed, the source installation is:

```bash
git clone --branch v3.0.0 --single-branch https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
uv sync --extra isaacsim --extra video
source .venv/bin/activate
cd ..
```

This uses the tagged release's dependency versions, including PyTorch 2.11.0 with CUDA 13.0 and Warp 1.17.0. The soccer task explicitly selects Isaac Sim's **PhysX** backend. Keep this environment active for the commands below.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/leggedrobotics/marladona-isaac-lab.git

# Option 2: SSH
git clone git@github.com:leggedrobotics/marladona-isaac-lab.git
```

```bash
# Enter the repository
cd marladona-isaac-lab
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
uv pip install -e source/isaaclab_marl -e source/rsl_marl
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/rsl_rl/train.py --task=Isaac-Soccer-v0 --headless
```

On the `testing-single-humanoid` branch, each field has **one learning blue player, two scripted blue teammates, and three scripted red opponents**. PPO collects transitions only from the learning player. Team sizes stay fixed; policy replay and self-play are disabled. The existing PPO runner and neighbor observations are reused for single-agent learning.

Train a new seven-output policy with:

```bash
python scripts/rsl_rl/train.py --task=Isaac-Soccer-v0 --headless
```

For a short GPU smoke test, add `--num_envs=16 --max_iterations=2`. Omit `--headless` to open the Isaac Sim viewer. `--video` uses Isaac Lab 3.0's internal video recorder. Runs are stored under `wks_logs/single_humanoid_soccer` by default.

Play a trained checkpoint with the same scripted teammates and opponents:

```bash
python scripts/rsl_rl/play.py --task=Isaac-Soccer-Play-v0 --load_run='<run-directory>'
```

The supplied five-output MARL checkpoints are incompatible with this branch's action and observation shapes; train a new policy. Playback defaults to 3v3 to match training.

## Policy and motion interface

The policy emits `[vx, vy, yaw_rate, kick, target_x, target_y, strength]`, with no `quick` flag.

| Output | Range | Meaning |
| --- | --- | --- |
| `vx` | `[-1, 1]` | Body-frame forward/backward walking request; scaled to +0.35 / -0.25 m/s. |
| `vy` | `[-1, 1]` | Body-frame sideways walking request; scaled to ±0.25 m/s. |
| `yaw_rate` | `[-1, 1]` | Rotation request; scaled to ±2.5 rad/s. |
| `kick` | `[0, 1]` | A rising crossing above 0.5 requests a complete approach, alignment, and kick. |
| `target_x`, `target_y` | `[-1, 1]` | Ball destination in normalized, team-relative field coordinates. |
| `strength` | `[0, 1]` | Fraction of `soccer_game.ball.max_kick_speed`; zero produces no strike. |

The target frame has its origin at the field center, with x scaled by half the field length and y by half the field width. It matches the policy's ball/neighbor observations: **both teams attack toward negative x**. `(-1, 0)` targets the center of the opponent's goal. Red's physical field coordinates are rotated by 180 degrees; arena placement in the world does not affect the target.

Walking requests use the existing combined translation limit. On a kick request, the motion action latches the destination and strength, walks behind the ball, turns toward the destination, and strikes only when the ball is in front, within 0.2 m, within 0.04 m laterally, and aligned within 0.12 rad. Walking outputs and subsequent destination changes are ignored while this motion is active. A request times out after 15 seconds; a completed strike has a 0.5-second cooldown. Lower `kick` to 0.5 or below, then raise it again after completion/cooldown to request another kick. Field resets clear pending motions. Policy and critic observations include the pending request, latched target/strength, timeout, and cooldown.

The default maximum outgoing ball speed is approximately 0.79 m/s (`BALL_MAX_SPEED`), with `speed = strength * max_kick_speed`. This controls outgoing speed, not arrival speed or guaranteed travel distance; friction and collisions still apply. Simultaneous eligible kicks are resolved to the closest player.

**The supplied robot assets remain planar rigid-body proxies.** Walking uses the existing velocity/force controller, and the proxy kick sets the ball's outgoing velocity. This branch supplies the high-level motion interface and simulated approach/kick behavior; an articulated humanoid needs its own locomotion and calibrated kick controller behind that interface. Quick kicks and relaxed alignment are omitted.

Tests cover single-learner masks, seven-output PPO, motion behavior, and the migrated Isaac Lab APIs. Full Isaac Sim/PhysX training, viewer/video recording, and new-checkpoint inference still require validation.

To run the focused compatibility tests in the installed environment:

```bash
uv pip install pytest
python -m pytest -q tests
```

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

The GUI assumes all logs are stored inside the `wks_logs` folder. It selects only the experiment folder prefix with digits, e.g., `00_example_policies`. Make sure all runs contain a non-empty `eval_traj` folder. This should be the case for all training runs that have finished the initialization. 

Note: The GUI application is built using pyqtgraph and PyQt5, so double-check your pip package version to see if the dependencies are not already auto-resolved by the setup.py.

### Value Function Visualizer ### 

![Value Functions](docs/gifs/value_function.gif)

The value function visualizer provides additional insight into the agent's intention. You can enable or disable the visualization via the `VISUALIZE_VALUE_FUN` flag in the play.py script.

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
