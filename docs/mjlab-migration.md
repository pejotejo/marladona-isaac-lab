# mjlab migration

The installed simulator uses mjlab 1.6 and MuJoCo-Warp. The original Isaac Lab code is retained for comparison and asset conversion; the active training and playback commands do not import it.

## Preserved behavior

- Two teams, configurable player count and active-player masks, training/evaluation split, per-player policy/replay/bot control.
- Five actions: body-frame forward/lateral velocity, yaw rate, and two ball-control commands. The original square-to-disc mapping, asymmetric speed limits, force gains, kick region, and rolling-drag torque remain.
- Physics timestep 0.02 s, four substeps per control step (0.08 s), and the original effective 30 s episode duration.
- Eighteen policy/critic features per player, team-relative mirroring, neighbor sorting and NaN padding, two-frame feature-major neighbor history, observation noise and symmetry indices. Checkpoint shapes are unchanged, including the bundled example policy.
- Original reward terms, weights, time scaling, score-triggered shaping decay, goal termination, border ball reset, initial-position distributions, and field curriculum.
- Original PPO, replay-policy selection, heuristic bots, evaluation trajectory format, and value-function plotting.

## Native simulation

Each player and the ball is a native mjlab `Entity` with a MuJoCo free joint. The player hull and visual meshes come directly from the existing USD assets, exported as OBJ files. The player mass is 5.480100154876709 kg, diagonal inertia is (0.01010004, 0.01010004, 0.0101) kg m², and the ball mass/radius/inertia are 0.044 kg / 0.05 m / (0.0001, 0.0001, 0.0001) kg m². As before, players are constrained to z=0.08 m with roll/pitch and vertical/roll/pitch velocities cleared at the start of each physics substep.

Controls write native world-frame body wrenches. Body-frame player forces are rotated by the current quaternion. Ball controls use the original front-of-player/low-ball eligibility checks, sum all eligible players' forces, and apply rolling drag. Ball sliding friction is sampled at startup from 64 buckets in [0.1, 0.5], matching the original configured startup event.

The static field bank contains one field per curriculum level, spaced by 8.32 m. These use the same border and goal-post dimensions as the generated Isaac terrain. A curriculum transition changes the selected origin, and only the reset match's players/ball move. MuJoCo-Warp worlds isolate parallel matches. Field markings and colored goal visuals are rebuilt with MuJoCo primitives; lighting and rendering are therefore different.

Regenerate the converted player and ball meshes with:

```bash
uv pip install -e '.[assets]'
python scripts/convert_assets.py
```

## Corrections during migration

The original `EnvData.step()` incremented score without publishing `goal_status`, so the score reward always read zero. The port publishes goal status and latches crossings over all substeps to award/count each goal once. Reset info records the completed score before clearing it. Player history, action, and reward buffers reset all player rows belonging to the selected match. Inactive players do not claim ball ownership. Explicit per-player controller assignments are honored. These fix observable inconsistencies in the previous implementation.

The training runner also handles CPU checkpoint loading, empty evaluation allocations, zero-teammate neighbor aggregation in 1v1, independent replay buffers, and current RSL logging APIs.

## Equivalence limits

MuJoCo and PhysX use different collision/contact solvers, friction combination rules, and integration. Exact trajectories, contact impulses, and policy performance are not claimed to match. The port uses the original physical parameters and control equations, but further contact calibration and training comparison require a working GPU/Isaac Lab reference installation. The screenshots/GIFs in the main README show historical gameplay, not renders of this port.

The validation suite exercises task math, history/masking/reset invariants, original checkpoint inference, and actual mjlab CPU physics. A full CPU PPO iteration, original-checkpoint playback in 1v1/3v3/5v5, Gym registration, and EGL offscreen rendering were also checked. CUDA throughput, GPU training convergence, native desktop rendering, and GUI plotting need validation on the target workstation.
