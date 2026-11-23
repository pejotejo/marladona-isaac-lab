# Subpolicy experiments

The blue outfield player attacks the goal at negative x. Each checkpoint must
run on every drill with the same observation and action interface. The other
active players use scripts. Record the initial poses, velocities and script
parameters so that a candidate and its competitors see the same trials.

## How to interpret the comparison

Use fresh training starts and seed 31001 for validation. Select checkpoints
using validation only. Reserve seed 91009 and at least 200 trials per drill
for the final comparison. Run every selected checkpoint and the scripted player
on each drill. Keep the simulator configuration and trial set fixed within each
comparison. A return on one drill has different units and scale from the
return on another. Compare policies on the same starts for each drill.

Report paired differences and 95% bootstrap intervals over starts. Report
actual goals, concessions, boundary exits and possession alongside return.
For Mid Field, check progress toward goal and scoring. For Ball Duel, inspect
possession against each opponent mode and whether control leads to progress.
For Positioning, count received passes and inspect passing lanes and spacing.
For Defense, count conceded goals, recoveries and clearance distance.

A positive difference against every other specialist on its home drill
supports specialization. An interval crossing zero leaves that comparison
unresolved. A good generalist may legitimately win multiple drills. Keep
the original test set and report those results even when a specialist loses.

## Running the policies

Run from this worktree using the repository's uv environment. The simulation
requires Isaac Lab and an NVIDIA GPU. The offline report only needs Python and
NumPy. Use a fresh output directory per run.

```bash
uv run --locked scripts/subpolicies.py train --drill duel --seed 422 --num-envs 128 \
  --updates 1200 --learning-starts 64 --num-updates 2 --batch-size 1024 \
  --gamma .99 --actor-lr .00007 --teacher-steps 320 --demo-weight 6 \
  --checkpoint-interval 300 --output runs/new-duel
uv run --locked scripts/subpolicies.py eval --drill duel --seed 31001 --episodes 128 \
  --num-envs 128 --checkpoint runs/new-duel/model_0001200.pt --output runs/new-duel-validation
```

For interactive playback, run the evaluation command with `--no-headless --num-envs 1`.
The actor receives the selected player's policy and neighbor observations. The
heading of a missing player has sine and cosine both zero. The native FastSAC
actor produces seven soccer motion and target-kick commands. Scripts and
reward calculations can read simulator state. Execution of the learned actor
requires neither a teacher nor the scenario identity.

Training uses one learner per match and independent episode resets. A native
checkpoint includes the actor, critic, target critic, optimizers, and running
observation statistics. Resuming starts a new replay buffer. Optional teacher
steps initialize the actor using physical rollout demonstrations. A positive
demo weight keeps an imitation term in the actor loss alongside the usual
FastSAC entropy and twin-critic objective. Recovery steps add teacher labels
on states reached by the current actor.

## Drill definitions

| Drill | Starting situation and scripts | What the score measures |
|---|---|---|
| Mid Field | Blue starts near the ball. A distant defender holds position or moves slowly to screen. | Distance gained toward goal, goals, boundary exits and time under opposing control. |
| Ball Duel | Both players start close. Red varies its angle and pace, approaches, blocks or challenges with a kick. | Goal progress, exclusive control gained and retained, time spent under opponent control, goals and boundary exits. |
| Positioning | A moving blue carrier offers a pass to an outfield receiver. Two defenders can hold or move to cover lanes. | Spacing, lane and shot quality at release, a verified reception and a subsequent goal. |
| Defense | A red attacker pursues a moving ball at varying speeds and distances from blue's goal. | Sustained recovery or a defender kick, a verified clearance, moving the ball upfield, and avoiding concessions. |

Proximity control is an estimate. The ball drills use a 0.30 m radius and a
0.02 m advantage over the opponent. Positioning credits a pass once per trial:
the carrier must physically kick, the ball must travel free and move at least
0.45 m, then the receiver must control it or execute a first-touch kick.
Defense counts one clearance after a blue kick moves the ball at least 0.75 m
upfield before an opponent recovers it. Timeouts earn no clearance bonus.
