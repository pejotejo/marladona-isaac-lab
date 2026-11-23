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

The four inference weights, selection record, raw episode archives and offline
reports ship under `artifacts/subpolicies1/`. Open the saved HTML report in a
browser to inspect the results. [Primary benchmark](../artifacts/subpolicies1/report.html)

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

To run the saved Ball Duel specialist on fresh starts:

```bash
uv run --locked scripts/subpolicies.py eval --drill duel --seed 173 --episodes 128 \
  --num-envs 128 --checkpoint artifacts/subpolicies1/weights/duel.pt --output runs/duel-173
```

To reproduce the primary matrix and render it:

```bash
weights=artifacts/subpolicies1/weights
uv run --locked scripts/subpolicy_benchmark.py --drills midfield duel positioning defense \
  --policy midfield "$weights/midfield.pt" --policy duel "$weights/duel.pt" \
  --policy positioning "$weights/positioning.pt" --policy defense "$weights/defense.pt" \
  --bot --seed 91009 --episodes 256 --num-envs 256 --output runs/reproduction
uv run --locked scripts/subpolicy_report.py runs/reproduction --output runs/reproduction.html
```

To regenerate the report from the archived episodes:

```bash
mkdir -p runs/saved-study
tar -xzf artifacts/subpolicies1/heldout.tar.gz -C runs/saved-study
uv run --locked scripts/subpolicy_report.py runs/saved-study/final --output runs/saved-study/report.html
```

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

## What changed during training

Sixteen FastSAC runs on Jarvis GPUs 0 and 1 produced 2,920,448 learner
transitions, excluding teacher data and evaluations. There were five runs
each for Mid Field and Ball Duel and three each for Positioning and Defense.
The saved run configurations identify the seeds and code revisions.

* Standard FastSAC reached the ball before it learned to finish attacks. At
  step 600 the Mid Field actor scored once in 128 validation episodes. At
  step 1,200 it scored 14 times. Ball Duel still scored zero at step 1,200.
* Imitation initialization taught the kick sequence, but continued RL could
  undo it. The initialized Mid Field actor scored in 82.8% of validation
  episodes; after 300 FastSAC steps this run scored in 5.5%. Later runs kept
  an auxiliary imitation loss and labeled recovery states reached by the actor.
  The selected policies execute the neural network alone.
* An early Positioning policy completed no verified receptions. In a 64-match
  trajectory probe with the scripted receiver, median pass travel during the
  2.5-second window was 0.46 m for a 1.25 m initial receiver gap. Stronger
  physical passes and a four-second window raised the scripted receiver's
  verified reception rate from 7.8% to 84.4% on the same validation starts.
  The travel, separation and receiver-control checks still apply.
* An idle defender conceded in 125 of 128 starts. The first learned defender
  conceded in 47. The revised drill gave clearance credit only for actual
  upfield travel and required a kick or sustained slow-ball control for a
  recovery. Training with lateral clearances and recovery demonstrations
  produced a specialist that could defend and regain possession.

The study uses single physical matches with one learned player. It evaluates
these particular checkpoints on randomized starts; intervals describe variation
across those starts. Optimizer settings and training seeds varied across runs.

## Held-out results

| Home drill | Mean return | Best other specialist | Bot return | Home specialist behavior on 256 starts |
|---|---:|---:|---:|---|
| Mid Field | 7.89 | 5.14 | 8.50 | Scored 222 goals and advanced the ball 2.48 m on average. |
| Ball Duel | 2.49 | 1.89 | 2.90 | Scored 104 goals, conceded 4, and averaged 2.85 s of exclusive proximity control. |
| Positioning | 5.25 | 2.73 | 2.96 | Completed 218 verified receptions and scored 3 goals. |
| Defense | 2.91 | 2.58 | 1.41 | Made 187 verified clearances, conceded 15 goals, and ended 150 trials at the boundary. |

The first three specialists' paired return differences against their nearest
learned competitors had positive 95% intervals. Defense led Ball Duel by 0.33
on the defensive drill, with an interval of −0.11 to +0.74. The scripted bot
was a stronger Mid Field finisher and had a higher mean Ball Duel return.
Positioning's strongest result was receiving passes; converting receptions
into goals remains an open training objective. Defense often cleared to touch.

The same four frozen weight files were subsequently repeated on seed 77101,
with 512 new starts per drill and all five controllers. All four specialists
again led the other learned policies on their own drill. The nearest-rival
margin for Defense was +0.50 (+0.23 to +0.79, paired 95% interval); for Ball
Duel it was +0.63 (+0.32 to +0.95). The bot again led the two attacking drills.
Across both held-out cohorts there are 15,360 scored episodes.
[Replication report](../artifacts/subpolicies1/replication.html)

The four small weight files include the actor and observation normalizer.
Full optimizer checkpoints and logs are in `~/sim/subpolicies1/runs/` inside
the Jarvis `johannes-isaac` distrobox; `selection.json` gives the relative
path and SHA-256 of each selected training checkpoint. The compressed archives
in the repository include trial starts, results and training/validation evidence.
