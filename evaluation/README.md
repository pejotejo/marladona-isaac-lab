# Evaluating soccer controllers

The evaluation script runs trials and saves their results. The report script opens those saved results later, including retained pitch replays. Reports can be generated with the simulator stopped and the checkpoint unavailable.

Run these commands from the repository root after following the project installation instructions. Choose a fresh output directory for each evaluation.

```bash
uv run scripts/evaluate.py --episodes 5 --num-envs 2 --output runs/diagnostic
uv run scripts/evaluation_report.py runs/diagnostic --output runs/diagnostic.html
uv run scripts/evaluate.py --episodes 5 --trials runs/diagnostic/trials.jsonl \
  --candidate policy --candidate-checkpoint path/to/model.pt --output runs/candidate
uv run scripts/evaluation_report.py runs/candidate --compare runs/diagnostic --output runs/paired.html
```

For standalone reporting in an existing environment with NumPy and PyYAML, run `python scripts/evaluation_report.py RUN --output report.html`.

A **trial** is one concrete starting state and one episode budget. Team A is always the candidate and attacks negative x. One record counts as one episode. The diagnostic fixture repeats a fixed formation to exercise the evaluation path. Reusing trials requires matching the scenario, parameters, seed, episode count and step budget. Controller identities and checkpoint SHA-256 hashes live in the run manifest and suite configuration.

A **slot** is one parallel match. A slot receives its next trial immediately after it finishes. A physics tick crossing a goal line or field boundary ends the trial. Otherwise it ends after the configured number of controller decisions. Goals determine win or loss from team A's perspective. Boundary exits and timeouts count as draws.

Settings start with the Python framework defaults, followed by the selected scenario's defaults. A suite overrides only the values it supplies. Explicit CLI flags take precedence over the suite, and omitted flags preserve the resolved values. The complete effective configuration is saved in `suite.json`.

Use `--suite FILE.yaml` to select a suite. For example:

```yaml
scenario: diagnostic
episodes: 8
num_envs: 3
max_steps: 100
replay_count: 2
proximity_radius_m: 0.25
metrics: [goals_a, goals_b, proximity_turnovers]
candidate: bot
opponent: bot
params:
  ball_y: 0.2
```

**Coordinates and signals.** Positions are in metres relative to the field centre, yaw in radians, velocity in metres/second or radians/second, and elapsed time in simulation seconds. Player indices are stable, team A then team B. Metrics use simulator truth. Policies receive the existing policy and neighbor observations. The proximity possession measure uses active players within the configured radius (default 0.25 m) of the ball at decision frames, including the initial and terminal frame. Both teams in range means contested, neither means unowned. A transfer counts when the next exclusive owner differs from the previous exclusive owner, including across contested or unowned intervals.

**Adding a scenario.** Add a module under `evaluation/scenarios/` and register its class in `SCENARIOS`. Subclass `Diagnostic` to inherit its formation, validation, defaults and scoring rules, then override the pieces the new experiment needs. The generator receives the seed, index and resolved parameters and produces a concrete initial state. Bump scenario and parameter versions when their meaning changes.

Set the scenario's `RULES` to a `SoccerRules` instance or subclass instance. Its `classify(frame, physics_event)` returns one integer code per slot at each physics tick. Extend `reasons` with custom code names and override `outcome(reason)` to score them. The backend applies the common timeout to slots that continue.

**Adding a metric.** Add a named, versioned metric in `evaluation/metrics/`, declare its units and required signals, and register it in `METRICS`. Select its name in the suite's `metrics` list.

Metrics that need observations throughout an episode can register an observer factory in `OBSERVERS`. Each trial gets a new observer. The metric declares the observer name among its required signals, and the observer declares its required frame fields. Its `classify(frame)` produces compact per-slot codes once per batch. Its `observe(code)` updates the episode state and can emit a categorical event. The metric reads that state when the trial ends.

**Archive.** Schema v1 stores manifest.json, suite.json, trials.jsonl, events.jsonl and results.jsonl. Each committed result makes that episode visible to offline readers, including after interruption. Replays are compressed, pickle-free NPZ files selected deterministically by seed and trial ID. Replay v2 includes player headings and three-component ball velocities. The offline report also reads position-only v1 replays. Every committed episode has metrics and timestamped events. Only retained episodes have the full sampled trajectory. New plots can use archived summaries and events for all trials. New trajectory metrics require retained replays containing their inputs or a fresh run.

The diagnostic ball starts strictly inside the 5.4 × 3.6 m field so its first exit has an observable edge. Inactive players must start stationary with |y| ≥ 2.5 m to keep their collision bodies clear of play. Each scenario's `validate_state` checks its physical constraints on both generated and loaded trials.
