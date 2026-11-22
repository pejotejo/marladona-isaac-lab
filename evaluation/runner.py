"""Schedule a finite queue across independently terminating simulator slots."""

import numpy as np

from .metrics import METRICS, OBSERVERS
from .scenarios import Diagnostic


def _sample(frame, slot):
    row = {}
    for key, value in frame.items():
        sample = value if key == "team" else value[slot]
        row[key] = np.array(
            sample.cpu().numpy() if hasattr(sample, "cpu") else sample, copy=True
        )
    return row


class Episode:
    def __init__(self, trial_id, retain, observers):
        self.trial_id = trial_id
        self.rows = [] if retain else None
        self.events = []
        self.signals = {name: factory() for name, factory in observers.items()}
        self.states = {}

    def observe(self, frame, slot, codes, times):
        if self.rows is not None:
            self.rows.append(_sample(frame, slot))
        for name, observer in self.signals.items():
            state = observer.observe(codes[name][slot])
            if state != self.states.get(name):
                self.events.append(
                    {
                        "trial_id": self.trial_id,
                        "time_s": times[slot],
                        "kind": name,
                        "state": state,
                    }
                )
            self.states[name] = state

    def complete(self, elapsed, reason, archive, selected, rules):
        metrics = {m.name: m.measure(elapsed, reason, self.signals) for m in selected}
        result = {
            "trial_id": self.trial_id,
            "elapsed_s": elapsed,
            "outcome": rules.outcome(reason),
            "termination": reason,
            "metrics": metrics,
        }
        self.events.append(
            {"trial_id": self.trial_id, "time_s": elapsed, "kind": reason}
        )
        replay = None
        if self.rows:
            replay = {
                k: self.rows[0][k]
                if k == "team"
                else np.stack([r[k] for r in self.rows])
                for k in self.rows[0]
            }
        archive.complete(result, self.events, replay)


def run_trials(
    backend,
    trials,
    num_envs,
    archive,
    metrics=None,
    proximity_radius_m=0.25,
):
    rules = getattr(backend, "rules", Diagnostic.RULES)
    selected = [m for m in METRICS if metrics is None or m.name in metrics]
    required = {key for m in selected for key in m.requires}
    observers = {name: OBSERVERS[name] for name in required & OBSERVERS.keys()}

    observer_params = {"possession": {"radius_m": proximity_radius_m}}
    needed = required.union(*(o.requires for o in observers.values()))

    def classify(frame):
        available = frame.keys() | {"terminal_event"} | observers.keys()
        if missing := needed - available:
            raise ValueError(f"missing metric signals: {sorted(missing)}")
        return {
            name: observer.classify(frame, **observer_params.get(name, {}))
            for name, observer in observers.items()
        }, frame["time_s"].tolist()

    queue = iter(trials)
    active = {}

    def assign(slots):
        assignments = dict(zip(slots, queue))
        if assignments:
            frame = backend.reset(assignments)
            codes, times = classify(frame)
            for slot, trial in assignments.items():
                trial_id = trial["trial_id"]
                episode = active[slot] = Episode(
                    trial_id, trial_id in archive.replay_ids, observers
                )
                episode.observe(frame, slot, codes, times)

    assign(range(num_envs))
    while active:
        frame, reasons = backend.step()
        codes, times = classify(frame)
        finished = []
        for slot, episode in active.items():
            episode.observe(frame, slot, codes, times)
            if reasons[slot] is not None:
                episode.complete(times[slot], reasons[slot], archive, selected, rules)
                finished.append(slot)
        for slot in finished:
            del active[slot]
        assign(finished)
