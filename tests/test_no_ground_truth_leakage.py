"""THE integrity test (CONSTRAINTS.md #1). Never weaken, never skip.

If either state encoding changes when only the hidden fields change, ground
truth is leaking into observations and every result the agent produces is
fraudulent. Method: build a snapshot, flip `is_true_incident` and
`deadline_min` on every alert, assert both encodings are bit-identical.
"""

import dataclasses

import numpy as np

from soc_triage.env import SOCTriageEnv
from soc_triage.state import EnvSnapshot, discretise, featurise


def _flip_hidden_fields(snap: EnvSnapshot) -> EnvSnapshot:
    """Return the same snapshot with every alert's ground truth inverted/shifted."""
    flipped = tuple(
        dataclasses.replace(
            alert,
            is_true_incident=not alert.is_true_incident,
            deadline_min=alert.deadline_min + 123.0,
        )
        for alert in snap.queue
    )
    return dataclasses.replace(snap, queue=flipped)


def test_no_ground_truth_leakage(cfg):
    """Both encoders must be blind to is_true_incident and deadline_min.

    Walked through three real episodes (random actions, fixed seeds) so the
    check covers many genuine queue states, not one artificial example. The
    per-state assertions are the integrity check; the floor at the end only
    guards against the walk silently checking too few states (a single
    episode's step count varies with the action mix, so three episodes).
    """
    env = SOCTriageEnv(cfg)
    rng = np.random.default_rng(0)

    checked = 0
    for episode_seed in (1, 2, 3):
        snap = env.reset(seed=episode_seed)
        done = False
        while not done:
            flipped = _flip_hidden_fields(snap)
            assert discretise(snap, cfg) == discretise(flipped, cfg), (
                "discretise() changed when only hidden ground truth changed — LEAK"
            )
            assert np.array_equal(featurise(snap, cfg), featurise(flipped, cfg)), (
                "featurise() changed when only hidden ground truth changed — LEAK"
            )
            checked += 1
            snap, _, done, _ = env.step(int(rng.integers(0, 5)))

    assert checked > 100, f"only {checked} states checked across 3 episodes — walk is broken"


def test_snapshot_carries_no_precomputed_truth_fields():
    """EnvSnapshot itself must not grow fields derived from ground truth.

    Guards against someone adding e.g. `n_true_in_queue` to the snapshot later.
    Whitelist the fields that are allowed to exist.
    """
    allowed = {"queue", "time_now", "shift_length", "alerts_handled", "incidents_confirmed"}
    actual = {f.name for f in dataclasses.fields(EnvSnapshot)}
    assert actual == allowed, (
        f"EnvSnapshot fields changed: {actual ^ allowed}. If deliberate, prove the new "
        "field encodes no ground truth, then update this whitelist AND EXPLAIN.md."
    )
