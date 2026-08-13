"""Environment correctness tests (ROADMAP Phase 0 list).

Covers: determinism under a fixed seed, reward accounting summing correctly,
bulk-close never exceeding its cap, and the step-ordering rule from FLOW.md.
"""

import numpy as np

from soc_triage.env import BULK_CLOSE_LOW_RISK, SOCTriageEnv
from soc_triage.state import discretise


def _run_episode(cfg, seed: int, action_seed: int):
    """Roll one full episode with seeded-random actions; return the trace."""
    env = SOCTriageEnv(cfg)
    rng = np.random.default_rng(action_seed)
    snap = env.reset(seed)
    trace = []
    done = False
    while not done:
        action = int(rng.integers(0, 5))
        snap, reward, done, info = env.step(action)
        trace.append((discretise(snap, cfg), action, reward, info["time_consumed"]))
    return trace, env.episode_outcome()


def test_env_deterministic_under_fixed_seed(cfg):
    """Same env seed + same action sequence => bit-identical trajectory and outcome."""
    trace_a, outcome_a = _run_episode(cfg, seed=7, action_seed=99)
    trace_b, outcome_b = _run_episode(cfg, seed=7, action_seed=99)
    assert trace_a == trace_b
    assert outcome_a == outcome_b


def test_different_seeds_differ(cfg):
    """Different env seeds must produce different shifts (sanity check on seeding)."""
    trace_a, _ = _run_episode(cfg, seed=7, action_seed=99)
    trace_b, _ = _run_episode(cfg, seed=8, action_seed=99)
    assert trace_a != trace_b


def test_reward_breakdown_sums_to_step_reward(cfg):
    """Every step's reward must equal the sum of its own breakdown dict.

    This is the accounting identity that makes reward_breakdown trustworthy
    for debugging and for the dashboard.
    """
    env = SOCTriageEnv(cfg)
    rng = np.random.default_rng(3)
    env.reset(seed=11)
    done = False
    while not done:
        _, reward, done, info = env.step(int(rng.integers(0, 5)))
        assert abs(reward - sum(info["reward_breakdown"].values())) < 1e-9


def test_bulk_close_never_exceeds_cap(cfg):
    """Action 4 closes at most max_alerts alerts, only eligible ones (ROADMAP)."""
    env = SOCTriageEnv(cfg)
    rng = np.random.default_rng(5)
    env.reset(seed=13)
    rules = cfg.actions.bulk_close
    done = False
    bulk_steps = 0
    while not done:
        # Bias toward bulk-close so the cap actually gets exercised on big queues.
        action = BULK_CLOSE_LOW_RISK if rng.random() < 0.5 else int(rng.integers(0, 4))
        _, _, done, info = env.step(action)
        closed = info["bulk_closed"]
        if closed:
            bulk_steps += 1
            assert len(closed) <= rules.max_alerts
            for alert in closed:
                assert alert.severity <= rules.max_severity
                assert alert.asset_criticality <= rules.max_asset_criticality
    assert bulk_steps > 0, "bulk close never fired — test exercised nothing"


def test_clock_terminates_at_shift_end(cfg):
    """Episode ends exactly when the clock passes 480; stepping after raises."""
    env = SOCTriageEnv(cfg)
    env.reset(seed=17)
    done = False
    steps = 0
    while not done:
        snap, _, done, _ = env.step(0)
        steps += 1
        assert steps < 10_000, "episode never terminated"
    assert snap.time_now >= cfg.shift.length_min
    try:
        env.step(0)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "step() after done must raise"
