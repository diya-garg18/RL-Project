"""Compare the tabular learners against Phase 1's DP solution (ROADMAP box 5).

Two questions, both asked over the states the agents actually visited:

  1. **Policy agreement %** — how often do two agents choose the same action?
  2. **Max-norm Q distance** — the largest disagreement about any single
     state-action value, `max |Q_a(s,a) - Q_b(s,a)|`.

**Why "over states actually visited" is the whole design.** Both sides have a
convention that fills in states they never saw, and both conventions look like
opinions when printed:

  - the learners leave an all-zero Q row, so `argmax` returns action 0 by the
    tie-break (FEATURE_005 / E-009);
  - DP treats an unvisited (s,a) as an absorbing self-loop with reward 0 (D-011).

Compared naively, two agents that have never visited a state would "agree"
there — and with 455 of 576 states unvisited by Q-learning and 443 by DP, that
manufactured agreement would dominate the headline number. This script reports
both figures precisely so the gap between them is visible.

A second caution the numbers cannot express: DP's Q is optimal for the
*estimated* model, not the true environment (D-004). Disagreement between a
learner and DP is therefore not evidence that the learner is wrong.

Usage:
    python scripts/compare_agents.py      # after run_dp.py and train.py --agent ...
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.config import load_env_config

RESULTS = ROOT / "results"
LEARNERS = ("q_learning", "sarsa", "monte_carlo")


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(Q, visits) for an agent, or exit with the command that would produce them."""
    q_path, v_path = RESULTS / f"{name}_Q.npy", RESULTS / f"{name}_visits.npy"
    if not q_path.exists() or not v_path.exists():
        how = "python scripts/run_dp.py" if name == "dp" else f"python scripts/train.py --agent {name}"
        raise SystemExit(f"missing {q_path.name} / {v_path.name} — run `{how}` first")
    return np.load(q_path), np.load(v_path)


def greedy(Q: np.ndarray) -> np.ndarray:
    """argmax with ties to the lower index — the convention every agent here uses."""
    policy = np.zeros(Q.shape[0], dtype=np.int64)
    for s in range(Q.shape[0]):
        best_a, best_v = 0, -np.inf
        for a in range(Q.shape[1]):
            if Q[s, a] > best_v:
                best_v, best_a = Q[s, a], a
        policy[s] = best_a
    return policy


def compare(name_a: str, name_b: str, data: dict) -> dict:
    Q_a, visits_a = data[name_a]
    Q_b, visits_b = data[name_b]

    seen_a = visits_a.sum(axis=1) > 0
    seen_b = visits_b.sum(axis=1) > 0
    both = seen_a & seen_b

    policy_a, policy_b = greedy(Q_a), greedy(Q_b)
    agree_all = float((policy_a == policy_b).mean())
    agree_seen = float((policy_a[both] == policy_b[both]).mean()) if both.any() else float("nan")

    # Max-norm over (s,a) pairs BOTH agents actually updated. An unvisited pair
    # holds a convention on one side and a real estimate on the other, so the
    # difference there measures the conventions, not the algorithms.
    pair_mask = (visits_a > 0) & (visits_b > 0)
    max_norm = float(np.abs(Q_a - Q_b)[pair_mask].max()) if pair_mask.any() else float("nan")

    return {
        "states_a": int(seen_a.sum()),
        "states_b": int(seen_b.sum()),
        "states_both": int(both.sum()),
        "agree_all": agree_all,
        "agree_seen": agree_seen,
        "max_norm": max_norm,
        "pairs_both": int(pair_mask.sum()),
    }


def main() -> None:
    load_env_config(ROOT / "config" / "env_default.yaml")  # fail early on a broken config
    data = {name: load(name) for name in (*LEARNERS, "dp")}

    print("Coverage — states each agent actually visited (of 576):")
    for name, (_, visits) in data.items():
        seen = int((visits.sum(axis=1) > 0).sum())
        pairs = int((visits > 0).sum())
        print(f"  {name:14s} {seen:3d} states   {pairs:4d}/2880 state-action pairs")

    print("\nPolicy agreement and max-norm Q distance")
    print("  'all 576' counts states neither agent has ever visited, where both")
    print("  fall back to a convention — it is the misleading number, shown for contrast.")
    print()
    print(f"  {'pair':28s} {'agree (both seen)':>18s} {'agree (all 576)':>16s} "
          f"{'max|dQ|':>10s} {'states':>7s}")

    pairs = [(learner, "dp") for learner in LEARNERS]
    pairs += [("q_learning", "sarsa"), ("q_learning", "monte_carlo"), ("sarsa", "monte_carlo")]

    for a, b in pairs:
        r = compare(a, b, data)
        print(f"  {a + ' vs ' + b:28s} {r['agree_seen'] * 100:17.1f}% "
              f"{r['agree_all'] * 100:15.1f}% {r['max_norm']:10.1f} {r['states_both']:7d}")

    print("\n  NOTE: DP's Q is optimal for the ESTIMATED model, not the true")
    print("  environment (D-004). A learner disagreeing with DP is not thereby wrong.")


if __name__ == "__main__":
    main()
