"""Print the five-state MRP worked example: hand arithmetic beside code output.

ROADMAP Phase 1, final box. Companion to
docs/features/phase1-mrp-worked-example.md and tests/test_mrp_bellman.py.

This script proves nothing the test suite does not already prove — its job is
to put the derivation on screen in a form that goes straight into the report
and can be read aloud in the viva. Run it before explaining Bellman to anyone.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.agents.dp import N_ACTIONS, value_iteration
from soc_triage.mrp_example import (
    GAMMA,
    HAND_COMPUTED_V,
    P_MRP,
    STATE_NAMES,
    TRANSITION_REWARD,
    as_degenerate_mdp,
    evaluate_iteratively,
    expected_rewards,
    solve_linear,
)


def main() -> None:
    R = expected_rewards(P_MRP, TRANSITION_REWARD)

    print(f"Five-state SOC Markov Reward Process  (gamma = {GAMMA})")
    print("An MRP has no actions - only states, transitions and rewards.\n")

    print("Transition matrix P(s'|s):")
    header = "                 " + "".join(f"{n[:5]:>8}" for n in STATE_NAMES)
    print(header)
    for s, name in enumerate(STATE_NAMES):
        row = "".join(f"{P_MRP[s, j]:>8.2f}" for j in range(len(STATE_NAMES)))
        print(f"  {name:<15}{row}")

    print("\nExpected one-step reward  R(s) = sum_s' P(s'|s) r(s,s')   [S&B eq. 3.5]")
    for s, name in enumerate(STATE_NAMES):
        terms = " + ".join(
            f"{P_MRP[s, j]:.2f}({TRANSITION_REWARD[s, j]:+.0f})"
            for j in range(len(STATE_NAMES))
            if P_MRP[s, j] > 0.0
        )
        print(f"  R({name:<14}) = {terms:<40} = {R[s]:+7.2f}")

    print("\nBellman equation   V(s) = R(s) + gamma * sum_s' P(s'|s) V(s')   [S&B eq. 3.14]")
    print("  V(CONFIRMED)     = 0                                    (absorbing, R = 0)")
    print("  V(MISSED)        = 0                                    (absorbing, R = 0)")
    print("  V(BACKLOG)       = -4  + 0.9[0.4(0) + 0.6(0)]           = -4")
    print("  V(INVESTIGATING) = +20 + 0.9[0.8(0) + 0.2(0)]           = +20")
    print("  V(QUIET)         = -1  + 0.9[0.5 V(QUIET) + 0.25(-4) + 0.25(20)]")
    print("                   = 2.6 + 0.45 V(QUIET)")
    print("    => 0.55 V(QUIET) = 2.6   =>   V(QUIET) = 52/11 = 4.7272...")

    v_closed = solve_linear(P_MRP, R, GAMMA)
    v_iter, sweeps = evaluate_iteratively(P_MRP, R, GAMMA, theta=1e-14, max_sweeps=10_000)
    P_mdp, R_mdp = as_degenerate_mdp(P_MRP, R, N_ACTIONS)
    v_dp, _, deltas = value_iteration(P_mdp, R_mdp, gamma=GAMMA, theta=1e-14, max_sweeps=10_000)

    print("\nFour routes to V, which must agree:")
    print(f"  {'state':<16}{'by hand':>12}{'closed form':>14}{'iterative':>12}{'agents/dp.py':>14}")
    for s, name in enumerate(STATE_NAMES):
        print(
            f"  {name:<16}{HAND_COMPUTED_V[s]:>12.6f}{v_closed[s]:>14.6f}"
            f"{v_iter[s]:>12.6f}{v_dp[s]:>14.6f}"
        )

    worst = max(
        float(np.abs(v_closed - HAND_COMPUTED_V).max()),
        float(np.abs(v_iter - HAND_COMPUTED_V).max()),
        float(np.abs(v_dp - HAND_COMPUTED_V).max()),
    )
    print(f"\n  iterative evaluation: {sweeps} sweeps")
    print(f"  agents/dp.py value iteration: {len(deltas)} sweeps, final delta {deltas[-1]:.2e}")
    print(f"  largest disagreement with the hand-derived answer: {worst:.2e}")
    print("\n  => the shipped Phase 1 solver reproduces a value function derived on paper.")


if __name__ == "__main__":
    main()
