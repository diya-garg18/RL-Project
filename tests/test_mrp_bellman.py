"""Phase 1 correctness anchor: a hand-solved MRP the shipped DP solver must reproduce.

ROADMAP Phase 1, final box. The 576-state MDP cannot be checked by hand, so
these tests check the Bellman machinery on the smallest object that can be:
the five-state MRP in src/soc_triage/mrp_example.py, whose value function is
derived with a pen in docs/features/phase1-mrp-worked-example.md.

The test that actually matters is `test_project_value_iteration_reproduces_hand_answer`
— it runs `agents.dp.value_iteration`, the same function that produced the
Phase 1 policy, and demands the number a human wrote down on paper.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soc_triage.agents.dp import N_ACTIONS, value_iteration  # noqa: E402
from soc_triage.mrp_example import (  # noqa: E402
    GAMMA,
    HAND_COMPUTED_V,
    N_MRP_STATES,
    P_MRP,
    TRANSITION_REWARD,
    as_degenerate_mdp,
    evaluate_iteratively,
    expected_rewards,
    solve_linear,
)

# Tight: every route below is exact arithmetic on five states, so any real
# disagreement will be enormous, not marginal. A loose tolerance here would
# hide precisely the kind of bug this file exists to catch.
TOL = 1e-9


@pytest.fixture(scope="module")
def R():
    """R(s) = sum_s' P(s'|s) r(s,s') — computed once, reused by every route."""
    return expected_rewards(P_MRP, TRANSITION_REWARD)


def test_transition_matrix_is_stochastic():
    """Rows must sum to 1, or the MRP is not a probability model at all."""
    row_sums = P_MRP.sum(axis=1)
    assert np.allclose(row_sums, np.ones(N_MRP_STATES), atol=TOL), row_sums
    assert (P_MRP >= 0.0).all()


def test_expected_rewards_match_hand_arithmetic(R):
    """The R(s) column of the worked example, done by hand:

        R(QUIET)         = 0.50(-1) + 0.25(-1) + 0.25(-1)  = -1
        R(BACKLOG)       = 0.40(+20) + 0.60(-20)           = -4
        R(INVESTIGATING) = 0.80(+30) + 0.20(-20)           = +20
        R(CONFIRMED)     = R(MISSED)                       =  0
    """
    expected = np.array([-1.0, -4.0, 20.0, 0.0, 0.0], dtype=np.float64)
    assert np.allclose(R, expected, atol=TOL), f"got {R}, hand-worked {expected}"


def test_closed_form_matches_hand_answer(R):
    """V = (I - gamma*P)^-1 R must equal the pen-and-paper value function."""
    V = solve_linear(P_MRP, R, GAMMA)
    assert np.allclose(V, HAND_COMPUTED_V, atol=TOL), f"got {V}, hand-worked {HAND_COMPUTED_V}"


def test_quiet_state_value_is_the_exact_fraction(R):
    """V(QUIET) = 52/11 exactly — the one non-integer, and the one that needs a solve.

    By hand:  V(QUIET) = -1 + 0.9[0.5 V(QUIET) + 0.25(-4) + 0.25(20)]
                       = 2.6 + 0.45 V(QUIET)
              0.55 V(QUIET) = 2.6   =>   V(QUIET) = 52/11
    """
    V = solve_linear(P_MRP, R, GAMMA)
    assert V[0] == pytest.approx(52.0 / 11.0, abs=TOL)


def test_iterative_evaluation_converges_to_the_same_answer(R):
    """Repeated Bellman backups must reach the closed-form fixed point.

    Closed form and iteration agreeing is the practical demonstration that the
    Bellman operator is a contraction (S&B §4.1) — the reason value iteration
    terminates at all.
    """
    V, sweeps = evaluate_iteratively(P_MRP, R, GAMMA, theta=1e-14, max_sweeps=10_000)
    assert np.allclose(V, HAND_COMPUTED_V, atol=TOL), f"got {V} after {sweeps} sweeps"
    assert sweeps > 1, "a self-loop on QUIET means this cannot converge in one sweep"


def test_absorbing_states_have_zero_value(R):
    """CONFIRMED and MISSED loop forever collecting nothing, so V must be 0.

    Not a triviality: it is the check that the payoff was booked on the
    *transition into* the terminal (eq. 3.5) and is not being paid out again on
    every subsequent absorbing step, which would send V to +/- infinity.
    """
    V = solve_linear(P_MRP, R, GAMMA)
    assert V[3] == pytest.approx(0.0, abs=TOL)
    assert V[4] == pytest.approx(0.0, abs=TOL)


def test_project_value_iteration_reproduces_hand_answer(R):
    """THE test: agents/dp.py's value_iteration, on the hand-solved MRP.

    The MRP is expanded to an MDP whose five actions are all identical, so the
    `max_a` in the value-iteration backup collapses and the optimal value
    function must equal the MRP's value function. This is the only place in
    Phase 1 where the shipped solver is checked against a number a human
    derived independently of the code.
    """
    P_mdp, R_mdp = as_degenerate_mdp(P_MRP, R, N_ACTIONS)
    V, policy, deltas = value_iteration(
        P_mdp, R_mdp, gamma=GAMMA, theta=1e-14, max_sweeps=10_000
    )

    assert np.allclose(V, HAND_COMPUTED_V, atol=TOL), (
        f"agents/dp.py value_iteration gave {V}, hand-worked answer is {HAND_COMPUTED_V}"
    )
    # Deltas must shrink monotonically toward the threshold — the convergence
    # curve plotted in results/dp_convergence.png, in miniature.
    assert deltas[-1] < 1e-14
    assert len(policy) == N_MRP_STATES
