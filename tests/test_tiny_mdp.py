"""Phase 2 correctness anchor: a hand-solved 2-state MDP the learners must reproduce.

ROADMAP Phase 2, final box. The 576-state MDP cannot be checked by hand, so
these tests establish a Q-table that *was* checked by hand — the two-state MDP
in src/soc_triage/tiny_mdp.py, derived with a pen in
docs/features/FEATURE_002_tiny_mdp_qstar.md.

This file verifies the anchor itself. It contains no learning code, because the
anchor has to be known-good *before* any learner is measured against it —
otherwise a disagreement between Q-learning and the fixture is ambiguous about
which of the two is wrong. The learner tests live in tests/test_tabular.py and
import HAND_COMPUTED_Q from here on out.

The test that carries the weight is
`test_project_value_iteration_reproduces_hand_answer` — it runs
`agents.dp.value_iteration`, already verified against an independent
hand-derived answer in Phase 1 (E-005), and demands it agree with this new
one. That cross-check is what promotes tiny_mdp from "some numbers someone
typed" to a trusted reference.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soc_triage.agents.dp import N_ACTIONS, value_iteration  # noqa: E402
from soc_triage.tiny_mdp import (  # noqa: E402
    BUSY,
    GAMMA,
    HAND_COMPUTED_POLICY,
    HAND_COMPUTED_Q,
    HAND_COMPUTED_V,
    MIN_ACTION_GAP,
    N_TINY_ACTIONS,
    N_TINY_STATES,
    P_TINY,
    QUIET,
    R_TINY,
    WAIT,
    WORK,
    bellman_optimality_residual,
    epsilon_soft_q,
    greedy_from_q,
    pad_actions,
    step,
)

# Two states and exact arithmetic: any genuine disagreement here will be large,
# not marginal. The only error present is float rounding on 0.9 * 13 (~1e-15),
# so a loose tolerance would hide exactly the bugs this file exists to catch.
TOL = 1e-12


# ---------------------------------------------------------------------------
# The model itself
# ---------------------------------------------------------------------------


def test_transition_model_is_a_probability_model():
    """Every P[s,a,:] must sum to 1, or this is not an MDP at all."""
    for s in range(N_TINY_STATES):
        for a in range(N_TINY_ACTIONS):
            assert P_TINY[s, a].sum() == pytest.approx(1.0, abs=TOL), (s, a)
    assert (P_TINY >= 0.0).all()


def test_transitions_are_deterministic():
    """Exactly one reachable successor per (s,a) — the design assumes it.

    If this ever fails, the tolerances everywhere else in Phase 2 are wrong:
    a stochastic fixture would need averaging, and the learners would no longer
    be expected to hit HAND_COMPUTED_Q exactly.
    """
    for s in range(N_TINY_STATES):
        for a in range(N_TINY_ACTIONS):
            assert np.count_nonzero(P_TINY[s, a]) == 1, (s, a)


def test_step_agrees_with_the_transition_matrix():
    """The sampler the learners call and the matrix the solvers read are one model.

    They are two separate code paths to the same dynamics. If they drift apart,
    every learner trains on one MDP and is graded against another — which would
    look like a broken algorithm rather than a broken fixture.
    """
    for s in range(N_TINY_STATES):
        for a in range(N_TINY_ACTIONS):
            next_state, reward = step(s, a)
            assert P_TINY[s, a, next_state] == 1.0, (s, a, next_state)
            assert reward == pytest.approx(R_TINY[s, a], abs=TOL)


def test_the_worked_transitions_are_the_ones_documented():
    """The four transitions, spelled out, exactly as FEATURE_002 describes them."""
    assert step(QUIET, WAIT) == (QUIET, +1.0)
    assert step(QUIET, WORK) == (BUSY, -5.0)
    assert step(BUSY, WAIT) == (BUSY, -1.0)
    assert step(BUSY, WORK) == (QUIET, +4.0)


# ---------------------------------------------------------------------------
# The hand-derived answer
# ---------------------------------------------------------------------------


def test_hand_computed_q_satisfies_bellman_optimality():
    """THE anchor check: q_* must satisfy S&B eq. 3.20 in every entry.

    This tests the frozen constants against the definition of optimality, with
    no solver involved at all. If HAND_COMPUTED_Q were merely a plausible table
    of numbers, this is what would catch it.
    """
    residual = bellman_optimality_residual(HAND_COMPUTED_Q, P_TINY, R_TINY, GAMMA)
    assert np.abs(residual).max() < TOL, f"residual {residual}"


def test_hand_computed_q_matches_the_pen_and_paper_arithmetic():
    """Each entry re-derived line by line, the way it appears in FEATURE_002.

    V(QUIET) = 1 + 0.9 V(QUIET)  =>  V(QUIET) = 10
    V(BUSY)  = 4 + 0.9 V(QUIET)  =>  V(BUSY)  = 13
    """
    assert HAND_COMPUTED_Q[QUIET, WAIT] == pytest.approx(1.0 + 0.9 * 10.0, abs=TOL)
    assert HAND_COMPUTED_Q[QUIET, WORK] == pytest.approx(-5.0 + 0.9 * 13.0, abs=TOL)
    assert HAND_COMPUTED_Q[BUSY, WAIT] == pytest.approx(-1.0 + 0.9 * 13.0, abs=TOL)
    assert HAND_COMPUTED_Q[BUSY, WORK] == pytest.approx(4.0 + 0.9 * 10.0, abs=TOL)


def test_v_is_the_row_max_of_q():
    """v_*(s) = max_a q_*(s,a) — S&B eq. 3.19. Two frozen constants, one relation.

    Not a triviality: HAND_COMPUTED_V and HAND_COMPUTED_Q are written out
    independently in the module, so this is the check that they were not
    transcribed inconsistently.
    """
    assert np.allclose(HAND_COMPUTED_Q.max(axis=1), HAND_COMPUTED_V, atol=TOL)


def test_greedy_policy_matches_the_hand_derivation():
    """WAIT when QUIET, WORK when BUSY."""
    assert np.array_equal(greedy_from_q(HAND_COMPUTED_Q), HAND_COMPUTED_POLICY)


def test_optimal_action_differs_between_the_two_states():
    """The property that makes this fixture able to fail a constant-action agent.

    If both states preferred the same action, an agent hard-wired to return 0
    would score a perfect policy match and the Phase 2 tests would be worthless.
    """
    assert HAND_COMPUTED_POLICY[QUIET] != HAND_COMPUTED_POLICY[BUSY]


def test_action_gap_is_wide_enough_to_test_against():
    """The optimal action must win by a clear margin in both states.

    A fixture whose best and second-best actions differ by 0.01 would produce a
    flaky test — a learner within 1% of q_* would flip the policy at random.
    MIN_ACTION_GAP records the real margin so it cannot be eroded unnoticed.
    """
    for s in range(N_TINY_STATES):
        ordered = np.sort(HAND_COMPUTED_Q[s])[::-1]
        gap = ordered[0] - ordered[1]
        assert gap >= MIN_ACTION_GAP - TOL, f"state {s} gap {gap}"


# ---------------------------------------------------------------------------
# Cross-check against the shipped Phase 1 solver
# ---------------------------------------------------------------------------


def test_padding_preserves_the_answer():
    """Widening 2 actions to 5 by duplication must not change the model's answer.

    Duplicated actions can only tie the maximum, never beat it, so q_* is
    untouched. This test is what licenses the next one to use the 5-action
    solver on a 2-action MDP.
    """
    P_wide, R_wide = pad_actions(P_TINY, R_TINY, N_ACTIONS)
    assert P_wide.shape == (N_TINY_STATES, N_ACTIONS, N_TINY_STATES)
    assert R_wide.shape == (N_TINY_STATES, N_ACTIONS)
    for a in range(N_ACTIONS):
        source = a % N_TINY_ACTIONS
        assert np.array_equal(P_wide[:, a, :], P_TINY[:, source, :])
        assert np.array_equal(R_wide[:, a], R_TINY[:, source])


def test_project_value_iteration_reproduces_hand_answer():
    """THE test: agents/dp.py's value_iteration, on the hand-solved 2-state MDP.

    `value_iteration` was checked against an independent hand-derived answer in
    Phase 1 (the five-state MRP, E-005). Running it here ties the new Phase 2
    anchor to that already-trusted solver, so any later disagreement between a
    learner and HAND_COMPUTED_Q is unambiguously the learner's fault.
    """
    P_wide, R_wide = pad_actions(P_TINY, R_TINY, N_ACTIONS)
    V, policy, deltas = value_iteration(
        P_wide, R_wide, gamma=GAMMA, theta=1e-14, max_sweeps=10_000
    )

    assert np.allclose(V, HAND_COMPUTED_V, atol=TOL), (
        f"agents/dp.py value_iteration gave {V}, hand-worked answer is {HAND_COMPUTED_V}"
    )
    assert np.array_equal(policy, HAND_COMPUTED_POLICY), (
        f"value_iteration policy {policy}, hand-worked {HAND_COMPUTED_POLICY}"
    )
    assert deltas[-1] < 1e-14
    # A self-loop on QUIET means the value has to be accumulated over many
    # sweeps, not back-substituted in one.
    assert len(deltas) > 1


def test_epsilon_soft_q_collapses_to_q_star_as_epsilon_goes_to_zero():
    """The on-policy target. At epsilon = 0 it must BE the hand-derived q_*.

    SARSA and first-visit Monte Carlo are on-policy: they converge to q_pi for
    the epsilon-greedy policy they follow, not to q_*. Grading them against
    HAND_COMPUTED_Q would be grading them against an answer they are not trying
    to reach — they would look broken while being correct.

    This helper computes that on-policy fixed point exactly. Anchoring it: at
    epsilon = 0 the epsilon-greedy policy IS the greedy policy, so the fixed
    point must coincide with q_* to machine precision. That ties the new target
    back to the pen-and-paper answer rather than letting it float free.
    """
    assert np.allclose(epsilon_soft_q(0.0), HAND_COMPUTED_Q, atol=TOL)


def test_epsilon_soft_q_is_worth_less_than_q_star_when_exploring():
    """Forced randomness costs value — and the cost must land in the right place.

    Under epsilon-greedy the agent sometimes takes the action it knows is worse,
    so every entry is worth no more than under q_*. The loss is largest in
    QUIET, where the wrong move (WORK, -5) is expensive; that asymmetry is the
    intuition for why SARSA is called the more 'cautious' algorithm.
    """
    soft = epsilon_soft_q(0.1)
    assert (soft <= HAND_COMPUTED_Q + TOL).all(), f"exploration gained value:\n{soft}"
    assert soft[QUIET, WAIT] < HAND_COMPUTED_Q[QUIET, WAIT] - 1e-6


def test_epsilon_soft_q_preserves_the_optimal_policy_at_this_epsilon():
    """At epsilon = 0.1 the ranking of actions is unchanged, so the greedy
    policy read off the soft values is still [WAIT, WORK].

    This is what makes the fixture usable for on-policy learners: their *values*
    are allowed to differ from q_*, but their *policy* must not. Any test
    asserting SARSA recovers HAND_COMPUTED_POLICY depends on this holding.
    """
    assert np.array_equal(greedy_from_q(epsilon_soft_q(0.1)), HAND_COMPUTED_POLICY)


def test_value_iteration_reconstructs_the_full_q_table():
    """V alone is not enough — Phase 2 is graded on Q, so check Q explicitly.

    Rebuilds q_*(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V(s') from the
    solver's V and demands the hand-derived table, entry for entry.
    """
    P_wide, R_wide = pad_actions(P_TINY, R_TINY, N_ACTIONS)
    V, _, _ = value_iteration(P_wide, R_wide, gamma=GAMMA, theta=1e-14, max_sweeps=10_000)

    Q = np.zeros((N_TINY_STATES, N_TINY_ACTIONS), dtype=np.float64)
    for s in range(N_TINY_STATES):
        for a in range(N_TINY_ACTIONS):
            Q[s, a] = R_TINY[s, a] + GAMMA * (P_TINY[s, a] @ V)

    assert np.allclose(Q, HAND_COMPUTED_Q, atol=TOL), f"got {Q}, hand-worked {HAND_COMPUTED_Q}"
