"""A two-state MDP, solved by hand, that every tabular learner must reproduce.

ROADMAP Phase 2, final box: *"Q-learning converges on a tiny hand-checkable
2-state MDP with a known answer."*

Why this file exists
--------------------
Phase 1 had exactly this problem and solved it the same way. The 576-state SOC
MDP cannot be checked with a pen, so a Q-table computed on it can only ever be
compared against *another program*. That catches disagreements, not shared
mistakes: SARSA and Q-learning both converging to the same wrong number would
look exactly like success.

This module is the smallest object that breaks the tie — an MDP whose optimal
action-value function is derived on paper in
docs/features/FEATURE_002_tiny_mdp_qstar.md and frozen below as
HAND_COMPUTED_Q. Every Phase 2 learner is pointed at it before it is pointed at
the real environment.

It is the action-value counterpart of `mrp_example.py`:

    mrp_example.py   ->  checks the Bellman backup for V   (Phase 1, dp.py)
    tiny_mdp.py      ->  checks the Bellman backup for Q   (Phase 2, the learners)

The design, and why each choice was made
----------------------------------------
Four properties were required, and each one rules out a test that would pass
while broken:

  1. **Solvable in exact arithmetic.** V(QUIET) = 10 and V(BUSY) = 13 exactly,
     so the expected answer is a number a human wrote down, not a number a
     program produced.
  2. **Continuing, not episodic.** There is no terminal state, so a learner
     cannot get the right answer by averaging complete returns without ever
     bootstrapping. TD methods must actually use V(s') to pass.
  3. **The optimal action differs between the two states** (WAIT in QUIET,
     WORK in BUSY). An agent stuck returning a constant action fails. If both
     states preferred the same action, `always return 0` would score 100%.
  4. **Deterministic transitions.** Nothing here is testing whether a learner
     can average out sampling noise — that is what the real environment is
     for. Determinism keeps the test exact and the runtime near zero.

A fifth property fell out of the design and is worth knowing about: under the
optimal policy the agent never leaves QUIET, so BUSY is only ever reached by
*exploring*. A learner with epsilon pinned to 0 will never see half the MDP and
will fail on Q(BUSY, .). That makes this fixture a live demonstration of why
exploration is not optional, which is a viva question waiting to happen.

Textbook anchors (Sutton & Barto, 2nd ed.):
  - action-value function q_pi and q_*          §3.5, §3.6
  - the Bellman optimality equation for q_*     eq. 3.20
  - why q_* makes the optimal policy trivial    §3.6 (argmax, no model needed)

Numbers note (CONSTRAINTS #9)
-----------------------------
The constants below are deliberately NOT in config/*.yaml, for the same reason
`mrp_example.py`'s are not (D-013, extended to this file as D-014). They are
not tunables; they are the *definition* of a worked example. The pen-and-paper
derivation is correct only for these exact values, so exposing them in YAML
would invite an edit that leaves the derivation and HAND_COMPUTED_Q silently
disagreeing — converting an external correctness anchor into a test that
checks nothing.
"""

import numpy as np

# ---------------------------------------------------------------------------
# The MDP: a SOC shift boiled down to two situations and two things to do.
# ---------------------------------------------------------------------------

STATE_NAMES: tuple[str, ...] = (
    "QUIET",  # 0 - queue under control, nothing urgent waiting
    "BUSY",   # 1 - a backlog has built up
)
ACTION_NAMES: tuple[str, ...] = (
    "WAIT",  # 0 - do not pull an alert this step
    "WORK",  # 1 - pull an alert and investigate it
)
N_TINY_STATES = len(STATE_NAMES)
N_TINY_ACTIONS = len(ACTION_NAMES)

QUIET, BUSY = 0, 1
WAIT, WORK = 0, 1

# Discount factor for this worked example only. NOT the project's 0.99 from
# config/training_default.yaml. 0.9 is chosen so V(QUIET) = 1/(1 - 0.9) = 10
# exactly — an integer a student can verify in one line, rather than a decimal
# that has to be copied from output. Same reasoning as mrp_example.GAMMA.
GAMMA = 0.9

# The horizon used when this continuing task is run in episodes.
#
# For a TD learner, 200 steps is ample: gamma^200 ~ 7e-10, so the discarded
# tail is worth nothing.
#
# The distinction that matters here is a classic bug: truncation is NOT
# termination. A TD learner must still bootstrap through the cut (done=False at
# the horizon), or it will learn that the world ends and drive every value
# toward the last reward it happened to see.
HORIZON = 200

# Monte Carlo needs a MUCH longer horizon, and the reason is worth understanding
# because the obvious argument for 200 is wrong.
#
# "gamma^200 ~ 7e-10, so truncation is harmless" is true for the return measured
# from t=0. But MC computes a return from EVERY timestep in the episode. The
# state visited at t=199 gets a return of one reward with no future at all,
# against a true value near 10. Roughly the last ~50 steps of any 200-step
# episode carry materially truncated returns, so ~25% of MC's samples are biased
# downward — and unlike variance, that bias does not average out with more
# episodes.
#
# Measured on this fixture (mean signed error against the epsilon-soft target,
# 5 seeds, alpha=0.01, 3000 episodes):
#     HORIZON =  50  ->  max |error| 2.75, every entry negative
#     HORIZON = 200  ->  max |error| 0.47
#     HORIZON = 800  ->  max |error| 0.09   (comparable to SARSA's noise floor)
#
# The bias shrinks with horizon exactly as the explanation predicts. 800 is
# where it drops below the constant-alpha noise the TD learners already carry,
# so it is the point past which lengthening the episode buys nothing.
MC_HORIZON = 800

# P[s, a, s'] = probability of landing in s' after taking a in s. Deterministic,
# so every row is a single 1.0 — written as a full matrix anyway so it plugs
# straight into the same solvers that take the estimated 576-state P_hat.
P_TINY = np.zeros((N_TINY_STATES, N_TINY_ACTIONS, N_TINY_STATES), dtype=np.float64)
P_TINY[QUIET, WAIT, QUIET] = 1.0  # stay calm, stay quiet
P_TINY[QUIET, WORK, BUSY] = 1.0   # dig into a quiet queue and a backlog forms behind you
P_TINY[BUSY, WAIT, BUSY] = 1.0    # ignoring a backlog does not clear it
P_TINY[BUSY, WORK, QUIET] = 1.0   # working the backlog clears it

# R[s, a] = the expected one-step reward. Signs mirror the real reward
# (brief §3.5) in miniature: idle time in a calm queue is cheap, needless
# investigation is expensive, an ignored backlog bleeds, clearing one pays.
R_TINY = np.array(
    [
        # WAIT   WORK
        [+1.0,  -5.0],  # from QUIET
        [-1.0,  +4.0],  # from BUSY
    ],
    dtype=np.float64,
)

# The answer derived on paper in docs/features/FEATURE_002_tiny_mdp_qstar.md.
#
#   V(QUIET) = 1 + 0.9 V(QUIET)          =>  0.1 V(QUIET) = 1   =>  V(QUIET) = 10
#   V(BUSY)  = 4 + 0.9 V(QUIET)          =>  V(BUSY) = 4 + 9    =>  V(BUSY)  = 13
#
#   Q(QUIET, WAIT) = +1 + 0.9(10) = 10.0      <- optimal in QUIET
#   Q(QUIET, WORK) = -5 + 0.9(13) =  6.7
#   Q(BUSY,  WAIT) = -1 + 0.9(13) = 10.7
#   Q(BUSY,  WORK) = +4 + 0.9(10) = 13.0      <- optimal in BUSY
#
# Frozen here so the tests compare code against the *human's* number, not
# against a rerun of the same code.
HAND_COMPUTED_Q = np.array(
    [
        # WAIT   WORK
        [10.0,   6.7],  # QUIET
        [10.7,  13.0],  # BUSY
    ],
    dtype=np.float64,
)

HAND_COMPUTED_V = np.array([10.0, 13.0], dtype=np.float64)

# argmax over each row of HAND_COMPUTED_Q. Written out rather than computed, so
# that a bug in the argmax helper cannot quietly define its own expected answer.
HAND_COMPUTED_POLICY = np.array([WAIT, WORK], dtype=np.int64)

# The margin by which the optimal action wins, per state: 10.0 - 6.7 = 3.3 in
# QUIET and 13.0 - 10.7 = 2.3 in BUSY. Both are large relative to the values
# themselves, so a learner that has roughly converged still gets the policy
# exactly right — a deliberate design choice, because a fixture with a 0.01
# margin would produce a flaky test rather than a strict one.
MIN_ACTION_GAP = 2.3


# ---------------------------------------------------------------------------
# Helpers. Explicit loops throughout (CONSTRAINTS #14) — this file is teaching
# material before it is anything else.
# ---------------------------------------------------------------------------


def step(state: int, action: int) -> tuple[int, float]:
    """Take one step. Deterministic, so no RNG and no seed.

    Returns (next_state, reward). There is no `done`: the task is continuing
    (see HORIZON above). Callers that need episodes must truncate and keep
    bootstrapping through the cut.
    """
    next_state = int(np.argmax(P_TINY[state, action]))
    return next_state, float(R_TINY[state, action])


def bellman_optimality_residual(
    Q: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """How badly Q fails the Bellman optimality equation, entry by entry.

    Sutton & Barto 2nd ed. eq. 3.20:
        q_*(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) * max_a' q_*(s',a')

    Returns Q - RHS. For the true q_* every entry is 0. This is the check that
    HAND_COMPUTED_Q is genuinely the optimal action-value function and not just
    a plausible table of numbers — it tests the frozen constants themselves,
    independently of any solver.
    """
    n_states, n_actions = R.shape
    residual = np.zeros((n_states, n_actions), dtype=np.float64)
    for s in range(n_states):
        for a in range(n_actions):
            expectation = 0.0
            for s_next in range(n_states):
                best_next = -np.inf
                for a_next in range(n_actions):
                    if Q[s_next, a_next] > best_next:
                        best_next = Q[s_next, a_next]
                expectation += P[s, a, s_next] * best_next
            residual[s, a] = Q[s, a] - (R[s, a] + gamma * expectation)
    return residual


def greedy_from_q(Q: np.ndarray) -> np.ndarray:
    """The greedy policy: argmax_a Q(s,a), ties broken toward the lower index.

    Deterministic tie-breaking matters for reproducibility across seeds — the
    same rule `agents.dp.greedy_policy` uses (strict `>`, first index wins).
    """
    n_states, n_actions = Q.shape
    policy = np.zeros(n_states, dtype=np.int64)
    for s in range(n_states):
        best_a = 0
        best_q = -np.inf
        for a in range(n_actions):
            if Q[s, a] > best_q:
                best_q = Q[s, a]
                best_a = a
        policy[s] = best_a
    return policy


def epsilon_soft_q(
    epsilon: float,
    tol: float = 1e-14,
    max_sweeps: int = 100_000,
) -> np.ndarray:
    """The action-value function an ON-POLICY learner actually converges to.

    Q-learning is off-policy and converges to q_* (S&B §6.5). SARSA (§6.4) and
    first-visit Monte Carlo control (§5.4) are on-policy: they converge to
    q_pi for the epsilon-greedy policy they are following, which is *not* q_*
    whenever epsilon > 0. Grading them against HAND_COMPUTED_Q would mark a
    correct implementation as broken.

    This computes that on-policy fixed point by iterating the expected-SARSA
    backup to convergence:

        Q(s,a) <- R(s,a) + gamma * sum_s' P(s'|s,a) * sum_a' pi_e(a'|s') Q(s',a')

    where pi_e is epsilon-greedy with respect to the current Q:

        pi_e(a|s) = 1 - epsilon + epsilon/n   for the greedy action
                  = epsilon/n                 otherwise

    Note this is a *different* computation from what SARSA does — SARSA samples
    the next action, this one takes the exact expectation — so agreement
    between them is evidence, not tautology. And at epsilon = 0 it reduces to
    the Bellman optimality equation, so it must reproduce HAND_COMPUTED_Q
    exactly; `test_epsilon_soft_q_collapses_to_q_star_as_epsilon_goes_to_zero`
    asserts precisely that, which is what keeps this target tied to the
    pen-and-paper answer instead of floating free.

    Explicit loops throughout (CONSTRAINTS #14) — this is the definition of
    on-policy value, and it should read like one.
    """
    Q = np.zeros((N_TINY_STATES, N_TINY_ACTIONS), dtype=np.float64)
    for _ in range(max_sweeps):
        delta = 0.0
        for s in range(N_TINY_STATES):
            for a in range(N_TINY_ACTIONS):
                expectation = 0.0
                for s_next in range(N_TINY_STATES):
                    greedy_next = 0
                    best = -np.inf
                    for a_next in range(N_TINY_ACTIONS):
                        if Q[s_next, a_next] > best:
                            best = Q[s_next, a_next]
                            greedy_next = a_next
                    # expected value of s' under the epsilon-greedy policy
                    value_next = 0.0
                    for a_next in range(N_TINY_ACTIONS):
                        prob = epsilon / N_TINY_ACTIONS
                        if a_next == greedy_next:
                            prob += 1.0 - epsilon
                        value_next += prob * Q[s_next, a_next]
                    expectation += P_TINY[s, a, s_next] * value_next
                updated = R_TINY[s, a] + GAMMA * expectation
                delta = max(delta, abs(updated - Q[s, a]))
                Q[s, a] = updated
        if delta < tol:
            return Q
    raise RuntimeError(f"epsilon-soft fixed point did not converge in {max_sweeps} sweeps")


def pad_actions(
    P: np.ndarray,
    R: np.ndarray,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Widen this 2-action MDP to `n_actions` so `agents.dp` can be run on it.

    `agents.dp.value_iteration` loops over its module-level N_ACTIONS = 5, so a
    2-action MDP has to be reshaped before the shipped solver will accept it.
    The padding *duplicates* the real actions cyclically (0,1,0,1,0) rather
    than inventing new ones.

    That choice is the whole trick, and it is not arbitrary: the `max_a` in the
    Bellman backup then ranges over a multiset containing only copies of the
    genuine actions, so the maximum — and therefore V, and therefore q_* — is
    unchanged. Padding with, say, zero-reward self-loops would have added an
    action worth 0, which sits above nothing here but would silently alter the
    answer in any MDP with negative values.

    Returns (P_wide, R_wide) with shapes (S, n_actions, S) and (S, n_actions).
    """
    n_states, n_real_actions = R.shape
    P_wide = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    R_wide = np.zeros((n_states, n_actions), dtype=np.float64)
    for s in range(n_states):
        for a in range(n_actions):
            source = a % n_real_actions  # 0,1,0,1,0 for n_actions=5
            P_wide[s, a] = P[s, source]
            R_wide[s, a] = R[s, source]
    return P_wide, R_wide
