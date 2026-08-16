"""A five-state Markov Reward Process, worked by hand and checked by code.

ROADMAP Phase 1, final box: *"hand-work a 5-state Markov Reward Process on
paper, show the Bellman equations explicitly, verify against code."*

Why this file exists
--------------------
The 576-state SOC MDP is far too large to check with a pen, so nothing else in
Phase 1 demonstrates that our Bellman backup is the *textbook* Bellman backup
rather than merely something that converges to a fixed point. This module is
the smallest object that can demonstrate it: an MRP small enough to solve by
hand, shaped so that the project's own `agents.dp.value_iteration()` can be
pointed straight at it.

Four independent routes to the same value function, all asserted equal in
`tests/test_mrp_bellman.py`:

  1. by hand      — the arithmetic in docs/features/phase1-mrp-worked-example.md,
                    frozen here as HAND_COMPUTED_V
  2. closed form  — V = (I - gamma*P)^-1 R                      `solve_linear`
  3. iterative    — repeated Bellman backups until convergence  `evaluate_iteratively`
  4. project DP   — `value_iteration()` from agents/dp.py, run on this MRP
                    expanded to a degenerate MDP whose five actions are identical

Route 4 is the one that carries the weight. Routes 2 and 3 only check this
file against itself; route 4 checks the *shipped Phase 1 solver* against a
number a human derived on paper. If `agents/dp.py` ever stops implementing the
textbook equation, that test is what fails.

Textbook anchors (Sutton & Barto, 2nd ed.):
  - return G_t and discounting            §3.3
  - state-value function and eq. 3.14     §3.5  (the Bellman equation for v_pi)
  - expected one-step reward, eq. 3.5     §3.1
  - iterative policy evaluation           §4.1
An MRP is an MDP with exactly one action available everywhere, so §4.1 applies
to it unchanged — which is precisely why the degenerate-MDP trick is valid.

Numbers note (CONSTRAINTS #9)
-----------------------------
The constants below are deliberately NOT in config/*.yaml. They are not
tunables; they are the *definition* of a worked example, and the pen-and-paper
derivation in the docs is only correct for these exact values. Moving them into
config would invite someone to change them and silently invalidate the
derivation the test is checking against. Recorded as D-013.
"""

import numpy as np

# ---------------------------------------------------------------------------
# The MRP: one alert's journey through a shift, coarse enough to solve by hand.
# ---------------------------------------------------------------------------

STATE_NAMES: tuple[str, ...] = (
    "QUIET",          # 0 - nothing pressing in the queue
    "BACKLOG",        # 1 - alerts waiting, none being worked
    "INVESTIGATING",  # 2 - an analyst is actively on one
    "CONFIRMED",      # 3 - a true incident was caught      (absorbing)
    "MISSED",         # 4 - a deadline expired un-triaged   (absorbing)
)
N_MRP_STATES = len(STATE_NAMES)

# Discount factor for the worked example only. NOT the project's gamma (0.99,
# in config/training_default.yaml). 0.9 is chosen so the hand arithmetic stays
# exact in small fractions: the only non-terminating value, V(QUIET) = 52/11,
# is still an exact rational a student can write down and verify.
GAMMA = 0.9

# P[s, s'] = probability of moving from s to s'. Rows sum to 1.
# QUIET dawdles (a self-loop, so the example needs a genuine solve rather than
# back-substitution); BACKLOG and INVESTIGATING both resolve in one step, but
# INVESTIGATING resolves well far more often — that contrast is the whole point.
P_MRP = np.array(
    [
        # QUIET  BACKLOG  INVESTIGATING  CONFIRMED  MISSED
        [0.50,   0.25,    0.25,          0.00,      0.00],  # from QUIET
        [0.00,   0.00,    0.00,          0.40,      0.60],  # from BACKLOG
        [0.00,   0.00,    0.00,          0.80,      0.20],  # from INVESTIGATING
        [0.00,   0.00,    0.00,          1.00,      0.00],  # CONFIRMED (absorbing)
        [0.00,   0.00,    0.00,          0.00,      1.00],  # MISSED (absorbing)
    ],
    dtype=np.float64,
)

# r[s, s'] = reward collected on the transition s -> s'.
# Signs mirror the real reward (brief §3.5) in miniature: idling costs a little,
# catching a true incident pays, catching it late pays less, missing it costs.
TRANSITION_REWARD = np.array(
    [
        # QUIET  BACKLOG  INVESTIGATING  CONFIRMED  MISSED
        [-1.0,   -1.0,    -1.0,           0.0,       0.0],  # a minute of clock burns
        [ 0.0,    0.0,     0.0,          20.0,     -20.0],  # caught late / missed
        [ 0.0,    0.0,     0.0,          30.0,     -20.0],  # caught fast / missed
        [ 0.0,    0.0,     0.0,           0.0,       0.0],  # absorbing, nothing more
        [ 0.0,    0.0,     0.0,           0.0,       0.0],  # absorbing, nothing more
    ],
    dtype=np.float64,
)

# The answer derived on paper in docs/features/phase1-mrp-worked-example.md.
# V(QUIET) is exactly 52/11; the others are exact integers. Frozen here so the
# test compares code against the *human's* number, not against a rerun of code.
HAND_COMPUTED_V = np.array([52.0 / 11.0, -4.0, 20.0, 0.0, 0.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# The three code routes. Explicit loops throughout (CONSTRAINTS #14): this file
# is teaching material before it is anything else.
# ---------------------------------------------------------------------------


def expected_rewards(P: np.ndarray, transition_reward: np.ndarray) -> np.ndarray:
    """Collapse per-transition rewards into R(s), the expected one-step reward.

    Sutton & Barto 2nd ed. eq. 3.5:
        R(s) = sum_s' P(s'|s) * r(s, s')

    This is the same quantity `agents.dp.estimate_model` produces as R_hat by
    averaging observed rewards — there it is estimated from samples, here it is
    computed exactly, which is why the two can be cross-checked at all.
    """
    n_states = P.shape[0]
    R = np.zeros(n_states, dtype=np.float64)
    for s in range(n_states):
        total = 0.0
        for s_next in range(n_states):
            total += P[s, s_next] * transition_reward[s, s_next]
        R[s] = total
    return R


def solve_linear(P: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """Solve the Bellman equation in closed form.

    The Bellman equation for an MRP, in vector form, is
        V = R + gamma * P @ V
    Rearranged:
        (I - gamma*P) V = R      =>      V = (I - gamma*P)^-1 R

    Only possible because the state space is tiny. The 576-state MDP cannot be
    solved this way with a max over actions in the loop — that non-linearity is
    exactly why value iteration exists (S&B §4.4).
    """
    n_states = P.shape[0]
    A = np.eye(n_states, dtype=np.float64) - gamma * P
    return np.linalg.solve(A, R)


def evaluate_iteratively(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    theta: float,
    max_sweeps: int,
) -> tuple[np.ndarray, int]:
    """Iterative policy evaluation (Sutton & Barto 2nd ed. §4.1) for an MRP.

    Applies the Bellman backup to every state, sweep after sweep:
        V(s) <- R(s) + gamma * sum_s' P(s'|s) * V(s')
    until the largest single-state change in a sweep falls below theta.

    Returns (V, sweeps_used). Identical in shape to the evaluation half of
    `agents.dp.policy_iteration` — deliberately, so a reader can put the two
    side by side and see that the big one is the same equation.
    """
    n_states = P.shape[0]
    V = np.zeros(n_states, dtype=np.float64)

    for sweep in range(1, max_sweeps + 1):
        delta = 0.0
        for s in range(n_states):
            expectation = 0.0
            for s_next in range(n_states):
                expectation += P[s, s_next] * V[s_next]
            v_new = R[s] + gamma * expectation
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            return V, sweep

    raise RuntimeError(
        f"MRP policy evaluation did not converge in {max_sweeps} sweeps (last delta {delta:.2e})"
    )


def as_degenerate_mdp(
    P: np.ndarray,
    R: np.ndarray,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-shape the MRP as an MDP in which every action does the same thing.

    Returns (P_mdp, R_mdp) with shapes (S, A, S) and (S, A), the exact shapes
    `agents.dp.value_iteration` expects from `estimate_model`.

    With all actions identical the `max_a` in the value-iteration backup is a
    max over equal numbers, so it collapses to the MRP Bellman equation and the
    optimal value function *must* equal the MRP's value function. That makes
    the big solver checkable against a hand-derived answer — the only such
    check that exists in Phase 1.
    """
    n_states = P.shape[0]
    P_mdp = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    R_mdp = np.zeros((n_states, n_actions), dtype=np.float64)
    for s in range(n_states):
        for a in range(n_actions):
            P_mdp[s, a] = P[s]
            R_mdp[s, a] = R[s]
    return P_mdp, R_mdp
