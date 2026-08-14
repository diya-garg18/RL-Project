"""Model-based Dynamic Programming (Phase 1, D-004).

The true transition dynamics of the queue are not analytically available, so:
  1. estimate_model()    — count transitions over random-policy rollouts to
                           build P-hat(s'|s,a) and R-hat(s,a)
  2. value_iteration()   — Sutton & Barto 2nd ed. §4.4, on the estimate
  3. policy_iteration()  — Sutton & Barto 2nd ed. §4.3 (evaluation per §4.1)

The resulting policy is optimal FOR THE ESTIMATED MODEL, not for the true
environment — say exactly that, every time (D-004).

Unvisited (s,a) pairs (D-011): treated as absorbing self-loops with reward 0.
A state-action we never saw under 50k random episodes is essentially
unreachable; pretending we know its dynamics would inject fiction into the
Bellman backups, and a neutral self-loop keeps its value at 0 so the greedy
policy never prefers the unknown.

Style note (CONSTRAINTS #14): the sweeps below use explicit loops over states
and actions — the vector `P[s, a] @ V` inside is the expectation term of the
Bellman equation written directly, not an optimisation trick.
"""

import numpy as np

from soc_triage.agents.base import Agent
from soc_triage.config import EnvConfig
from soc_triage.env import SOCTriageEnv
from soc_triage.state import N_STATES, discretise

N_ACTIONS = 5


def estimate_model(
    cfg: EnvConfig,
    n_episodes: int,
    seed_start: int,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count transitions under a uniform-random policy.

    Returns (P_hat, R_hat, visits):
      P_hat  (576, 5, 576)  estimated transition probabilities
      R_hat  (576, 5)       estimated expected one-step reward
      visits (576, 5)       raw visit counts (coverage evidence for the report)

    Episode seeds run seed_start .. seed_start + n_episodes - 1 (a block kept
    disjoint from train/eval/calibration — enforced in load_training_config).
    The action RNG is seeded once from seed_start so the whole estimation is
    reproducible.
    """
    env = SOCTriageEnv(cfg)
    counts = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.float64)
    reward_sum = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    action_rng = np.random.default_rng(seed_start)

    for episode in range(n_episodes):
        snap = env.reset(seed_start + episode)
        s = discretise(snap, cfg)
        done = False
        while not done:
            a = int(action_rng.integers(0, N_ACTIONS))
            snap, reward, done, _ = env.step(a)
            s_next = discretise(snap, cfg)
            counts[s, a, s_next] += 1.0
            reward_sum[s, a] += reward
            s = s_next
        if progress_every and (episode + 1) % progress_every == 0:
            print(f"  estimation: {episode + 1}/{n_episodes} episodes")

    visits = counts.sum(axis=2)

    P_hat = np.zeros_like(counts)
    R_hat = np.zeros_like(reward_sum)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if visits[s, a] > 0:
                P_hat[s, a] = counts[s, a] / visits[s, a]
                R_hat[s, a] = reward_sum[s, a] / visits[s, a]
            else:
                P_hat[s, a, s] = 1.0  # unvisited: absorbing self-loop, reward 0 (D-011)

    return P_hat, R_hat, visits


def value_iteration(
    P_hat: np.ndarray,
    R_hat: np.ndarray,
    gamma: float,
    theta: float,
    max_sweeps: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Value iteration (Sutton & Barto 2nd ed., §4.4).

    Update rule, applied to every state each sweep:
        V(s) <- max_a [ R_hat(s,a) + gamma * sum_s' P_hat(s'|s,a) * V(s') ]
    Stops when the largest single-state change in a sweep drops below theta.

    Returns (V, greedy_policy, per-sweep deltas — the convergence curve).
    """
    n_states = R_hat.shape[0]
    V = np.zeros(n_states, dtype=np.float64)
    deltas: list[float] = []

    for _ in range(max_sweeps):
        delta = 0.0
        for s in range(n_states):
            best = -np.inf
            for a in range(N_ACTIONS):
                q_sa = R_hat[s, a] + gamma * (P_hat[s, a] @ V)  # Bellman expectation
                if q_sa > best:
                    best = q_sa
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        deltas.append(delta)
        if delta < theta:
            break
    else:
        raise RuntimeError(f"value iteration did not converge in {max_sweeps} sweeps (last delta {delta:.2e})")

    policy = greedy_policy(P_hat, R_hat, gamma, V)
    return V, policy, deltas


def greedy_policy(P_hat: np.ndarray, R_hat: np.ndarray, gamma: float, V: np.ndarray) -> np.ndarray:
    """The policy that is greedy with respect to V: argmax_a of the Bellman RHS."""
    n_states = R_hat.shape[0]
    policy = np.zeros(n_states, dtype=np.int64)
    for s in range(n_states):
        best_a = 0
        best_q = -np.inf
        for a in range(N_ACTIONS):
            q_sa = R_hat[s, a] + gamma * (P_hat[s, a] @ V)
            if q_sa > best_q:
                best_q = q_sa
                best_a = a
        policy[s] = best_a
    return policy


def policy_iteration(
    P_hat: np.ndarray,
    R_hat: np.ndarray,
    gamma: float,
    eval_theta: float,
    max_sweeps: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Policy iteration (Sutton & Barto 2nd ed., §4.3).

    Alternates iterative policy evaluation (§4.1) with greedy improvement until
    the policy stops changing. Returns (V, policy, n_improvement_rounds).

    Cross-check (FLOW.md Flow C): must agree with value_iteration on ≥95% of
    states — disagreement beyond ties means one of them is wrong.
    """
    n_states = R_hat.shape[0]
    policy = np.zeros(n_states, dtype=np.int64)  # start: always action 0
    V = np.zeros(n_states, dtype=np.float64)

    for rounds in range(1, max_sweeps + 1):
        # --- policy evaluation (§4.1): V <- V^pi by iterative sweeps
        for _ in range(max_sweeps):
            delta = 0.0
            for s in range(n_states):
                a = policy[s]
                v_new = R_hat[s, a] + gamma * (P_hat[s, a] @ V)
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            if delta < eval_theta:
                break
        else:
            raise RuntimeError(f"policy evaluation did not converge in {max_sweeps} sweeps")

        # --- policy improvement: greedy w.r.t. the evaluated V
        new_policy = greedy_policy(P_hat, R_hat, gamma, V)
        if np.array_equal(new_policy, policy):
            return V, policy, rounds
        policy = new_policy

    raise RuntimeError(f"policy iteration did not stabilise in {max_sweeps} improvement rounds")


class DPAgent(Agent):
    """Acts by table lookup on the DP-computed policy. Optimal for the
    estimated model, NOT for the true environment (D-004)."""

    name = "dp"
    obs_kind = "disc"

    def __init__(self, policy: np.ndarray):
        self._policy = policy

    def act(self, obs: int) -> int:
        return int(self._policy[obs])
