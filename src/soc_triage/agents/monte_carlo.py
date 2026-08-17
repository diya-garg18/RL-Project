"""First-visit Monte Carlo control (Sutton & Barto 2nd ed. §5.4).

The update rule:

    G_t = r_{t+1} + gamma*r_{t+2} + gamma^2*r_{t+3} + ...     (the actual return)
    Q(s,a) <- Q(s,a) + alpha [ G_t - Q(s,a) ]

**No bootstrapping.** This is the one algorithm in Phase 2 that never uses its
own estimate of a successor's value. It waits for the episode to finish and uses
the return that actually happened. That makes it unbiased — and high variance,
because a single unlucky shift moves the estimate for every state-action pair
visited during it. TD methods trade a little bias for a lot less variance, which
is the central comparison of S&B chapters 5 and 6 and the reason the report runs
all three.

**Structural consequence: MC cannot learn until an episode ends.** `update()`
can only buffer; the real work happens in `end_episode()`. That is not an
implementation choice, it is what "use the actual return" requires, and it means
MC cannot learn at all on a task that never terminates. On the continuing tiny
MDP the caller truncates at `tiny_mdp.HORIZON`; with gamma = 0.9 the discarded
tail is worth ~7e-10, so a truncated return is indistinguishable from the
infinite one.

**First-visit, not every-visit** (§5.1): if a state-action pair occurs more than
once in an episode, only the return following its *first* occurrence is used.
`config/training_default.yaml` sets `first_visit: true`. The two variants both
converge but differ on finite data, and `test_monte_carlo_is_first_visit_not_every_visit`
pins which one this is — from identical episode data they produce 2.615 and
2.8075 respectively.

MC is on-policy, so like SARSA it converges to `tiny_mdp.epsilon_soft_q(eps)`
rather than to q_*.

Style: the return is accumulated by walking the episode **backwards**
(G = r + gamma*G), which is both the textbook formulation and O(T) instead of
the O(T^2) of recomputing each return from scratch. Written as an explicit loop
(CONSTRAINTS #14).
"""

import numpy as np

from soc_triage.agents.tabular import TabularAgent


class MonteCarloAgent(TabularAgent):
    """First-visit MC control with an epsilon-greedy behaviour policy."""

    name = "monte_carlo"
    obs_kind = "disc"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # (state, action, reward) for the episode in progress. Cleared at every
        # episode boundary — a leaked buffer folds the previous episode's
        # rewards into this one's returns, a bug whose only symptom is "MC is
        # oddly noisy" (test_monte_carlo_clears_its_buffer_between_episodes).
        self._episode: list[tuple[int, int, float]] = []

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        done: bool,
    ) -> None:
        """Buffer the step. MC cannot learn anything yet.

        `next_obs` and `done` are unused: the return is computed from rewards
        alone, and MC never bootstraps off a successor's value. They stay in the
        signature because the runner calls every agent identically — the runner
        must not need to know which algorithm it is driving (CONSTRAINTS #10).
        """
        self._episode.append((obs, action, float(reward)))

    def end_episode(self) -> None:
        """Compute returns, apply first-visit updates, then decay epsilon.

        Two passes, in this order:
          1. find the FIRST occurrence index of each (s,a) in the episode
          2. walk backwards accumulating G = r + gamma*G, updating only at those
             first occurrences

        The backwards walk is what makes this O(T): each return is the previous
        one discounted plus the current reward, rather than a fresh sum.
        """
        first_occurrence: dict[tuple[int, int], int] = {}
        for t, (state, action, _) in enumerate(self._episode):
            if (state, action) not in first_occurrence:
                first_occurrence[(state, action)] = t

        G = 0.0
        for t in range(len(self._episode) - 1, -1, -1):
            state, action, reward = self._episode[t]
            G = reward + self.gamma * G
            if first_occurrence[(state, action)] == t:
                self.Q[state, action] += self.alpha * (G - self.Q[state, action])
                self.visits[state, action] += 1

        self._episode.clear()
        super().end_episode()  # the epsilon decay every learner shares (D-015)

    def q_table(self) -> np.ndarray:
        """Alias kept for symmetry with the other agents' `.Q` access in scripts."""
        return self.Q
