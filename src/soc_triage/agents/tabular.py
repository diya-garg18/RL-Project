"""Shared machinery for the three tabular learners (Phase 2).

Q-learning, SARSA and first-visit Monte Carlo differ in **one method**:
`update`. Everything else — the Q-table, the epsilon-greedy behaviour policy,
the per-episode epsilon decay, the tie-breaking argmax, the visit counts — is
identical across all three.

Extracted once the third implementation appeared, not before (rule of three).
The point is not to save typing: it is that each agent file now contains
essentially nothing but its own update rule, which is the part the students
must be able to write from memory and the part an examiner will ask about.
Reading `sarsa.py` next to `q_learning.py` shows the difference in a few lines
rather than burying it in forty lines of shared boilerplate.

What deliberately stays here rather than in `agents/base.py`: `base.Agent` is
the interface every agent implements, baselines included, and baselines have no
Q-table, no exploration schedule and nothing to decay. Widening that interface
to suit three subclasses would push dead members onto five agents that do not
want them (CONSTRAINTS #18).

Style (CONSTRAINTS #14): the argmax loops are explicit rather than `np.argmax`,
because the tie-breaking rule is load-bearing for reproducibility and `np.argmax`
hides it.
"""

import numpy as np

from soc_triage.agents.base import Agent


class TabularAgent(Agent):
    """Q-table + epsilon-greedy behaviour. Subclasses implement `update`.

    All hyperparameters are injected; none are defaulted, because every tunable
    number in this project lives in `config/*.yaml` (CONSTRAINTS #9) and a
    constructor default is a magic number in disguise.
    """

    name = "tabular"
    obs_kind = "disc"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        seed: int,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        # Zero init, not optimistic init. Optimistic values are a legitimate
        # exploration technique but a different algorithm, and would change
        # every number this project reports — it would need a DECISIONS entry.
        self.Q: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)

        # Visit counts play no part in learning. They exist so the policy table
        # can print "never seen" instead of silently reporting the argmax
        # tie-break — see FEATURE_005, where 455 of 576 unvisited states would
        # otherwise have rendered as a confident preference (E-009).
        self.visits: np.ndarray = np.zeros((n_states, n_actions), dtype=np.int64)

        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # One RNG, seeded once, owned by the agent. The environment has its own
        # seeded stream; keeping them separate means changing the exploration
        # schedule cannot shift the alert sequence, so two runs stay comparable.
        self._rng = np.random.default_rng(seed)

    # -- acting ------------------------------------------------------------

    def act(self, obs: int) -> int:
        """Epsilon-greedy action selection (S&B §2.2)."""
        return self._epsilon_greedy(obs)

    def _epsilon_greedy(self, state: int) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return self._argmax(state)

    def _argmax(self, state: int) -> int:
        """Greedy action, ties broken toward the lower action index.

        The tie-break is not a detail. A zero-initialised table ties on every
        action at step one of every run, so an arbitrary rule would make runs
        unreproducible across seeds. Strict `>` keeps the first index — the same
        convention as `agents.dp.greedy_policy`, so the learned policies and the
        DP policy are comparable rather than merely similar.
        """
        best_action = 0
        best_value = -np.inf
        for action in range(self.n_actions):
            if self.Q[state, action] > best_value:
                best_value = self.Q[state, action]
                best_action = action
        return best_action

    # -- learning ----------------------------------------------------------

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        done: bool,
    ) -> None:
        raise NotImplementedError("each tabular agent implements its own update rule")

    def end_episode(self) -> None:
        """Decay epsilon one step, floored (D-015).

        Called once per EPISODE, not per step — `config/training_default.yaml`
        specifies `decay: 0.9995  # per episode`. Applying it per step would
        drive epsilon to its floor within a single shift, and the symptom (a
        learning curve that plateaus early) looks nothing like the cause.

        It cannot be inferred from `done` either: the tiny MDP is continuing and
        never sets it. Subclasses that need an episode boundary for their own
        reasons — Monte Carlo does — override this and call `super()`.
        """
        decayed = self.epsilon * self.epsilon_decay
        self.epsilon = decayed if decayed > self.epsilon_min else self.epsilon_min

    # -- reading the result -------------------------------------------------

    def greedy_policy(self) -> np.ndarray:
        """The learned policy: argmax_a Q(s,a) for every state.

        What gets printed as the readable policy table (FEATURE_005) — arguably
        the real deliverable of Phase 2, more than the Q-table itself.
        """
        policy = np.zeros(self.n_states, dtype=np.int64)
        for state in range(self.n_states):
            policy[state] = self._argmax(state)
        return policy
