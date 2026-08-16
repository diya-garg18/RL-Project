"""Tabular Q-learning — off-policy TD control (Sutton & Barto 2nd ed. §6.5).

The update rule, which is the whole algorithm:

    Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Everything else in this file is bookkeeping around that one line.

**Off-policy** is the word that matters, and it is the single difference from
SARSA (§6.4, `agents/sarsa.py`). The target uses `max_a' Q(s',a')` — the value
of the *best* next action — regardless of which action the epsilon-greedy
behaviour policy will actually take. So the agent explores with one policy and
learns the value of another: the greedy one. SARSA instead bootstraps off the
action it really takes, which makes it learn the value of its own exploratory
behaviour. `test_update_bootstraps_off_the_max_not_the_behaviour_action` pins
this distinction at the update itself, because the two algorithms agree at the
optimum and a convergence test alone cannot tell them apart.

Style (CONSTRAINTS #13, #14): the max and the argmax below are written as
explicit loops rather than `Q[s].max()` / `np.argmax`. Slower, and deliberate —
this is one of the functions both students must be able to write from memory in
an interview, and `np.argmax` hides the tie-breaking rule that reproducibility
depends on.

Verified against a hand-solved 2-state MDP whose q_* was derived on paper
(`tiny_mdp.py`, FEATURE_002, E-006) before ever being pointed at the real
environment.
"""

import numpy as np

from soc_triage.agents.base import Agent


class QLearningAgent(Agent):
    """Off-policy TD control with an epsilon-greedy behaviour policy.

    All hyperparameters are injected — none are defaulted here, because every
    tunable number in this project lives in `config/training_default.yaml`
    (CONSTRAINTS #9) and a default in the constructor is a magic number wearing
    a disguise.
    """

    name = "q_learning"
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

        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # One RNG, seeded once, owned by the agent. The environment has its own
        # seeded stream; keeping them separate means changing the exploration
        # schedule cannot shift the alert sequence, so two runs stay comparable.
        self._rng = np.random.default_rng(seed)

    # -- acting ------------------------------------------------------------

    def act(self, obs: int) -> int:
        """Epsilon-greedy action selection (S&B §2.2).

        With probability epsilon, act uniformly at random; otherwise take the
        greedy action. Exploration is not optional here: on the tiny MDP the
        greedy policy never leaves QUIET, so with epsilon = 0 half the state
        space is never observed at all.
        """
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return self._argmax(obs)

    def _argmax(self, state: int) -> int:
        """Greedy action, ties broken toward the lower action index.

        The tie-break is not a detail. A zero-initialised table ties on every
        action at step one of every run, so an arbitrary rule here would make
        runs unreproducible across seeds. Strict `>` keeps the first index —
        the same convention as `agents.dp.greedy_policy`, so the two modules'
        policies are comparable rather than merely similar.
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
        """One Q-learning backup (S&B §6.5).

            Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

        `done` must mean *terminated*, never merely truncated. On termination
        there is no successor, so the target is the reward alone; bootstrapping
        past a terminal state invents value that does not exist. The mirror-image
        bug is just as costly — passing done=True at an episode-length cutoff on
        a continuing task teaches the agent the world ends, dragging every value
        toward the last reward it happened to see.
        """
        if done:
            target = reward
        else:
            best_next_value = -np.inf
            for next_action in range(self.n_actions):
                if self.Q[next_obs, next_action] > best_next_value:
                    best_next_value = self.Q[next_obs, next_action]
            target = reward + self.gamma * best_next_value

        td_error = target - self.Q[obs, action]
        self.Q[obs, action] += self.alpha * td_error

    def end_episode(self) -> None:
        """Decay epsilon one step, floored.

        Called once per EPISODE, not per step — `config/training_default.yaml`
        specifies `decay: 0.9995  # per episode`. Applying it per step instead
        would drive epsilon to its floor within a single shift, and the symptom
        (a learning curve that plateaus early) looks nothing like the cause.
        This is why decay lives in its own method rather than inside `update()`:
        the episode boundary has to be stated by the caller, and on a continuing
        task like the tiny MDP there is no `done` flag to infer it from.
        """
        decayed = self.epsilon * self.epsilon_decay
        self.epsilon = decayed if decayed > self.epsilon_min else self.epsilon_min

    # -- reading the result -------------------------------------------------

    def greedy_policy(self) -> np.ndarray:
        """The learned policy: argmax_a Q(s,a) for every state.

        This is what gets printed as the readable policy table for the report
        and the viva — the deliverable of Phase 2, more than the Q-table itself.
        """
        policy = np.zeros(self.n_states, dtype=np.int64)
        for state in range(self.n_states):
            policy[state] = self._argmax(state)
        return policy
