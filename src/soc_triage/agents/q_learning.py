"""Tabular Q-learning — off-policy TD control (Sutton & Barto 2nd ed. §6.5).

The update rule, which is the whole algorithm:

    Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

**Off-policy** is the word that matters, and it is the single difference from
SARSA (§6.4, `sarsa.py`). The target uses `max_a' Q(s',a')` — the value of the
*best* next action — regardless of which action the epsilon-greedy behaviour
policy will actually take. So the agent explores with one policy and learns the
value of another: the greedy one.

The measurable consequence, on the tiny MDP: Q-learning converges to q_* itself
(9.24e-14, E-007), while SARSA converges to `tiny_mdp.epsilon_soft_q(epsilon)`,
which at epsilon = 0.1 is lower by more than 1.5 in places. Q-learning does not
pay for its own exploration; SARSA does.

`test_update_bootstraps_off_the_max_not_the_behaviour_action` pins this at a
single backup, because the two agree at the optimum and a convergence test on
this fixture cannot tell them apart.

Everything except `update` lives in `TabularAgent` — deliberately, so this file
is almost entirely the update rule (see `tabular.py` for why).

Style (CONSTRAINTS #14): the max below is an explicit loop rather than
`Q[next_obs].max()`. Slower, and deliberate — this is one of the functions both
students must be able to write from memory in an interview.
"""

import numpy as np

from soc_triage.agents.tabular import TabularAgent


class QLearningAgent(TabularAgent):
    """Off-policy TD control with an epsilon-greedy behaviour policy."""

    name = "q_learning"
    obs_kind = "disc"

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
        self.visits[obs, action] += 1
