"""SARSA — on-policy TD control (Sutton & Barto 2nd ed. §6.4).

The update rule:

    Q(s,a) <- Q(s,a) + alpha [ r + gamma * Q(s',a') - Q(s,a) ]

The name is the tuple it needs: **S**tate, **A**ction, **R**eward, next
**S**tate, next **A**ction. That final `a'` is the entire difference from
Q-learning, which uses `max_a' Q(s',a')` instead.

**On-policy** means the agent evaluates the policy it is actually following,
exploration included. If epsilon-greedy is about to do something foolish, SARSA
bootstraps off the foolish action's value and learns that this state is worth
less than it would be under perfect play. Q-learning ignores its own
exploration and learns the value of behaving optimally from here on.

The measurable consequence: SARSA converges to `tiny_mdp.epsilon_soft_q(eps)`,
not to q_*. At epsilon = 0.1 that is lower by more than 1.5 in places. The gap
is not error — it is the price of exploring, correctly accounted for. This is
also why SARSA is described as the more "cautious" algorithm: on a cliff-edge
problem it learns to walk away from the edge, because it knows it will
occasionally step randomly.

**The `a'` problem, and how this file solves it.** `Agent.update(obs, action,
reward, next_obs, done)` provides no `a'` — the runner does not know SARSA needs
one, and widening the interface would force every other agent to supply
something meaningless (CONSTRAINTS #10, #18). So the agent selects `a'` itself
during `update` and caches it, and the next `act()` returns exactly that cached
action. The on-policy property depends entirely on that cache being honoured:
if `act()` re-sampled instead, the agent would become a strange hybrid that
still converges and still passes every value test.
`test_sarsa_actually_takes_the_action_it_bootstrapped_off` exists to catch it.
"""

from soc_triage.agents.tabular import TabularAgent


class SarsaAgent(TabularAgent):
    """On-policy TD control. Bootstraps off the action it is about to take."""

    name = "sarsa"
    obs_kind = "disc"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The a' chosen during update(), waiting to be returned by the next
        # act(). None means "no commitment outstanding" — at an episode start,
        # or after a terminal step, act() chooses freshly.
        self._committed_action: int | None = None
        # Kept for the test that asserts the commitment is honoured. Unlike
        # _committed_action it is not cleared, so it can be inspected after the
        # fact without changing behaviour.
        self.last_bootstrap_action: int | None = None

    def act(self, obs: int) -> int:
        """Return the action this agent already committed to, if there is one.

        SARSA chose `a'` at update time and bootstrapped off its value. Choosing
        a *different* action now would break the on-policy property: the value
        it learned would refer to an action it never took.
        """
        if self._committed_action is not None:
            action = self._committed_action
            self._committed_action = None
            return action
        return self._epsilon_greedy(obs)

    def force_next_action(self, action: int) -> None:
        """Pin the next action, for tests that need a specific `a'`.

        Present so `test_sarsa_bootstraps_off_a_worse_action_when_exploration_picks_one`
        can construct the case where the behaviour policy explores, instead of
        fishing for a seed that happens to produce it. A seed-hunted test breaks
        the moment the RNG stream shifts for an unrelated reason.
        """
        self._committed_action = action

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        done: bool,
    ) -> None:
        """One SARSA backup (S&B §6.4).

            Q(s,a) <- Q(s,a) + alpha [ r + gamma * Q(s',a') - Q(s,a) ]

        On termination there is no `a'` and no successor, so the target is the
        reward alone and no commitment is carried into the next episode.
        """
        if done:
            target = reward
            self._committed_action = None
        else:
            # Choose a' now — under the SAME epsilon-greedy policy the agent
            # acts with — bootstrap off it, and commit to taking it next.
            if self._committed_action is None:
                self._committed_action = self._epsilon_greedy(next_obs)
            next_action = self._committed_action
            self.last_bootstrap_action = next_action
            target = reward + self.gamma * self.Q[next_obs, next_action]

        td_error = target - self.Q[obs, action]
        self.Q[obs, action] += self.alpha * td_error
        self.visits[obs, action] += 1
