"""One-step actor-critic — bootstrapped policy gradient, with separate heads.

Sutton & Barto 2nd ed. §13.5 ("Actor-Critic Methods"), the boxed episodic
one-step algorithm.

The update rule
---------------
Every step, not every episode:

    delta = r + gamma*v(s',w) - v(s,w)            (if s' is terminal, gamma*v(s') = 0)
    w     <- w     + alpha_w     * delta * grad v(s,w)
    theta <- theta + alpha_theta * I * delta * grad ln pi(a|s,theta)
    I     <- gamma * I                           (reset to 1 at the episode start)

What actually separates this from REINFORCE
-------------------------------------------
The temptation is to say "actor-critic has two networks". `reinforce.py` also
has two networks — a policy and a learned baseline — and it is not an
actor-critic. The difference is one term:

    REINFORCE      coefficient = G_t - b(s_t)             G_t is the OBSERVED return
    actor-critic   coefficient = r + gamma*v(s') - v(s)   the successor's ESTIMATE

REINFORCE waits for the episode to finish because G_t does not exist until then,
and it uses what actually happened. This agent never waits, because it replaces
the rest of the episode with the critic's guess about it. **Bootstrapping is the
criterion, not the network count** — `tests/test_reinforce.py` pins that
REINFORCE does not bootstrap, and `tests/test_actor_critic.py` pins that this
does. The two tests are mirror images on purpose.

What that buys and what it costs
--------------------------------
It buys **variance**: G_t is a sum of ~50 noisy rewards, while
`r + gamma*v(s')` is one reward plus one estimate, so the coefficient stops
swinging with every unlucky shift. It costs **bias**: v(s') is wrong early in
training, and an update scaled by a wrong number is a wrong update. REINFORCE is
unbiased and noisy; this is biased and quiet. That trade is the whole content of
the ROADMAP's variance box and the answer to the obvious viva question.

It also buys **online learning**, worth stating separately because it is what
makes the sample-efficiency comparison interesting: this agent improves inside an
episode, where REINFORCE cannot improve until the shift is over.

The entropy bonus is NOT in the textbook
----------------------------------------
S&B §13.5 has no entropy term. This one does, and `entropy_coef: 0.0` recovers
the textbook algorithm exactly.

The reason it is here is E-018: REINFORCE's greedy policy was degenerate — a
single action in every state — by 300 episodes, and nothing in its configuration
resisted a policy sharpening early. E-019 then ruled out the gradient clip as the
cause. A policy-gradient method has no epsilon to hold exploration open, so the
only thing that can keep pi spread out is a term that pays for spread. It is
declared here rather than assumed, because an addition to a textbook algorithm
that goes unmentioned is how a student ends up defending code they cannot name.

The I accumulator
-----------------
S&B's boxed algorithm carries `I`, decayed by gamma each step and reset at the
episode boundary. It is the same gamma^t factor `reinforce.py` applies, and it is
kept here for the same reason: it is in the algorithm as stated, and a student
should be able to say what it is for — a step taken at time t contributes to the
objective discounted from the episode's start, not from t.

Style (CONSTRAINTS #14)
-----------------------
`update()` is written as the four steps of the boxed algorithm in the order S&B
states them, with no batching and no vectorisation: this agent genuinely does one
step at a time, so the readable form and the fast form are the same form.
"""

import numpy as np
import torch

from soc_triage.agents.base import Agent
from soc_triage.agents.reinforce import _mlp
from soc_triage.config import ActorCriticConfig

# Same reasoning as dqn.py and reinforce.py: multi-threaded CPU kernels reorder
# float reductions, so two runs with the same seed can differ in the last bits.
torch.set_num_threads(1)


class ActorCriticAgent(Agent):
    """One-step actor-critic over the continuous state.

    Every hyperparameter is injected from config (CONSTRAINTS #9); none is
    defaulted, because a constructor default is a magic number in disguise.
    """

    name = "actor_critic"
    obs_kind = "cont"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        accfg: ActorCriticConfig,
        gamma: float,
        seed: int,
        feature_scales: np.ndarray,
    ) -> None:
        if feature_scales.shape != (obs_dim,):
            raise ValueError(
                f"feature_scales has shape {feature_scales.shape}, expected ({obs_dim},)"
            )

        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.accfg = accfg
        self._scales = feature_scales.astype(np.float32)

        # Seed torch before constructing the networks: initial weights are drawn
        # here, so two agents with the same seed must start from the same point
        # for any comparison between them to mean anything.
        torch.manual_seed(seed)
        # Two SEPARATE networks rather than a shared trunk with two heads. A
        # shared trunk is the more common engineering choice and it is the wrong
        # one here: it couples the actor's and the critic's gradients through
        # weights they both own, and the resulting behaviour cannot be explained
        # in five minutes (CONSTRAINTS #13). Separate is also what makes the
        # comparison against reinforce.py's policy+baseline pair like-for-like.
        self.actor_net = _mlp(obs_dim, accfg.hidden_layers, n_actions, accfg.activation)
        self.critic_net = _mlp(obs_dim, accfg.hidden_layers, 1, accfg.activation)
        self.actor_optimiser = torch.optim.Adam(self.actor_net.parameters(), lr=accfg.actor_lr)
        self.critic_optimiser = torch.optim.Adam(self.critic_net.parameters(), lr=accfg.critic_lr)

        # The agent's own RNG for action sampling, separate from the
        # environment's stream — changing exploration must not shift the alert
        # sequence, or two runs stop being comparable.
        self._rng = np.random.default_rng(seed)

        # S&B's I. Starts at 1 for every episode and decays by gamma per step.
        self.discount_accumulator: float = 1.0

        # Diagnostics from the most recent step, plus the collected per-episode
        # series. These exist for the variance demonstration (ROADMAP box 4), the
        # trainer's logging, and the tests — nothing in the algorithm reads them.
        self.last_td_error: float = 0.0
        self.last_entropy: float = 0.0
        self.last_actor_grad_norm: float = 0.0
        self.last_critic_grad_norm: float = 0.0
        self.last_td_errors: np.ndarray = np.zeros(0, dtype=np.float64)
        self._episode_td_errors: list[float] = []

    # -- acting ------------------------------------------------------------

    def _scaled(self, obs: np.ndarray) -> torch.Tensor:
        """Divide each column by its domain constant (D-023/D-032), then to torch."""
        return torch.from_numpy(obs.astype(np.float32) / self._scales)

    def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """pi(.|s) as a numpy vector. No gradient — this is for acting and for
        reading the learned policy, never for the update."""
        with torch.no_grad():
            probs = torch.softmax(self.actor_net(self._scaled(obs)), dim=-1)
        return probs.numpy().astype(np.float64)

    def act(self, obs: np.ndarray) -> int:
        """Sample from the policy. **No argmax** — see the module docstring."""
        probs = self.action_probabilities(obs)
        # Renormalise defensively: softmax output can miss 1.0 by ~1e-8 in
        # float32, and numpy's choice() rejects a vector that does not sum to 1.
        probs = probs / probs.sum()
        return int(self._rng.choice(self.n_actions, p=probs))

    def reseed(self, seed: int) -> None:
        """Restart the action-sampling stream, leaving actor and critic alone.

        Same contract and same reason as `ReinforceAgent.reseed` — D-036 reports
        Phase 4 on the sampled policy, so the evaluation draws are pinned to a
        seed rather than inherited from wherever training left the stream.
        """
        self._rng = np.random.default_rng(seed)

    def state_value(self, obs: np.ndarray) -> float:
        """The critic's v(s). Diagnostic, and the quantity the target bootstraps off."""
        with torch.no_grad():
            return float(self.critic_net(self._scaled(obs)).item())

    # -- learning ----------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """One step of S&B §13.5, in the order the book states it.

        Unlike `ReinforceAgent.update`, this is the entire algorithm — nothing is
        left for the episode boundary except resetting I. `next_obs` and `done`
        are load-bearing here, where REINFORCE ignored both.
        """
        observation = self._scaled(obs)
        next_observation = self._scaled(next_obs)

        # 1. delta = r + gamma*v(s') - v(s).
        #
        #    `done` kills the bootstrap. The shift is over at 480 minutes and the
        #    end-of-shift penalty has already been charged, so there is no future
        #    left to value; carrying gamma*v(s') across that boundary would have
        #    the agent valuing a state that does not exist. (Worth knowing for a
        #    viva: bootstrapping through a TIME LIMIT rather than a true terminal
        #    is a real bias in many implementations. Here the shift end is a
        #    genuine episode end, not a truncation, so treating it as terminal is
        #    correct rather than merely conventional.)
        #
        #    Computed under no_grad: delta is a scalar COEFFICIENT everywhere it
        #    appears, never something either network differentiates through.
        with torch.no_grad():
            value_now = self.critic_net(observation).squeeze(-1)
            bootstrap = (
                torch.zeros(())
                if done
                else self.gamma * self.critic_net(next_observation).squeeze(-1)
            )
            td_error = reward + bootstrap - value_now
        self.last_td_error = float(td_error.item())

        # 2. The critic step: move v(s) toward the target r + gamma*v(s').
        #    Written as the squared TD error rather than as `delta * grad v`,
        #    which is the same update — d/dw (target - v)^2 / 2 = -(target - v) *
        #    grad v — and lets torch produce the gradient instead of hand-rolling
        #    it. `target` is a constant here, so it is built from the detached
        #    prediction plus delta.
        predicted = self.critic_net(observation).squeeze(-1)
        target = predicted.detach() + td_error
        critic_loss = (target - predicted) ** 2
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.last_critic_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.accfg.grad_clip_norm)
        )
        self.critic_optimiser.step()

        # 3. The actor step: I * delta * grad ln pi(a|s), ascended.
        #    The minus sign is the only thing turning gradient ASCENT on the
        #    objective into the descent torch performs. Losing it trains the agent
        #    to do the opposite of what worked, confidently and without error.
        logits = self.actor_net(observation)
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_prob = log_probs[action]
        actor_loss = -self.discount_accumulator * td_error * chosen_log_prob

        # The entropy bonus — NOT S&B. Subtracted, i.e. entropy is MAXIMISED, so
        # the policy pays a price for collapsing onto one action. At
        # entropy_coef = 0 this term is exactly zero and the update is the
        # textbook's (test_zero_entropy_coefficient_is_the_textbook_update).
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum()
        self.last_entropy = float(entropy.item())
        actor_loss = actor_loss - self.accfg.entropy_coef * entropy

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.last_actor_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.accfg.grad_clip_norm)
        )
        self.actor_optimiser.step()

        # 4. I <- gamma * I.
        self.discount_accumulator *= self.gamma
        self._episode_td_errors.append(self.last_td_error)

    def end_episode(self) -> None:
        """Reset I and publish the episode's TD errors.

        Almost nothing happens here, and that near-emptiness is the point: for
        REINFORCE this method IS the algorithm, and for actor-critic it is
        bookkeeping. Forgetting to call it does not stop this agent learning — it
        leaves I decayed from the previous shift, so the early steps of every
        subsequent episode are weighted as though they were late ones.
        """
        self.discount_accumulator = 1.0
        self.last_td_errors = np.array(self._episode_td_errors, dtype=np.float64)
        self._episode_td_errors = []

    # -- reading the result -------------------------------------------------

    def policy_parameters(self) -> np.ndarray:
        """Every actor weight flattened into one vector.

        For tests that need to prove the policy did or did not move, and for
        checking two seeded agents started from the same point.
        """
        return np.concatenate([p.detach().numpy().ravel() for p in self.actor_net.parameters()])

    def greedy_policy(self, observations: np.ndarray) -> np.ndarray:
        """argmax_a pi(a|s) for a batch of observations.

        Only for *reading* the learned policy — the policy table and the
        evaluation protocol. `act()` never calls this: sampling is the algorithm.
        """
        return np.array(
            [int(np.argmax(self.action_probabilities(obs))) for obs in observations],
            dtype=np.int64,
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor_net.state_dict(),
                "critic": self.critic_net.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, weights_only=True)
        self.actor_net.load_state_dict(state["actor"])
        self.critic_net.load_state_dict(state["critic"])
