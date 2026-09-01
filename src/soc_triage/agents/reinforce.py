"""REINFORCE — Monte Carlo policy gradient, with an optional learned baseline.

Sutton & Barto 2nd ed. §13.3 (REINFORCE) and §13.4 (REINFORCE with Baseline).

This is the first agent in the project that does not learn a value and then act
greedily with respect to it. It learns the policy directly, and `act()` contains
no argmax at all.

The update rule
---------------
    G_t   = r_{t+1} + gamma*r_{t+2} + ... + gamma^{T-t-1}*r_T      (actual return)
    theta <- theta + alpha * gamma^t * (G_t - b(s_t)) * grad ln pi(a_t|s_t,theta)

Three factors multiplying a direction, and it is worth naming them separately
because the whole algorithm is in the middle one:

  grad ln pi(a_t|s_t)   the direction that makes the action ACTUALLY TAKEN more
                        likely. Not the best action — the taken one.
  (G_t - b(s_t))        how much better the episode went than expected. Positive:
                        push that way. Negative: push the other way.
  gamma^t               see "the gamma^t factor" below.

Why the baseline reduces variance without adding bias
-----------------------------------------------------
Subtracting any function of the state alone leaves the gradient's expectation
untouched, because

    E[ b(s) * grad ln pi(a|s) ] = b(s) * sum_a grad pi(a|s) = b(s) * grad 1 = 0.

What it changes is the spread. Without it, every action taken during a good
episode is reinforced — including the bad ones — because G_t is large and
positive for all of them. With it, only actions that did better than the state's
own average keep pushing. Same expectation, smaller variance.

This is NOT an actor-critic (S&B §13.5)
---------------------------------------
The most likely thing to be caught out on, so it is stated in the file that
could be mistaken for one. `b(s_t)` here is subtracted from the **full observed
return** and never appears inside the target: nothing bootstraps. Actor-critic
replaces G_t with `r + gamma*v(s')`, which does bootstrap, and that is what buys
it lower variance and costs it bias. **Two networks is not the criterion —
bootstrapping is.** `test_the_baseline_is_not_a_critic` fails if that changes.

The gamma^t factor
------------------
S&B's boxed algorithm includes it; most published implementations drop it. Kept
here, because the teaching constraint (CONSTRAINTS #13) says the code should be
the textbook's algorithm and a student should be able to say what the factor is
for: a return earned at step t is worth gamma^t from the episode's start, and
dropping it optimises the undiscounted objective using discounted returns. With
gamma = 0.99 over ~50-step shifts it reaches ~0.6 by the end of an episode —
real but not crippling. `test_the_gamma_to_the_t_factor_is_applied` pins it, so
removing it has to be a decision rather than an accident.

Exploration
-----------
There is no epsilon anywhere in this file, and that absence is deliberate rather
than an omission. Value-based agents explore by sometimes ignoring their policy;
a policy-gradient agent explores by *having* a stochastic policy and sampling
from it. Exploration therefore decays only as the policy itself sharpens, with
nothing to schedule.

Style (CONSTRAINTS #14)
-----------------------
The return accumulation is an explicit backwards loop, the same shape as
`monte_carlo.py`: G = r + gamma*G, which is O(T) rather than the O(T^2) of
recomputing each return from scratch. The per-step log-probabilities are batched
into one tensor because a Python loop over `.backward()` calls would be slower
without being any clearer — what is kept readable is the structure of
`end_episode()`, which is the algorithm's five steps in the order S&B states
them.
"""

import numpy as np
import torch
import torch.nn as nn

from soc_triage.agents.base import Agent
from soc_triage.config import ReinforceConfig

# Same reasoning as dqn.py: multi-threaded CPU kernels reorder float reductions,
# so two runs with the same seed can differ in the last bits. Reproducibility is
# worth more here than throughput on a network this small.
torch.set_num_threads(1)

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


def _mlp(
    input_dim: int,
    hidden_layers: tuple[int, ...],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    """An MLP with no activation on the output.

    Used for both heads. The policy head's outputs are logits (unbounded, signed,
    turned into probabilities by softmax) and the value head's output is a state
    value (unbounded, signed, and in this environment frequently negative), so
    neither may be squashed.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}")
    make_activation = _ACTIVATIONS[activation]

    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_layers:
        layers.append(nn.Linear(width, hidden))
        layers.append(make_activation())
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class ReinforceAgent(Agent):
    """Monte Carlo policy gradient over the continuous state.

    Every hyperparameter is injected from config (CONSTRAINTS #9); none is
    defaulted, because a constructor default is a magic number in disguise.
    """

    name = "reinforce"
    obs_kind = "cont"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        rcfg: ReinforceConfig,
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
        self.rcfg = rcfg
        self._scales = feature_scales.astype(np.float32)

        # Seed torch before constructing the networks: initial weights are drawn
        # here, so two agents with the same seed must start from the same point
        # for any comparison between them to mean anything.
        torch.manual_seed(seed)
        self.policy_net = _mlp(obs_dim, rcfg.hidden_layers, n_actions, rcfg.activation)
        self.policy_optimiser = torch.optim.Adam(self.policy_net.parameters(), lr=rcfg.lr)

        # The baseline is built even when switched off, so that turning it on is
        # a config change rather than a code path that has never run. It simply
        # contributes nothing to the update while `use_baseline` is false.
        self.value_net = _mlp(obs_dim, rcfg.hidden_layers, 1, rcfg.activation)
        self.value_optimiser = torch.optim.Adam(self.value_net.parameters(), lr=rcfg.baseline_lr)

        # The agent's own RNG for action sampling, separate from the
        # environment's stream — changing the exploration behaviour must not
        # shift the alert sequence, or two runs stop being comparable.
        self._rng = np.random.default_rng(seed)

        # (observation, action, reward) for the episode in progress. Cleared at
        # every episode boundary: a leaked buffer folds the previous shift's
        # rewards into this one's returns, and the only symptom is extra noise
        # (test_the_buffer_is_cleared_between_episodes).
        self._episode: list[tuple[np.ndarray, int, float]] = []

        # Diagnostics from the most recently completed episode. These exist for
        # the variance demonstration Phase 4 owes (ROADMAP box 4) and for the
        # tests — nothing in the algorithm reads them.
        self.last_returns: np.ndarray = np.zeros(0, dtype=np.float64)
        self.last_coefficients: np.ndarray = np.zeros(0, dtype=np.float64)
        self.last_policy_grad_norm: float = 0.0

    # -- acting ------------------------------------------------------------

    def _scaled(self, obs: np.ndarray) -> torch.Tensor:
        """Divide each column by its domain constant (D-023/D-032), then to torch."""
        return torch.from_numpy(obs.astype(np.float32) / self._scales)

    def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """pi(.|s) as a numpy vector. No gradient — this is for acting and for
        reading the learned policy, never for the update."""
        with torch.no_grad():
            logits = self.policy_net(self._scaled(obs))
            probs = torch.softmax(logits, dim=-1)
        return probs.numpy().astype(np.float64)

    def act(self, obs: np.ndarray) -> int:
        """Sample from the policy. **No argmax** — see the module docstring.

        The agent's own RNG draws the action rather than torch.multinomial, so
        the action stream depends only on this seed and stays reproducible
        independently of anything torch does internally.
        """
        probs = self.action_probabilities(obs)
        # Renormalise defensively: softmax output can miss 1.0 by ~1e-8 in
        # float32, and numpy's choice() rejects a vector that does not sum to 1.
        probs = probs / probs.sum()
        return int(self._rng.choice(self.n_actions, p=probs))

    def reseed(self, seed: int) -> None:
        """Restart the action-sampling stream, leaving the learned policy alone.

        D-036 reports Phase 4 on the **sampled** policy, which makes the random
        draws part of the reported number. Left alone, evaluation would simply
        continue whatever stream training ended on, so the same weights measured
        after 200 and after 20000 episodes would be read through different draws
        — a reported value that moves with the training budget for a reason that
        has nothing to do with learning.

        Only `self._rng` is touched. The networks and optimisers are untouched,
        because evaluation must measure the agent that trained rather than one
        perturbed on the way to being measured.
        """
        self._rng = np.random.default_rng(seed)

    def state_value(self, obs: np.ndarray) -> float:
        """The baseline's estimate of v(s). Diagnostic; also what gets subtracted."""
        with torch.no_grad():
            return float(self.value_net(self._scaled(obs)).item())

    # -- learning ----------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Buffer the step. REINFORCE cannot learn anything yet.

        `next_obs` and `done` are unused: the update needs G_t, which is built
        from rewards alone once the episode has finished, and REINFORCE never
        bootstraps off a successor's value. They stay in the signature because
        the runner drives every agent identically and must not need to know
        which algorithm it holds (CONSTRAINTS #10).
        """
        self._episode.append((np.asarray(obs, dtype=np.float64), int(action), float(reward)))

    def _returns(self) -> np.ndarray:
        """G_t for every t, by one backwards pass: G = r + gamma*G.

        The textbook formulation, and O(T) rather than the O(T^2) of summing
        each return from scratch.
        """
        returns = np.zeros(len(self._episode), dtype=np.float64)
        G = 0.0
        for t in range(len(self._episode) - 1, -1, -1):
            G = self._episode[t][2] + self.gamma * G
            returns[t] = G
        return returns

    def end_episode(self) -> None:
        """The whole algorithm, in the five steps S&B states it in.

        1. compute the returns G_t
        2. subtract the baseline, giving the advantage (G_t - v(s_t))
        3. weight each step by gamma^t
        4. one gradient step on -sum_t coefficient_t * ln pi(a_t|s_t)
        5. one gradient step fitting the baseline to those same returns

        Step 4's minus sign is the only thing that turns gradient ASCENT on the
        objective into the descent that torch optimisers perform. Losing it
        trains the agent to do the opposite of what worked, which produces a
        confidently wrong policy and no error.
        """
        if not self._episode:
            return

        returns = self._returns()
        observations = torch.from_numpy(
            np.stack([obs for obs, _, _ in self._episode]).astype(np.float32) / self._scales
        )
        actions = torch.tensor([action for _, action, _ in self._episode], dtype=torch.int64)
        returns_t = torch.from_numpy(returns.astype(np.float32))

        # 2. The baseline. Detached: the value head is trained by its own loss in
        #    step 5, and letting the policy's gradient flow into it would make
        #    one network chase two different objectives.
        if self.rcfg.use_baseline:
            baseline = self.value_net(observations).squeeze(-1).detach()
        else:
            baseline = torch.zeros(len(self._episode))
        advantages = returns_t - baseline

        # 3. gamma^t — the discount on the step itself, not on the return.
        discounts = torch.tensor(
            [self.gamma**t for t in range(len(self._episode))], dtype=torch.float32
        )
        coefficients = discounts * advantages

        # 4. The policy step.
        logits = self.policy_net(observations)
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(coefficients * chosen_log_probs).sum()

        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        # Clipping is load-bearing, not decoration: returns in this environment
        # reach +-500, so a single unbaselined episode can otherwise produce a
        # step large enough to destroy the policy in one move.
        self.last_policy_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.rcfg.grad_clip_norm)
        )
        self.policy_optimiser.step()

        # 5. The baseline step: fit v(s_t) to the returns actually observed.
        #    Runs only when the baseline is in use — training a value head that
        #    nothing consumes would burn time and, worse, make the ablation look
        #    like it still had one.
        if self.rcfg.use_baseline:
            predicted = self.value_net(observations).squeeze(-1)
            value_loss = ((returns_t - predicted) ** 2).mean()
            self.value_optimiser.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.rcfg.grad_clip_norm)
            self.value_optimiser.step()

        self.last_returns = returns
        self.last_coefficients = coefficients.detach().numpy().astype(np.float64)
        self._episode.clear()

    # -- reading the result -------------------------------------------------

    def policy_parameters(self) -> np.ndarray:
        """Every policy weight flattened into one vector.

        For tests that need to prove the policy did or did not move, and for
        checking two seeded agents started from the same point.
        """
        return np.concatenate([p.detach().numpy().ravel() for p in self.policy_net.parameters()])

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
                "policy": self.policy_net.state_dict(),
                "value": self.value_net.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, weights_only=True)
        self.policy_net.load_state_dict(state["policy"])
        self.value_net.load_state_dict(state["value"])
