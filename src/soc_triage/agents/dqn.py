"""Deep Q-Network — Q-learning with a neural network and two stabilisers.

Mnih et al. (2015), "Human-level control through deep reinforcement learning";
Sutton & Barto 2nd ed. §16.5, which presents DQN as exactly what this docstring
claims it is: the §6.5 Q-learning backup, with the table replaced by a function
approximator, plus the two devices that make that replacement stable.

The update rule, in the same form `q_learning.py` writes it:

    tabular:  Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
    DQN:      minimise  ( r + gamma * max_a' Q_target(s',a')  -  Q_online(s,a) )^2
              by gradient descent on the online network's weights

The bracketed quantity is the same TD error in both. What changed is that
`alpha` is no longer a step size on one table cell — the gradient step moves
every weight, and therefore moves the estimate for *every* state at once. That
generalisation is the entire reason to do this, and also the entire reason it
is unstable without help.

The two stabilisers, and what each one fixes
--------------------------------------------
1. **Experience replay** (`replay.py`) breaks the correlation between
   consecutive transitions and lets each one be learned from more than once.
2. **A target network** — a frozen copy of the online network, refreshed by a
   hard copy every `target_update_every` GRADIENT steps — supplies
   `max_a' Q(s',a')`. Without it the regression target moves every time the
   weights move, and the network chases a value it is itself changing. This is
   the one both students should be able to explain in a sentence: *the target
   must not be a function of the parameters currently being updated.*

Both are switchable off, because Phase 3's roadmap requires demonstrating what
each is worth rather than asserting it (`dcfg.no_replay`, `dcfg.no_target_network`).

Style note (CONSTRAINTS #14)
----------------------------
The batched `max` and `gather` below are tensor operations, not explicit loops —
a per-sample Python loop over a minibatch would be roughly two orders of
magnitude slower and would not make the algorithm any clearer. What is kept
readable is the *structure*: `update()` is a short, named sequence of the eight
steps the algorithm actually has, in the order the paper states them.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from soc_triage.agents.base import Agent
from soc_triage.agents.replay import ReplayBuffer
from soc_triage.config import DQNConfig

# Single-threaded on purpose. Multi-threaded CPU kernels reorder float
# reductions, so two runs with the same seed can differ in the last bits and
# `test_same_seed_gives_identical_parameters` becomes flaky. Reproducibility is
# worth more here than throughput — the network is tiny and the bottleneck is
# the environment, not the matmul.
torch.set_num_threads(1)

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class QNetwork(nn.Module):
    """An MLP mapping a state vector to one Q-value per action.

    One output head per action rather than a network taking (state, action) and
    returning a scalar: with this shape a single forward pass yields every
    action's value, so both `act()` and the `max_a'` in the target cost one pass
    instead of `n_actions` of them.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...],
        n_actions: int,
        activation: str,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        make_activation = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_layers:
            layers.append(nn.Linear(width, hidden))
            layers.append(make_activation())
            width = hidden
        # No activation on the output: Q-values are unbounded and signed. A
        # ReLU here would forbid negative values, and this environment's
        # missed-incident penalty makes most of them negative.
        layers.append(nn.Linear(width, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, input_dim) -> (B, n_actions)."""
        return self.net(x)


class DQNAgent(Agent):
    """Q-learning over the continuous 17-dim state, with replay and a target net."""

    name = "dqn"
    obs_kind = "cont"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        dcfg: DQNConfig,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        seed: int,
        feature_scales: np.ndarray,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.dcfg = dcfg

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Inverted so the code below reads positively; the YAML states the
        # ablations as things turned OFF, which is how they are described.
        self.use_replay = not dcfg.no_replay
        self.use_target_network = not dcfg.no_target_network

        if feature_scales.shape != (obs_dim,):
            raise ValueError(
                f"feature_scales has shape {feature_scales.shape}, expected ({obs_dim},)"
            )
        self._scales = feature_scales.astype(np.float32)

        # Seed torch before constructing the network: the weights are drawn at
        # construction, so seeding afterwards would not make them reproducible.
        torch.manual_seed(seed)
        self.online = QNetwork(obs_dim, dcfg.hidden_layers, n_actions, dcfg.activation)
        if self.use_target_network:
            self.target = QNetwork(obs_dim, dcfg.hidden_layers, n_actions, dcfg.activation)
            self.target.load_state_dict(self.online.state_dict())
        else:
            # None rather than an unused copy, so any code path that reaches for
            # a target network under this ablation fails loudly instead of
            # silently bootstrapping off a stale network nobody updates.
            self.target = None

        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=dcfg.lr)
        self.buffer = ReplayBuffer(dcfg.replay_capacity, obs_dim, seed)

        # Exploration and replay sampling draw from separate generators so that
        # changing one does not shift the other's stream.
        self._rng = np.random.default_rng(seed)

        self.env_steps = 0
        self.gradient_steps = 0
        # Diagnostics, read by the training script's curves and by the tests.
        # last_grad_norm is the norm BEFORE clipping, which is the only version
        # from which "clipping actually engaged" can be determined.
        self.last_loss = float("nan")
        self.last_grad_norm = float("nan")

    # -- acting ------------------------------------------------------------

    def _scale(self, obs: np.ndarray) -> np.ndarray:
        """Divide each column by its domain constant (D-023).

        Not a learned or running normalisation: the divisors are fixed domain
        facts, so a state's encoding is identical in episode 1 and episode
        20000, and identical between training and evaluation.
        """
        return obs.astype(np.float32) / self._scales

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Q(obs, ·) for every action, as a plain (n_actions,) array."""
        x = torch.from_numpy(self._scale(obs)).unsqueeze(0)
        with torch.no_grad():
            return self.online(x).numpy().ravel()

    def act(self, obs: np.ndarray) -> int:
        """Epsilon-greedy over the online network's outputs.

        Ties broken toward the LOWER action index — the same rule as
        `TabularAgent._argmax` and `agents.dp.greedy_policy`. Matching matters:
        Task 5 compares the DQN's greedy policy against the tabular one, and a
        different tie-break would show up as a policy disagreement that is
        really just two conventions.
        """
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        values = self.q_values(obs)
        best_action = 0
        best_value = -np.inf
        for action in range(self.n_actions):
            if values[action] > best_value:  # strict >, so the first index wins
                best_value = values[action]
                best_action = action
        return best_action

    def end_episode(self) -> None:
        """Decay epsilon once per EPISODE, floored at epsilon_min (D-015).

        Mirrors `TabularAgent.end_episode` exactly. Decaying per step instead
        would collapse exploration within the first shift; `train.py` must call
        this, and a trainer that forgets produces an agent that never stops
        exploring.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -- learning ----------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """One DQN learning step, in the order the algorithm states it.

        `done` must mean *terminated*, never merely truncated — carried
        verbatim from `q_learning.py`, because the bug is identical here. On
        termination there is no successor and the target is the reward alone;
        passing done=True at an episode-length cutoff on a continuing task
        teaches the agent the world ends.
        """
        # 1. count the environment step
        self.env_steps += 1

        # 2. store the transition, already scaled so scaling happens once per
        #    transition rather than once per time it is sampled
        scaled_obs = self._scale(obs)
        scaled_next_obs = self._scale(next_obs)
        self.buffer.push(scaled_obs, action, reward, scaled_next_obs, done)

        # 3. wait for enough data, then train only every train_freq steps
        if len(self.buffer) < self.dcfg.learning_starts:
            return
        if self.env_steps % self.dcfg.train_freq != 0:
            return

        # 4. assemble the batch
        if self.use_replay:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = (
                self.buffer.sample(self.dcfg.batch_size)
            )
        else:
            # The ablation: learn from the single transition just observed, the
            # way online Q-learning does. Kept as a batch of 1 rather than a
            # separate code path, so everything below is shared and the only
            # difference between the conditions is the data.
            batch_obs = scaled_obs[None, :]
            batch_action = np.array([action], dtype=np.int64)
            batch_reward = np.array([reward], dtype=np.float32)
            batch_next_obs = scaled_next_obs[None, :]
            batch_done = np.array([float(done)], dtype=np.float32)

        obs_t = torch.from_numpy(batch_obs)
        action_t = torch.from_numpy(batch_action)
        reward_t = torch.from_numpy(batch_reward)
        next_obs_t = torch.from_numpy(batch_next_obs)
        done_t = torch.from_numpy(batch_done)

        # 5. the TD target. no_grad because the target is a fixed number for
        #    this step — letting gradients flow into it is the bug that turns
        #    the regression into a moving-target chase even WITH a target net.
        bootstrap_net = self.target if self.use_target_network else self.online
        with torch.no_grad():
            best_next_value = bootstrap_net(next_obs_t).max(dim=1).values
            # (1 - done) is what stops a terminal state from bootstrapping.
            target = reward_t + self.gamma * (1.0 - done_t) * best_next_value

        # 6. the prediction: Q(s,a) for the action actually taken. gather picks
        #    one column per row — the batched form of Q[s, a].
        predicted = self.online(obs_t).gather(1, action_t.unsqueeze(1)).squeeze(1)
        # Huber rather than MSE, but the delta must be matched to the reward
        # scale — see `dqn.huber_delta` in training_default.yaml and E-016.
        # Below the delta Huber is quadratic and the gradient scales with the
        # error, which is what lets the agent distinguish "wrong by 1" from
        # "wrong by 150"; above it the gradient is flat, which is what tames the
        # rare compound miss. Leaving delta at torch's default of 1.0 put every
        # penalty in the flat regime and cost Phase 3 its first sweep.
        loss = F.huber_loss(predicted, target, delta=self.dcfg.huber_delta)

        # 7. one optimiser step, with the gradient norm clipped
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), self.dcfg.grad_clip_norm
        )
        self.optimiser.step()

        self.last_loss = float(loss.detach())
        self.last_grad_norm = float(grad_norm)

        # 8. refresh the target network on a schedule counted in GRADIENT
        #    steps, not environment steps — with train_freq=4 the two differ by
        #    a factor of four, and confusing them silently changes the stabiliser
        #    this phase is measuring.
        self.gradient_steps += 1
        if (
            self.use_target_network
            and self.gradient_steps % self.dcfg.target_update_every == 0
        ):
            self.target.load_state_dict(self.online.state_dict())

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the online network plus the counters a resumed run needs."""
        torch.save(
            {
                "online": self.online.state_dict(),
                "epsilon": self.epsilon,
                "env_steps": self.env_steps,
                "gradient_steps": self.gradient_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore a saved network; the target is re-synced to it.

        weights_only=True because a checkpoint is data, and torch.load's default
        unpickles arbitrary objects.
        """
        state = torch.load(path, weights_only=True)
        self.online.load_state_dict(state["online"])
        if self.use_target_network:
            self.target.load_state_dict(self.online.state_dict())
        self.epsilon = float(state["epsilon"])
        self.env_steps = int(state["env_steps"])
        self.gradient_steps = int(state["gradient_steps"])
