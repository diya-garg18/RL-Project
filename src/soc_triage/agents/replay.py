"""Uniform experience replay — the first of DQN's two stabilisers.

Implements the replay memory of Mnih et al. (2015), "Human-level control
through deep reinforcement learning"; Sutton & Barto 2nd ed. §16.5 describes
the same mechanism in its treatment of DQN.

Written by hand (CONSTRAINTS #7). It buys two distinct things, and it is worth
being able to say which is which in an interview:

1. **Decorrelation.** Consecutive transitions inside one shift are strongly
   related — the queue at step t+1 is the queue at step t minus one alert. Fed
   to a network in order, each gradient step sees a batch that is nearly one
   sample, and the network chases whatever the agent is doing right now. Uniform
   sampling from a large buffer mixes transitions from thousands of different
   shifts into every batch.
2. **Sample reuse.** Each transition costs one environment step to generate and
   can then be learned from many times, instead of once and discarded.

Design choice: five parallel numpy arrays, not a `deque` of tuples. Sampling
then returns contiguous arrays ready for `torch.from_numpy` with no per-sample
Python work, and the capacity semantics are two integers you can print
(`_cursor`, `_size`) rather than container behaviour you have to trust.
"""

import numpy as np


class ReplayBuffer:
    """A fixed-capacity ring buffer of transitions, sampled uniformly.

    Once full, the oldest transition is the one overwritten: `_cursor` wraps
    modulo capacity, so the write position always points at the least recent
    entry.
    """

    def __init__(self, capacity: int, obs_dim: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.obs_dim = obs_dim

        # float32 throughout because that is what torch wants; storing float64
        # would double the memory and be cast away on every sample anyway.
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros(capacity, dtype=np.int64)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        # `done` is stored as a float, not a bool, because it is used
        # arithmetically in the target: y = r + gamma * (1 - done) * max Q'.
        self._done = np.zeros(capacity, dtype=np.float32)

        self._cursor = 0  # where the next push writes
        self._size = 0    # how many slots hold real data; stops at capacity

        # Own generator, not the global numpy one: two buffers with the same
        # seed must sample identically regardless of what else ran first.
        self._rng = np.random.default_rng(seed)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, overwriting the oldest if the buffer is full."""
        i = self._cursor
        self._obs[i] = obs
        self._action[i] = action
        self._reward[i] = reward
        self._next_obs[i] = next_obs
        self._done[i] = float(done)

        self._cursor = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draw a uniform minibatch, with replacement, from the filled slots.

        With replacement is the standard choice and the cheap one; at a buffer
        size of 100000 and a batch of 64 the chance of a repeat inside one batch
        is negligible, and it keeps sampling a single vectorised draw.

        Indexing all five arrays with the same index array is what keeps a
        transition's five fields together — that alignment is the property
        `test_transitions_stay_aligned_across_the_five_arrays` exists to pin.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            self._obs[idx],
            self._action[idx],
            self._reward[idx],
            self._next_obs[idx],
            self._done[idx],
        )

    def __len__(self) -> int:
        """Transitions currently stored — not capacity, and not pushes seen."""
        return self._size
