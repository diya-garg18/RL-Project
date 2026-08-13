"""The Agent interface every algorithm implements (ARCHITECTURE.md §2).

Agents see only encoded observations and emit action indices. They never touch
environment internals (CONSTRAINTS.md #10). The one sanctioned exception is the
oracle baseline, which declares obs_kind='snapshot' — see baselines.py.
"""

from typing import Any


class Agent:
    """Base class: a no-op agent. Baselines override act(); learners also update().

    obs_kind tells the runner which encoding to feed act():
      'disc'     -> int state id from state.discretise()
      'cont'     -> np.ndarray from state.featurise()
      'snapshot' -> the raw EnvSnapshot (oracle ONLY — it is the point of an upper bound)
    """

    name: str = "base"
    obs_kind: str = "disc"

    def act(self, obs: Any) -> int:
        raise NotImplementedError

    def update(self, obs: Any, action: int, reward: float, next_obs: Any, done: bool) -> None:
        """Learning hook. Baselines have nothing to learn — default is a no-op,
        so the runner never needs to special-case them (FLOW.md Flow B)."""

    def save(self, path: str) -> None:
        """Persist learned parameters. No-op for baselines."""

    def load(self, path: str) -> None:
        """Restore learned parameters. No-op for baselines."""
