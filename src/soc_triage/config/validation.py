"""The error type and the three checks every config loader shares.

Split out of the former single-file `config.py` so the environment and training
loaders can both use them without either importing the other (CONSTRAINTS #12 —
the file had reached 657 lines).

Every function here exists to turn a silent config mistake into a loud one at
load time. A missing key that surfaces mid-training costs an evening; the same
key named in a ConfigError costs a line.
"""

from typing import Any


class ConfigError(Exception):
    """Raised when the YAML is missing a key or a value fails validation.

    The message always names the full dotted path (e.g. 'incident.base_rate')
    so the fix is findable without reading this module.
    """


def _require(mapping: dict[str, Any], key: str, path: str) -> Any:
    """Fetch a required key, or raise naming the exact dotted path that is missing."""
    if not isinstance(mapping, dict) or key not in mapping:
        raise ConfigError(f"missing required config key: '{path}.{key}'")
    return mapping[key]


def _check_prior(prior: tuple[float, ...], expected_len: int, path: str) -> None:
    """A prior must be a probability distribution matching its levels list."""
    if len(prior) != expected_len:
        raise ConfigError(
            f"'{path}': prior has {len(prior)} entries but {expected_len} levels"
        )
    if any(p < 0.0 or p > 1.0 for p in prior):
        raise ConfigError(f"'{path}': prior entries must be in [0, 1], got {prior}")
    total = sum(prior)
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"'{path}': prior must sum to 1, sums to {total}")


def _check_ascending(values: tuple[float, ...], path: str) -> None:
    """Bucket boundaries must strictly ascend or bucketing is ambiguous."""
    if any(b >= a for b, a in zip(values, values[1:])):
        raise ConfigError(f"'{path}': boundaries must strictly ascend, got {values}")
