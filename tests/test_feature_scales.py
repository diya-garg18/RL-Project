"""The 17 input divisors are shared by every function-approximation agent.

They were written under `dqn:` in Phase 3 because the DQN was the only thing
that could use them. They are not DQN hyperparameters: they are domain constants
(the shift is 480 minutes long, severity runs 0-3), and Phase 4's REINFORCE and
actor-critic read the same 17-column vector from `state.featurise`. Two copies
of a domain constant is one copy that can silently drift, so they moved to a
shared `features:` block (D-032).

Every check here exists because the failure it prevents is silent: an unscaled
column does not raise, it just trains worse.
"""

from pathlib import Path

import pytest
import yaml

from soc_triage.config import ConfigError, load_training_config
from soc_triage.state import FEATURE_NAMES, feature_scale_vector

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "training_default.yaml"


def _raw() -> dict:
    """The YAML as a plain dict, so a test can corrupt one key and reload."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_scales_cover_every_column_exactly():
    features = load_training_config(CONFIG_PATH).features
    assert {name for name, _ in features.scales} == set(FEATURE_NAMES)


def test_scale_vector_is_ordered_like_feature_names():
    features = load_training_config(CONFIG_PATH).features
    vector = feature_scale_vector(features.scales)
    lookup = dict(features.scales)
    assert vector.shape == (len(FEATURE_NAMES),)
    for i, name in enumerate(FEATURE_NAMES):
        assert vector[i] == lookup[name]


def test_scale_vector_rejects_a_missing_column():
    partial = tuple((n, 1.0) for n in FEATURE_NAMES[:-1])
    with pytest.raises(ValueError, match=FEATURE_NAMES[-1]):
        feature_scale_vector(partial)


def test_scale_vector_rejects_an_unknown_column():
    extra = tuple((n, 1.0) for n in FEATURE_NAMES) + (("not_a_feature", 1.0),)
    with pytest.raises(ValueError, match="not_a_feature"):
        feature_scale_vector(extra)


def test_zero_divisor_is_rejected(tmp_path):
    raw = _raw()
    raw["features"]["scales"]["queue_len"] = 0.0
    with pytest.raises(ConfigError, match="queue_len"):
        load_training_config(_write(tmp_path, raw))


def test_a_missing_features_block_is_refused(tmp_path):
    """Not a hypothetical: the block moved out of `dqn:` in D-032, so an old
    config file pulled from before that change has no `features:` at all. It
    must refuse to load rather than train an agent on unscaled columns."""
    raw = _raw()
    del raw["features"]
    with pytest.raises(ConfigError, match="features"):
        load_training_config(_write(tmp_path, raw))


def test_the_dqn_section_does_not_keep_a_private_copy_of_the_scales():
    """The point of the promotion (D-032). If someone later re-adds
    `dqn.feature_scales`, the DQN and REINFORCE can be scaled differently
    without anything failing, and the sample-efficiency comparison Phase 4 is
    built on stops being a comparison of algorithms."""
    raw = _raw()
    assert "feature_scales" not in raw["dqn"]
