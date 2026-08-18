"""The DQN config section fails loudly at load time, not mid-training.

Every check here exists because the failure it prevents is silent: a
learning_starts below batch_size deadlocks the first sample() call, a
feature_scales typo silently leaves one column unscaled, and a duplicated
train_seed_start makes two experiments share alert streams (D-016).
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


def test_dqn_section_loads():
    dqn = load_training_config(CONFIG_PATH).dqn
    assert dqn.hidden_layers == (128, 128)
    assert dqn.batch_size == 64
    assert dqn.train_freq >= 1
    assert dqn.loss == "huber"
    assert dqn.no_replay is False
    assert dqn.no_target_network is False


def test_feature_scales_cover_every_column_exactly():
    dqn = load_training_config(CONFIG_PATH).dqn
    assert {name for name, _ in dqn.feature_scales} == set(FEATURE_NAMES)


def test_scale_vector_is_ordered_like_feature_names():
    dqn = load_training_config(CONFIG_PATH).dqn
    vector = feature_scale_vector(dqn.feature_scales)
    lookup = dict(dqn.feature_scales)
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
    raw["dqn"]["feature_scales"]["queue_len"] = 0.0
    with pytest.raises(ConfigError, match="queue_len"):
        load_training_config(_write(tmp_path, raw))


def test_learning_starts_below_batch_size_is_rejected(tmp_path):
    raw = _raw()
    raw["dqn"]["learning_starts"] = 8
    raw["dqn"]["batch_size"] = 64
    with pytest.raises(ConfigError, match="learning_starts"):
        load_training_config(_write(tmp_path, raw))


def test_replay_capacity_below_batch_size_is_rejected(tmp_path):
    raw = _raw()
    raw["dqn"]["replay_capacity"] = 32
    with pytest.raises(ConfigError, match="replay_capacity"):
        load_training_config(_write(tmp_path, raw))


def test_unimplemented_loss_is_rejected(tmp_path):
    raw = _raw()
    raw["dqn"]["loss"] = "mse"
    with pytest.raises(ConfigError, match="huber"):
        load_training_config(_write(tmp_path, raw))


def test_dqn_seed_block_must_be_distinct_from_the_tabular_learners(tmp_path):
    raw = _raw()
    raw["dqn"]["train_seed_start"] = raw["q_learning"]["train_seed_start"]
    with pytest.raises(ConfigError):
        load_training_config(_write(tmp_path, raw))


def test_ablation_seed_block_must_differ_from_the_main_dqn_block(tmp_path):
    raw = _raw()
    raw["dqn"]["ablation_seed_start"] = raw["dqn"]["train_seed_start"]
    with pytest.raises(ConfigError, match="ablation_seed_start"):
        load_training_config(_write(tmp_path, raw))
