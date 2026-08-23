"""The DQN config section fails loudly at load time, not mid-training.

Every check here exists because the failure it prevents is silent: a
learning_starts below batch_size deadlocks the first sample() call, a
duplicated train_seed_start makes two experiments share alert streams (D-016).

Input scaling is no longer tested here: it moved to a shared `features:` block
in D-032, and its tests moved with it to tests/test_feature_scales.py.
"""

from pathlib import Path

import pytest
import yaml

from soc_triage.config import ConfigError, load_training_config

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


def test_huber_delta_below_the_measured_collapse_threshold_is_refused(tmp_path):
    """torch's default delta of 1.0 is what cost Phase 3 its first sweep (E-016):
    the -150 and -200 penalties in env_default.yaml landed in Huber's linear
    regime and produced the same gradient as a routine +-1 error, so all 20 runs
    collapsed to BULK_CLOSE. A 5x3 delta sweep collapsed 3/3 seeds at 10 and 1/3
    at 25, and 0/3 from 50 up. The loader refuses anything under 50 so the
    failure cannot recur silently in an unattended overnight run.
    """
    for bad in (1.0, 10.0, 25.0):
        raw = _raw()
        raw["dqn"]["huber_delta"] = bad
        with pytest.raises(ConfigError, match="huber_delta"):
            load_training_config(_write(tmp_path, raw))


def test_huber_delta_must_be_positive(tmp_path):
    raw = _raw()
    raw["dqn"]["huber_delta"] = 0.0
    with pytest.raises(ConfigError, match="huber_delta"):
        load_training_config(_write(tmp_path, raw))


def test_the_shipped_config_keeps_every_named_penalty_quadratic():
    """200 is not a round number picked for tidiness: it is the largest NAMED
    single-event penalty in env_default.yaml (end_of_shift_missed -200, before
    the asset multiplier; bulk_close_true_incident -150). Keeping delta at or
    above it means every individual penalty the agent must learn stays in the
    quadratic regime where the gradient still carries magnitude, and only the
    compound multi-miss tail is linearised."""
    dqn = load_training_config(CONFIG_PATH).dqn
    assert dqn.huber_delta >= 150.0
