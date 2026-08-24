"""The REINFORCE config section fails loudly at load time, not mid-training.

Same principle as tests/test_dqn_config.py: every check here prevents a failure
that is silent at runtime. A learning rate of zero trains a policy that never
moves and looks like a hard problem; a seed block shared with the DQN makes the
two agents train on identical alert streams, which would make Phase 4's
sample-efficiency comparison look better-controlled than it is (D-016).
"""

from pathlib import Path

import pytest
import yaml

from soc_triage.config import ConfigError, load_training_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "training_default.yaml"


def _raw() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_reinforce_section_loads():
    reinforce = load_training_config(CONFIG_PATH).reinforce
    assert reinforce.hidden_layers == (128, 128)
    assert reinforce.lr > 0
    assert reinforce.baseline_lr > 0
    assert reinforce.use_baseline is True
    assert reinforce.grad_clip_norm > 0


def test_reinforce_has_no_epsilon_of_its_own():
    """Policy gradient explores by sampling from its own policy. If an epsilon
    key ever appears under `reinforce:`, someone has bolted epsilon-greedy onto
    an algorithm that does not have it, and the agent stops being REINFORCE."""
    assert "epsilon" not in _raw()["reinforce"]


def test_reinforce_seed_block_is_distinct_from_every_other_learner(tmp_path):
    raw = _raw()
    for other in ("q_learning", "sarsa", "monte_carlo", "dqn"):
        raw = _raw()
        raw["reinforce"]["train_seed_start"] = raw[other]["train_seed_start"]
        with pytest.raises(ConfigError):
            load_training_config(_write(tmp_path, raw))


def test_reinforce_seed_block_stays_clear_of_the_low_blocks(tmp_path):
    raw = _raw()
    raw["reinforce"]["train_seed_start"] = 105
    with pytest.raises(ConfigError, match="reinforce"):
        load_training_config(_write(tmp_path, raw))


def test_non_positive_learning_rates_are_rejected(tmp_path):
    for key in ("lr", "baseline_lr"):
        raw = _raw()
        raw["reinforce"][key] = 0.0
        with pytest.raises(ConfigError, match=key):
            load_training_config(_write(tmp_path, raw))


def test_empty_hidden_layers_is_rejected(tmp_path):
    raw = _raw()
    raw["reinforce"]["hidden_layers"] = []
    with pytest.raises(ConfigError, match="hidden_layers"):
        load_training_config(_write(tmp_path, raw))


def test_unknown_activation_is_rejected(tmp_path):
    raw = _raw()
    raw["reinforce"]["activation"] = "gelu"
    with pytest.raises(ConfigError, match="activation"):
        load_training_config(_write(tmp_path, raw))


def test_reinforce_ablation_block_differs_from_every_other_block(tmp_path):
    """The no-baseline ablation (ROADMAP box 4) must not share alert streams
    with the control, or the variance comparison is confounded by the shifts
    rather than by the baseline (D-027)."""
    for other_section, key in (
        ("reinforce", "train_seed_start"),
        ("dqn", "train_seed_start"),
        ("dqn", "ablation_seed_start"),
    ):
        raw = _raw()
        raw["reinforce"]["ablation_seed_start"] = raw[other_section][key]
        with pytest.raises(ConfigError):
            load_training_config(_write(tmp_path, raw))


def test_clip_experiment_has_its_own_seed_block():
    """E-019's clip sweep is a tuning exercise, so it trains on shifts no
    headline run has seen. If it shared the control's block, a clip value chosen
    here would have been chosen on the same alert streams the reported run uses,
    which is how tuning launders itself into a result (D-016)."""
    reinforce = load_training_config(CONFIG_PATH).reinforce
    assert reinforce.clip_experiment_seed_start >= 100_000


def test_clip_experiment_block_differs_from_every_other_block(tmp_path):
    for other_section, key in (
        ("reinforce", "train_seed_start"),
        ("reinforce", "ablation_seed_start"),
        ("dqn", "train_seed_start"),
        ("dqn", "ablation_seed_start"),
        ("q_learning", "train_seed_start"),
    ):
        raw = _raw()
        raw["reinforce"]["clip_experiment_seed_start"] = raw[other_section][key]
        with pytest.raises(ConfigError):
            load_training_config(_write(tmp_path, raw))


def test_clip_experiment_block_stays_clear_of_the_low_blocks(tmp_path):
    raw = _raw()
    raw["reinforce"]["clip_experiment_seed_start"] = 130
    with pytest.raises(ConfigError, match="clip_experiment_seed_start"):
        load_training_config(_write(tmp_path, raw))
