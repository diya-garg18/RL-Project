"""The actor-critic config section fails loudly at load time, not mid-training.

Same principle as tests/test_dqn_config.py and tests/test_reinforce_config.py:
every check here prevents a failure that is silent at runtime. A zero critic
learning rate leaves the critic at its initialisation forever, which makes every
TD error a constant offset and reads as "actor-critic does not work on this
environment" rather than as a typo.

One check here has no counterpart in the other two files. `entropy_coef` is
allowed to be **zero**, because zero is the textbook algorithm — S&B §13.5 has
no entropy term at all. The bonus is this project's addition (E-018 found
REINFORCE collapsing to a single action with nothing resisting it, and E-019
found that collapse is not caused by the gradient clip), so the config must be
able to express "run the textbook version" without being wrong.
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


def test_actor_critic_section_loads():
    actor_critic = load_training_config(CONFIG_PATH).actor_critic
    assert actor_critic.hidden_layers == (128, 128)
    assert actor_critic.actor_lr > 0
    assert actor_critic.critic_lr > 0
    assert actor_critic.entropy_coef >= 0
    assert actor_critic.grad_clip_norm > 0


def test_the_networks_match_the_other_function_approximators():
    """D-032's reasoning applied to capacity rather than to scaling. The Phase 4
    sample-efficiency comparison is between ALGORITHMS, so the DQN, REINFORCE and
    actor-critic carry the same hidden layers; differing here would make the plot
    a comparison of network size wearing three algorithm names."""
    tcfg = load_training_config(CONFIG_PATH)
    assert tcfg.actor_critic.hidden_layers == tcfg.reinforce.hidden_layers
    assert tcfg.actor_critic.hidden_layers == tcfg.dqn.hidden_layers


def test_actor_critic_has_no_epsilon_of_its_own():
    """Same structural point as REINFORCE: a policy-gradient method explores by
    sampling its own policy. An epsilon key here would mean someone has bolted
    epsilon-greedy onto an algorithm that does not have one."""
    assert "epsilon" not in _raw()["actor_critic"]


def test_the_critic_learns_faster_than_the_actor():
    """Not a taste bound. The actor's update is scaled by the TD error the critic
    produces, so an actor moving faster than its critic is chasing an estimate
    that has not caught up — the same reasoning that sets reinforce.baseline_lr
    above reinforce.lr."""
    actor_critic = load_training_config(CONFIG_PATH).actor_critic
    assert actor_critic.critic_lr > actor_critic.actor_lr


def test_zero_entropy_coefficient_is_allowed(tmp_path):
    """Zero is S&B §13.5 exactly. The loader must not reject the textbook."""
    raw = _raw()
    raw["actor_critic"]["entropy_coef"] = 0.0
    assert load_training_config(_write(tmp_path, raw)).actor_critic.entropy_coef == 0.0


def test_negative_entropy_coefficient_is_rejected(tmp_path):
    """A negative coefficient does not error — it rewards CERTAINTY, driving the
    policy deterministic as fast as it can and deleting exploration. It trains,
    it logs a curve, and it does the opposite of what the key is named for."""
    raw = _raw()
    raw["actor_critic"]["entropy_coef"] = -0.01
    with pytest.raises(ConfigError, match="entropy_coef"):
        load_training_config(_write(tmp_path, raw))


def test_non_positive_learning_rates_are_rejected(tmp_path):
    for key in ("actor_lr", "critic_lr"):
        raw = _raw()
        raw["actor_critic"][key] = 0.0
        with pytest.raises(ConfigError, match=key):
            load_training_config(_write(tmp_path, raw))


def test_empty_hidden_layers_is_rejected(tmp_path):
    raw = _raw()
    raw["actor_critic"]["hidden_layers"] = []
    with pytest.raises(ConfigError, match="hidden_layers"):
        load_training_config(_write(tmp_path, raw))


def test_unknown_activation_is_rejected(tmp_path):
    raw = _raw()
    raw["actor_critic"]["activation"] = "gelu"
    with pytest.raises(ConfigError, match="activation"):
        load_training_config(_write(tmp_path, raw))


def test_actor_critic_seed_block_is_distinct_from_every_other_block(tmp_path):
    for other_section, key in (
        ("q_learning", "train_seed_start"),
        ("sarsa", "train_seed_start"),
        ("monte_carlo", "train_seed_start"),
        ("dqn", "train_seed_start"),
        ("dqn", "ablation_seed_start"),
        ("reinforce", "train_seed_start"),
        ("reinforce", "ablation_seed_start"),
        ("reinforce", "clip_experiment_seed_start"),
    ):
        raw = _raw()
        raw["actor_critic"]["train_seed_start"] = raw[other_section][key]
        with pytest.raises(ConfigError):
            load_training_config(_write(tmp_path, raw))


def test_actor_critic_seed_block_stays_clear_of_the_low_blocks(tmp_path):
    raw = _raw()
    raw["actor_critic"]["train_seed_start"] = 120
    with pytest.raises(ConfigError, match="actor_critic"):
        load_training_config(_write(tmp_path, raw))
