"""The Phase 5a config section fails at load time, not after generating episodes.

Same principle as `tests/test_actor_critic_config.py`: every check here prevents
a failure that is silent at runtime and expensive to trace. The two that earn
their place most clearly are the seed-block collision — pairs built on a
learner's own training seeds would show labellers exactly the shifts that
learner was fitted to, and nothing would raise — and the capacity arithmetic,
which otherwise surfaces only after 9 policies have been run on 12 seeds.
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


def _load_with(tmp_path: Path, **overrides):
    raw = _raw()
    raw["rlhf"].update(overrides)
    return load_training_config(_write(tmp_path, raw))


# --------------------------------------------------------------------------
# The shipped values
# --------------------------------------------------------------------------

def test_the_shipped_rlhf_section_loads_and_matches_the_brief():
    """PROJECT_BRIEF §6.2 asks for 300 pairs with 50 double-labelled."""
    rlhf = load_training_config(CONFIG_PATH).rlhf
    assert rlhf.target_pairs == 300
    assert rlhf.double_labelled_pairs == 50
    assert rlhf.pair_must_share_seed is True
    assert len(rlhf.policies) == 9


def test_the_shipped_pool_excludes_the_oracle():
    assert "oracle_greedy" not in load_training_config(CONFIG_PATH).rlhf.policies


def test_the_shipped_target_fits_in_the_shipped_capacity():
    """9 policies give 36 unordered pairings; 36 x 12 seeds = 432 >= 300.

    Arithmetic done here rather than trusted: a target above capacity is only
    discoverable after the episodes have been generated.
    """
    rlhf = load_training_config(CONFIG_PATH).rlhf
    n = len(rlhf.policies)
    assert n * (n - 1) // 2 == 36
    assert 36 * rlhf.n_pair_seeds == 432
    assert rlhf.target_pairs <= 432


# --------------------------------------------------------------------------
# Seed blocks
# --------------------------------------------------------------------------

def test_the_pair_seed_block_is_disjoint_from_every_other_block():
    """CONSTRAINTS #2, and the D-016 one-block-per-consumer convention."""
    cfg = load_training_config(CONFIG_PATH)
    others = {
        cfg.q_learning.train_seed_start, cfg.sarsa.train_seed_start,
        cfg.monte_carlo.train_seed_start, cfg.dqn.train_seed_start,
        cfg.reinforce.train_seed_start, cfg.actor_critic.train_seed_start,
        cfg.dqn.ablation_seed_start, cfg.reinforce.ablation_seed_start,
        cfg.reinforce.clip_experiment_seed_start,
        cfg.actor_critic.entropy_experiment_seed_start,
        cfg.dp.estimation_seed_start,
    }
    assert cfg.rlhf.pair_seed_start not in others


def test_a_pair_seed_block_colliding_with_a_training_block_is_refused(tmp_path):
    """The collision whose consequence is least visible: labellers would be
    shown the exact shifts a learner was trained on."""
    q_block = load_training_config(CONFIG_PATH).q_learning.train_seed_start
    with pytest.raises(ConfigError, match="pair_seed_start"):
        _load_with(tmp_path, pair_seed_start=q_block)


def test_a_pair_seed_block_in_the_low_range_is_refused(tmp_path):
    """Below 100000 is where the train, eval, calibration and DP estimation
    blocks live — including eval [101..130]."""
    with pytest.raises(ConfigError, match="pair_seed_start"):
        _load_with(tmp_path, pair_seed_start=101)


# --------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------

def test_a_target_above_capacity_is_refused_with_the_arithmetic(tmp_path):
    """36 pairings x 3 seeds = 108 candidates; 300 cannot be met without
    repeating a comparison."""
    with pytest.raises(ConfigError) as excinfo:
        _load_with(tmp_path, n_pair_seeds=3)
    message = str(excinfo.value)
    assert "108" in message and "300" in message


def test_more_double_labels_than_pairs_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="double_labelled_pairs"):
        _load_with(tmp_path, double_labelled_pairs=301)


def test_fewer_than_two_double_labels_is_refused(tmp_path):
    """Cohen's kappa is undefined below two shared pairs (rlhf/agreement.py).
    A config guaranteeing an undefined kappa guarantees the ROADMAP 5a box
    cannot be ticked, which is a result failure, not a crash."""
    with pytest.raises(ConfigError, match="kappa"):
        _load_with(tmp_path, double_labelled_pairs=1)


def test_a_single_pair_seed_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="n_pair_seeds"):
        _load_with(tmp_path, n_pair_seeds=1)


# --------------------------------------------------------------------------
# The policy pool
# --------------------------------------------------------------------------

def test_including_the_oracle_is_refused(tmp_path):
    """It reads is_true_incident by design, so its pairs are foregone
    conclusions and the Bradley-Terry gradient there is ~0."""
    pool = list(_raw()["rlhf"]["policies"]) + ["oracle_greedy"]
    with pytest.raises(ConfigError, match="oracle_greedy"):
        _load_with(tmp_path, policies=pool)


def test_a_duplicated_policy_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="duplicates"):
        _load_with(tmp_path, policies=["random", "random", "sarsa"])


def test_fewer_than_two_policies_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="at least 2"):
        _load_with(tmp_path, policies=["random"])


# --------------------------------------------------------------------------
# The key that must not be switched off
# --------------------------------------------------------------------------

def test_turning_off_pair_must_share_seed_is_refused(tmp_path):
    """Same shape of guard as `dqn.loss must be huber`. Flipping this raises
    nowhere: it silently produces pairs whose two sides ran different alert
    streams, and every preference collected afterwards is a judgement about
    luck rather than about policy."""
    with pytest.raises(ConfigError, match="pair_must_share_seed"):
        _load_with(tmp_path, pair_must_share_seed=False)


# --------------------------------------------------------------------------
# The section is mandatory
# --------------------------------------------------------------------------

def test_a_config_with_no_rlhf_section_is_refused(tmp_path):
    """A missing section means someone is running Phase 5 against a stale
    config, which is worth failing on rather than defaulting through."""
    raw = _raw()
    del raw["rlhf"]
    with pytest.raises(ConfigError, match="rlhf"):
        load_training_config(_write(tmp_path, raw))


# --------------------------------------------------------------------------
# The labelling page (FEATURE_012, D-040 to D-042)
# --------------------------------------------------------------------------

def test_the_shipped_labeller_ids_are_opaque_and_there_are_two():
    """Two, because kappa needs two. Opaque, because CONSTRAINTS #23 says so."""
    labellers = load_training_config(CONFIG_PATH).rlhf.labellers
    assert labellers == ("L1", "L2")


def test_the_shipped_timer_cap_and_bind_address_load():
    rlhf = load_training_config(CONFIG_PATH).rlhf
    assert rlhf.max_seconds_per_pair == 300
    assert rlhf.ui_host == "127.0.0.1"
    assert rlhf.ui_port == 8000


def test_a_single_labeller_is_refused(tmp_path):
    """One annotator cannot produce an agreement statistic at all."""
    with pytest.raises(ConfigError, match="at least two"):
        _load_with(tmp_path, labellers=["L1"])


def test_duplicate_labeller_ids_are_refused(tmp_path):
    """Two entries for one person would make kappa compare them with themselves."""
    with pytest.raises(ConfigError, match="duplicate"):
        _load_with(tmp_path, labellers=["L1", "L1"])


def test_an_empty_labeller_id_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="labellers"):
        _load_with(tmp_path, labellers=["L1", ""])


def test_the_labellers_must_divide_the_single_label_pairs_evenly(tmp_path):
    """250 singles over 2 labellers is 125 each. Three would not divide.

    Not a hard requirement of the code — the round-robin handles a remainder —
    but an uneven split is worth being told about at load rather than
    discovering in the report that one person did 84 and another 83.
    """
    with pytest.raises(ConfigError, match="evenly"):
        _load_with(tmp_path, labellers=["L1", "L2", "L3"])


def test_a_zero_timer_cap_is_refused(tmp_path):
    """A cap of zero would store None for every answer, losing the column."""
    with pytest.raises(ConfigError, match="max_seconds_per_pair"):
        _load_with(tmp_path, max_seconds_per_pair=0)


def test_a_negative_timer_cap_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="max_seconds_per_pair"):
        _load_with(tmp_path, max_seconds_per_pair=-5)


def test_a_port_outside_the_usable_range_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="ui_port"):
        _load_with(tmp_path, ui_port=70000)


def test_a_privileged_port_is_refused(tmp_path):
    """Nothing about a local labelling page needs a port below 1024."""
    with pytest.raises(ConfigError, match="ui_port"):
        _load_with(tmp_path, ui_port=80)


def test_binding_the_wildcard_address_is_refused(tmp_path):
    """0.0.0.0 would put unlabelled shift data and a writable database of
    irreplaceable labels on the local network, for no benefit at all. Any
    specific host is allowed; the wildcard is not.
    """
    with pytest.raises(ConfigError, match="ui_host"):
        _load_with(tmp_path, ui_host="0.0.0.0")
