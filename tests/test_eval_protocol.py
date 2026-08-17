"""The evaluation protocol itself, enforced in code (D-019).

Every headline number in this project is a mean over the evaluation seeds, so the
evaluation seed block is not a configuration detail — it is the measuring
instrument. E-008 and E-012 showed the original 5-seed block was too small to
resolve the effects being reported: per-seed standard deviation of ±218 reward
for severity-sort against differences of ~100, and hyperparameter sweeps whose
between-config spread was smaller than the spread between repeats of one config.

CONSTRAINTS #3 requires "at least 5 seeds". These tests encode what measurement
subsequently showed that floor is not enough for, and pin the properties the
widened block must keep.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soc_triage.config import load_env_config  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "env_default.yaml"

# Set from measurement, not preference. Severity-sort's per-seed std is ~218
# reward and the effects of interest are ~100, so the standard error of the mean
# needs to be well under that: 218/sqrt(30) ~ 40. Thirty also matches the
# 30-seed diagnostic already used in E-003, so the two are directly comparable.
MIN_EVAL_SEEDS = 30


def test_eval_block_is_large_enough_to_resolve_the_effects_being_reported():
    """5 seeds was too few, and that was a measured fact, not an opinion (E-008).

    With ~218 per-seed reward std, five seeds give a standard error near 100 —
    the same size as the differences between agents the report wants to discuss.
    """
    cfg = load_env_config(CONFIG_PATH)
    assert len(cfg.seeds.eval) >= MIN_EVAL_SEEDS, (
        f"eval block has {len(cfg.seeds.eval)} seeds; measurement in E-008 showed "
        f"at least {MIN_EVAL_SEEDS} are needed to resolve ~100-reward effects"
    )


def test_the_original_five_eval_seeds_are_still_in_the_block():
    """101-105 must remain a SUBSET of the widened block.

    This is what keeps the pre-2026-08-17 results (E-002, E-003, E-004, E-008,
    E-010) coherent rather than orphaned: every old number is a sub-sample of the
    new one, so old and new can be discussed together and the widening can be
    described as adding seeds rather than replacing them. Dropping any of the
    five would silently invalidate that continuity.
    """
    cfg = load_env_config(CONFIG_PATH)
    for seed in (101, 102, 103, 104, 105):
        assert seed in cfg.seeds.eval, f"original eval seed {seed} was dropped"


def test_eval_seeds_are_unique():
    """A repeated seed would be double-counted in every mean and would shrink the
    reported std for free — the exact direction of error nobody would notice."""
    cfg = load_env_config(CONFIG_PATH)
    assert len(set(cfg.seeds.eval)) == len(cfg.seeds.eval)


def test_eval_and_train_blocks_stay_disjoint():
    """CONSTRAINTS #2, re-checked after the widening.

    The loader already enforces this, but widening a seed block is exactly the
    edit most likely to break it, so the property is asserted where someone
    changing the block will see it.
    """
    cfg = load_env_config(CONFIG_PATH)
    assert not (set(cfg.seeds.eval) & set(cfg.seeds.train))


def test_eval_block_avoids_every_other_reserved_seed_range():
    """Calibration (1000-3099), DP estimation (10000-59999) and the learner
    training blocks (200000+) must all stay clear.

    An eval seed colliding with a training shift would mean evaluating on a shift
    the agent had trained on — the single most damaging silent error available in
    this project, and one no test result would reveal.
    """
    cfg = load_env_config(CONFIG_PATH)
    for seed in cfg.seeds.eval:
        assert not (1_000 <= seed <= 3_099), f"eval seed {seed} is in the calibration block"
        assert not (10_000 <= seed <= 59_999), f"eval seed {seed} is in the DP estimation block"
        assert seed < 100_000, f"eval seed {seed} is in a learner training block"
