"""`across_runs_summary` — the aggregation CONSTRAINTS #3 is actually about.

BUG_004 shipped because nobody checked what the std beside a Phase 4 number
described. These tests pin the distinction with arithmetic worked out by hand
rather than by running the function and recording what it said.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.evaluation.metrics import (  # noqa: E402
    MIN_RUNS_TO_REPORT,
    across_runs_summary,
)


def _summary(recall: float, mttd: float | None) -> dict:
    """One run's summary, in the shape `summarise` produces."""
    return {
        "recall_at_deadline": {"mean": recall, "std": 0.0},
        "mttd_min": {"mean": mttd, "std": 0.0},
        "n_episodes": 30,  # a non-dict entry, which must be ignored
    }


def test_the_mean_and_std_are_over_RUNS_not_over_pooled_episodes():
    """The BUG_004 distinction, with the arithmetic done by hand.

    Three runs whose means are 0.2, 0.5 and 0.8. Mean = 1.5/3 = 0.5. Population
    std = sqrt(((0.3)^2 + 0 + (0.3)^2)/3) = sqrt(0.18/3) = sqrt(0.06)
        = 0.244948974968...
    Each run's OWN std is 0.0 here, so a function that pooled episodes instead
    would report 0.0 and look far more precise than the evidence allows.
    """
    per_run = [_summary(0.2, 10.0), _summary(0.5, 20.0), _summary(0.8, 30.0)]
    result = across_runs_summary(per_run)

    assert result["recall_at_deadline"]["mean"] == pytest.approx(0.5)
    assert result["recall_at_deadline"]["std"] == pytest.approx(np.sqrt(0.06))
    assert result["recall_at_deadline"]["std"] > 0.0, (
        "a zero std here means episodes were pooled instead of runs -- BUG_004"
    )
    assert result["recall_at_deadline"]["n_runs"] == 3


def test_a_single_run_reports_zero_spread_and_says_it_is_one_run():
    """Not an error — `--only-repeat` is legitimate (D-027) — but the n_runs
    field has to make the one-run-ness visible, because a std of 0.00 beside a
    mean reads as precision rather than as absence of evidence."""
    result = across_runs_summary([_summary(0.42, 12.0)])
    assert result["recall_at_deadline"]["mean"] == pytest.approx(0.42)
    assert result["recall_at_deadline"]["std"] == 0.0
    assert result["recall_at_deadline"]["n_runs"] == 1
    assert result["recall_at_deadline"]["n_runs"] < MIN_RUNS_TO_REPORT


def test_runs_with_an_undefined_metric_are_dropped_from_that_metric_only():
    """`summarise` reports mttd_min as None when a run caught no incidents.

    Averaging None as zero would invent a perfect detection time for a policy
    that detected nothing — and E-020's collapsed actor-critic runs are exactly
    that case, so this is a real path. recall is defined on all three runs and
    must still aggregate over three.
    """
    per_run = [_summary(0.0, None), _summary(0.6, 20.0), _summary(0.9, 40.0)]
    result = across_runs_summary(per_run)

    assert result["mttd_min"]["n_runs"] == 2
    assert result["mttd_min"]["mean"] == pytest.approx(30.0)
    assert result["recall_at_deadline"]["n_runs"] == 3
    assert result["recall_at_deadline"]["mean"] == pytest.approx(0.5)


def test_a_metric_undefined_everywhere_is_reported_undefined_not_zero():
    """The whole-collapse case: no run caught anything. The honest answer is
    'undefined', never 0.0, which would read as instantaneous detection."""
    result = across_runs_summary([_summary(0.0, None), _summary(0.0, None)])
    assert result["mttd_min"]["mean"] is None
    assert result["mttd_min"]["std"] is None
    assert result["mttd_min"]["n_runs"] == 0


def test_non_metric_entries_are_ignored_rather_than_crashing():
    """`summarise` payloads carry scalars alongside the metric dicts."""
    result = across_runs_summary([_summary(0.3, 5.0), _summary(0.7, 15.0)])
    assert "n_episodes" not in result
    assert set(result) == {"recall_at_deadline", "mttd_min"}
