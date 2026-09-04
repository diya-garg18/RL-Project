"""`rlhf.summary` — turning an EpisodeRecord into something a person can judge.

The load-bearing test in this file is `test_summary_never_shows_a_reward`.
Phase 5 exists because the hand-written reward numbers are invented
(PROJECT_BRIEF §6.1). If a labeller can see the hand reward, the preferences
they express are partly a readback of it, and the Bradley-Terry model trained on
those preferences becomes an expensive re-derivation of `env_default.yaml`.
FEATURE_011 §7 states the rule; this file is where it is enforced.

Fixtures here are built from the real shape observed in
`results/runs/severity_sort-seed101.json` on 2026-09-04, not from the shape the
renderer happens to want.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.rlhf.summary import render_text, summarise_episode  # noqa: E402


def _step(
    action_name: str,
    time_consumed: float,
    reward: float = 0.0,
    alert: dict | None = None,
    was_true_incident: bool | None = None,
    delay_min: float | None = None,
    n_bulk_closed: int = 0,
    bulk_closed_ids: list[int] | None = None,
) -> dict:
    """One step in the shape `runner.run_episode` writes (runner.py:76)."""
    return {
        "state_disc": 151,
        "action": 0,
        "reward": reward,
        "info": {
            "action_name": action_name,
            "alert_investigated": alert,
            "was_true_incident": was_true_incident,
            "delay_min": delay_min,
            "n_bulk_closed": n_bulk_closed,
            "bulk_closed_ids": bulk_closed_ids or [],
            "time_consumed": time_consumed,
            "reward_breakdown": {"false_positive_cost": reward} if reward else {},
        },
    }


def _alert(
    alert_id: int = 1,
    severity: int = 1,
    asset_criticality: int = 1,
    verify_cost_min: int = 10,
    alert_type: str = "phishing_click",
    is_true_incident: bool = False,
) -> dict:
    """An Alert as `runner._alert_to_dict` serialises it — ground truth included."""
    return {
        "id": alert_id,
        "arrival_time": 13.33,
        "severity": severity,
        "asset_criticality": asset_criticality,
        "verify_cost_min": verify_cost_min,
        "alert_type": alert_type,
        "is_true_incident": is_true_incident,
        "deadline_min": 60.0 if is_true_incident else 0.0,
    }


@pytest.fixture
def record() -> dict:
    """A three-step shift: one false positive, one caught incident, one bulk close.

    Times are 10 + 20 + 5 = 35 minutes, so the three timeline rows must start at
    minute 0.0, 10.0 and 30.0 — a step's minute is when it STARTED, which is the
    sum of everything before it, not including itself.
    """
    return {
        "run_id": "sarsa-seed3000004",
        "agent_name": "sarsa",
        "seed": 3000004,
        "config_hash": "abc123def456",
        "steps": [
            _step("PULL_HIGHEST_SEVERITY", 10.0, reward=-10.0,
                  alert=_alert(alert_id=1, severity=1, is_true_incident=False),
                  was_true_incident=False),
            _step("PULL_HIGHEST_SEVERITY", 20.0, reward=340.0,
                  alert=_alert(alert_id=2, severity=3, asset_criticality=2,
                               verify_cost_min=20, alert_type="c2_beacon",
                               is_true_incident=True),
                  was_true_incident=True, delay_min=42.5),
            _step("BULK_CLOSE_LOW", 5.0, reward=-95.0,
                  n_bulk_closed=4, bulk_closed_ids=[7, 8, 9, 10]),
        ],
        "outcome": {
            "incidents_total": 6,
            "incidents_caught": 1,
            "incidents_caught_in_time": 1,
            "incidents_missed": 5,
            "incidents_buried_by_bulk_close": 2,
            "critical_missed": 2,
            "missed_by_criticality": [1, 2, 2],
            "mttd_min": 42.5,
            "wasted_minutes": 10.0,
            "total_reward": 235.0,
        },
    }


# --------------------------------------------------------------------------
# The rule this whole feature turns on
# --------------------------------------------------------------------------

def _walk(node, path="summary"):
    """Yield every (path, key, value) in a nested dict/list structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield f"{path}.{key}", key, value
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield f"{path}[{i}]", None, value
            yield from _walk(value, f"{path}[{i}]")


def test_summary_never_shows_a_reward(record):
    """No key mentions reward, and the total_reward value appears nowhere.

    Both halves matter. Checking only the key names would miss a renderer that
    passed the number through under an innocent name; checking only the value
    would miss an empty `reward_breakdown` dict that a later edit fills in.
    """
    summary = summarise_episode(record)

    offending_keys = [
        path for path, key, _ in _walk(summary)
        if key is not None and "reward" in key.lower()
    ]
    assert offending_keys == [], f"reward-named keys leaked into the summary: {offending_keys}"

    total_reward = record["outcome"]["total_reward"]
    offending_values = [
        path for path, _, value in _walk(summary)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and value == total_reward
    ]
    assert offending_values == [], f"the total reward leaked as a value at: {offending_values}"


def test_rendered_text_never_shows_a_reward(record):
    """The text rendering is a second surface, so it gets its own check."""
    text = render_text(summarise_episode(record))
    assert "reward" not in text.lower()
    assert "235" not in text


# --------------------------------------------------------------------------
# The summary reports; it does not recompute
# --------------------------------------------------------------------------

def test_outcome_counts_are_copied_not_recomputed(record):
    """Every outcome key except total_reward survives, unchanged.

    The environment computed these with full ground truth (env.py:239). A
    renderer that recounted them from `steps` would disagree, because the steps
    do not contain the alerts left unhandled in the queue at the end of the
    shift — FEATURE_011 §7 'Known limitation'.
    """
    summary = summarise_episode(record)
    expected = {k: v for k, v in record["outcome"].items() if k != "total_reward"}
    assert summary["outcome"] == expected


def test_total_reward_is_the_only_outcome_key_dropped(record):
    summary = summarise_episode(record)
    dropped = set(record["outcome"]) - set(summary["outcome"])
    assert dropped == {"total_reward"}


# --------------------------------------------------------------------------
# The timeline
# --------------------------------------------------------------------------

def test_timeline_minutes_are_cumulative_start_times(record):
    """10 + 20 + 5 minute steps start at 0, 10 and 30 — see the fixture docstring."""
    summary = summarise_episode(record)
    assert [row["minute"] for row in summary["timeline"]] == [0.0, 10.0, 30.0]


def test_shift_minutes_is_the_total_time_consumed(record):
    assert summarise_episode(record)["shift_minutes"] == 35.0


def test_investigated_alert_keeps_only_the_four_fields_a_labeller_needs(record):
    """`id` and `deadline_min` are dropped: an opaque id is noise on screen, and
    the deadline is a parameter of the reward we are trying to replace."""
    summary = summarise_episode(record)
    assert set(summary["timeline"][0]["alert"]) == {
        "severity", "asset_criticality", "alert_type", "verify_cost_min",
    }


def test_ground_truth_is_shown_because_labellers_judge_outcomes(record):
    """PROJECT_BRIEF §6.2: 'Ground truth *is* shown to the labeller.'

    This is the one place in the project where showing `is_true_incident` is
    correct rather than a CONSTRAINTS #1 violation — a summary is environment-side
    output for a human, never an agent observation.
    """
    summary = summarise_episode(record)
    assert summary["timeline"][0]["was_true_incident"] is False
    assert summary["timeline"][1]["was_true_incident"] is True


def test_a_step_that_investigated_nothing_has_no_alert(record):
    """The bulk close swallowed four alerts and investigated none of them."""
    row = summarise_episode(record)["timeline"][2]
    assert row["alert"] is None
    assert row["n_bulk_closed"] == 4


# --------------------------------------------------------------------------
# The caught-incident cards
# --------------------------------------------------------------------------

def test_caught_cards_come_from_steps_and_carry_the_delay(record):
    """One incident caught, at minute 10, 42.5 minutes after it arrived."""
    caught = summarise_episode(record)["caught"]
    assert len(caught) == 1
    assert caught[0]["minute"] == 10.0
    assert caught[0]["delay_min"] == 42.5
    assert caught[0]["alert_type"] == "c2_beacon"


def test_caught_card_count_agrees_with_the_outcome_block(record):
    """If these two disagree the summary is lying to the labeller in one of two
    places, and there is no way for them to tell which."""
    summary = summarise_episode(record)
    assert len(summary["caught"]) == summary["outcome"]["incidents_caught"]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def test_summary_carries_run_id_and_config_hash(record):
    """Traceability: a rendered pair must be tied back to the exact run and the
    exact config that produced it (`runner.config_hash`)."""
    summary = summarise_episode(record)
    assert summary["run_id"] == "sarsa-seed3000004"
    assert summary["config_hash"] == "abc123def456"


def test_summary_does_not_carry_the_agent_name(record):
    """Blinding, FEATURE_011 §6. The policy name lives in pairs_key.json only.

    `run_id` does contain it, which is why `pairs.py` must not copy `run_id`
    into `pairs.json` — that is tested in test_rlhf_pairs.py. Here we only pin
    that the summary adds no *separate* name field a UI might render.
    """
    assert "agent_name" not in summarise_episode(record)


def test_render_text_is_readable_and_mentions_the_headline_counts(record):
    text = render_text(summarise_episode(record))
    assert "c2_beacon" in text
    assert "BULK_CLOSE_LOW" in text
    assert "5" in text          # incidents missed
    assert "10.0" in text       # wasted minutes


def test_summarise_does_not_mutate_the_record(record):
    """The caller may be iterating over records it intends to use again."""
    before = record["outcome"].copy()
    summarise_episode(record)
    assert record["outcome"] == before
    assert "total_reward" in record["outcome"]
