"""EpisodeRecord -> EpisodeSummary: what a human labeller actually reads.

FEATURE_011 §7. A summary is a *report* of an episode, not a recomputation of
it: the counts come straight from `record["outcome"]`, which the environment
built with full ground truth (`env.py:239`), and the timeline comes straight
from `record["steps"]`.

The one thing this module removes is the reward. Every reward field — the
per-step `reward`, the `reward_breakdown`, and `outcome["total_reward"]` — is
dropped on the way through, and `tests/test_rlhf_summary.py` fails if any of
them reappears. The reason is the point of Phase 5: the hand-written reward
numbers are invented (PROJECT_BRIEF §3.5, §6.1), and a labeller who can see
them is no longer expressing an independent judgement. A reward model fitted to
preferences contaminated that way would rediscover `config/env_default.yaml`
at considerable expense.

Showing ground truth, by contrast, is *required* here — "Ground truth is shown
to the labeller — they are judging outcomes, not guessing" (PROJECT_BRIEF §6.2).
That is not a CONSTRAINTS #1 problem: #1 forbids ground truth in an **agent
observation**, and a summary is environment-side output for a person.
"""

from typing import Any

# The alert fields a labeller needs in order to judge whether an investigation
# was a good use of the shift. `id` is an opaque number that only adds noise on
# screen, and `deadline_min` is a parameter of the reward we are replacing.
_ALERT_FIELDS = ("severity", "asset_criticality", "alert_type", "verify_cost_min")


def _alert_view(alert: dict | None) -> dict | None:
    """Project an investigated alert down to the fields worth showing."""
    if alert is None:
        return None
    return {field: alert[field] for field in _ALERT_FIELDS}


def summarise_episode(record: dict) -> dict:
    """Render one EpisodeRecord as a labeller-facing summary.

    Returns a plain dict so it serialises straight into `pairs.json` with no
    encoder — the labelling UI is a separate program (Diya's, PROJECT_BRIEF §9)
    and JSON is the whole contract between us.

    The record is not mutated; callers commonly iterate over records they mean
    to use again.
    """
    timeline: list[dict] = []
    caught: list[dict] = []

    # A step's `minute` is when it STARTED, so it is the sum of every earlier
    # step's duration and does not include its own. Written as an explicit
    # running total rather than an itertools.accumulate one-liner
    # (CONSTRAINTS #14 — the readable form is the shipped form).
    elapsed = 0.0
    for step in record["steps"]:
        info = step["info"]
        alert = _alert_view(info["alert_investigated"])

        timeline.append({
            "minute": elapsed,
            "action": info["action_name"],
            "alert": alert,
            "was_true_incident": info["was_true_incident"],
            "n_bulk_closed": info["n_bulk_closed"],
        })

        # A caught incident is an investigation that turned out to be real. The
        # delay is the environment's own figure, not one recomputed here.
        if info["was_true_incident"] and alert is not None:
            caught.append({
                "minute": elapsed,
                "delay_min": info["delay_min"],
                **alert,
            })

        elapsed += info["time_consumed"]

    outcome = {k: v for k, v in record["outcome"].items() if k != "total_reward"}

    return {
        "run_id": record["run_id"],
        "config_hash": record["config_hash"],
        "seed": record["seed"],
        "n_steps": len(record["steps"]),
        "shift_minutes": elapsed,
        "timeline": timeline,
        "caught": caught,
        "outcome": outcome,
    }


def _format_alert(alert: dict | None) -> str:
    if alert is None:
        return ""
    return (f"sev {alert['severity']} · crit {alert['asset_criticality']} · "
            f"{alert['alert_type']} · {alert['verify_cost_min']} min")


def render_text(summary: dict) -> str:
    """A plain-text rendering of a summary.

    This is not the labelling UI — it exists so a pair can be eyeballed from a
    terminal while building the pair set, and so a CLI fallback is possible if
    the web page slips (ROADMAP 5a allows "even a CLI with rendered text").
    """
    lines: list[str] = []
    lines.append(f"SHIFT — {summary['shift_minutes']:.0f} analyst-minutes, "
                 f"{summary['n_steps']} actions")
    lines.append("")
    lines.append("TIMELINE")
    for row in summary["timeline"]:
        parts = [f"  {row['minute']:>6.1f}  {row['action']:<24}"]
        if row["alert"] is not None:
            verdict = "REAL INCIDENT" if row["was_true_incident"] else "false positive"
            parts.append(f"{_format_alert(row['alert'])}  -> {verdict}")
        if row["n_bulk_closed"]:
            parts.append(f"closed {row['n_bulk_closed']} alerts unread")
        lines.append("  ".join(parts).rstrip())

    outcome = summary["outcome"]
    lines.append("")
    lines.append("OUTCOME")
    lines.append(f"  incidents caught   : {outcome['incidents_caught']} "
                 f"of {outcome['incidents_total']} "
                 f"({outcome['incidents_caught_in_time']} before their deadline)")
    for card in summary["caught"]:
        lines.append(f"      caught at {card['minute']:.0f} min, "
                     f"{card['delay_min']:.0f} min after it arrived — "
                     f"{_format_alert(card)}")
    lines.append(f"  incidents missed   : {outcome['incidents_missed']} "
                 f"({outcome['critical_missed']} on crown-jewel assets)")
    lines.append(f"  buried unread      : {outcome['incidents_buried_by_bulk_close']}")
    lines.append(f"  analyst minutes wasted on false positives: "
                 f"{outcome['wasted_minutes']}")
    mttd = outcome["mttd_min"]
    lines.append(f"  mean time to detect: "
                 f"{'n/a — nothing caught' if mttd is None else f'{mttd:.1f} min'}")
    return "\n".join(lines)


def summarise_all(records: list[dict]) -> list[dict]:
    """Summarise a list of records. Explicit loop, per CONSTRAINTS #14."""
    summaries: list[Any] = []
    for record in records:
        summaries.append(summarise_episode(record))
    return summaries
