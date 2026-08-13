"""Evaluation metrics over EpisodeRecords (PROJECT_BRIEF.md §8).

Consumes outcome dicts the runner produced; trains nothing, mutates nothing.

Metrics:
  MTTD              mean time-to-detect over caught incidents (minutes; lower better)
  recall@deadline   caught-before-deadline / total incidents (higher better)
  wasted_minutes    analyst time burned on false positives (lower better)
  critical_misses   crown-jewel (criticality 2) incidents missed (lower better)
  composite_cost    rupees, under the stated cost assumptions in config
                    metrics.composite_cost_inr (lower better)
"""

import numpy as np

from soc_triage.config import EnvConfig


def episode_metrics(record: dict, cfg: EnvConfig) -> dict:
    """The five headline metrics for one EpisodeRecord."""
    outcome = record["outcome"]

    total = outcome["incidents_total"]
    recall = outcome["incidents_caught_in_time"] / total if total > 0 else 1.0

    # Composite cost = breach cost per missed incident (by its exact asset tier)
    #                + wasted analyst minutes + dwell-delay cost on caught incidents.
    # All three rupee rates are stated assumptions from config, not learned values.
    cost_cfg = cfg.composite_cost
    missed_cost = 0.0
    for tier, n_missed in enumerate(outcome["missed_by_criticality"]):
        missed_cost += n_missed * cost_cfg.missed_incident_by_criticality[tier]

    delay_cost = 0.0
    if outcome["mttd_min"] is not None:
        delay_cost = (
            outcome["mttd_min"] * outcome["incidents_caught"] * cost_cfg.detection_delay_per_min
        )

    composite = (
        missed_cost
        + outcome["wasted_minutes"] * cost_cfg.wasted_analyst_minute
        + delay_cost
    )

    return {
        "mttd_min": outcome["mttd_min"],
        "recall_at_deadline": recall,
        "wasted_minutes": outcome["wasted_minutes"],
        "critical_misses": outcome["critical_missed"],
        "composite_cost_inr": composite,
        "total_reward": outcome["total_reward"],
    }


METRIC_NAMES = (
    "mttd_min",
    "recall_at_deadline",
    "wasted_minutes",
    "critical_misses",
    "composite_cost_inr",
    "total_reward",
)


def summarise(records: list[dict], cfg: EnvConfig) -> dict:
    """mean ± std of each metric over a list of episodes (one agent, many seeds).

    MTTD is averaged only over episodes that caught at least one incident; the
    count of episodes excluded that way is reported as mttd_undefined_episodes.
    """
    per_episode = [episode_metrics(r, cfg) for r in records]
    summary: dict = {"n_episodes": len(per_episode)}

    for name in METRIC_NAMES:
        values = [m[name] for m in per_episode if m[name] is not None]
        if values:
            summary[name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        else:
            summary[name] = {"mean": None, "std": None}
    summary["mttd_undefined_episodes"] = sum(1 for m in per_episode if m["mttd_min"] is None)
    return summary
