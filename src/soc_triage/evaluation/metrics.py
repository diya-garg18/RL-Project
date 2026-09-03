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

# CONSTRAINTS #3: "Never report a single run. Every headline number is mean ± std
# over at least 5 seeds." This is a protocol floor rather than a tunable, which is
# why it lives beside the metrics instead of in config/ — the same reasoning that
# puts MIN_EVAL_SEEDS in tests/test_eval_protocol.py. Trainers cite it when they
# refuse to present a one-run aggregate as a result (docs/bugs/BUG_004).
MIN_RUNS_TO_REPORT = 5


def across_runs_summary(per_run_summaries: list[dict]) -> dict[str, dict]:
    """Mean and std ACROSS runs of each run's mean — never a single run.

    CONSTRAINTS #3. The convention matches `train.py` and `train_dqn.py`
    deliberately, so a Phase 4 number and a Phase 2 number mean the same thing:
    each run contributes ONE value (its own mean over the eval seeds), and the
    std reported is the spread of those values. A std computed over the pooled
    episodes instead would describe seed difficulty and would be much smaller —
    the mistake BUG_004 shipped.

    A run's mean is None when the metric is undefined for it: `summarise`
    reports mttd_min as None when no episode caught an incident, which is the
    honest answer rather than zero. Those runs are dropped from that metric's
    average, and the metric reads undefined if none survive. A policy that
    catches nothing is a real Phase 4 outcome — E-020's collapsed actor-critic
    runs scored recall 0.0000 — so this path is load-bearing, not padding.
    """
    aggregated: dict[str, dict] = {}
    for metric, value in per_run_summaries[0].items():
        if not isinstance(value, dict):
            continue
        means = [s[metric]["mean"] for s in per_run_summaries if s[metric]["mean"] is not None]
        if not means:
            aggregated[metric] = {"mean": None, "std": None, "n_runs": 0}
        else:
            aggregated[metric] = {"mean": float(np.mean(means)),
                                  "std": float(np.std(means)),
                                  "n_runs": len(means)}
    return aggregated


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
