"""Load and validate config/env_default.yaml into typed, frozen objects.

Design rule (CONSTRAINTS.md #9): every tunable number lives in YAML, none here.
This module's job is to fail loudly and specifically the moment the YAML is
missing a key or contains a nonsensical value — a config error caught at load
time is a one-line fix; the same error surfacing mid-training is an evening lost.

The dataclasses are frozen so no code can silently mutate configuration after
startup. If a value needs to change, change the YAML and rerun.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from soc_triage.config.validation import (
    ConfigError,
    _check_ascending,
    _check_prior,
    _require,
)

# ---------------------------------------------------------------------------
# One frozen dataclass per YAML section. Field names mirror the YAML keys
# exactly, so the YAML file itself doubles as documentation of these shapes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShiftConfig:
    length_min: float
    empty_queue_wait_min: float


@dataclass(frozen=True)
class ArrivalsConfig:
    rate_per_min: float  # Poisson lambda


@dataclass(frozen=True)
class IncidentConfig:
    base_rate: float
    target_severity_corr: tuple[float, float]  # (low, high) acceptance band
    severity_lift: tuple[float, ...]           # P(true) multiplier per severity level
    asset_lift: tuple[float, ...]              # P(true) multiplier per criticality level
    dwell_deadline_low_min: float
    dwell_deadline_high_min: float


@dataclass(frozen=True)
class SeverityConfig:
    levels: tuple[int, ...]
    prior: tuple[float, ...]


@dataclass(frozen=True)
class AssetCriticalityConfig:
    levels: tuple[int, ...]
    prior: tuple[float, ...]
    reward_multiplier: tuple[float, ...]


@dataclass(frozen=True)
class VerifyCostConfig:
    options: tuple[int, ...]
    prior: tuple[float, ...]


@dataclass(frozen=True)
class AlertTypeConfig:
    name: str
    prior: float
    incident_lift: float  # multiplies incident.base_rate for this type


@dataclass(frozen=True)
class StateBucketsConfig:
    # Each tuple holds ascending boundaries; N boundaries define N+1 buckets.
    queue_len: tuple[float, ...]
    oldest_age: tuple[float, ...]
    time_left: tuple[float, ...]


@dataclass(frozen=True)
class BulkCloseConfig:
    max_alerts: int
    time_cost_min: float
    max_severity: int
    max_asset_criticality: int


@dataclass(frozen=True)
class ActionsConfig:
    names: tuple[str, ...]  # order defines action indices 0..4
    bulk_close: BulkCloseConfig


@dataclass(frozen=True)
class RewardConfig:
    true_incident_base: float
    true_incident_decay_min: float
    false_positive_per_min: float
    bulk_close_fp: float
    bulk_close_true_incident: float
    end_of_shift_missed: float


@dataclass(frozen=True)
class CompositeCostConfig:
    # Evaluation-only rupee assumptions for the composite metric (brief §8).
    missed_incident_by_criticality: tuple[float, ...]
    wasted_analyst_minute: float
    detection_delay_per_min: float


@dataclass(frozen=True)
class SeedsConfig:
    train: tuple[int, ...]
    eval: tuple[int, ...]


@dataclass(frozen=True)
class EnvConfig:
    shift: ShiftConfig
    arrivals: ArrivalsConfig
    incident: IncidentConfig
    severity: SeverityConfig
    asset_criticality: AssetCriticalityConfig
    verify_cost_min: VerifyCostConfig
    alert_types: tuple[AlertTypeConfig, ...]
    state_buckets: StateBucketsConfig
    actions: ActionsConfig
    reward: RewardConfig
    composite_cost: CompositeCostConfig
    seeds: SeedsConfig



def load_env_config(path: str | Path) -> EnvConfig:
    """Read the environment YAML, validate every section, return typed config.

    Raises ConfigError (with the offending dotted key path) on any missing key,
    malformed prior, non-ascending bucket boundary, or train/eval seed overlap.
    """
    raw_text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} did not parse to a mapping")

    shift_raw = _require(raw, "shift", "env")
    shift = ShiftConfig(
        length_min=float(_require(shift_raw, "length_min", "shift")),
        empty_queue_wait_min=float(_require(shift_raw, "empty_queue_wait_min", "shift")),
    )

    arrivals_raw = _require(raw, "arrivals", "env")
    arrivals = ArrivalsConfig(
        rate_per_min=float(_require(arrivals_raw, "rate_per_min", "arrivals")),
    )

    incident_raw = _require(raw, "incident", "env")
    corr_band = tuple(_require(incident_raw, "target_severity_corr", "incident"))
    dwell_raw = _require(incident_raw, "dwell_deadline_min", "incident")
    incident = IncidentConfig(
        base_rate=float(_require(incident_raw, "base_rate", "incident")),
        target_severity_corr=(float(corr_band[0]), float(corr_band[1])),
        severity_lift=tuple(_require(incident_raw, "severity_lift", "incident")),
        asset_lift=tuple(_require(incident_raw, "asset_lift", "incident")),
        dwell_deadline_low_min=float(_require(dwell_raw, "low", "incident.dwell_deadline_min")),
        dwell_deadline_high_min=float(_require(dwell_raw, "high", "incident.dwell_deadline_min")),
    )

    severity_raw = _require(raw, "severity", "env")
    severity = SeverityConfig(
        levels=tuple(_require(severity_raw, "levels", "severity")),
        prior=tuple(_require(severity_raw, "prior", "severity")),
    )

    asset_raw = _require(raw, "asset_criticality", "env")
    asset_criticality = AssetCriticalityConfig(
        levels=tuple(_require(asset_raw, "levels", "asset_criticality")),
        prior=tuple(_require(asset_raw, "prior", "asset_criticality")),
        reward_multiplier=tuple(_require(asset_raw, "reward_multiplier", "asset_criticality")),
    )

    verify_raw = _require(raw, "verify_cost_min", "env")
    verify_cost = VerifyCostConfig(
        options=tuple(_require(verify_raw, "options", "verify_cost_min")),
        prior=tuple(_require(verify_raw, "prior", "verify_cost_min")),
    )

    types_raw = _require(raw, "alert_types", "env")
    alert_types = tuple(
        AlertTypeConfig(
            name=str(_require(t, "name", f"alert_types[{i}]")),
            prior=float(_require(t, "prior", f"alert_types[{i}]")),
            incident_lift=float(_require(t, "incident_lift", f"alert_types[{i}]")),
        )
        for i, t in enumerate(types_raw)
    )

    buckets_raw = _require(raw, "state_buckets", "env")
    state_buckets = StateBucketsConfig(
        queue_len=tuple(_require(buckets_raw, "queue_len", "state_buckets")),
        oldest_age=tuple(_require(buckets_raw, "oldest_age", "state_buckets")),
        time_left=tuple(_require(buckets_raw, "time_left", "state_buckets")),
    )

    actions_raw = _require(raw, "actions", "env")
    bulk_raw = _require(actions_raw, "bulk_close", "actions")
    actions = ActionsConfig(
        names=tuple(_require(actions_raw, "names", "actions")),
        bulk_close=BulkCloseConfig(
            max_alerts=int(_require(bulk_raw, "max_alerts", "actions.bulk_close")),
            time_cost_min=float(_require(bulk_raw, "time_cost_min", "actions.bulk_close")),
            max_severity=int(_require(bulk_raw, "max_severity", "actions.bulk_close")),
            max_asset_criticality=int(
                _require(bulk_raw, "max_asset_criticality", "actions.bulk_close")
            ),
        ),
    )

    reward_raw = _require(raw, "reward", "env")
    reward = RewardConfig(
        true_incident_base=float(_require(reward_raw, "true_incident_base", "reward")),
        true_incident_decay_min=float(_require(reward_raw, "true_incident_decay_min", "reward")),
        false_positive_per_min=float(_require(reward_raw, "false_positive_per_min", "reward")),
        bulk_close_fp=float(_require(reward_raw, "bulk_close_fp", "reward")),
        bulk_close_true_incident=float(
            _require(reward_raw, "bulk_close_true_incident", "reward")
        ),
        end_of_shift_missed=float(_require(reward_raw, "end_of_shift_missed", "reward")),
    )

    metrics_raw = _require(raw, "metrics", "env")
    composite_raw = _require(metrics_raw, "composite_cost_inr", "metrics")
    composite_cost = CompositeCostConfig(
        missed_incident_by_criticality=tuple(
            _require(composite_raw, "missed_incident_by_criticality", "metrics.composite_cost_inr")
        ),
        wasted_analyst_minute=float(
            _require(composite_raw, "wasted_analyst_minute", "metrics.composite_cost_inr")
        ),
        detection_delay_per_min=float(
            _require(composite_raw, "detection_delay_per_min", "metrics.composite_cost_inr")
        ),
    )

    seeds_raw = _require(raw, "seeds", "env")
    seeds = SeedsConfig(
        train=tuple(_require(seeds_raw, "train", "seeds")),
        eval=tuple(_require(seeds_raw, "eval", "seeds")),
    )

    config = EnvConfig(
        shift=shift,
        arrivals=arrivals,
        incident=incident,
        severity=severity,
        asset_criticality=asset_criticality,
        verify_cost_min=verify_cost,
        alert_types=alert_types,
        state_buckets=state_buckets,
        actions=actions,
        reward=reward,
        composite_cost=composite_cost,
        seeds=seeds,
    )
    _validate(config)
    return config


def _validate(cfg: EnvConfig) -> None:
    """Cross-field sanity checks. Anything that fails here is a config bug."""
    if cfg.shift.length_min <= 0:
        raise ConfigError("'shift.length_min' must be positive")
    if cfg.arrivals.rate_per_min <= 0:
        raise ConfigError("'arrivals.rate_per_min' must be positive")
    if not 0.0 < cfg.incident.base_rate < 1.0:
        raise ConfigError("'incident.base_rate' must be a probability in (0, 1)")

    lo, hi = cfg.incident.target_severity_corr
    if not 0.0 <= lo < hi <= 1.0:
        raise ConfigError(
            f"'incident.target_severity_corr' must be an ascending band in [0, 1], got ({lo}, {hi})"
        )

    if len(cfg.incident.severity_lift) != len(cfg.severity.levels):
        raise ConfigError("'incident.severity_lift' length must match severity.levels")
    if len(cfg.incident.asset_lift) != len(cfg.asset_criticality.levels):
        raise ConfigError("'incident.asset_lift' length must match asset_criticality.levels")
    if any(x < 0 for x in cfg.incident.severity_lift + cfg.incident.asset_lift):
        raise ConfigError("incident lift values must be non-negative")

    _check_prior(cfg.severity.prior, len(cfg.severity.levels), "severity.prior")
    _check_prior(
        cfg.asset_criticality.prior,
        len(cfg.asset_criticality.levels),
        "asset_criticality.prior",
    )
    if len(cfg.asset_criticality.reward_multiplier) != len(cfg.asset_criticality.levels):
        raise ConfigError("'asset_criticality.reward_multiplier' length must match levels")
    _check_prior(cfg.verify_cost_min.prior, len(cfg.verify_cost_min.options), "verify_cost_min.prior")

    type_priors = tuple(t.prior for t in cfg.alert_types)
    _check_prior(type_priors, len(cfg.alert_types), "alert_types[*].prior")
    if any(t.incident_lift < 0 for t in cfg.alert_types):
        raise ConfigError("'alert_types[*].incident_lift' must be non-negative")

    _check_ascending(cfg.state_buckets.queue_len, "state_buckets.queue_len")
    _check_ascending(cfg.state_buckets.oldest_age, "state_buckets.oldest_age")
    _check_ascending(cfg.state_buckets.time_left, "state_buckets.time_left")

    if len(cfg.actions.names) != 5:
        raise ConfigError(f"exactly 5 actions required, got {len(cfg.actions.names)}")

    if len(cfg.composite_cost.missed_incident_by_criticality) != len(cfg.asset_criticality.levels):
        raise ConfigError(
            "'metrics.composite_cost_inr.missed_incident_by_criticality' length must match asset levels"
        )

    # CONSTRAINTS.md #2: train/eval seed separation is enforced here, in code,
    # not by convention. Overlapping seeds refuse to load at all.
    overlap = set(cfg.seeds.train) & set(cfg.seeds.eval)
    if overlap:
        raise ConfigError(f"train and eval seeds must be disjoint; both contain {sorted(overlap)}")
