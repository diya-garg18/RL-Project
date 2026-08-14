"""Alert stream generator — Poisson arrivals with hidden ground truth.

Produces the full alert list for one 480-minute shift up front. The environment
releases alerts to the agent as the clock passes each arrival time; generating
everything ahead of time keeps the stream a pure function of (config, seed),
which is what makes paired policy comparisons possible (brief §8: same alert
stream across policies).

The truth model (brief §4.2, tunables in config/env_default.yaml):

    P(true incident) = base_rate × type_lift × severity_lift[sev] × asset_lift[crit]

Severity gets only a mild lift — the informative signal deliberately lives in
the *combination* of type, asset and cost. The Phase 0 calibration check tunes
these until incidence ≈ 3% and Pearson r(severity, truth) lands in 0.30–0.40.

Simplification, recorded in DECISIONS.md: no time-of-day modulation (the brief
mentions it as optional colour; it affects neither calibration target).
"""

import numpy as np

from soc_triage.alerts import Alert
from soc_triage.config import EnvConfig

# Probabilities are capped below 1 so no alert is ever *certain* to be an
# incident — even the worst-looking alert can be a false alarm.
_P_TRUE_CAP = 0.95


def generate_shift(cfg: EnvConfig, seed: int) -> list[Alert]:
    """Generate the complete, time-ordered alert stream for one shift.

    Pure function of (cfg, seed): same inputs, same alerts, always. All
    randomness flows through one numpy Generator seeded here.
    """
    rng = np.random.default_rng(seed)

    # Poisson process: exponential(1/λ) gaps between consecutive arrivals
    # (standard queueing theory: the inter-arrival times of a Poisson process
    # are exponentially distributed). Drawn in chunks and cumulative-summed —
    # vectorised because profiling showed per-alert draws were 82% of episode
    # runtime (CONSTRAINTS #14: optimise the simulator, not the algorithms).
    mean_gap = 1.0 / cfg.arrivals.rate_per_min
    chunk = int(cfg.shift.length_min * cfg.arrivals.rate_per_min * 1.5) + 50
    gaps = rng.exponential(mean_gap, size=chunk)
    while gaps.sum() < cfg.shift.length_min:  # astronomically rare undershoot
        gaps = np.concatenate([gaps, rng.exponential(mean_gap, size=chunk)])
    arrivals_all = np.cumsum(gaps)
    arrival_times = arrivals_all[arrivals_all < cfg.shift.length_min]
    n = len(arrival_times)

    # One vectorised draw per feature — same distributions as the per-alert
    # version, so calibration properties are unchanged (re-verified in E-003).
    severities = rng.choice(cfg.severity.levels, size=n, p=cfg.severity.prior)
    criticalities = rng.choice(cfg.asset_criticality.levels, size=n, p=cfg.asset_criticality.prior)
    verify_costs = rng.choice(cfg.verify_cost_min.options, size=n, p=cfg.verify_cost_min.prior)
    type_names = [at.name for at in cfg.alert_types]
    type_priors = [at.prior for at in cfg.alert_types]
    type_lift = {at.name: at.incident_lift for at in cfg.alert_types}
    types = rng.choice(type_names, size=n, p=type_priors)

    # Truth model (D-007): multiplicative lifts on the base rate, capped.
    severity_lift = np.array(cfg.incident.severity_lift)
    asset_lift = np.array(cfg.incident.asset_lift)
    type_lift_arr = np.array([type_lift[t] for t in types])
    p_true = (
        cfg.incident.base_rate
        * type_lift_arr
        * severity_lift[severities]
        * asset_lift[criticalities]
    )
    is_true = rng.random(n) < np.minimum(p_true, _P_TRUE_CAP)

    # Dwell deadline only exists for real incidents; false positives get 0.0
    # (documented as meaningless for them — alerts.py).
    deadlines = np.where(
        is_true,
        rng.uniform(cfg.incident.dwell_deadline_low_min, cfg.incident.dwell_deadline_high_min, size=n),
        0.0,
    )

    alerts: list[Alert] = []
    for alert_id in range(n):
        alerts.append(
            Alert(
                id=alert_id,
                arrival_time=float(arrival_times[alert_id]),
                severity=int(severities[alert_id]),
                asset_criticality=int(criticalities[alert_id]),
                verify_cost_min=int(verify_costs[alert_id]),
                alert_type=str(types[alert_id]),
                is_true_incident=bool(is_true[alert_id]),
                deadline_min=float(deadlines[alert_id]),
            )
        )
    return alerts
