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
    # are exponentially distributed).
    arrival_times: list[float] = []
    t = rng.exponential(1.0 / cfg.arrivals.rate_per_min)
    while t < cfg.shift.length_min:
        arrival_times.append(t)
        t += rng.exponential(1.0 / cfg.arrivals.rate_per_min)

    type_names = [at.name for at in cfg.alert_types]
    type_priors = [at.prior for at in cfg.alert_types]
    type_lift = {at.name: at.incident_lift for at in cfg.alert_types}

    alerts: list[Alert] = []
    for alert_id, arrival in enumerate(arrival_times):
        # Independent draws from the config priors. Realism note: features are
        # sampled independently; only the *truth* depends on their combination.
        severity = int(rng.choice(cfg.severity.levels, p=cfg.severity.prior))
        criticality = int(rng.choice(cfg.asset_criticality.levels, p=cfg.asset_criticality.prior))
        verify_cost = int(rng.choice(cfg.verify_cost_min.options, p=cfg.verify_cost_min.prior))
        alert_type = str(rng.choice(type_names, p=type_priors))

        p_true = (
            cfg.incident.base_rate
            * type_lift[alert_type]
            * cfg.incident.severity_lift[severity]
            * cfg.incident.asset_lift[criticality]
        )
        is_true = bool(rng.random() < min(p_true, _P_TRUE_CAP))

        # Dwell deadline only exists for real incidents. False positives get
        # 0.0 — the field is documented as meaningless for them (alerts.py).
        if is_true:
            deadline = float(
                rng.uniform(cfg.incident.dwell_deadline_low_min, cfg.incident.dwell_deadline_high_min)
            )
        else:
            deadline = 0.0

        alerts.append(
            Alert(
                id=alert_id,
                arrival_time=float(arrival),
                severity=severity,
                asset_criticality=criticality,
                verify_cost_min=verify_cost,
                alert_type=alert_type,
                is_true_incident=is_true,
                deadline_min=deadline,
            )
        )

    return alerts
