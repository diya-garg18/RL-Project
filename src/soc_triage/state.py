"""State encoders — the only bridge between the environment and the agent.

Two encodings of the same situation (PROJECT_BRIEF.md §3.2 / §3.3):

  discretise(snapshot) -> int in [0, 576)   for tabular methods
  featurise(snapshot)  -> np.ndarray (17,)  for DQN and policy gradients

GROUND-TRUTH FIREWALL (CONSTRAINTS.md #1): this module reads Alert fields the
agent is allowed to see — severity, age, asset criticality, verify cost, type —
and must NEVER read `is_true_incident` or `deadline_min`, directly or via
proxy. `test_no_ground_truth_leakage` flips those hidden fields and asserts
both encodings are unchanged.
"""

from dataclasses import dataclass

import numpy as np

from soc_triage.alerts import Alert
from soc_triage.config import EnvConfig


@dataclass(frozen=True)
class EnvSnapshot:
    """Everything the agent is allowed to observe, at one decision point.

    Built by the environment after each step. Frozen so encoders (or agents)
    cannot mutate the environment through it.
    """

    queue: tuple[Alert, ...]        # alerts currently waiting (agent may see their visible fields)
    time_now: float                 # minutes into the shift
    shift_length: float             # total shift minutes (480)
    alerts_handled: int             # investigations completed so far
    incidents_confirmed: int        # of those, how many turned out real (known AFTER investigating — not leakage)


def bucket(value: float, boundaries: tuple[float, ...]) -> int:
    """Map a value to a bucket index given ascending boundaries.

    N boundaries define N+1 buckets: [0, b0) -> 0, [b0, b1) -> 1, ... [bN, inf) -> N.
    One shared helper so every discretisation uses the identical convention
    (HANDOVER warned: an off-by-one here silently corrupts all 576 states).
    """
    index = 0
    for boundary in boundaries:
        if value >= boundary:
            index += 1
    return index


def discretise(snap: EnvSnapshot, cfg: EnvConfig) -> int:
    """Encode the situation as one integer state id in [0, 576).

    The five features and their bucket counts (brief §3.2):
      max_severity_in_queue   4   (empty queue -> 0)
      queue_len               4   boundaries [10, 40, 100]
      oldest_age              4   boundaries [30, 90, 180] minutes
      time_left               3   boundaries [60, 240]  (0 = <60 min left — the crunch)
      max_asset_criticality   3   (empty queue -> 0)

    Mixed-radix packing, exactly like reading a 5-digit number where each digit
    has its own base: id = ((((sev*4 + qlen)*4 + age)*3 + tleft)*3 + crit.
    """
    if snap.queue:
        max_severity = 0
        max_criticality = 0
        oldest_age = 0.0
        for alert in snap.queue:
            if alert.severity > max_severity:
                max_severity = alert.severity
            if alert.asset_criticality > max_criticality:
                max_criticality = alert.asset_criticality
            age = snap.time_now - alert.arrival_time
            if age > oldest_age:
                oldest_age = age
    else:
        # Empty-queue convention: all queue-derived features take their lowest bucket.
        max_severity = 0
        max_criticality = 0
        oldest_age = 0.0

    queue_len_bucket = bucket(float(len(snap.queue)), cfg.state_buckets.queue_len)
    age_bucket = bucket(oldest_age, cfg.state_buckets.oldest_age)
    time_left = snap.shift_length - snap.time_now
    time_left_bucket = bucket(time_left, cfg.state_buckets.time_left)

    state_id = max_severity
    state_id = state_id * 4 + queue_len_bucket
    state_id = state_id * 4 + age_bucket
    state_id = state_id * 3 + time_left_bucket
    state_id = state_id * 3 + max_criticality
    return state_id


N_STATES = 4 * 4 * 4 * 3 * 3  # 576 — must match the packing in discretise()

# featurise() layout — index: meaning. Kept as a module constant so DQN code
# and any debugging tooling agree on what each column is.
FEATURE_NAMES: tuple[str, ...] = (
    "queue_len",
    "mean_severity",
    "max_severity",
    "mean_age_min",
    "max_age_min",
    "mean_asset_criticality",
    "max_asset_criticality",
    "mean_verify_cost_min",
    "frac_type_0",
    "frac_type_1",
    "frac_type_2",
    "frac_type_3",
    "frac_type_4",
    "frac_type_5",
    "time_left_norm",
    "alerts_handled",
    "incidents_confirmed",
)


def feature_scale_vector(scales: tuple[tuple[str, float], ...]) -> np.ndarray:
    """Order config's (name, divisor) pairs into a vector matching FEATURE_NAMES.

    The ordering check lives here rather than in config.py because this module
    owns FEATURE_NAMES, and because config.py cannot import state.py without a
    cycle (state.py imports EnvConfig).

    Silently accepting a partial mapping would leave one column unscaled — the
    network would still train and the bug would surface only as a worse result,
    so both a missing and an unknown name raise.
    """
    provided = dict(scales)
    missing = [name for name in FEATURE_NAMES if name not in provided]
    unknown = [name for name in provided if name not in FEATURE_NAMES]
    if missing or unknown:
        raise ValueError(
            f"feature_scales must name every column exactly once; "
            f"missing {missing}, unknown {unknown}"
        )
    return np.array([provided[name] for name in FEATURE_NAMES], dtype=np.float64)


def featurise(snap: EnvSnapshot, cfg: EnvConfig) -> np.ndarray:
    """Encode the situation as a 17-float vector for function approximation.

    Same information as discretise() plus queue composition detail the buckets
    throw away (brief §3.3 — the honest motivation for DQN). Empty queue: all
    queue statistics are 0.
    """
    type_index = {at.name: i for i, at in enumerate(cfg.alert_types)}
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    n = len(snap.queue)
    features[0] = float(n)
    if n > 0:
        severities = []
        ages = []
        criticalities = []
        costs = []
        type_counts = [0] * len(cfg.alert_types)
        for alert in snap.queue:
            severities.append(alert.severity)
            ages.append(snap.time_now - alert.arrival_time)
            criticalities.append(alert.asset_criticality)
            costs.append(alert.verify_cost_min)
            type_counts[type_index[alert.alert_type]] += 1
        features[1] = float(np.mean(severities))
        features[2] = float(np.max(severities))
        features[3] = float(np.mean(ages))
        features[4] = float(np.max(ages))
        features[5] = float(np.mean(criticalities))
        features[6] = float(np.max(criticalities))
        features[7] = float(np.mean(costs))
        for i, count in enumerate(type_counts):
            features[8 + i] = count / n

    features[14] = (snap.shift_length - snap.time_now) / snap.shift_length
    features[15] = float(snap.alerts_handled)
    features[16] = float(snap.incidents_confirmed)
    return features
