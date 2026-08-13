"""The Alert dataclass — the atomic unit everything else moves around.

Contract defined in ARCHITECTURE.md §4. Changing any field here is a
DECISIONS.md-worthy event, because the generator, environment, state encoders,
and evaluation all consume this shape.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Alert:
    """One security alert as it lands in the SOC queue.

    Frozen: an alert is a fact about the world; nothing may rewrite it after
    generation. The queue changes, alerts do not.

    GROUND-TRUTH WARNING (CONSTRAINTS.md #1): `is_true_incident` and
    `deadline_min` exist so the *environment* can compute rewards and detect
    missed deadlines. They must NEVER be encoded into an agent observation,
    directly or via proxy. `state.py` is the enforcement point and
    `test_no_ground_truth_leakage` is the tripwire. The only reader allowed to
    act on them is the `oracle` baseline, which exists to be an upper bound.
    """

    id: int                      # unique per shift; ties broken by id for reproducibility
    arrival_time: float          # minutes into the shift
    severity: int                # 0..3 — vendor label, deliberately noisy (brief §4.2)
    asset_criticality: int       # 0=dev box, 1=standard, 2=crown jewel
    verify_cost_min: int         # analyst minutes to investigate: 5 | 10 | 20 | 40
    alert_type: str              # one of the 6 names in config alert_types
    is_true_incident: bool       # HIDDEN from the agent — see class docstring
    deadline_min: float          # dwell budget; only meaningful when is_true_incident
