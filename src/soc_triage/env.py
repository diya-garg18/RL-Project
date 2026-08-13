"""SOCTriageEnv — the MDP itself (PROJECT_BRIEF.md §3, ARCHITECTURE.md §2).

Gymnasium-style interface (reset/step), interface only — no gymnasium import.
The environment owns the clock, the queue, the hidden ground truth, and the
reward arithmetic. It never knows which agent is acting (CONSTRAINTS.md #10).

Step ordering (FLOW.md Flow A — the documented trap):
  1. resolve the action on the CURRENT queue
  2. advance the clock by the action's time cost
  3. admit alerts whose arrival_time has now passed
  4. build the next observation from the fresh queue
Getting 3 and 4 backwards would show the agent a stale queue.

Reward semantics decided this session (D-009 in DECISIONS.md):
  - detection delay is measured at the moment investigation STARTS
  - the end-of-shift miss penalty applies only to never-investigated true
    incidents whose deadline expired within the shift
  - a bulk-closed true incident is charged once (-150 x mult), not again at shift end
"""

import math

from soc_triage.alerts import Alert
from soc_triage.config import EnvConfig
from soc_triage.generator import generate_shift
from soc_triage.state import EnvSnapshot

# Action indices — order fixed by config actions.names (asserted in __init__).
PULL_HIGHEST_SEVERITY = 0
PULL_OLDEST = 1
PULL_MOST_CRITICAL_ASSET = 2
PULL_CHEAPEST = 3
BULK_CLOSE_LOW_RISK = 4


class SOCTriageEnv:
    """One 8-hour analyst shift as a Markov Decision Process.

    Usage:
        env = SOCTriageEnv(cfg)
        snap = env.reset(seed)
        snap, reward, done, info = env.step(action)
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        expected = (
            "PULL_HIGHEST_SEVERITY",
            "PULL_OLDEST",
            "PULL_MOST_CRITICAL_ASSET",
            "PULL_CHEAPEST",
            "BULK_CLOSE_LOW_RISK",
        )
        if cfg.actions.names != expected:
            raise ValueError(f"action names/order mismatch with env constants: {cfg.actions.names}")
        self._reset_called = False

    # ------------------------------------------------------------------ reset

    def reset(self, seed: int) -> EnvSnapshot:
        """Start a fresh shift: generate the full alert stream, zero the clock."""
        self._pending: list[Alert] = generate_shift(self.cfg, seed)  # sorted by arrival
        self._queue: list[Alert] = []
        self._clock: float = 0.0
        self._alerts_handled: int = 0
        self._incidents_confirmed: int = 0
        self._wasted_minutes: float = 0.0
        # (alert, minutes_into_shift_when_investigation_started) for every investigation
        self._investigated: list[tuple[Alert, float]] = []
        self._bulk_closed: list[Alert] = []
        self._done = False
        self._reset_called = True
        self._admit_arrivals()
        return self._snapshot()

    # ------------------------------------------------------------------- step

    def step(self, action: int) -> tuple[EnvSnapshot, float, bool, dict]:
        """Apply one action. Returns (snapshot, reward, done, info)."""
        if not self._reset_called:
            raise RuntimeError("call reset(seed) before step()")
        if self._done:
            raise RuntimeError("episode is over — call reset(seed)")
        if action not in (0, 1, 2, 3, 4):
            raise ValueError(f"invalid action {action}")

        breakdown: dict[str, float] = {}
        info: dict = {"action_name": self.cfg.actions.names[action]}

        if not self._queue:
            # Empty queue: any action just waits (brief §3.4).
            time_cost = self.cfg.shift.empty_queue_wait_min
            reward = 0.0
            info.update(alert_investigated=None, was_true_incident=None,
                        delay_min=None, bulk_closed=[])
        elif action == BULK_CLOSE_LOW_RISK:
            reward, time_cost = self._do_bulk_close(info, breakdown)
        else:
            reward, time_cost = self._do_investigate(action, info, breakdown)

        # Clock advances, THEN new arrivals are admitted (FLOW.md ordering).
        self._clock += time_cost
        self._admit_arrivals()

        if self._clock >= self.cfg.shift.length_min:
            self._done = True
            reward += self._end_of_shift_penalty(breakdown)

        info["time_consumed"] = time_cost
        info["reward_breakdown"] = breakdown
        return self._snapshot(), reward, self._done, info

    # -------------------------------------------------------- action handlers

    def _do_investigate(self, action: int, info: dict, breakdown: dict) -> tuple[float, float]:
        """Actions 0-3: pick one alert by the chosen rule and verify it."""
        alert = self._select_alert(action)
        self._queue.remove(alert)

        start_time = self._clock  # delay measured at investigation start (D-009)
        self._investigated.append((alert, start_time))
        self._alerts_handled += 1

        if alert.is_true_incident:
            self._incidents_confirmed += 1
            delay = start_time - alert.arrival_time
            multiplier = self.cfg.asset_criticality.reward_multiplier[alert.asset_criticality]
            reward = (
                self.cfg.reward.true_incident_base
                * math.exp(-delay / self.cfg.reward.true_incident_decay_min)
                * multiplier
            )
            breakdown["true_incident_caught"] = reward
            info["delay_min"] = delay
        else:
            self._wasted_minutes += alert.verify_cost_min
            reward = self.cfg.reward.false_positive_per_min * alert.verify_cost_min
            breakdown["false_positive_cost"] = reward
            info["delay_min"] = None

        info["alert_investigated"] = alert
        info["was_true_incident"] = alert.is_true_incident
        info["bulk_closed"] = []
        return reward, float(alert.verify_cost_min)

    def _do_bulk_close(self, info: dict, breakdown: dict) -> tuple[float, float]:
        """Action 4: auto-close up to max_alerts low-severity, low-criticality alerts.

        Eligible = severity <= 1 AND criticality == 0 (config). Oldest first
        (lowest id), so the closure order is deterministic. The deliberate trap
        (brief §3.5): small + rewards per false positive, a large penalty if a
        real incident gets buried.
        """
        rules = self.cfg.actions.bulk_close
        eligible: list[Alert] = []
        for alert in self._queue:  # queue is id-ordered (arrival order)
            if (alert.severity <= rules.max_severity
                    and alert.asset_criticality <= rules.max_asset_criticality):
                eligible.append(alert)
                if len(eligible) == rules.max_alerts:
                    break

        reward = 0.0
        fp_credit = 0.0
        buried_penalty = 0.0
        for alert in eligible:
            self._queue.remove(alert)
            self._bulk_closed.append(alert)
            if alert.is_true_incident:
                multiplier = self.cfg.asset_criticality.reward_multiplier[alert.asset_criticality]
                buried_penalty += self.cfg.reward.bulk_close_true_incident * multiplier
            else:
                fp_credit += self.cfg.reward.bulk_close_fp
        reward = fp_credit + buried_penalty
        if fp_credit:
            breakdown["bulk_close_fp_credit"] = fp_credit
        if buried_penalty:
            breakdown["bulk_close_buried_incident"] = buried_penalty

        info["alert_investigated"] = None
        info["was_true_incident"] = None
        info["delay_min"] = None
        info["bulk_closed"] = list(eligible)
        return reward, rules.time_cost_min

    def _select_alert(self, action: int) -> Alert:
        """The four pull rules. Ties broken by lowest alert id (reproducibility)."""
        best = self._queue[0]
        for alert in self._queue[1:]:
            if action == PULL_HIGHEST_SEVERITY:
                better = alert.severity > best.severity
            elif action == PULL_OLDEST:
                better = alert.arrival_time < best.arrival_time
            elif action == PULL_MOST_CRITICAL_ASSET:
                better = alert.asset_criticality > best.asset_criticality
            else:  # PULL_CHEAPEST
                better = alert.verify_cost_min < best.verify_cost_min
            if better:  # note: equal keeps `best` (earlier id wins — queue is id-ordered)
                best = alert
        return best

    # ------------------------------------------------------------- internals

    def _admit_arrivals(self) -> None:
        """Move pending alerts whose arrival time has passed into the queue."""
        while self._pending and self._pending[0].arrival_time <= self._clock:
            self._queue.append(self._pending.pop(0))

    def _end_of_shift_penalty(self, breakdown: dict) -> float:
        """Charge every never-triaged true incident whose deadline expired in-shift.

        Bulk-closed true incidents were already charged -150x at closure time
        and are NOT charged again here (D-009). Incidents whose dwell budget
        outlives the shift are the next shift's problem — no charge.
        """
        penalty = 0.0
        untriaged = self._queue + self._pending  # pending: arrived too late to ever be seen
        for alert in untriaged:
            if alert.is_true_incident:
                deadline_at = alert.arrival_time + alert.deadline_min
                if deadline_at <= self.cfg.shift.length_min:
                    multiplier = self.cfg.asset_criticality.reward_multiplier[alert.asset_criticality]
                    penalty += self.cfg.reward.end_of_shift_missed * multiplier
        if penalty:
            breakdown["end_of_shift_missed"] = penalty
        return penalty

    def _snapshot(self) -> EnvSnapshot:
        return EnvSnapshot(
            queue=tuple(self._queue),
            time_now=self._clock,
            shift_length=self.cfg.shift.length_min,
            alerts_handled=self._alerts_handled,
            incidents_confirmed=self._incidents_confirmed,
        )

    # ------------------------------------------------- outcome (for the runner)

    def episode_outcome(self) -> dict:
        """Ground-truth summary of the finished episode (EpisodeRecord.outcome).

        Environment-side bookkeeping — this is allowed to see the truth because
        it is evaluation output, never an agent observation.
        """
        if not self._done:
            raise RuntimeError("episode_outcome() only valid after the episode ends")

        all_alerts = ([a for a, _ in self._investigated] + self._bulk_closed
                      + self._queue + self._pending)
        incidents_total = sum(a.is_true_incident for a in all_alerts)

        caught_delays: list[float] = []
        caught_in_time = 0
        for alert, start_time in self._investigated:
            if alert.is_true_incident:
                delay = start_time - alert.arrival_time
                caught_delays.append(delay)
                if start_time <= alert.arrival_time + alert.deadline_min:
                    caught_in_time += 1

        missed = [a for a in self._queue + self._pending if a.is_true_incident]
        buried = [a for a in self._bulk_closed if a.is_true_incident]
        critical_missed = sum(1 for a in missed + buried if a.asset_criticality == 2)

        return {
            "incidents_total": int(incidents_total),
            "incidents_caught": len(caught_delays),
            "incidents_caught_in_time": int(caught_in_time),
            "incidents_missed": len(missed) + len(buried),
            "incidents_buried_by_bulk_close": len(buried),
            "critical_missed": int(critical_missed),
            "mttd_min": float(sum(caught_delays) / len(caught_delays)) if caught_delays else None,
            "wasted_minutes": float(self._wasted_minutes),
        }
