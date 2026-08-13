"""The five Phase 0 baselines (PROJECT_BRIEF.md §8).

random          — the floor. If anything loses to this, it's broken.
fifo            — the naive human default: always work the oldest alert.
severity_sort   — what the industry actually does. THE baseline to beat.
cheapest_first  — throughput-maximising strawman: burn through the queue.
oracle_greedy   — CHEATS by design: sees hidden ground truth to give an upper
                  bound. Named 'oracle', lives here, never presented as an
                  achieved result (CONSTRAINTS.md #1 exception clause).
"""

import numpy as np

from soc_triage.agents.base import Agent
from soc_triage.config import EnvConfig
from soc_triage.env import (
    BULK_CLOSE_LOW_RISK,
    PULL_CHEAPEST,
    PULL_HIGHEST_SEVERITY,
    PULL_MOST_CRITICAL_ASSET,
    PULL_OLDEST,
)
from soc_triage.state import EnvSnapshot


class RandomAgent(Agent):
    """Uniform random action each step. Seeded so runs are reproducible."""

    name = "random"
    obs_kind = "disc"

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        return int(self._rng.integers(0, 5))


class FifoAgent(Agent):
    """Always investigate the longest-waiting alert."""

    name = "fifo"
    obs_kind = "disc"

    def act(self, obs) -> int:
        return PULL_OLDEST


class SeveritySortAgent(Agent):
    """Always investigate the highest-severity alert — the industry default."""

    name = "severity_sort"
    obs_kind = "disc"

    def act(self, obs) -> int:
        return PULL_HIGHEST_SEVERITY


class CheapestFirstAgent(Agent):
    """Always investigate the fastest-to-verify alert — maximise throughput."""

    name = "cheapest_first"
    obs_kind = "disc"

    def act(self, obs) -> int:
        return PULL_CHEAPEST


class OracleGreedyAgent(Agent):
    """Upper bound. Reads hidden ground truth (sanctioned breach — see module docstring).

    Greedy logic, one step at a time, still restricted to the same 5 actions:
      1. If some pull rule would land on a true incident, take it — most urgent
         deadline first (recall@deadline is the metric an upper bound must max).
         It knows what each rule would select because the pull rules are
         deterministic, using the same tie-breaks as env._select_alert.
      2. Else, if a true incident is in the queue but no rule reaches it, clear
         a path: find the rule needing the fewest removals before that incident
         becomes its argmax, and pull along that path (each pull removes that
         path's top blocker). Safe bulk-close (sweeps only false positives —
         the oracle can check) clears junk faster: 2 minutes for up to 10.
      3. Else (no incident in queue): wait as cheaply as possible.

    Still not optimal (greedy, no lookahead over arrivals) — but no honest
    policy can see more than it sees.
    """

    name = "oracle_greedy"
    obs_kind = "snapshot"  # the sanctioned exception

    def __init__(self, cfg: EnvConfig):
        # Bulk-close eligibility rules come from config, not hardcoded copies.
        self._bulk = cfg.actions.bulk_close

    def act(self, obs: EnvSnapshot) -> int:
        if not obs.queue:
            return PULL_CHEAPEST  # empty queue: action is a wait regardless

        # Reproduce each pull rule's deterministic selection (ties: lowest id,
        # i.e. first in the id-ordered queue — same convention as the env).
        selected = {}
        best_sev = obs.queue[0]
        best_old = obs.queue[0]
        best_crit = obs.queue[0]
        best_cheap = obs.queue[0]
        for alert in obs.queue[1:]:
            if alert.severity > best_sev.severity:
                best_sev = alert
            if alert.arrival_time < best_old.arrival_time:
                best_old = alert
            if alert.asset_criticality > best_crit.asset_criticality:
                best_crit = alert
            if alert.verify_cost_min < best_cheap.verify_cost_min:
                best_cheap = alert
        selected[PULL_HIGHEST_SEVERITY] = best_sev
        selected[PULL_OLDEST] = best_old
        selected[PULL_MOST_CRITICAL_ASSET] = best_crit
        selected[PULL_CHEAPEST] = best_cheap

        # 1. Any rule that lands on a real incident? Catch the most time-critical
        # one: unexpired deadlines first (those still count for recall@deadline
        # and decay less), earliest deadline first; expired ones still caught
        # (avoids the end-of-shift miss penalty).
        best_action = None
        best_key: tuple[bool, float] | None = None
        for action, alert in selected.items():
            if alert.is_true_incident:
                deadline_at = alert.arrival_time + alert.deadline_min
                key = (deadline_at < obs.time_now, deadline_at)  # False (unexpired) sorts first
                if best_key is None or key < best_key:
                    best_key = key
                    best_action = action
        if best_action is not None:
            return best_action

        safe_bulk_close = self._safe_bulk_close(obs)

        # 2. A real incident is in the queue but no rule reaches it: clear a path.
        incidents = [a for a in obs.queue if a.is_true_incident]
        if incidents:
            # Target the most urgent incident and find the rule whose argmax it
            # is closest to becoming (fewest blocking alerts on that path).
            target = min(incidents, key=lambda a: a.arrival_time + a.deadline_min)
            rules = (PULL_HIGHEST_SEVERITY, PULL_OLDEST, PULL_MOST_CRITICAL_ASSET, PULL_CHEAPEST)
            blockers = {rule: 0 for rule in rules}
            for alert in obs.queue:
                if alert.id == target.id:
                    continue
                for rule in rules:
                    if self._blocks(alert, target, rule):
                        blockers[rule] += 1
            best_rule = min(rules, key=lambda rule: blockers[rule])

            # Bulk-close is only useful if it actually removes blockers on the
            # chosen path — it can never touch high-severity or high-criticality
            # blockers, so blindly sweeping junk would loop forever on hygiene
            # while the incident sits (the bug the first oracle version had).
            if safe_bulk_close:
                would_close = self._bulk_eligible(obs)[: self._bulk.max_alerts]
                removes_blockers = any(self._blocks(a, target, best_rule) for a in would_close)
                if removes_blockers:
                    return BULK_CLOSE_LOW_RISK  # 2 min, clears several blockers at once
            return best_rule  # pulls that path's current argmax = its top blocker

        # 3. No real incident visible: wait as cheaply as possible.
        if safe_bulk_close:
            return BULK_CLOSE_LOW_RISK  # 2 min beats a 5-minute cheapest pull
        return PULL_CHEAPEST

    @staticmethod
    def _blocks(alert, target, rule: int) -> bool:
        """Would `rule` pick `alert` over `target`? Strictly better key, or an
        equal key with an earlier id — the environment's first-in-queue tie-break."""
        earlier = alert.id < target.id
        if rule == PULL_HIGHEST_SEVERITY:
            return alert.severity > target.severity or (alert.severity == target.severity and earlier)
        if rule == PULL_OLDEST:
            return alert.arrival_time < target.arrival_time
        if rule == PULL_MOST_CRITICAL_ASSET:
            return (alert.asset_criticality > target.asset_criticality
                    or (alert.asset_criticality == target.asset_criticality and earlier))
        return (alert.verify_cost_min < target.verify_cost_min
                or (alert.verify_cost_min == target.verify_cost_min and earlier))

    def _bulk_eligible(self, obs: EnvSnapshot) -> list:
        """Alerts bulk-close may touch, in the order the environment closes them."""
        return [
            a for a in obs.queue
            if a.severity <= self._bulk.max_severity
            and a.asset_criticality <= self._bulk.max_asset_criticality
        ]

    def _safe_bulk_close(self, obs: EnvSnapshot) -> bool:
        """True if bulk-close would fire AND sweep only false positives."""
        would_close = self._bulk_eligible(obs)[: self._bulk.max_alerts]
        return bool(would_close) and not any(a.is_true_incident for a in would_close)


def make_baselines(cfg: EnvConfig, seed: int) -> list[Agent]:
    """All five baselines, with the random agent seeded for reproducibility."""
    return [
        RandomAgent(seed),
        FifoAgent(),
        SeveritySortAgent(),
        CheapestFirstAgent(),
        OracleGreedyAgent(cfg),
    ]
