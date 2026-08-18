"""Test E-014's hypothesis for why the DP policy collapses on the wider eval block.

**The claim to test.** DP's policy is optimal for a transition model estimated
from random rollouts that reached only 133 of 576 states (E-004). D-011 fills
every unvisited (s,a) with an absorbing self-loop worth 0, so on shifts that
wander outside that estimated core the policy is not making informed choices —
it is falling back on a convention. If that is what drives the collapse from
+305.9 (5 seeds) to −201.2 (30 seeds), then **per-seed DP reward should fall as
the share of off-core steps rises.**

**Why this needs a control.** A negative correlation on its own proves nothing:
seeds where DP strays off-core might simply be *harder* seeds, on which every
agent scores badly. So the script measures three things, not one:

  1. corr(off-core share, DP reward)          — the hypothesis
  2. corr(off-core share, severity-sort reward) — is off-core just "hard seed"?
  3. corr(severity-sort reward, DP reward)      — general seed difficulty

The hypothesis is supported only if (1) is strongly negative **and** (2) is not.
If both are negative, off-core-ness is a proxy for difficulty and the test says
nothing about DP specifically.

Two off-core measures are reported because they mean different things:
  - **state** off-core: the state was never seen during estimation at all
  - **pair**  off-core: the action DP chose from that state was never observed,
    so its value came from the D-011 convention rather than from data

Usage:
    python scripts/dp_collapse.py        # needs scripts/run_dp.py to have run
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.agents.baselines import make_baselines
from soc_triage.agents.dp import DPAgent
from soc_triage.config import load_env_config
from soc_triage.env import SOCTriageEnv
from soc_triage.runner import config_hash, run_episode


def pearson(x: list[float], y: list[float]) -> float:
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def main() -> None:
    policy_path = ROOT / "results" / "dp_policy.npy"
    visits_path = ROOT / "results" / "dp_visits.npy"
    if not policy_path.exists() or not visits_path.exists():
        raise SystemExit("missing results/dp_policy.npy or dp_visits.npy — run `python scripts/run_dp.py` first")

    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    cfg_hash = config_hash(ROOT / "config" / "env_default.yaml")
    policy = np.load(policy_path)
    visits = np.load(visits_path)

    state_seen = visits.sum(axis=1) > 0
    print(f"DP estimation coverage: {int(state_seen.sum())}/{len(state_seen)} states, "
          f"{int((visits > 0).sum())}/{visits.size} state-action pairs\n")

    env = SOCTriageEnv(cfg)
    dp_agent = DPAgent(policy)
    severity = next(a for a in make_baselines(cfg, 12345) if a.name == "severity_sort")

    rows = []
    for seed in cfg.seeds.eval:
        record = run_episode(env, dp_agent, seed, cfg, cfg_hash)
        steps = record["steps"]
        n = len(steps)

        off_state = sum(1 for s in steps if not state_seen[s["state_disc"]]) / n
        # The action DP actually chose was never observed from that state, so its
        # value is the D-011 convention, not data.
        off_pair = sum(1 for s in steps if visits[s["state_disc"], s["action"]] == 0) / n

        sev_reward = run_episode(env, severity, seed, cfg, cfg_hash)["outcome"]["total_reward"]

        rows.append({
            "seed": seed,
            "dp_reward": record["outcome"]["total_reward"],
            "sev_reward": sev_reward,
            "off_state": off_state,
            "off_pair": off_pair,
            "steps": n,
        })

    print(f"{'seed':>5} {'steps':>6} {'off-core state':>15} {'off-core pair':>14} "
          f"{'DP reward':>10} {'severity':>10}")
    for r in rows:
        print(f"{r['seed']:>5} {r['steps']:>6} {r['off_state'] * 100:14.1f}% "
              f"{r['off_pair'] * 100:13.1f}% {r['dp_reward']:10.1f} {r['sev_reward']:10.1f}")

    dp = [r["dp_reward"] for r in rows]
    sev = [r["sev_reward"] for r in rows]
    off_s = [r["off_state"] for r in rows]
    off_p = [r["off_pair"] for r in rows]

    print("\n--- the test ---")
    print(f"  (1) corr(off-core STATE share, DP reward)        = {pearson(off_s, dp):+.3f}   <- hypothesis")
    print(f"      corr(off-core PAIR  share, DP reward)        = {pearson(off_p, dp):+.3f}")
    print(f"  (2) corr(off-core STATE share, severity reward)  = {pearson(off_s, sev):+.3f}   <- control")
    print(f"  (3) corr(severity reward, DP reward)             = {pearson(sev, dp):+.3f}   <- seed difficulty")
    print(f"\n  mean off-core state share: {100 * float(np.mean(off_s)):.1f}%"
          f"   mean off-core pair share: {100 * float(np.mean(off_p)):.1f}%")

    print("\n  Reading it: the hypothesis is supported only if (1) is strongly negative")
    print("  AND (2) is not. If both are negative, off-core-ness is just a proxy for")
    print("  seed difficulty and this says nothing about DP specifically.")


if __name__ == "__main__":
    main()
