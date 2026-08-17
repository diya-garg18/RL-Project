"""Phase 1 pipeline (FLOW.md Flow C): estimate the model, solve it, evaluate honestly.

  1. estimate P-hat/R-hat from random-policy rollouts (seeds from config dp block)
  2. report state-action coverage (the estimate is only as good as its visits)
  3. value iteration until delta < theta; save the convergence curve
  4. policy iteration as an independent cross-check — policies must agree ~everywhere
  5. evaluate the DP policy in the REAL environment on the eval seeds

The DP policy is optimal for the ESTIMATED model, not the true environment (D-004).
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; we only write a PNG
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.agents.baselines import make_baselines
from soc_triage.agents.dp import DPAgent, estimate_model, policy_iteration, value_iteration
from soc_triage.config import load_env_config, load_training_config
from soc_triage.env import SOCTriageEnv
from soc_triage.evaluation.metrics import summarise
from soc_triage.runner import config_hash, run_episodes, save_records
from soc_triage.state import N_STATES


def main() -> None:
    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")
    cfg_hash = config_hash(ROOT / "config" / "env_default.yaml")

    # --- 1. model estimation
    print(f"estimating model: {tcfg.dp.n_estimation_episodes} random episodes "
          f"(seeds {tcfg.dp.estimation_seed_start}+) ...")
    t0 = time.perf_counter()
    P_hat, R_hat, visits = estimate_model(
        cfg,
        n_episodes=tcfg.dp.n_estimation_episodes,
        seed_start=tcfg.dp.estimation_seed_start,
        progress_every=10_000,
    )
    print(f"  done in {(time.perf_counter() - t0) / 60:.1f} min")

    # --- 2. coverage
    state_visits = visits.sum(axis=1)
    visited_states = int((state_visits > 0).sum())
    visited_pairs = int((visits > 0).sum())
    print(f"coverage: {visited_states}/{N_STATES} states visited, "
          f"{visited_pairs}/{N_STATES * 5} state-action pairs")
    print(f"  visit counts over visited states: min {int(state_visits[state_visits > 0].min())}, "
          f"median {int(np.median(state_visits[state_visits > 0]))}, "
          f"max {int(state_visits.max())}")
    print(f"  unvisited (s,a) pairs -> absorbing self-loop, reward 0 (D-011)")

    # --- 3. value iteration + convergence curve
    t0 = time.perf_counter()
    V_vi, policy_vi, deltas = value_iteration(
        P_hat, R_hat, tcfg.common.gamma,
        tcfg.dp.value_iteration_theta, tcfg.dp.max_sweeps,
    )
    print(f"value iteration: converged in {len(deltas)} sweeps "
          f"({time.perf_counter() - t0:.1f}s), final delta {deltas[-1]:.2e}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(range(1, len(deltas) + 1), deltas)
    ax.set_xlabel("sweep")
    ax.set_ylabel("max |V change| (log scale)")
    ax.set_title(f"Value iteration convergence (gamma={tcfg.common.gamma}, "
                 f"theta={tcfg.dp.value_iteration_theta})")
    fig.tight_layout()
    plot_path = ROOT / "results" / "dp_convergence.png"
    fig.savefig(plot_path, dpi=120)
    print(f"convergence curve -> {plot_path}")

    # --- 4. policy iteration cross-check
    t0 = time.perf_counter()
    V_pi, policy_pi, rounds = policy_iteration(
        P_hat, R_hat, tcfg.common.gamma,
        tcfg.dp.policy_eval_theta, tcfg.dp.max_sweeps,
    )
    agreement = float((policy_vi == policy_pi).mean())
    print(f"policy iteration: stabilised after {rounds} improvement rounds "
          f"({time.perf_counter() - t0:.1f}s)")
    print(f"VI/PI policy agreement: {agreement * 100:.1f}%  (must be >= 95% — FLOW.md Flow C)")
    if agreement < 0.95:
        raise SystemExit("VI and PI disagree beyond tie tolerance — one of them is wrong. STOP.")

    # --- 5. evaluate in the real environment, alongside the two reference baselines
    env = SOCTriageEnv(cfg)
    dp_agent = DPAgent(policy_vi)
    records = run_episodes(env, dp_agent, cfg.seeds.eval, cfg, cfg_hash)
    save_records(records, ROOT / "results" / "runs")
    dp_summary = summarise(records, cfg)

    print("\nDP policy vs references on eval seeds (mean ± std):")
    reference = {a.name: a for a in make_baselines(cfg, 12345)
                 if a.name in ("severity_sort", "oracle_greedy")}
    rows = {"dp": dp_summary}
    for name, agent in reference.items():
        rows[name] = summarise(run_episodes(env, agent, cfg.seeds.eval, cfg, cfg_hash), cfg)
    for name, s in rows.items():
        r = s["recall_at_deadline"]; tr = s["total_reward"]; m = s["mttd_min"]
        print(f"  {name:14s} recall {r['mean']:.2f}±{r['std']:.2f}   "
              f"reward {tr['mean']:7.1f}±{tr['std']:.1f}   mttd {m['mean']:6.1f}")

    np.save(ROOT / "results" / "dp_policy.npy", policy_vi)

    # Reconstruct the full action-value table from the converged V:
    #     Q(s,a) = R_hat(s,a) + gamma * sum_s' P_hat(s'|s,a) V(s')
    # Value iteration returns V, but ROADMAP Phase 2 box 5 compares Q-tables, and
    # P_hat/R_hat exist only inside this script. Saved here so the comparison does
    # not have to re-estimate the model (50k episodes) just to get Q.
    # Visit counts go with it: a DP "preference" on a state-action pair never
    # observed is the D-011 self-loop convention, not a decision, and the
    # comparison must be able to exclude those (same principle as FEATURE_005).
    Q_dp = np.zeros((N_STATES, 5), dtype=np.float64)
    for s in range(N_STATES):
        for a in range(5):
            Q_dp[s, a] = R_hat[s, a] + tcfg.common.gamma * (P_hat[s, a] @ V_vi)
    np.save(ROOT / "results" / "dp_Q.npy", Q_dp)
    np.save(ROOT / "results" / "dp_visits.npy", visits)
    print(f"\npolicy saved -> results/dp_policy.npy (gitignored, regenerable by this script)")
    print(f"Q-table      -> results/dp_Q.npy      (for ROADMAP box 5)")
    print(f"visit counts -> results/dp_visits.npy")


if __name__ == "__main__":
    main()
