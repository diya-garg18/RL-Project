"""Phase 3's headline comparison: DQN on continuous state vs tabular Q-learning.

The two agents see the same environment through different eyes. Tabular
Q-learning sees `state.discretise()` — one of 576 buckets. The DQN sees
`state.featurise()` — 17 continuous numbers, including queue composition detail
the buckets throw away. Whether that extra detail is worth anything is the
question Phase 3 exists to answer.

Two comparisons, and they answer different questions:

  1. **Do they behave the same?** Policy agreement, measured ONLY over states
     actually visited during evaluation.
  2. **Does one do better?** The metric table, reported both unpaired
     (mean ± std, as Phase 2 reported it) and PAIRED per eval seed.

On (1), restricting to visited states is not a refinement, it is the whole
result. E-011 found the naive figure over all 576 buckets was 83-86% and that
the agreement was manufactured almost entirely by states neither agent had ever
seen, where both fall back to the same argmax tie-break on an all-zero row.
Over commonly-visited states it collapsed to 22-44%. With ~450 unvisited
buckets, the artefact dominates any average taken over all of them.

Rejected alternative, recorded so it is not re-proposed: projecting the DQN
onto all 576 buckets by inventing a representative feature vector per bucket.
That manufactures inputs the network never saw and invites reading structure
into a figure built on almost no data — which is exactly what E-013 retracted.

On (2), pairing is the point. Both agents run on the identical 30 eval shifts,
so the per-seed difference cancels the shift-to-shift variance that makes the
unpaired spreads (severity_sort 40.4 ± 220.1) wider than any effect being
measured. E-014 found five seeds could not resolve the differences being
claimed; pairing attacks the same problem from the other side, and costs
nothing because the runs already exist.

Usage:
    python scripts/compare_dqn_tabular.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from soc_triage.agents.q_learning import QLearningAgent
from soc_triage.config import load_env_config, load_training_config
from soc_triage.env import SOCTriageEnv
from soc_triage.evaluation.metrics import summarise
from soc_triage.runner import config_hash, run_episodes
from soc_triage.state import N_STATES, discretise, featurise

from train_dqn import N_ACTIONS, build_agent


def load_dqn(tcfg, checkpoint: Path):
    """Rebuild the agent from config, then restore its weights.

    Config first, weights second: the architecture has to match what was saved,
    and reading it from the same YAML the run used is what guarantees that.
    """
    agent = build_agent(tcfg, seed=0, no_replay=False, no_target_network=False)
    agent.load(str(checkpoint))
    agent.epsilon = 0.0  # report the learned policy, not a partly-random one
    return agent


def load_tabular(tcfg):
    """Tabular Q-learning restored from the Phase 2 artefacts.

    Both files are needed. Q alone would let an unvisited state's all-zero row
    render as a confident preference for action 0 — the E-009 mistake, which is
    why `visits` is saved beside it (FEATURE_005).
    """
    agent = QLearningAgent(
        n_states=N_STATES, n_actions=N_ACTIONS,
        alpha=tcfg.q_learning.alpha, gamma=tcfg.common.gamma,
        epsilon_start=0.0, epsilon_min=0.0, epsilon_decay=1.0, seed=0,
    )
    q_path = ROOT / "results" / "q_learning_Q.npy"
    v_path = ROOT / "results" / "q_learning_visits.npy"
    for path in (q_path, v_path):
        if not path.exists():
            raise SystemExit(
                f"missing {path.relative_to(ROOT).as_posix()} — regenerate with\n"
                f"  .\\.venv\\Scripts\\python.exe scripts/train.py --agent q_learning"
            )
    agent.Q = np.load(q_path)
    agent.visits = np.load(v_path)
    return agent


def dqn_actions_by_state(env, dqn, cfg, cfg_hash) -> dict[int, list[int]]:
    """Roll the DQN over the eval seeds, recording (bucket, action) per step.

    The DQN chooses from the continuous features, so many distinct situations
    map to one bucket and it may legitimately pick different actions within
    one. Every visit is therefore recorded, not one action per bucket, and the
    ambiguity is reported rather than averaged away.
    """
    by_state: dict[int, list[int]] = {}
    for seed in cfg.seeds.eval:
        snap = env.reset(seed)
        done = False
        while not done:
            bucket = discretise(snap, cfg)
            action = dqn.act(featurise(snap, cfg))
            by_state.setdefault(bucket, []).append(action)
            snap, _, done, _ = env.step(action)
    return by_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the DQN against tabular Q-learning.")
    parser.add_argument("--checkpoint", default=None,
                        help="DQN .pt to compare (default: repeat0 of the control sweep)")
    args = parser.parse_args()

    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")
    cfg_hash = config_hash(ROOT / "config" / "env_default.yaml")
    env = SOCTriageEnv(cfg)

    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        ROOT / "results" / "dqn_runs" / "dqn" / "repeat0.pt")
    if not checkpoint.exists():
        raise SystemExit(
            f"missing {checkpoint} — run the sweep first:\n"
            f"  .\\.venv\\Scripts\\python.exe scripts/run_dqn_sweep.py"
        )

    dqn = load_dqn(tcfg, checkpoint)
    tabular = load_tabular(tcfg)
    tabular_policy = tabular.greedy_policy()
    tabular_seen = tabular.visits.sum(axis=1) > 0

    print(f"DQN checkpoint : {checkpoint.relative_to(ROOT).as_posix()}")
    print(f"eval seeds     : {len(cfg.seeds.eval)} "
          f"({min(cfg.seeds.eval)}..{max(cfg.seeds.eval)})")
    print(f"tabular states visited during TRAINING: {int(tabular_seen.sum())}/{N_STATES}")

    # --- 1. policy agreement, over visited states only
    by_state = dqn_actions_by_state(env, dqn, cfg, cfg_hash)
    visited = sorted(by_state)
    both_seen = [s for s in visited if tabular_seen[s]]

    visit_agree = visit_total = 0
    state_agree = 0
    for state in both_seen:
        actions = by_state[state]
        visit_total += len(actions)
        visit_agree += sum(1 for a in actions if a == tabular_policy[state])
        modal = max(set(actions), key=actions.count)
        state_agree += int(modal == tabular_policy[state])

    print(f"\npolicy agreement (E-011: visited states ONLY — see module docstring)")
    print(f"  states the DQN visited at eval      : {len(visited)}/{N_STATES}")
    print(f"  of those, also seen by tabular      : {len(both_seen)}")
    if both_seen:
        print(f"  agreement per VISIT                 : "
              f"{visit_agree}/{visit_total} = {100 * visit_agree / visit_total:.1f}%")
        print(f"  agreement per STATE (modal action)  : "
              f"{state_agree}/{len(both_seen)} = {100 * state_agree / len(both_seen):.1f}%")
        multi = sum(1 for s in both_seen if len(set(by_state[s])) > 1)
        print(f"  buckets where the DQN chose >1 action: {multi}/{len(both_seen)} "
              f"— the continuous state distinguishing situations the buckets merge, "
              f"which is the entire premise of this phase")
    else:
        print("  NO overlap — nothing to compare. Treat any agreement claim as void.")

    # --- 2. metrics, unpaired and paired
    print("\nrolling both agents over the eval seeds ...")
    dqn_records = run_episodes(env, dqn, cfg.seeds.eval, cfg, cfg_hash, learn=False)
    tab_records = run_episodes(env, tabular, cfg.seeds.eval, cfg, cfg_hash, learn=False)
    dqn_summary = summarise(dqn_records, cfg)
    tab_summary = summarise(tab_records, cfg)

    def fmt(summary, metric):
        entry = summary[metric]
        if entry["mean"] is None:
            return "  undefined"
        return f"{entry['mean']:8.2f} ± {entry['std']:.2f}"

    print(f"\nunpaired, across the {len(cfg.seeds.eval)} eval seeds (one run each):")
    print(f"  {'metric':22s}{'DQN':>18}{'tabular q_learning':>22}")
    for metric in ("recall_at_deadline", "total_reward", "mttd_min"):
        print(f"  {metric:22s}{fmt(dqn_summary, metric):>18}"
              f"{fmt(tab_summary, metric):>22}")

    dqn_seed = np.array([r["outcome"]["total_reward"] for r in dqn_records])
    tab_seed = np.array([r["outcome"]["total_reward"] for r in tab_records])
    diff = dqn_seed - tab_seed
    wins = int((diff > 0).sum())
    sem = diff.std(ddof=1) / np.sqrt(len(diff))

    print(f"\nPAIRED per eval seed — total reward, DQN minus tabular:")
    print(f"  mean difference : {diff.mean():+8.2f}")
    print(f"  std of the diff : {diff.std(ddof=1):8.2f}   "
          f"(vs unpaired spreads of ±52 and ±220 — this is what pairing buys)")
    print(f"  standard error  : {sem:8.2f}")
    print(f"  DQN wins on     : {wins}/{len(diff)} seeds")
    print(f"  |mean| / SEM    : {abs(diff.mean()) / sem:8.2f}"
          if sem > 0 else "  SEM is zero — identical policies")
    print("\n  Read the ratio before the sign (R6, and the standing lesson of E-014):")
    print("  below ~2 the difference is not resolvable at 30 seeds, whichever way")
    print("  it points, and reporting a winner would repeat the E-008 mistake.")

    out = ROOT / "results" / "dqn_vs_tabular.md"
    out.write_text(
        "# DQN vs tabular Q-learning\n\n"
        f"Checkpoint: `{checkpoint.relative_to(ROOT).as_posix()}`  \n"
        f"Eval seeds: {len(cfg.seeds.eval)}  \n"
        f"Config hash: `{cfg_hash}`\n\n"
        "## Policy agreement (visited states only, E-011)\n\n"
        f"- DQN visited {len(visited)}/{N_STATES} buckets at eval; "
        f"{len(both_seen)} also seen by tabular during training\n"
        + (f"- per visit: **{100 * visit_agree / visit_total:.1f}%**\n"
           f"- per state (modal): **{100 * state_agree / len(both_seen):.1f}%**\n"
           if both_seen else "- no overlap; agreement undefined\n")
        + "\n## Paired total reward (DQN − tabular, per eval seed)\n\n"
        f"| quantity | value |\n|---|---|\n"
        f"| mean difference | {diff.mean():+.2f} |\n"
        f"| std of difference | {diff.std(ddof=1):.2f} |\n"
        f"| standard error | {sem:.2f} |\n"
        f"| DQN wins | {wins}/{len(diff)} seeds |\n\n"
        "Pairing cancels the shift-to-shift variance that makes the unpaired\n"
        "spreads wider than the effect; a |mean|/SEM below ~2 is not resolvable\n"
        "at this sample size regardless of sign.\n",
        encoding="utf-8",
    )
    print(f"\nwritten -> {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
