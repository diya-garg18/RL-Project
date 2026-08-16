"""Phase 2 training entry point (FLOW.md Flow A): train a tabular learner, honestly.

  1. train on a dedicated seed block — one fresh shift per episode (D-016)
  2. every `eval_every` episodes, freeze exploration and measure the greedy
     policy on the TRAIN-diagnostic seeds — never on the eval seeds
  3. repeat the whole run over several agent seeds, because a single run is not
     a result (CONSTRAINTS #3)
  4. only at the very end, evaluate on the eval seeds and print the comparison

The ordering of 2 and 4 is the point. Evaluation-seed numbers are computed once,
after every training decision has already been made, so nothing in this script
can tune against them (CONSTRAINTS #2).

Usage:
    python scripts/train.py                      # full run, from config
    python scripts/train.py --episodes 200 --repeats 1 --no-plot    # smoke test
"""

import argparse
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
from soc_triage.agents.q_learning import QLearningAgent
from soc_triage.config import EnvConfig, TrainingConfig, load_env_config, load_training_config
from soc_triage.env import SOCTriageEnv
from soc_triage.evaluation.metrics import summarise
from soc_triage.runner import config_hash, run_episode, run_episodes, save_records
from soc_triage.state import N_STATES

N_ACTIONS = 5


def build_agent(tcfg: TrainingConfig, seed: int) -> QLearningAgent:
    """Every hyperparameter comes from config; only the seed is injected here."""
    return QLearningAgent(
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        alpha=tcfg.q_learning.alpha,
        gamma=tcfg.common.gamma,
        epsilon_start=tcfg.epsilon.start,
        epsilon_min=tcfg.epsilon.min,
        epsilon_decay=tcfg.epsilon.decay,
        seed=seed,
    )


def greedy_diagnostic(
    env: SOCTriageEnv,
    agent: QLearningAgent,
    cfg: EnvConfig,
    cfg_hash: str,
) -> float:
    """Mean total reward of the CURRENT greedy policy on the train-diagnostic seeds.

    Two things are deliberately switched off. Exploration is pinned to 0, so the
    curve shows what the learned policy is worth rather than what a partly-random
    agent happens to score. And `learn=False`, so measuring never changes what is
    being measured.

    The seeds here are `cfg.seeds.train` (1-10) — training seeds, held out from
    the 200000+ block the agent actually trains on, and disjoint from the eval
    seeds. Plotting a curve against evaluation seeds would be tuning against
    them by eye, which CONSTRAINTS #2 forbids just as firmly as doing it in code.
    """
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        rewards = [
            run_episode(env, agent, seed, cfg, cfg_hash, learn=False)["outcome"]["total_reward"]
            for seed in cfg.seeds.train
        ]
    finally:
        agent.epsilon = saved_epsilon
    return float(np.mean(rewards))


def train_one_run(
    cfg: EnvConfig,
    tcfg: TrainingConfig,
    cfg_hash: str,
    repeat_index: int,
    n_episodes: int,
    eval_every: int,
) -> tuple[QLearningAgent, list[float], list[tuple[int, float]]]:
    """One complete training run. Returns (agent, per-episode rewards, diagnostic curve).

    Each repeat gets its own agent seed AND its own slice of the training seed
    block, so the repeats differ in both the exploration randomness and the
    shifts encountered. Varying only the agent seed would leave all five runs
    facing an identical alert stream, and the resulting std would understate the
    real variability.
    """
    env = SOCTriageEnv(cfg)
    agent = build_agent(tcfg, seed=repeat_index)
    seed_base = tcfg.q_learning.train_seed_start + repeat_index * n_episodes

    episode_rewards: list[float] = []
    curve: list[tuple[int, float]] = []
    t0 = time.perf_counter()

    for episode in range(n_episodes):
        record = run_episode(env, agent, seed_base + episode, cfg, cfg_hash, learn=True)
        episode_rewards.append(record["outcome"]["total_reward"])
        # The one line the whole epsilon schedule depends on (D-015). Without it
        # epsilon stays at its start value forever and the agent never converges
        # — silently, with no error.
        agent.end_episode()

        if (episode + 1) % eval_every == 0:
            curve.append((episode + 1, greedy_diagnostic(env, agent, cfg, cfg_hash)))
            elapsed = time.perf_counter() - t0
            print(
                f"  repeat {repeat_index}  ep {episode + 1}/{n_episodes}  "
                f"eps {agent.epsilon:.3f}  greedy(train-diag) {curve[-1][1]:8.1f}  "
                f"[{elapsed / 60:.1f} min]"
            )

    return agent, episode_rewards, curve


def smooth(values: list[float], window: int) -> np.ndarray:
    """Trailing moving average — the smoothing the roadmap asks for on learning curves."""
    if len(values) < window:
        return np.array(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override common.n_episodes (for smoke tests)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="independent training runs; >=5 for a reportable number")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="override common.eval_every")
    parser.add_argument("--no-plot", action="store_true", help="skip the PNG")
    args = parser.parse_args()

    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")
    cfg_hash = config_hash(ROOT / "config" / "env_default.yaml")

    n_episodes = args.episodes if args.episodes is not None else tcfg.common.n_episodes
    eval_every = args.eval_every if args.eval_every is not None else tcfg.common.eval_every

    print(f"Q-learning: {args.repeats} runs x {n_episodes} episodes")
    print(f"  alpha {tcfg.q_learning.alpha}  gamma {tcfg.common.gamma}  "
          f"eps {tcfg.epsilon.start} -> {tcfg.epsilon.min} @ {tcfg.epsilon.decay}/episode")
    print(f"  training seeds from {tcfg.q_learning.train_seed_start} "
          f"(disjoint from train-diag {list(cfg.seeds.train)} and eval {list(cfg.seeds.eval)})")

    agents: list[QLearningAgent] = []
    all_rewards: list[list[float]] = []
    all_curves: list[list[tuple[int, float]]] = []

    t_start = time.perf_counter()
    for repeat in range(args.repeats):
        agent, rewards, curve = train_one_run(
            cfg, tcfg, cfg_hash, repeat, n_episodes, eval_every
        )
        agents.append(agent)
        all_rewards.append(rewards)
        all_curves.append(curve)
    print(f"training done in {(time.perf_counter() - t_start) / 60:.1f} min")

    # --- learning curve
    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for repeat, rewards in enumerate(all_rewards):
            smoothed = smooth(rewards, tcfg.common.log_smoothing_window)
            ax.plot(range(len(smoothed)), smoothed, alpha=0.7, linewidth=1, label=f"run {repeat}")
        ax.set_xlabel(f"episode (trailing mean over {tcfg.common.log_smoothing_window})")
        ax.set_ylabel("total reward per episode")
        ax.set_title(f"Q-learning on SOC triage — {args.repeats} runs, alpha={tcfg.q_learning.alpha}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        (ROOT / "results").mkdir(exist_ok=True)
        plot_path = ROOT / "results" / "q_learning_curve.png"
        fig.savefig(plot_path, dpi=120)
        print(f"learning curve -> {plot_path}")

    # --- FINAL evaluation. First and only look at the eval seeds.
    env = SOCTriageEnv(cfg)
    print("\nevaluating on eval seeds (greedy, no learning) ...")
    per_run_summaries = []
    for repeat, agent in enumerate(agents):
        agent.epsilon = 0.0  # report the learned policy, not a partly-random one
        records = run_episodes(env, agent, cfg.seeds.eval, cfg, cfg_hash, learn=False)
        if repeat == 0:
            save_records(records, ROOT / "results" / "runs")
        per_run_summaries.append(summarise(records, cfg))

    def across_runs(metric: str) -> tuple[float, float]:
        """Mean and std ACROSS runs of each run's mean — never a single run
        (CONSTRAINTS #3)."""
        means = [s[metric]["mean"] for s in per_run_summaries]
        return float(np.mean(means)), float(np.std(means))

    print(f"\nQ-learning vs references on eval seeds "
          f"(mean ± std across {args.repeats} runs):")
    rows: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
        "q_learning": (
            across_runs("recall_at_deadline"),
            across_runs("total_reward"),
            across_runs("mttd_min"),
        )
    }
    for agent in make_baselines(cfg, 12345):
        if agent.name not in ("severity_sort", "oracle_greedy"):
            continue
        s = summarise(run_episodes(env, agent, cfg.seeds.eval, cfg, cfg_hash), cfg)
        rows[agent.name] = (
            (s["recall_at_deadline"]["mean"], s["recall_at_deadline"]["std"]),
            (s["total_reward"]["mean"], s["total_reward"]["std"]),
            (s["mttd_min"]["mean"], s["mttd_min"]["std"]),
        )

    for name, (recall, reward, mttd) in rows.items():
        print(f"  {name:14s} recall {recall[0]:.2f}±{recall[1]:.2f}   "
              f"reward {reward[0]:7.1f}±{reward[1]:.1f}   mttd {mttd[0]:6.1f}±{mttd[1]:.1f}")

    print("\n  NOTE: the baseline rows show std across EVAL SEEDS (one deterministic")
    print("  run each); the q_learning row shows std across TRAINING RUNS. Different")
    print("  quantities — do not read the two spreads as comparable.")

    # Visits are saved alongside Q because the policy table is not honest without
    # them: an unvisited state has an all-zero Q row, so argmax returns action 0
    # and thousands of never-seen states would print as a confident preference.
    np.save(ROOT / "results" / "q_learning_Q.npy", agents[0].Q)
    np.save(ROOT / "results" / "q_learning_visits.npy", agents[0].visits)
    print(f"\nrun-0 Q-table  -> results/q_learning_Q.npy (gitignored, regenerable)")
    print(f"run-0 visits   -> results/q_learning_visits.npy")


if __name__ == "__main__":
    main()
