"""Hyperparameter ablations for Q-learning (ROADMAP box 7): alpha, gamma, epsilon-decay.

**Measured on the TRAIN-DIAGNOSTIC seeds (1-10), never on the eval seeds.**
An ablation is a tuning exercise, and CONSTRAINTS #2 forbids tuning against the
evaluation seeds. Reading eval numbers while choosing hyperparameters would be
exactly that, whether or not the choice is made by a program.

**Every table here prints the noise floor beside the effect.** E-008 established
that this environment's shift-to-shift variance is several times larger than the
differences typically being reported, so an ablation that shows only means is
close to worthless — it invites reading a random draw as a finding. Each row
therefore carries the std across independent training runs, and the summary
states plainly whether the spread *between* configurations exceeds the spread
*within* one.

Reduced budget, stated up front: these runs use fewer episodes and fewer repeats
than a headline result would (see BUDGET below). That is a deliberate trade to
keep the sweep under ten minutes, and it means these numbers rank
configurations — they are not comparable to E-008's.

Usage:
    python scripts/ablations.py                    # full sweep
    python scripts/ablations.py --episodes 500     # quick check
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.agents.q_learning import QLearningAgent
from soc_triage.config import EnvConfig, load_env_config, load_training_config
from soc_triage.env import SOCTriageEnv
from soc_triage.runner import config_hash, run_episode
from soc_triage.state import N_STATES

N_ACTIONS = 5

# Reduced relative to a headline run (20000 episodes x 5 repeats). Enough to
# rank configurations, not enough to quote against E-008.
BUDGET_EPISODES = 6000
BUDGET_REPEATS = 3

# Ablation seeds sit in their own block again (D-016), so a sweep never trains
# on the same shifts as the headline runs and cannot launder tuning into them.
ABLATION_SEED_START = 800_000


def evaluate_greedy(env: SOCTriageEnv, agent, cfg: EnvConfig, cfg_hash: str) -> float:
    """Mean total reward of the greedy policy on the train-diagnostic seeds."""
    saved = agent.epsilon
    agent.epsilon = 0.0
    try:
        return float(np.mean([
            run_episode(env, agent, seed, cfg, cfg_hash, learn=False)["outcome"]["total_reward"]
            for seed in cfg.seeds.train
        ]))
    finally:
        agent.epsilon = saved


def train_once(cfg, cfg_hash, gamma, alpha, decay, eps_start, eps_min,
               n_episodes, repeat) -> float:
    env = SOCTriageEnv(cfg)
    agent = QLearningAgent(
        n_states=N_STATES, n_actions=N_ACTIONS,
        alpha=alpha, gamma=gamma,
        epsilon_start=eps_start, epsilon_min=eps_min, epsilon_decay=decay,
        seed=repeat,
    )
    base = ABLATION_SEED_START + repeat * n_episodes
    for episode in range(n_episodes):
        run_episode(env, agent, base + episode, cfg, cfg_hash, learn=True)
        agent.end_episode()
    return evaluate_greedy(env, agent, cfg, cfg_hash)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-learning hyperparameter ablations.")
    parser.add_argument("--episodes", type=int, default=BUDGET_EPISODES)
    parser.add_argument("--repeats", type=int, default=BUDGET_REPEATS)
    args = parser.parse_args()

    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")
    cfg_hash = config_hash(ROOT / "config" / "env_default.yaml")

    base = dict(
        gamma=tcfg.common.gamma,
        alpha=tcfg.q_learning.alpha,
        decay=tcfg.epsilon.decay,
        eps_start=tcfg.epsilon.start,
        eps_min=tcfg.epsilon.min,
    )

    sweeps = {
        "alpha": ("alpha", [0.02, 0.10, 0.30]),
        "gamma": ("gamma", [0.90, 0.95, 0.99]),
        "epsilon_decay": ("decay", [0.999, 0.9995, 0.9999]),
    }

    print(f"Q-learning ablations — {args.repeats} runs x {args.episodes} episodes per config")
    print(f"baseline: alpha={base['alpha']} gamma={base['gamma']} decay={base['decay']}")
    print(f"measured on TRAIN-DIAGNOSTIC seeds {list(cfg.seeds.train)} (never eval)\n")

    results: dict[str, list[tuple[float, float, float]]] = {}
    t_start = time.perf_counter()

    for sweep_name, (key, values) in sweeps.items():
        print(f"--- {sweep_name} ---")
        rows = []
        for value in values:
            settings = dict(base)
            settings[key] = value
            scores = [
                train_once(cfg, cfg_hash, n_episodes=args.episodes, repeat=r, **settings)
                for r in range(args.repeats)
            ]
            mean, std = float(np.mean(scores)), float(np.std(scores))
            rows.append((value, mean, std))
            marker = "  <- config default" if value == base[key] else ""
            print(f"  {key}={value:<8g} reward {mean:8.1f} +/- {std:6.1f}"
                  f"   runs {[round(s) for s in scores]}{marker}")
        results[sweep_name] = rows
        print()

    print(f"sweep completed in {(time.perf_counter() - t_start) / 60:.1f} min\n")

    # The part that decides whether any of the above means anything.
    print("Does any effect clear the noise?")
    print("  'between' = spread of config means within a sweep")
    print("  'within'  = typical spread across repeats of a single config")
    print("  A sweep whose between-spread is not clearly larger than its")
    print("  within-spread has produced a ranking of random draws.\n")
    for sweep_name, rows in results.items():
        means = [m for _, m, _ in rows]
        stds = [s for _, _, s in rows]
        between = float(np.std(means))
        within = float(np.mean(stds))
        verdict = "SIGNAL" if between > 2 * within else "NOT DISTINGUISHABLE FROM NOISE"
        print(f"  {sweep_name:15s} between {between:7.1f}   within {within:7.1f}   -> {verdict}")


if __name__ == "__main__":
    main()
