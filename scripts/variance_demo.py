"""ROADMAP Phase 4, box 4 - REINFORCE's variance, shown rather than asserted.

The box asks to "show REINFORCE's variance explicitly - it is high, and being able
to say *why* (full-return estimates, no bootstrapping) is a strong interview
answer." This measures it, in the one place where the three algorithms genuinely
differ: **the coefficient that multiplies `grad ln pi(a|s)`.**

Everything else about the update is shared. The networks are the same shape, the
inputs carry the same scaling (D-032), the optimiser is the same Adam. The
algorithms differ in exactly one number, and that number is what is sampled here:

  REINFORCE, no baseline   coefficient = G_t                      a whole episode of rewards
  REINFORCE, with baseline coefficient = G_t - v(s_t)             the same, re-centred
  actor-critic             coefficient = r + gamma*v(s') - v(s)   ONE reward + one estimate

The prediction, from theory, before any number is read
------------------------------------------------------
1. **No baseline is the widest.** `G_t` sums ~50 noisy rewards, and every action
   in a lucky episode is reinforced by the same large positive number - including
   the bad ones.
2. **The baseline narrows it without moving its mean.** Subtracting `b(s)`
   changes the spread, not the expectation, because
   `E[b(s) * grad ln pi] = b(s) * grad 1 = 0`. This is the variance-reduction
   claim S&B 13.4 makes, and it is checkable rather than quotable.
3. **The actor-critic is narrowest by far.** It never sums an episode at all.

That ordering is the deliverable. If the measurement contradicts it, the
measurement wins and the disagreement is the finding - this project has already
retracted one claim that failed to replicate (E-013).

What the spread does NOT tell you
---------------------------------
Low variance is not the same as good. The actor-critic's coefficient is **biased**
early in training, because `v(s')` is wrong and the update is scaled by a wrong
number. REINFORCE's is unbiased and noisy. Neither column of this table says
which agent triages better - that is the sample-efficiency comparison's job
(`scripts/compare_sample_efficiency.py`), and the two should be read together.

Budget and seeds
----------------
Short by design: this measures a property of the ESTIMATOR, visible within a few
episodes, not a training outcome that needs 20000. Each condition trains on its
own block so no two share alert streams (D-016/D-027), and the evaluation seeds
are never touched - `_assert_no_eval_seeds` makes that a runtime failure rather
than a comment.

Usage:
    python scripts/variance_demo.py
    python scripts/variance_demo.py --episodes 10 --repeats 1   # smoke
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.agents.actor_critic import ActorCriticAgent
from soc_triage.agents.reinforce import ReinforceAgent
from soc_triage.config import EnvConfig, load_env_config, load_training_config
from soc_triage.env import SOCTriageEnv
from soc_triage.runner import config_hash, run_episode
from soc_triage.state import FEATURE_NAMES, feature_scale_vector

N_ACTIONS = 5

# Enough episodes for the coefficient distribution to be well sampled (each
# episode contributes ~50 coefficients, so 30 episodes is ~1500 per condition per
# seed) and far short of a training run. A property of the estimator, not an
# outcome.
BUDGET_EPISODES = 30
BUDGET_REPEATS = 3


def _assert_no_eval_seeds(seeds: tuple[int, ...], cfg: EnvConfig) -> None:
    """The eval block is never touched, enforced rather than intended."""
    overlap = set(seeds) & set(cfg.seeds.eval)
    if overlap:
        raise ValueError(
            f"variance demo would train on evaluation seeds {sorted(overlap)} - "
            "CONSTRAINTS #2 forbids it"
        )


def collect_reinforce(tcfg, cfg, cfg_hash, use_baseline, seed, n_episodes, seed_start):
    """Every update coefficient REINFORCE produces over `n_episodes`.

    `last_coefficients` is `gamma^t * (G_t - b(s_t))` for the episode just
    finished - exactly the vector that multiplies `grad ln pi`.
    """
    env = SOCTriageEnv(cfg)
    agent = ReinforceAgent(
        obs_dim=len(FEATURE_NAMES), n_actions=N_ACTIONS,
        rcfg=replace(tcfg.reinforce, use_baseline=use_baseline),
        gamma=tcfg.common.gamma, seed=seed,
        feature_scales=feature_scale_vector(tcfg.features.scales),
    )
    base = seed_start + seed * n_episodes
    _assert_no_eval_seeds(tuple(range(base, base + n_episodes)), cfg)

    coefficients: list[float] = []
    for episode in range(n_episodes):
        run_episode(env, agent, base + episode, cfg, cfg_hash, learn=True)
        agent.end_episode()
        coefficients.extend(agent.last_coefficients.tolist())
    return np.asarray(coefficients, dtype=np.float64)


def collect_actor_critic(tcfg, cfg, cfg_hash, seed, n_episodes, seed_start):
    """Every TD error the actor-critic produces over `n_episodes`.

    Not multiplied by `I` here, deliberately: `I` discounts a step by its position
    in the episode and REINFORCE's `gamma^t` does the identical job, so including
    it on one side only would compare two different things. What is compared is
    the ESTIMATOR - full return against bootstrapped one-step target.
    """
    env = SOCTriageEnv(cfg)
    agent = ActorCriticAgent(
        obs_dim=len(FEATURE_NAMES), n_actions=N_ACTIONS,
        accfg=tcfg.actor_critic, gamma=tcfg.common.gamma, seed=seed,
        feature_scales=feature_scale_vector(tcfg.features.scales),
    )
    base = seed_start + seed * n_episodes
    _assert_no_eval_seeds(tuple(range(base, base + n_episodes)), cfg)

    errors: list[float] = []
    for episode in range(n_episodes):
        run_episode(env, agent, base + episode, cfg, cfg_hash, learn=True)
        agent.end_episode()
        errors.extend(agent.last_td_errors.tolist())
    return np.asarray(errors, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 box 4: the variance demonstration.")
    parser.add_argument("--episodes", type=int, default=BUDGET_EPISODES)
    parser.add_argument("--repeats", type=int, default=BUDGET_REPEATS)
    args = parser.parse_args()

    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")
    cfg_hash = config_hash(ROOT / "config" / "env_default.yaml")

    print("Phase 4 box 4 - the update-signal variance, measured")
    print(f"  {args.repeats} seeds x {args.episodes} episodes per condition")
    print("  the quantity is the coefficient multiplying grad ln pi(a|s)\n")

    # Each condition on its own block: the two REINFORCE conditions must not
    # share alert streams (D-027), and the actor-critic gets its own.
    conditions = {
        "REINFORCE (no baseline)": (
            lambda seed: collect_reinforce(tcfg, cfg, cfg_hash, False, seed,
                                           args.episodes, tcfg.reinforce.ablation_seed_start)),
        "REINFORCE (with baseline)": (
            lambda seed: collect_reinforce(tcfg, cfg, cfg_hash, True, seed,
                                           args.episodes, tcfg.reinforce.train_seed_start)),
        "actor-critic (TD error)": (
            lambda seed: collect_actor_critic(tcfg, cfg, cfg_hash, seed,
                                              args.episodes, tcfg.actor_critic.train_seed_start)),
    }

    results: dict[str, dict] = {}
    samples: dict[str, np.ndarray] = {}
    for label, collect in conditions.items():
        per_seed_std: list[float] = []
        pooled: list[np.ndarray] = []
        for seed in range(args.repeats):
            values = collect(seed)
            per_seed_std.append(float(values.std()))
            pooled.append(values)
        allv = np.concatenate(pooled)
        samples[label] = allv
        results[label] = {
            "std_mean_over_seeds": float(np.mean(per_seed_std)),
            "std_spread_over_seeds": float(np.std(per_seed_std)),
            "abs_mean": float(np.abs(allv).mean()),
            "p99_abs": float(np.percentile(np.abs(allv), 99)),
            "max_abs": float(np.abs(allv).max()),
            "n_samples": int(allv.size),
        }
        row = results[label]
        print(f"  {label:28s} std {row['std_mean_over_seeds']:10.2f}  "
              f"|mean| {row['abs_mean']:9.2f}  p99 {row['p99_abs']:10.2f}  "
              f"max {row['max_abs']:11.2f}  n={row['n_samples']}", flush=True)

    print("\n| condition | coefficient std | 99th pct magnitude | largest magnitude |")
    print("|---|---|---|---|")
    for label, row in results.items():
        print(f"| {label} | {row['std_mean_over_seeds']:.2f} | "
              f"{row['p99_abs']:.2f} | {row['max_abs']:.2f} |")

    plain = results["REINFORCE (no baseline)"]["std_mean_over_seeds"]
    based = results["REINFORCE (with baseline)"]["std_mean_over_seeds"]
    critic = results["actor-critic (TD error)"]["std_mean_over_seeds"]

    print("\n--- the two claims this script exists to check ---")
    verdict_one = "CONFIRMED" if based < plain else "NOT CONFIRMED"
    verdict_two = "CONFIRMED" if critic < based else "NOT CONFIRMED"
    print(f"1. the baseline reduces REINFORCE's spread: {plain:.2f} -> {based:.2f}"
          f"   [{verdict_one}, ratio {plain / based:.2f}x]")
    print(f"2. bootstrapping is narrower still:         {based:.2f} -> {critic:.2f}"
          f"   [{verdict_two}, ratio {based / critic:.2f}x]")
    print("\nLower is NOT better on its own. The actor-critic's coefficient is biased")
    print("early in training - v(s') is wrong and the update is scaled by a wrong")
    print("number - where REINFORCE's is unbiased and noisy. Read this beside")
    print("scripts/compare_sample_efficiency.py, never instead of it.")

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    for label, values in samples.items():
        # Clipped at its own 99th percentile so one outlier does not flatten the
        # plot; the unclipped maximum is in the table above and in the JSON.
        limit = float(np.percentile(np.abs(values), 99))
        left.hist(np.clip(values, -limit, limit), bins=80, alpha=0.55, density=True,
                  label=f"{label} (std {values.std():.1f})")
    left.set_xlabel("update coefficient (clipped at its own 99th percentile)")
    left.set_ylabel("density")
    left.set_title("What multiplies grad ln pi")
    left.legend(fontsize=7)

    right.bar(range(len(results)), [r["std_mean_over_seeds"] for r in results.values()],
              yerr=[r["std_spread_over_seeds"] for r in results.values()], capsize=4)
    right.set_xticks(range(len(results)))
    right.set_xticklabels([lbl.replace(" (", "\n(") for lbl in results], fontsize=8)
    right.set_ylabel("coefficient std")
    right.set_yscale("log")
    right.set_title("Spread, log scale (error bars = spread across seeds)")

    figure.tight_layout()
    out = ROOT / "results" / "variance_demo"
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(out / "variance_demo.png", dpi=140)
    (out / "variance_demo.json").write_text(json.dumps({
        "episodes": args.episodes, "repeats": args.repeats,
        "config_hash": cfg_hash, "conditions": results,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'variance_demo.png'}")


if __name__ == "__main__":
    main()
