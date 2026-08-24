"""E-019 — does `grad_clip_norm` change which algorithm REINFORCE is running?

E-018 logged pre-clip policy-gradient norms of **1584-2228** against a configured
`grad_clip_norm` of **10.0**. At that ratio the clip fires on essentially every
step, and the update stops being

    theta <- theta + alpha * gamma^t * (G_t - b(s_t)) * grad ln pi

and becomes a **fixed-size step along the gradient's direction**: the
`(G_t - b(s_t))` factor still chooses the direction, but its magnitude — the
part that says *how much* better the episode went than expected — is divided out
before the optimiser ever sees it. Nothing errors. The agent trains. It is
simply not the algorithm the docstring describes, which is the same shape of
problem as the Phase 3 Huber delta (E-016, BUG_002).

This script settles it by measurement rather than argument: train the same agent
at three clip values and report what changes.

**Measured on the TRAIN-DIAGNOSTIC seeds (1-10), trained on a dedicated block,
and the evaluation seeds are never touched.** Choosing a hyperparameter is
tuning, and CONSTRAINTS #2 forbids tuning against the evaluation seeds — whether
or not the choice is made by a program. `_assert_no_eval_seeds` makes that a
runtime failure instead of a convention.

**Reduced budget, stated up front.** 1500 episodes x 3 repeats per value, against
a headline run's 20000 x 5. These numbers **rank clip values**; they are not
comparable to any reported REINFORCE result and must not be quoted beside one.
The same trade `scripts/ablations.py` makes, for the same reason: a sweep nobody
can afford to run is a sweep that never happens.

**Read the noise floor before reading the ranking.** E-008 established that this
environment's shift-to-shift variance is several times the differences usually
being reported, so the summary prints the spread *between* configurations beside
the spread *within* one and says plainly when the first does not exceed the
second.

Usage:
    python scripts/reinforce_clip_experiment.py                  # the sweep
    python scripts/reinforce_clip_experiment.py --episodes 50    # smoke
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.agents.reinforce import ReinforceAgent
from soc_triage.config import EnvConfig, TrainingConfig, load_env_config, load_training_config
from soc_triage.env import SOCTriageEnv
from soc_triage.runner import config_hash, run_episode
from soc_triage.state import FEATURE_NAMES, feature_scale_vector

# Structural, not tunable: the action set is the MDP's (PROJECT_BRIEF.md §3).
N_ACTIONS = 5

# The three values, and why these three rather than a finer grid. 10.0 is what
# the config ships and what E-018 measured; 100.0 still clips at the observed
# norms but only by ~20x rather than ~200x; 2000.0 sits at the top of the
# observed pre-clip range, so it is effectively "unclipped" without removing the
# guard that stops a single pathological episode destroying the policy.
CLIP_VALUES = (10.0, 100.0, 2000.0)

# Reduced relative to a headline run (20000 x 5). Enough to rank three values
# against the noise floor, not enough to quote. Same trade as ablations.py.
BUDGET_EPISODES = 1500
BUDGET_REPEATS = 3


def _assert_no_eval_seeds(seeds: tuple[int, ...], cfg: EnvConfig) -> None:
    """Refuse to run if a tuning seed collides with the evaluation block.

    A convention that lives only in a comment is one careless edit from being
    false. This is the check CONSTRAINTS #2 asks to live in code.
    """
    overlap = set(seeds) & set(cfg.seeds.eval)
    if overlap:
        raise ValueError(
            f"clip experiment would train on evaluation seeds {sorted(overlap)} — "
            "tuning against the eval block is exactly what CONSTRAINTS #2 forbids"
        )


class _GreedyView:
    """Deterministic read of the policy, for diagnostics only.

    Duplicated from `train_reinforce.py` rather than imported: importing a
    private helper out of a training script would couple two entry points that
    are deliberately separate (D-025), and the class is six lines.
    """

    name = "reinforce_greedy"
    obs_kind = "cont"

    def __init__(self, agent: ReinforceAgent) -> None:
        self._agent = agent

    def act(self, obs: np.ndarray) -> int:
        return int(np.argmax(self._agent.action_probabilities(obs)))

    def update(self, obs, action, reward, next_obs, done) -> None:
        """No-op. Diagnostics never learn."""

    def save(self, path: str) -> None:
        """No-op. The wrapper owns no parameters."""

    def load(self, path: str) -> None:
        """No-op. The wrapper owns no parameters."""


def build_agent(tcfg: TrainingConfig, seed: int, clip: float) -> ReinforceAgent:
    """The shipped REINFORCE config with one value overridden.

    Everything else — network shape, learning rates, the baseline — is left
    exactly as configured, so any difference between conditions is attributable
    to the clip and nothing else.
    """
    rcfg = replace(tcfg.reinforce, grad_clip_norm=clip)
    return ReinforceAgent(
        obs_dim=len(FEATURE_NAMES),
        n_actions=N_ACTIONS,
        rcfg=rcfg,
        gamma=tcfg.common.gamma,
        seed=seed,
        feature_scales=feature_scale_vector(tcfg.features.scales),
    )


def greedy_diagnostic(
    env: SOCTriageEnv,
    agent: ReinforceAgent,
    cfg: EnvConfig,
    cfg_hash: str,
) -> float:
    """Mean total reward of the policy's ARGMAX on the train-diagnostic seeds.

    The argmax is a *reading* of a stochastic policy, not how the agent behaves —
    the same distinction `train_reinforce.py` draws. Read here because the
    question is what the policy has learned, not what it collected while
    exploring.
    """
    view = _GreedyView(agent)
    total = 0.0
    for seed in cfg.seeds.train:
        record = run_episode(env, view, seed, cfg, cfg_hash, learn=False)
        total += record["outcome"]["total_reward"]
    return total / len(cfg.seeds.train)


def train_once(
    cfg: EnvConfig,
    tcfg: TrainingConfig,
    cfg_hash: str,
    clip: float,
    repeat: int,
    n_episodes: int,
    seed_start: int,
) -> dict:
    """One training run at one clip value.

    Returns the end-of-run greedy diagnostic plus the two numbers that say
    whether the clip was actually doing anything: the mean PRE-clip gradient
    norm (`clip_grad_norm_` returns the norm before it scales) and the fraction
    of updates on which it exceeded the threshold.
    """
    env = SOCTriageEnv(cfg)
    agent = build_agent(tcfg, repeat, clip)

    # Each (value, repeat) pair gets a disjoint slice of the experiment block, so
    # no two conditions ever see the same alert stream.
    offset = (CLIP_VALUES.index(clip) * BUDGET_REPEATS + repeat) * n_episodes
    base = seed_start + offset
    _assert_no_eval_seeds(tuple(range(base, base + n_episodes)), cfg)

    norms: list[float] = []
    rewards: list[float] = []
    for episode in range(n_episodes):
        record = run_episode(env, agent, base + episode, cfg, cfg_hash, learn=True)
        rewards.append(record["outcome"]["total_reward"])
        agent.end_episode()
        norms.append(agent.last_policy_grad_norm)

    norms_array = np.asarray(norms, dtype=np.float64)
    return {
        "clip": clip,
        "repeat": repeat,
        "greedy_train_diag": greedy_diagnostic(env, agent, cfg, cfg_hash),
        "sampled_reward_last_100": float(np.mean(rewards[-100:])),
        "mean_pre_clip_norm": float(norms_array.mean()),
        "clip_active_fraction": float((norms_array > clip).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E-019: REINFORCE gradient-clip sweep.")
    parser.add_argument("--episodes", type=int, default=BUDGET_EPISODES,
                        help="episodes per run (default: the reduced sweep budget)")
    parser.add_argument("--repeats", type=int, default=BUDGET_REPEATS,
                        help="independent runs per clip value")
    args = parser.parse_args()

    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")
    cfg_hash = config_hash(ROOT / "config" / "env_default.yaml")
    seed_start = tcfg.reinforce.clip_experiment_seed_start

    print(f"E-019 clip sweep: {len(CLIP_VALUES)} values x {args.repeats} repeats "
          f"x {args.episodes} episodes")
    print(f"  shipped value is {tcfg.reinforce.grad_clip_norm}; "
          f"E-018 measured pre-clip norms of 1584-2228 against it")
    print(f"  training seeds from {seed_start} (its own block — "
          f"the eval block is never touched)")
    print(f"  measured on train-diagnostic seeds {list(cfg.seeds.train)}\n")

    t0 = time.perf_counter()
    rows: list[dict] = []
    for clip in CLIP_VALUES:
        for repeat in range(args.repeats):
            row = train_once(cfg, tcfg, cfg_hash, clip, repeat, args.episodes, seed_start)
            rows.append(row)
            print(f"  clip {clip:7.1f}  repeat {repeat}  "
                  f"greedy {row['greedy_train_diag']:8.1f}  "
                  f"sampled(last 100) {row['sampled_reward_last_100']:8.1f}  "
                  f"pre-clip norm {row['mean_pre_clip_norm']:9.1f}  "
                  f"clip fired {row['clip_active_fraction']:6.1%}  "
                  f"[{(time.perf_counter() - t0) / 60:.1f} min]", flush=True)

    # --- The summary, with the noise floor printed beside the effect.
    print("\n| clip | greedy reward (mean +- std) | mean pre-clip norm | clip fired |")
    print("|---|---|---|---|")
    per_value_means: list[float] = []
    within_stds: list[float] = []
    for clip in CLIP_VALUES:
        group = [r for r in rows if r["clip"] == clip]
        greedy = np.array([r["greedy_train_diag"] for r in group])
        per_value_means.append(float(greedy.mean()))
        within_stds.append(float(greedy.std()))
        print(f"| {clip} | {greedy.mean():.1f} +- {greedy.std():.1f} | "
              f"{np.mean([r['mean_pre_clip_norm'] for r in group]):.1f} | "
              f"{np.mean([r['clip_active_fraction'] for r in group]):.1%} |")

    between = float(np.std(per_value_means))
    within = float(np.mean(within_stds))
    print(f"\nspread BETWEEN clip values: {between:.1f}")
    print(f"spread WITHIN a clip value: {within:.1f}")
    if between <= within:
        print("VERDICT: the between-value spread does NOT exceed the noise floor. "
              "This sweep does not distinguish the three values on reward, and "
              "any ranking read off it would be a random draw (E-008, E-012).")
    else:
        print("VERDICT: the between-value spread exceeds the within-value spread. "
              "The ranking is worth reporting — with the reduced budget attached.")

    results_dir = ROOT / "results" / "reinforce_clip"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "E-019",
        "config_hash": cfg_hash,
        "clip_values": list(CLIP_VALUES),
        "episodes": args.episodes,
        "repeats": args.repeats,
        "seed_start": seed_start,
        "rows": rows,
        "between_value_spread": between,
        "within_value_spread": within,
    }
    (results_dir / "clip_experiment.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )
    print(f"\nwrote {results_dir / 'clip_experiment.json'}")


if __name__ == "__main__":
    main()
