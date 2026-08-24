"""E-020 — is `entropy_coef: 0.01` large enough to be doing anything?

The smoke run of `train_actor_critic.py` collapsed. At the shipped `entropy_coef`
of 0.01 the policy's entropy fell from 0.911 to 0.0003 within five episodes, the
actor's gradient norm went to 0.00, and the greedy diagnostic sat on **-515.4** —
the exact value Phase 3's collapsed DQN produced (E-016).

The suspected mechanism, and what this script exists to test: the actor's loss is
`I * delta * ln pi`, and `delta` in this environment reaches **1410** (measured).
The entropy term is `entropy_coef * H(pi)`, and `H` is bounded above by
`ln(5) = 1.609`. At coefficient 0.01 the bonus can contribute at most 0.016
against a term three to four orders of magnitude larger, so it cannot resist the
policy saturating — after which `grad ln pi` vanishes and learning stops.

**This is the third time this project has met the same shape of problem**, and
naming the pattern is worth more than the fix. E-016: `huber_delta` left at
torch's default of 1.0 against penalties of -150 to -200, flattening every
catastrophe to the size of a routine error. E-019: `grad_clip_norm` at 10.0
against gradient norms of 1584-2228, replacing the update's magnitude with a
constant. Now this. In all three cases nothing errors, the loss curve looks fine,
and a defensible-looking default silently replaces the algorithm with a different
one. **A hyperparameter whose scale was never checked against the environment's
reward scale is not a tuning choice — it is an untested assumption.**

**Measured on the TRAIN-DIAGNOSTIC seeds (1-10), trained on a dedicated block,
and the evaluation seeds are never touched.** Choosing entropy_coef is tuning,
and CONSTRAINTS #2 forbids tuning against the evaluation seeds.

**Reduced budget, stated up front.** 80 episodes x 3 repeats per value against a
headline run's 20000 x 5. That is far smaller even than E-019's sweep, and
deliberately so: the actor-critic updates every STEP rather than once per
episode, measured at ~0.6 s/episode against REINFORCE's ~0.016 — about 37x the
cost per episode. The budget is set by the ~10-minute limit that does not need
human approval (CLAUDE.md), not by what would be ideal.

What that budget CAN answer: whether the policy stays non-degenerate, which the
diagnostic showed separating within 5-20 episodes. What it CANNOT answer: which
value earns the most reward. The summary prints the noise floor beside the
ranking and is expected to say the reward difference does not clear it.

Usage:
    python scripts/actor_critic_entropy_experiment.py               # the sweep
    python scripts/actor_critic_entropy_experiment.py --episodes 5  # smoke
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

from soc_triage.agents.actor_critic import ActorCriticAgent
from soc_triage.config import EnvConfig, TrainingConfig, load_env_config, load_training_config
from soc_triage.env import SOCTriageEnv
from soc_triage.runner import config_hash, run_episode
from soc_triage.state import FEATURE_NAMES, feature_scale_vector

# Structural, not tunable: the action set is the MDP's (PROJECT_BRIEF.md §3).
N_ACTIONS = 5

# Four values spanning three orders of magnitude, because the question is one of
# SCALE and a fine grid around a value suspected of being 1000x too small would
# answer nothing. 0.01 is what the config ships; 10.0 is near the point where the
# diagnostic showed the bonus starting to dominate (at 50.0 entropy pins to
# ln(5) and the policy never commits to anything).
ENTROPY_VALUES = (0.01, 0.1, 1.0, 10.0)

# Set by the ~10-minute no-approval limit at a measured ~0.6 s/episode, not by
# what would be ideal. See the module docstring for what it can and cannot answer.
BUDGET_EPISODES = 80
BUDGET_REPEATS = 3


def _assert_no_eval_seeds(seeds: tuple[int, ...], cfg: EnvConfig) -> None:
    """Refuse to run if a tuning seed collides with the evaluation block.

    A convention that lives only in a comment is one careless edit from being
    false. This is the check CONSTRAINTS #2 asks to live in code.
    """
    overlap = set(seeds) & set(cfg.seeds.eval)
    if overlap:
        raise ValueError(
            f"entropy experiment would train on evaluation seeds {sorted(overlap)} — "
            "tuning against the eval block is exactly what CONSTRAINTS #2 forbids"
        )


class _GreedyView:
    """Deterministic read of the policy, for diagnostics only.

    Duplicated from `train_actor_critic.py` rather than imported, for the reason
    `reinforce_clip_experiment.py` gives: importing a private helper out of a
    training script couples two entry points that are deliberately separate.
    """

    name = "actor_critic_greedy"
    obs_kind = "cont"

    def __init__(self, agent: ActorCriticAgent) -> None:
        self._agent = agent

    def act(self, obs: np.ndarray) -> int:
        return int(np.argmax(self._agent.action_probabilities(obs)))

    def update(self, obs, action, reward, next_obs, done) -> None:
        """No-op. Diagnostics never learn."""

    def save(self, path: str) -> None:
        """No-op. The wrapper owns no parameters."""

    def load(self, path: str) -> None:
        """No-op. The wrapper owns no parameters."""


def build_agent(tcfg: TrainingConfig, seed: int, entropy_coef: float) -> ActorCriticAgent:
    """The shipped actor-critic config with one value overridden.

    Everything else — network shape, both learning rates, the clip — is left as
    configured, so any difference between conditions is attributable to the
    entropy coefficient and nothing else.
    """
    accfg = replace(tcfg.actor_critic, entropy_coef=entropy_coef)
    return ActorCriticAgent(
        obs_dim=len(FEATURE_NAMES),
        n_actions=N_ACTIONS,
        accfg=accfg,
        gamma=tcfg.common.gamma,
        seed=seed,
        feature_scales=feature_scale_vector(tcfg.features.scales),
    )


def greedy_diagnostic(
    env: SOCTriageEnv,
    agent: ActorCriticAgent,
    cfg: EnvConfig,
    cfg_hash: str,
) -> float:
    """Mean total reward of the policy's ARGMAX on the train-diagnostic seeds."""
    view = _GreedyView(agent)
    total = 0.0
    for seed in cfg.seeds.train:
        total += run_episode(env, view, seed, cfg, cfg_hash, learn=False)["outcome"]["total_reward"]
    return total / len(cfg.seeds.train)


def train_once(
    cfg: EnvConfig,
    tcfg: TrainingConfig,
    cfg_hash: str,
    entropy_coef: float,
    repeat: int,
    n_episodes: int,
    seed_start: int,
) -> dict:
    """One training run at one entropy coefficient.

    Returns the end-of-run greedy diagnostic plus the numbers that answer the
    actual question: the policy's entropy (mean over the run and at the end) and
    the mean absolute TD error, which is the magnitude the bonus has to compete
    against.
    """
    env = SOCTriageEnv(cfg)
    agent = build_agent(tcfg, repeat, entropy_coef)

    # Each (value, repeat) pair gets a disjoint slice, so no two conditions ever
    # see the same alert stream.
    offset = (ENTROPY_VALUES.index(entropy_coef) * BUDGET_REPEATS + repeat) * n_episodes
    base = seed_start + offset
    _assert_no_eval_seeds(tuple(range(base, base + n_episodes)), cfg)

    entropies: list[float] = []
    rewards: list[float] = []
    abs_td: list[float] = []
    for episode in range(n_episodes):
        record = run_episode(env, agent, base + episode, cfg, cfg_hash, learn=True)
        rewards.append(record["outcome"]["total_reward"])
        entropies.append(agent.last_entropy)
        agent.end_episode()
        if agent.last_td_errors.size:
            abs_td.append(float(np.abs(agent.last_td_errors).mean()))

    return {
        "entropy_coef": entropy_coef,
        "repeat": repeat,
        "greedy_train_diag": greedy_diagnostic(env, agent, cfg, cfg_hash),
        "sampled_reward_last_20": float(np.mean(rewards[-20:])),
        "mean_entropy": float(np.mean(entropies)),
        "final_entropy": float(entropies[-1]),
        "mean_abs_td_error": float(np.mean(abs_td)) if abs_td else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E-020: actor-critic entropy_coef sweep.")
    parser.add_argument("--episodes", type=int, default=BUDGET_EPISODES,
                        help="episodes per run (default: the reduced sweep budget)")
    parser.add_argument("--repeats", type=int, default=BUDGET_REPEATS,
                        help="independent runs per value")
    args = parser.parse_args()

    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")
    cfg_hash = config_hash(ROOT / "config" / "env_default.yaml")
    seed_start = tcfg.actor_critic.entropy_experiment_seed_start
    uniform = float(np.log(N_ACTIONS))

    print(f"E-020 entropy sweep: {len(ENTROPY_VALUES)} values x {args.repeats} repeats "
          f"x {args.episodes} episodes")
    print(f"  shipped value is {tcfg.actor_critic.entropy_coef}; the smoke run "
          f"collapsed to entropy 0.0003 in 5 episodes")
    print(f"  uniform policy entropy = ln({N_ACTIONS}) = {uniform:.4f}; "
          f"0.0 means one action in every state")
    print(f"  training seeds from {seed_start} (its own block — "
          f"the eval block is never touched)\n")

    t0 = time.perf_counter()
    rows: list[dict] = []
    for entropy_coef in ENTROPY_VALUES:
        for repeat in range(args.repeats):
            row = train_once(cfg, tcfg, cfg_hash, entropy_coef, repeat, args.episodes, seed_start)
            rows.append(row)
            print(f"  coef {entropy_coef:6.2f}  repeat {repeat}  "
                  f"greedy {row['greedy_train_diag']:8.1f}  "
                  f"sampled(last 20) {row['sampled_reward_last_20']:8.1f}  "
                  f"entropy mean {row['mean_entropy']:6.4f} final {row['final_entropy']:6.4f}  "
                  f"|TD| {row['mean_abs_td_error']:7.2f}  "
                  f"[{(time.perf_counter() - t0) / 60:.1f} min]", flush=True)

    print("\n| entropy_coef | greedy reward (mean +- std) | mean entropy | "
          "final entropy | mean abs TD |")
    print("|---|---|---|---|---|")
    per_value_means: list[float] = []
    within_stds: list[float] = []
    for entropy_coef in ENTROPY_VALUES:
        group = [r for r in rows if r["entropy_coef"] == entropy_coef]
        greedy = np.array([r["greedy_train_diag"] for r in group])
        per_value_means.append(float(greedy.mean()))
        within_stds.append(float(greedy.std()))
        print(f"| {entropy_coef} | {greedy.mean():.1f} +- {greedy.std():.1f} | "
              f"{np.mean([r['mean_entropy'] for r in group]):.4f} | "
              f"{np.mean([r['final_entropy'] for r in group]):.4f} | "
              f"{np.mean([r['mean_abs_td_error'] for r in group]):.2f} |")

    between = float(np.std(per_value_means))
    within = float(np.mean(within_stds))
    print(f"\nspread BETWEEN values: {between:.1f}")
    print(f"spread WITHIN a value:  {within:.1f}")
    if between <= within:
        print("VERDICT on REWARD: does NOT clear the noise floor at this budget. "
              "Any reward ranking read off this sweep would be a random draw.")
    else:
        print("VERDICT on REWARD: clears the within-value spread — reportable, "
              "with the reduced budget attached.")

    # The entropy verdict is separate, and it is the one this sweep is FOR. A
    # policy at entropy ~0 has one action in every state and has stopped learning,
    # whatever its reward did; that is a structural claim, not a noisy mean.
    collapsed = [
        value
        for value, group in (
            (v, [r for r in rows if r["entropy_coef"] == v]) for v in ENTROPY_VALUES
        )
        if float(np.mean([r["final_entropy"] for r in group])) < 0.01
    ]
    print(f"VERDICT on COLLAPSE: values whose policy went degenerate "
          f"(final entropy < 0.01): {collapsed if collapsed else 'none'}")

    results_dir = ROOT / "results" / "actor_critic_entropy"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "E-020",
        "config_hash": cfg_hash,
        "entropy_values": list(ENTROPY_VALUES),
        "episodes": args.episodes,
        "repeats": args.repeats,
        "seed_start": seed_start,
        "rows": rows,
        "between_value_spread": between,
        "within_value_spread": within,
        "collapsed_values": collapsed,
    }
    (results_dir / "entropy_experiment.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )
    print(f"\nwrote {results_dir / 'entropy_experiment.json'}")


if __name__ == "__main__":
    main()
