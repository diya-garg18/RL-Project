"""Phase 4 training entry point: train the one-step actor-critic, honestly.

A separate file from `train_reinforce.py` for the reason D-025 gives, and the
separation earns itself here more than anywhere: this trainer logs the TD-error
spread and the policy entropy, which REINFORCE has no equivalent of, and it has
no baseline ablation, which REINFORCE's entire `--no-baseline` path exists for.
Folding them together would produce a script whose every second line asks which
algorithm it is driving.

The honesty machinery is carried over unchanged, because it is the part that must
not vary between phases:

  1. train on a dedicated seed block — one fresh shift per episode (D-016)
  2. every `eval_every` episodes, measure the *greedy* policy on the
     TRAIN-diagnostic seeds — never on the eval seeds
  3. repeat over several agent seeds; one run is not a result (CONSTRAINTS #3)
  4. only at the very end, evaluate on the eval seeds

The order of 2 and 4 is the point: evaluation-seed numbers are computed once,
after every training decision has already been made (CONSTRAINTS #2).

**Two diagnostics are logged that no earlier trainer has, and both are here
because of what E-018 and E-019 found.**

`td_error_std` is the per-episode standard deviation of the TD errors. It is the
actor-critic half of the variance demonstration ROADMAP box 4 asks for: put
beside REINFORCE's coefficient spread, it is the bias-variance trade made
visible rather than asserted.

`entropy` is the policy entropy. E-018 found REINFORCE's greedy policy
degenerate — one action in every state — by 300 episodes, and E-019 showed the
gradient clip was not the cause. Entropy is the number that says whether that is
happening here, *while* it happens, rather than after the fact from a policy
table. A run whose entropy falls to ~0 has stopped exploring, whatever its reward
curve is doing.

Usage:
    python scripts/train_actor_critic.py --episodes 200 --repeats 1 --no-plot  # smoke
    python scripts/train_actor_critic.py                                       # full run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; we only write a PNG
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.agents.actor_critic import ActorCriticAgent
from soc_triage.config import EnvConfig, TrainingConfig, load_env_config, load_training_config
from soc_triage.env import SOCTriageEnv
from soc_triage.evaluation.metrics import MIN_RUNS_TO_REPORT, summarise
from soc_triage.runner import config_hash, run_episode, run_episodes, save_records
from soc_triage.state import FEATURE_NAMES, feature_scale_vector

# Structural, not tunable: the action set is defined by the MDP in
# PROJECT_BRIEF.md §3, and changing it changes the problem, not a knob.
N_ACTIONS = 5

TAG = "actor_critic"


def build_agent(tcfg: TrainingConfig, seed: int) -> ActorCriticAgent:
    """Every hyperparameter comes from config; only the seed is injected here."""
    return ActorCriticAgent(
        obs_dim=len(FEATURE_NAMES),
        n_actions=N_ACTIONS,
        accfg=tcfg.actor_critic,
        gamma=tcfg.common.gamma,
        seed=seed,
        # The SHARED scales (D-032), the same vector the DQN and REINFORCE get,
        # so the sample-efficiency comparison compares algorithms rather than
        # preprocessing.
        feature_scales=feature_scale_vector(tcfg.features.scales),
    )


class _GreedyView:
    """The agent's policy read deterministically, for diagnostics only.

    Same wrapper, and same justification, as `train_reinforce.py`'s: a
    policy-gradient agent has no epsilon to pin to zero, so "what has it learned"
    and "what does it do" are genuinely two questions. This answers the first
    without touching the agent's own RNG or its accumulator. `update()` is a
    no-op because the runner is called with learn=False; it exists so this
    satisfies the Agent interface (CONSTRAINTS #10).
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


def greedy_diagnostic(
    env: SOCTriageEnv,
    agent: ActorCriticAgent,
    cfg: EnvConfig,
    cfg_hash: str,
) -> float:
    """Mean total reward of the current policy's ARGMAX on the train-diagnostic seeds.

    The train-diagnostic seeds (1-10), never the eval block: a curve measured on
    evaluation seeds is a tuning signal, and looking at it even once compromises
    the final number (CONSTRAINTS #2).
    """
    view = _GreedyView(agent)
    records = run_episodes(env, view, tuple(cfg.seeds.train), cfg, cfg_hash, learn=False)
    return float(np.mean([r["outcome"]["total_reward"] for r in records]))


def train_one_run(
    cfg: EnvConfig,
    tcfg: TrainingConfig,
    cfg_hash: str,
    repeat_index: int,
    n_episodes: int,
    eval_every: int,
    seed_start: int,
) -> tuple[ActorCriticAgent, list[float], list[tuple[int, float]], list[float], list[float]]:
    """One complete training run.

    Returns (agent, per-episode sampled reward, greedy diagnostic curve,
    per-episode TD-error std, per-episode policy entropy).

    Each repeat gets its own agent seed AND its own slice of the training seed
    block, so repeats differ in both the sampling randomness and the shifts
    encountered. Varying only the agent seed would leave every run facing an
    identical alert stream, and the resulting std would understate the real
    variability.
    """
    env = SOCTriageEnv(cfg)
    agent = build_agent(tcfg, repeat_index)
    seed_base = seed_start + repeat_index * n_episodes

    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    td_error_stds: list[float] = []
    entropies: list[float] = []
    curve: list[tuple[int, float]] = []
    t0 = time.perf_counter()

    for episode in range(n_episodes):
        record = run_episode(env, agent, seed_base + episode, cfg, cfg_hash, learn=True)
        episode_rewards.append(record["outcome"]["total_reward"])
        # Sample efficiency is measured against STEPS, not episodes: a
        # bulk-closing policy makes more decisions per 480-minute shift
        # than a verifying one, so episodes are not a common currency
        # (ROADMAP box 3).
        episode_steps.append(len(record["steps"]))
        # Read once per episode rather than averaged over every step:
        # last_entropy is a scalar the agent overwrites each update, and
        # accumulating it per step would cost a list of ~50 floats per episode to
        # report a quantity that moves on the scale of episodes, not steps.
        entropies.append(agent.last_entropy)
        # Unlike REINFORCE, end_episode() is NOT where this agent learns -- it
        # only resets I and publishes the episode's TD errors. Forgetting it does
        # not stop training; it silently leaves I decayed from the previous
        # shift, so early steps are weighted as though they were late ones.
        agent.end_episode()
        td_error_stds.append(
            float(agent.last_td_errors.std()) if agent.last_td_errors.size else 0.0
        )

        if (episode + 1) % eval_every == 0:
            curve.append((episode + 1, greedy_diagnostic(env, agent, cfg, cfg_hash)))
            elapsed = time.perf_counter() - t0
            recent = float(np.mean(episode_rewards[-eval_every:]))
            print(
                f"  repeat {repeat_index}  ep {episode + 1}/{n_episodes}  "
                f"sampled {recent:8.1f}  td_err_std {td_error_stds[-1]:8.2f}  "
                f"entropy {entropies[-1]:5.3f}  "
                f"greedy(train-diag) {curve[-1][1]:8.1f}  [{elapsed / 60:.1f} min]",
                flush=True,
            )

    return agent, episode_rewards, episode_steps, curve, td_error_stds, entropies


def smooth(values: list[float], window: int) -> np.ndarray:
    """Trailing moving average — the smoothing the roadmap asks for on curves."""
    if len(values) < window:
        return np.array(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the actor-critic agent on SOC triage.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override common.n_episodes (for smoke tests)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="independent training runs; >=5 for a reportable number")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="override common.eval_every")
    parser.add_argument("--only-repeat", type=int, default=None,
                        help="run ONLY this repeat index and write its result as JSON, so "
                             "repeats can run as parallel processes (each uses one core)")
    parser.add_argument("--no-plot", action="store_true", help="skip the PNGs")
    args = parser.parse_args()

    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")
    cfg_hash = config_hash(ROOT / "config" / "env_default.yaml")

    n_episodes = args.episodes if args.episodes is not None else tcfg.common.n_episodes
    eval_every = args.eval_every if args.eval_every is not None else tcfg.common.eval_every
    seed_start = tcfg.actor_critic.train_seed_start

    ac = tcfg.actor_critic
    print(f"{TAG}: {args.repeats} runs x {n_episodes} episodes")
    print(f"  actor {list(ac.hidden_layers)} {ac.activation} lr {ac.actor_lr}  "
          f"critic lr {ac.critic_lr}  gamma {tcfg.common.gamma}")
    print(f"  entropy_coef {ac.entropy_coef} "
          f"({'S&B 13.5 exactly' if ac.entropy_coef == 0 else 'NOT in S&B 13.5'})  "
          f"grad_clip {ac.grad_clip_norm}")
    print("  exploration: none scheduled — the policy is stochastic; the entropy "
          "bonus is what resists it sharpening")
    print(f"  training seeds from {seed_start} "
          f"(disjoint from train-diag {list(cfg.seeds.train)} and the eval block)")

    results_dir = ROOT / "results" / "actor_critic_runs" / TAG
    results_dir.mkdir(parents=True, exist_ok=True)

    repeats = [args.only_repeat] if args.only_repeat is not None else list(range(args.repeats))
    all_rewards: list[list[float]] = []
    all_steps: list[list[int]] = []
    all_curves: list[list[tuple[int, float]]] = []
    all_td_stds: list[list[float]] = []
    all_entropies: list[list[float]] = []
    trained_agents: list[ActorCriticAgent] = []

    for repeat_index in repeats:
        agent, rewards, steps, curve, td_stds, entropies = train_one_run(
            cfg, tcfg, cfg_hash, repeat_index, n_episodes, eval_every, seed_start
        )
        all_rewards.append(rewards)
        all_steps.append(steps)
        all_curves.append(curve)
        all_td_stds.append(td_stds)
        all_entropies.append(entropies)
        trained_agents.append(agent)
        agent.save(str(results_dir / f"{TAG}_repeat{repeat_index}.pt"))

    assert trained_agents

    # --- Only now does anything touch the evaluation seeds (CONSTRAINTS #2).
    #
    # EVERY repeat is evaluated, not just the last one. See the identical note in
    # train_reinforce.py and docs/bugs/BUG_004: evaluating a single `final_agent`
    # reported one training run as though it were a result, with a spread that
    # described seed difficulty rather than run-to-run variation.
    # D-036: the SAMPLED policy is Phase 4's headline and the greedy read is a
    # named diagnostic. Both are computed on the same agents and seeds. E-020
    # makes the reason concrete for this agent in particular: the argmax was
    # constant BULK_CLOSE at every entropy coefficient tested, including the one
    # where the sampled policy was healthy, so the greedy column alone would say
    # the same thing about an agent that learned and one that collapsed.
    print(f"\nevaluating {len(trained_agents)} trained policies on the eval seeds "
          f"(first look, last step)")
    env = SOCTriageEnv(cfg)
    sampled_per_run: list[dict] = []
    greedy_per_run: list[dict] = []
    for position, (repeat_index, agent) in enumerate(zip(repeats, trained_agents)):
        # Pin the evaluation draws. `repeat_index` is the agent's own
        # construction seed (see build_agent), so this restarts its own stream
        # rather than continuing wherever training happened to leave it.
        agent.reseed(repeat_index)
        # learn=False, so the runner calls neither update() nor end_episode() —
        # the agent samples and learns nothing (see runner.run_episode).
        sampled_records = run_episodes(
            env, agent, tuple(cfg.seeds.eval), cfg, cfg_hash, learn=False
        )
        greedy_records = run_episodes(
            env, _GreedyView(agent), tuple(cfg.seeds.eval), cfg, cfg_hash, learn=False
        )
        if position == 0:
            save_records(sampled_records, results_dir / "eval_records_sampled")
            save_records(greedy_records, results_dir / "eval_records_greedy")
        sampled = summarise(sampled_records, cfg)
        greedy = summarise(greedy_records, cfg)
        sampled_per_run.append(sampled)
        greedy_per_run.append(greedy)
        print(f"  repeat {repeat_index}:  SAMPLED recall "
              f"{sampled['recall_at_deadline']['mean']:.4f} reward "
              f"{sampled['total_reward']['mean']:8.1f}   |   greedy recall "
              f"{greedy['recall_at_deadline']['mean']:.4f} reward "
              f"{greedy['total_reward']['mean']:8.1f}")

    def across_runs(per_run_summaries: list[dict], metric: str) -> dict[str, float | int | None]:
        """Mean and std ACROSS runs of each run's mean — never a single run
        (CONSTRAINTS #3). Same convention as `train.py` and `train_dqn.py`,
        deliberately, so a Phase 4 number and a Phase 2 number mean the same thing.

        A run's mean is None when the metric is undefined for it: `summarise`
        reports mttd_min as None when no episode caught an incident, which is the
        honest answer rather than zero. Those runs are dropped from the average
        and, if none survive, the metric is reported undefined. A policy that
        catches nothing is a real Phase 4 outcome, not a defensive edge case —
        E-020's collapsed runs scored recall 0.0000.
        """
        means = [s[metric]["mean"] for s in per_run_summaries if s[metric]["mean"] is not None]
        if not means:
            return {"mean": None, "std": None, "n_runs": 0}
        return {"mean": float(np.mean(means)), "std": float(np.std(means)),
                "n_runs": len(means)}

    def aggregate(per_run: list[dict]) -> dict[str, dict]:
        return {metric: across_runs(per_run, metric) for metric in per_run[0]
                if isinstance(per_run[0][metric], dict)}

    eval_sampled = aggregate(sampled_per_run)
    eval_greedy = aggregate(greedy_per_run)

    def report(heading: str, aggregated: dict[str, dict], n_runs: int) -> None:
        print(f"\n  {heading}")
        for metric, stats in aggregated.items():
            if stats["mean"] is None:
                print(f"    {metric}: undefined on all {n_runs} runs")
            else:
                print(f"    {metric}: {stats['mean']:.4f} +- {stats['std']:.4f}"
                      f"  (over {stats['n_runs']} run(s))")

    report(f"SAMPLED, ACROSS {len(sampled_per_run)} RUNS "
           f"-- Phase 4's reported number (D-036):", eval_sampled, len(sampled_per_run))
    report(f"greedy (argmax), ACROSS {len(greedy_per_run)} RUNS "
           f"-- DIAGNOSTIC ONLY, not the headline (D-036):",
           eval_greedy, len(greedy_per_run))

    if len(sampled_per_run) < MIN_RUNS_TO_REPORT:
        # Not an error: --only-repeat is the parallel-sweep pattern (D-027), and
        # those per-run files are aggregated later. It is a refusal to let a
        # one-run std of 0.00 be quoted as if it were a measured spread.
        print(f"\n  WARNING: {len(sampled_per_run)} run(s) only - NOT a reportable result "
              f"(CONSTRAINTS #3 wants at least {MIN_RUNS_TO_REPORT}). "
              f"Aggregate the per-run files before quoting anything.")

    payload = {
        "tag": TAG,
        "config_hash": cfg_hash,
        "n_episodes": n_episodes,
        "repeats": repeats,
        "seed_start": seed_start,
        "episode_rewards": all_rewards,
        "episode_steps": all_steps,
        "curves": all_curves,
        "td_error_std": all_td_stds,
        "entropies": all_entropies,
        # Named rather than positional, because "eval_summary" is exactly the
        # key a reader would quote without checking which policy produced it.
        # D-036: sampled is the headline, greedy is the diagnostic.
        "eval_policy_convention": "sampled is the reported number (D-036); greedy is diagnostic",
        "eval_sampled_per_run": sampled_per_run,
        "eval_sampled_across_runs": eval_sampled,
        "eval_greedy_per_run": greedy_per_run,
        "eval_greedy_across_runs": eval_greedy,
    }
    suffix = "" if args.only_repeat is None else f"_repeat{args.only_repeat}"
    (results_dir / f"{TAG}{suffix}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    if args.no_plot:
        return

    window = tcfg.common.log_smoothing_window
    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for i, rewards in zip(repeats, all_rewards):
        top.plot(smooth(rewards, window), label=f"repeat {i}", linewidth=0.9)
    top.set_ylabel(f"total reward (sampled, {window}-episode mean)")
    top.set_title(f"{TAG}: training reward")
    top.legend(fontsize=7)

    for i, stds in zip(repeats, all_td_stds):
        middle.plot(smooth(stds, window), label=f"repeat {i}", linewidth=0.9)
    middle.set_ylabel("TD-error std within episode")
    # The actor-critic half of the variance story (ROADMAP box 4). REINFORCE's
    # coefficients are full returns; these are one-step errors, and the
    # difference in scale IS the bias-variance trade.
    middle.set_title("update-signal spread — one-step TD errors, not full returns")

    for i, entropy_series in zip(repeats, all_entropies):
        bottom.plot(smooth(entropy_series, window), label=f"repeat {i}", linewidth=0.9)
    bottom.set_ylabel("policy entropy (nats)")
    bottom.set_xlabel("episode")
    # E-018's degenerate greedy policy, made visible while it happens. ln(5) is
    # the maximum for five actions; a curve heading to 0 has stopped exploring.
    bottom.axhline(np.log(N_ACTIONS), linestyle=":", linewidth=0.8,
                   label=f"uniform = ln({N_ACTIONS})")
    bottom.set_title("policy entropy — 0 means one action in every state (E-018)")
    bottom.legend(fontsize=7)

    figure.tight_layout()
    figure.savefig(results_dir / f"{TAG}_training.png", dpi=140)
    print(f"\nwrote {results_dir / f'{TAG}_training.png'}")


if __name__ == "__main__":
    main()
