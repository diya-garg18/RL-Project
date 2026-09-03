"""Phase 4 training entry point: train REINFORCE, honestly.

A separate file from `train.py` (tabular) and `train_dqn.py` (Phase 3), for the
same reason those two are separate — D-025. This trainer saves two networks
rather than one, has no epsilon to report, and carries a baseline ablation the
others do not. Folding it into either would produce a script whose every second
line asks which algorithm it is driving.

The honesty machinery is carried over unchanged, because it is the part that
must not vary between phases:

  1. train on a dedicated seed block — one fresh shift per episode (D-016)
  2. every `eval_every` episodes, measure the *greedy* policy on the
     TRAIN-diagnostic seeds — never on the eval seeds
  3. repeat over several agent seeds; one run is not a result (CONSTRAINTS #3)
  4. only at the very end, evaluate on the eval seeds

The order of 2 and 4 is the point: evaluation-seed numbers are computed once,
after every training decision has already been made (CONSTRAINTS #2).

One thing here differs from the DQN trainer and it is not cosmetic. REINFORCE's
policy is stochastic, so "the greedy policy" is a *reading* of it (argmax over
the action probabilities), not how the agent behaves. Both are logged: the
greedy diagnostic says what the policy has learned, and the sampled training
reward says what the agent actually collected while learning. When those two
diverge the agent is still exploring; when they converge the policy has sharpened.

Usage:
    python scripts/train_reinforce.py --episodes 200 --repeats 1 --no-plot  # smoke
    python scripts/train_reinforce.py                                       # full run
    python scripts/train_reinforce.py --no-baseline                         # ablation
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; we only write a PNG
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.agents.reinforce import ReinforceAgent
from soc_triage.config import EnvConfig, TrainingConfig, load_env_config, load_training_config
from soc_triage.env import SOCTriageEnv
from soc_triage.evaluation.metrics import MIN_RUNS_TO_REPORT, summarise
from soc_triage.runner import config_hash, run_episode, run_episodes, save_records
from soc_triage.state import FEATURE_NAMES, feature_scale_vector

# Structural, not tunable: the action set is defined by the MDP in
# PROJECT_BRIEF.md §3, and changing it changes the problem, not a knob.
N_ACTIONS = 5


def run_tag(no_baseline: bool) -> str:
    """Ablation runs get their own filename so one can never overwrite the
    control's. Same reasoning as the results/smoke/ guard (D-018): the
    corruption is silent and the stale file looks entirely valid."""
    return "reinforce_no_baseline" if no_baseline else "reinforce"


def build_agent(tcfg: TrainingConfig, seed: int, no_baseline: bool) -> ReinforceAgent:
    """Every hyperparameter comes from config; only the seed and the ablation
    switch are injected here."""
    rcfg = replace(tcfg.reinforce, use_baseline=not no_baseline)
    return ReinforceAgent(
        obs_dim=len(FEATURE_NAMES),
        n_actions=N_ACTIONS,
        rcfg=rcfg,
        gamma=tcfg.common.gamma,
        seed=seed,
        # The SHARED scales (D-032), the same vector the DQN gets, so the Phase 4
        # sample-efficiency comparison compares algorithms rather than
        # preprocessing.
        feature_scales=feature_scale_vector(tcfg.features.scales),
    )


class _GreedyView:
    """The agent's policy read deterministically, for diagnostics only.

    The DQN needed no equivalent: pinning epsilon to 0 made it greedy. A
    policy-gradient agent has no epsilon — its randomness *is* the policy — so
    "what has it learned" and "what does it do" are genuinely two questions, and
    this wrapper answers the first without touching the agent's own RNG or its
    buffer. `update()` is a no-op because the runner is called with learn=False;
    it exists so this satisfies the Agent interface (CONSTRAINTS #10).
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


def greedy_diagnostic(
    env: SOCTriageEnv,
    agent: ReinforceAgent,
    cfg: EnvConfig,
    cfg_hash: str,
) -> float:
    """Mean total reward of the current policy's ARGMAX on the train-diagnostic seeds.

    As in the DQN trainer these are the train-diagnostic seeds (1-10), never the
    eval block: a curve measured on evaluation seeds is a tuning signal, and
    looking at it even once compromises the final number (CONSTRAINTS #2).
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
    no_baseline: bool,
) -> tuple[ReinforceAgent, list[float], list[tuple[int, float]], list[float]]:
    """One complete training run.

    Returns (agent, per-episode sampled reward, greedy diagnostic curve,
    per-episode policy gradient norm).

    Each repeat gets its own agent seed AND its own slice of the training seed
    block, so repeats differ in both the sampling randomness and the shifts
    encountered. Varying only the agent seed would leave every run facing an
    identical alert stream, and the resulting std would understate the real
    variability.
    """
    env = SOCTriageEnv(cfg)
    agent = build_agent(tcfg, repeat_index, no_baseline)
    seed_base = seed_start + repeat_index * n_episodes

    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    grad_norms: list[float] = []
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
        # The entire update happens here. Forgetting this line trains nothing at
        # all — REINFORCE buffers during the episode and learns only at its end
        # (the same D-015 trap the tabular learners have, with a worse failure:
        # not a stuck epsilon but no learning whatsoever).
        agent.end_episode()
        # Logged per episode because the gradient's magnitude IS the variance
        # story ROADMAP box 4 asks for — with and without the baseline.
        grad_norms.append(agent.last_policy_grad_norm)

        if (episode + 1) % eval_every == 0:
            curve.append((episode + 1, greedy_diagnostic(env, agent, cfg, cfg_hash)))
            elapsed = time.perf_counter() - t0
            recent = float(np.mean(episode_rewards[-eval_every:]))
            print(
                f"  repeat {repeat_index}  ep {episode + 1}/{n_episodes}  "
                f"sampled {recent:8.1f}  grad_norm {agent.last_policy_grad_norm:8.2f}  "
                f"greedy(train-diag) {curve[-1][1]:8.1f}  [{elapsed / 60:.1f} min]"
            )

    return agent, episode_rewards, episode_steps, curve, grad_norms


def smooth(values: list[float], window: int) -> np.ndarray:
    """Trailing moving average — the smoothing the roadmap asks for on curves."""
    if len(values) < window:
        return np.array(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the REINFORCE agent on SOC triage.")
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
    parser.add_argument("--no-baseline", action="store_true",
                        help="ABLATION: subtract nothing; coefficients are the raw returns")
    args = parser.parse_args()

    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")
    cfg_hash = config_hash(ROOT / "config" / "env_default.yaml")

    # The CLI flag is the only way to run the ablation, so the shipped YAML
    # always describes the control and an ablation is always visible in the
    # command that produced it.
    no_baseline = args.no_baseline
    tag = run_tag(no_baseline)

    n_episodes = args.episodes if args.episodes is not None else tcfg.common.n_episodes
    eval_every = args.eval_every if args.eval_every is not None else tcfg.common.eval_every
    seed_start = (
        tcfg.reinforce.ablation_seed_start if no_baseline else tcfg.reinforce.train_seed_start
    )

    r = tcfg.reinforce
    print(f"{tag}: {args.repeats} runs x {n_episodes} episodes")
    print(f"  policy net {list(r.hidden_layers)} {r.activation}  lr {r.lr}  "
          f"gamma {tcfg.common.gamma}")
    print(f"  baseline {'OFF (ablation)' if no_baseline else f'ON, lr {r.baseline_lr}'}  "
          f"grad_clip {r.grad_clip_norm}")
    print("  exploration: none scheduled — the policy is stochastic and sharpens on its own")
    print(f"  training seeds from {seed_start} "
          f"(disjoint from train-diag {list(cfg.seeds.train)} and eval {list(cfg.seeds.eval)})")

    # A reduced run must NEVER land where a full one's artefacts go (D-018,
    # CONSTRAINTS #4). The tabular and DQN trainers have had this guard since a
    # `--episodes 200` smoke test silently replaced a real 20000-episode
    # Q-table; Phase 4 was built without it, and this session's own smoke runs
    # had to be deleted by hand from the full run's directory (BUG_004).
    #
    # `--only-repeat` is exempt on purpose: it is one slice of a full parallel
    # sweep (D-027), so it is a full run that happens to compute a single
    # repeat. Judging it by `--repeats` would divert the real sweep into smoke/.
    is_full_run = (
        n_episodes == tcfg.common.n_episodes
        and eval_every == tcfg.common.eval_every
        and (args.repeats >= MIN_RUNS_TO_REPORT or args.only_repeat is not None)
    )
    base_dir = ROOT / "results" if is_full_run else ROOT / "results" / "smoke"
    results_dir = base_dir / "reinforce_runs" / tag
    if not is_full_run:
        print(f"\n  REDUCED RUN - writing to {results_dir.relative_to(ROOT).as_posix()}/ "
              f"so it cannot be mistaken for, or overwrite, a full run's artefacts.")
    results_dir.mkdir(parents=True, exist_ok=True)

    repeats = [args.only_repeat] if args.only_repeat is not None else list(range(args.repeats))
    all_rewards: list[list[float]] = []
    all_steps: list[list[int]] = []
    all_curves: list[list[tuple[int, float]]] = []
    all_grad_norms: list[list[float]] = []
    trained_agents: list[ReinforceAgent] = []

    for repeat_index in repeats:
        agent, rewards, steps, curve, grad_norms = train_one_run(
            cfg, tcfg, cfg_hash, repeat_index, n_episodes, eval_every, seed_start, no_baseline
        )
        all_rewards.append(rewards)
        all_steps.append(steps)
        all_curves.append(curve)
        all_grad_norms.append(grad_norms)
        trained_agents.append(agent)
        agent.save(str(results_dir / f"{tag}_repeat{repeat_index}.pt"))

    assert trained_agents

    # --- Only now does anything touch the evaluation seeds (CONSTRAINTS #2).
    #
    # EVERY repeat is evaluated, not just the last one. Until 2026-09-01 this
    # block evaluated a single `final_agent`, which made the reported number one
    # training run dressed up as a result — 30 eval seeds, but one agent, so its
    # spread described seed difficulty and said nothing about run-to-run
    # variation. CONSTRAINTS #3 forbids exactly that, and every other phase
    # already did it properly (`train.py`'s across_runs, `aggregate_dqn.py`).
    # See docs/bugs/BUG_004.
    # D-036: the SAMPLED policy is Phase 4's headline and the greedy read is a
    # named diagnostic. Both are computed here, on the same agents and the same
    # seeds, so neither can later be quietly substituted for the other. The
    # reason the two differ is not subtle — E-019 found the sampled policy
    # earning positive reward while its own argmax scored -515.4 in nine runs of
    # nine — so reporting one without the other hides the finding.
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
        # Keyed on the REPEAT INDEX, not the loop position. Under --only-repeat
        # every process is at position 0, so five parallel repeats would each
        # write these same two directories at the same time and interleave into
        # each other's files. Only repeat 0 dumps records; the numbers that get
        # reported live in the JSON payload, which is per-process and safe.
        if repeat_index == 0:
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
        "tag": tag,
        "config_hash": cfg_hash,
        "n_episodes": n_episodes,
        "repeats": repeats,
        "no_baseline": no_baseline,
        "seed_start": seed_start,
        "episode_rewards": all_rewards,
        "episode_steps": all_steps,
        "curves": all_curves,
        "grad_norms": all_grad_norms,
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
    (results_dir / f"{tag}{suffix}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    if args.no_plot:
        return

    window = tcfg.common.log_smoothing_window
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for i, rewards in zip(repeats, all_rewards):
        top.plot(smooth(rewards, window), label=f"repeat {i}", linewidth=0.9)
    top.set_ylabel(f"total reward (sampled, {window}-episode mean)")
    top.set_title(f"{tag}: training reward")
    top.legend(fontsize=7)

    for i, norms in zip(repeats, all_grad_norms):
        bottom.plot(smooth(norms, window), label=f"repeat {i}", linewidth=0.9)
    bottom.set_ylabel("policy gradient norm")
    bottom.set_xlabel("episode")
    # The variance story (ROADMAP box 4) is read off this panel: without a
    # baseline the norm is both larger and noisier, because every coefficient is
    # a full return rather than a deviation from one.
    bottom.set_title("gradient magnitude — the variance the baseline is there to reduce")
    figure.tight_layout()
    figure.savefig(results_dir / f"{tag}_training.png", dpi=140)
    print(f"\nwrote {results_dir / f'{tag}_training.png'}")


if __name__ == "__main__":
    main()
