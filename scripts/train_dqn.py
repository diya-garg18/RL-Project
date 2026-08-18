"""Phase 3 training entry point: train the DQN, honestly.

Structurally parallel to `scripts/train.py`, and deliberately a separate file
rather than a branch inside it (rule of three — see ARCHITECTURE.md). The
tabular trainer saves a Q-table and a visit count; this one saves network
weights, plots a loss curve the tabular learners do not have, and carries two
ablation switches. Merging them would mean a script whose every second line is
an `if isinstance(agent, ...)`.

The honesty machinery is identical, and carried over deliberately:

  1. train on a dedicated seed block — one fresh shift per episode (D-016)
  2. every `eval_every` episodes, freeze exploration and measure the greedy
     policy on the TRAIN-diagnostic seeds — never on the eval seeds
  3. repeat over several agent seeds, because a single run is not a result
     (CONSTRAINTS #3)
  4. only at the very end, evaluate on the eval seeds

The ordering of 2 and 4 is the point. Evaluation-seed numbers are computed once,
after every training decision has been made, so nothing here can tune against
them (CONSTRAINTS #2).

Usage:
    python scripts/train_dqn.py                                  # full run
    python scripts/train_dqn.py --episodes 40 --repeats 1 --no-plot   # smoke test
    python scripts/train_dqn.py --no-replay                      # ablation
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

from soc_triage.agents.baselines import make_baselines
from soc_triage.agents.dqn import DQNAgent
from soc_triage.config import EnvConfig, TrainingConfig, load_env_config, load_training_config
from soc_triage.env import SOCTriageEnv
from soc_triage.evaluation.metrics import summarise
from soc_triage.runner import config_hash, run_episode, run_episodes, save_records
from soc_triage.state import FEATURE_NAMES, feature_scale_vector

# Structural, not tunable: the action set is defined by the MDP in
# PROJECT_BRIEF.md §3 and changing it is a change to the problem, not a knob.
N_ACTIONS = 5


def run_tag(no_replay: bool, no_target_network: bool) -> str:
    """Artefact name for this condition.

    Ablation runs get their own filename so one can never overwrite the
    control's `dqn.pt`. Same reasoning as the results/smoke/ guard (D-018): the
    corruption is silent, and the stale file looks entirely valid.
    """
    if no_replay and no_target_network:
        return "dqn_no_replay_no_target_network"
    if no_replay:
        return "dqn_no_replay"
    if no_target_network:
        return "dqn_no_target_network"
    return "dqn"


def build_agent(tcfg: TrainingConfig, seed: int, no_replay: bool, no_target_network: bool) -> DQNAgent:
    """Every hyperparameter comes from config; only the seed and the ablation
    switches are injected here."""
    dcfg = replace(tcfg.dqn, no_replay=no_replay, no_target_network=no_target_network)
    return DQNAgent(
        obs_dim=len(FEATURE_NAMES),
        n_actions=N_ACTIONS,
        dcfg=dcfg,
        gamma=tcfg.common.gamma,
        epsilon_start=tcfg.epsilon.start,
        epsilon_min=tcfg.epsilon.min,
        epsilon_decay=tcfg.epsilon.decay,
        seed=seed,
        feature_scales=feature_scale_vector(dcfg.feature_scales),
    )


def greedy_diagnostic(
    env: SOCTriageEnv,
    agent: DQNAgent,
    cfg: EnvConfig,
    cfg_hash: str,
) -> float:
    """Mean total reward of the CURRENT greedy policy on the train-diagnostic seeds.

    Two things are deliberately switched off. Exploration is pinned to 0, so the
    curve shows what the learned policy is worth rather than what a partly-random
    agent happens to score. And `learn=False`, so measuring never changes what is
    being measured.

    The seeds here are `cfg.seeds.train` (1-10) — training seeds, held out from
    the 1000000+ block the agent actually trains on, and disjoint from the eval
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
    seed_start: int,
    no_replay: bool,
    no_target_network: bool,
) -> tuple[DQNAgent, list[float], list[tuple[int, float]], list[float]]:
    """One complete training run.

    Returns (agent, per-episode rewards, diagnostic curve, per-episode loss).

    Each repeat gets its own agent seed AND its own slice of the training seed
    block, so the repeats differ in both the exploration randomness and the
    shifts encountered. Varying only the agent seed would leave every run facing
    an identical alert stream, and the resulting std would understate the real
    variability.
    """
    env = SOCTriageEnv(cfg)
    agent = build_agent(tcfg, repeat_index, no_replay, no_target_network)
    seed_base = seed_start + repeat_index * n_episodes

    episode_rewards: list[float] = []
    episode_losses: list[float] = []
    curve: list[tuple[int, float]] = []
    t0 = time.perf_counter()

    for episode in range(n_episodes):
        record = run_episode(env, agent, seed_base + episode, cfg, cfg_hash, learn=True)
        episode_rewards.append(record["outcome"]["total_reward"])
        # nan until the buffer passes learning_starts; plotted as a gap rather
        # than as a zero, which would read as "the loss was low here".
        episode_losses.append(agent.last_loss)
        # The one line the whole epsilon schedule depends on (D-015). Without it
        # epsilon stays at its start value forever and the agent never converges
        # — silently, with no error.
        agent.end_episode()

        if (episode + 1) % eval_every == 0:
            curve.append((episode + 1, greedy_diagnostic(env, agent, cfg, cfg_hash)))
            elapsed = time.perf_counter() - t0
            print(
                f"  repeat {repeat_index}  ep {episode + 1}/{n_episodes}  "
                f"eps {agent.epsilon:.3f}  grad steps {agent.gradient_steps}  "
                f"loss {agent.last_loss:8.2f}  "
                f"greedy(train-diag) {curve[-1][1]:8.1f}  [{elapsed / 60:.1f} min]"
            )

    return agent, episode_rewards, curve, episode_losses


def smooth(values: list[float], window: int) -> np.ndarray:
    """Trailing moving average — the smoothing the roadmap asks for on learning curves."""
    if len(values) < window:
        return np.array(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DQN agent on SOC triage.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override common.n_episodes (for smoke tests)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="independent training runs; >=5 for a reportable number")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="override common.eval_every")
    parser.add_argument("--only-repeat", type=int, default=None,
                        help="run ONLY this repeat index and write its result as JSON, so "
                             "repeats can run as parallel processes (each uses one core). "
                             "Combine them afterwards with scripts/aggregate_dqn.py.")
    parser.add_argument("--no-plot", action="store_true", help="skip the PNGs")
    parser.add_argument("--no-replay", action="store_true",
                        help="ABLATION: train on the latest transition only")
    parser.add_argument("--no-target-network", action="store_true",
                        help="ABLATION: bootstrap off the online network")
    args = parser.parse_args()

    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")
    cfg_hash = config_hash(ROOT / "config" / "env_default.yaml")

    # A CLI flag overrides the config's ablation booleans, so the shipped YAML
    # always describes the control condition and an ablation is always visible
    # in the command that produced it.
    no_replay = args.no_replay or tcfg.dqn.no_replay
    no_target_network = args.no_target_network or tcfg.dqn.no_target_network
    is_ablation = no_replay or no_target_network
    tag = run_tag(no_replay, no_target_network)

    n_episodes = args.episodes if args.episodes is not None else tcfg.common.n_episodes
    eval_every = args.eval_every if args.eval_every is not None else tcfg.common.eval_every
    # Ablations train on their own seed block so the control and the ablated
    # conditions never share alert streams — the one confound an ablation
    # cannot have.
    seed_start = tcfg.dqn.ablation_seed_start if is_ablation else tcfg.dqn.train_seed_start

    d = tcfg.dqn
    print(f"{tag}: {args.repeats} runs x {n_episodes} episodes")
    print(f"  net {list(d.hidden_layers)} {d.activation}  lr {d.lr}  batch {d.batch_size}  "
          f"gamma {tcfg.common.gamma}")
    print(f"  replay {'OFF (ablation)' if no_replay else f'{d.replay_capacity}'}  "
          f"target net {'OFF (ablation)' if no_target_network else f'every {d.target_update_every} grad steps'}")
    print(f"  train_freq {d.train_freq}  learning_starts {d.learning_starts}  "
          f"grad_clip {d.grad_clip_norm}  loss {d.loss}")
    print(f"  eps {tcfg.epsilon.start} -> {tcfg.epsilon.min} @ {tcfg.epsilon.decay}/episode")
    print(f"  training seeds from {seed_start} "
          f"(disjoint from train-diag {list(cfg.seeds.train)} and eval {list(cfg.seeds.eval)})")

    # --- single-repeat mode: this process IS one repeat.
    #
    # Each training process is single-threaded (dqn.py pins
    # torch.set_num_threads(1), which measured fastest on a net this small), so
    # N repeats can run as N processes on N cores instead of sequentially. At
    # 20000 episodes that is the difference between ~70 min and ~11 hours for a
    # 10-run condition.
    #
    # seed_base below depends only on repeat_index and n_episodes — never on how
    # many repeats are running — so a repeat computed in parallel gets exactly
    # the same alert stream it would have got sequentially. That is what makes
    # the two paths comparable rather than merely similar.
    if args.only_repeat is not None:
        repeat = args.only_repeat
        t0 = time.perf_counter()
        agent, rewards, curve, losses = train_one_run(
            cfg, tcfg, cfg_hash, repeat, n_episodes, eval_every,
            seed_start, no_replay, no_target_network,
        )
        env = SOCTriageEnv(cfg)
        agent.epsilon = 0.0  # report the learned policy, not a partly-random one
        records = run_episodes(env, agent, cfg.seeds.eval, cfg, cfg_hash, learn=False)
        summary = summarise(records, cfg)

        out_dir = ROOT / "results" / "dqn_runs" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        agent.save(str(out_dir / f"repeat{repeat}.pt"))
        # Per-seed total rewards are kept so the aggregator can do a PAIRED
        # comparison against tabular Q-learning on the identical eval shifts.
        # Pairing cancels most of the shift-to-shift variance that makes the
        # unpaired spreads (severity_sort +/-220.1) wider than the effects being
        # measured — the lower-variance protocol E-013 asked for.
        (out_dir / f"repeat{repeat}.json").write_text(
            json.dumps(
                {
                    "tag": tag,
                    "repeat": repeat,
                    "n_episodes": n_episodes,
                    "eval_every": eval_every,
                    "seed_base": seed_start + repeat * n_episodes,
                    "config_hash": cfg_hash,
                    "no_replay": no_replay,
                    "no_target_network": no_target_network,
                    "wall_min": (time.perf_counter() - t0) / 60,
                    "episode_rewards": rewards,
                    "episode_losses": losses,
                    "curve": curve,
                    "summary": summary,
                    "eval_seeds": list(cfg.seeds.eval),
                    "per_seed_total_reward": [
                        r["outcome"]["total_reward"] for r in records
                    ],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        rel = out_dir.relative_to(ROOT).as_posix()
        print(f"\nrepeat {repeat} done in {(time.perf_counter() - t0) / 60:.1f} min "
              f"-> {rel}/repeat{repeat}.json")
        print("  combine the repeats with: python scripts/aggregate_dqn.py")
        return

    agents: list[DQNAgent] = []
    all_rewards: list[list[float]] = []
    all_curves: list[list[tuple[int, float]]] = []
    all_losses: list[list[float]] = []

    t_start = time.perf_counter()
    for repeat in range(args.repeats):
        agent, rewards, curve, losses = train_one_run(
            cfg, tcfg, cfg_hash, repeat, n_episodes, eval_every,
            seed_start, no_replay, no_target_network,
        )
        agents.append(agent)
        all_rewards.append(rewards)
        all_curves.append(curve)
        all_losses.append(losses)
    wall_min = (time.perf_counter() - t_start) / 60
    print(f"training done in {wall_min:.1f} min")

    # --- learning curve and loss curve
    if not args.no_plot:
        (ROOT / "results").mkdir(exist_ok=True)
        fig, (ax, ax_loss) = plt.subplots(2, 1, figsize=(8, 7), sharex=False)
        for repeat, rewards in enumerate(all_rewards):
            smoothed = smooth(rewards, tcfg.common.log_smoothing_window)
            ax.plot(range(len(smoothed)), smoothed, alpha=0.7, linewidth=1, label=f"run {repeat}")
        ax.set_xlabel(f"episode (trailing mean over {tcfg.common.log_smoothing_window})")
        ax.set_ylabel("total reward per episode")
        ax.set_title(f"{tag} on SOC triage — {args.repeats} runs")
        ax.legend(fontsize=8)

        # The loss curve is the plot that makes an ablation visible. A DQN whose
        # target moves with its own weights shows a loss that will not settle,
        # which the reward curve alone can hide.
        for repeat, losses in enumerate(all_losses):
            smoothed = smooth(losses, tcfg.common.log_smoothing_window)
            ax_loss.plot(range(len(smoothed)), smoothed, alpha=0.7, linewidth=1,
                         label=f"run {repeat}")
        ax_loss.set_xlabel("episode")
        ax_loss.set_ylabel("Huber loss (last step of episode)")
        ax_loss.set_yscale("log")
        ax_loss.legend(fontsize=8)

        fig.tight_layout()
        plot_path = ROOT / "results" / f"{tag}_curve.png"
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

    def across_runs(metric: str) -> tuple[float | None, float | None]:
        """Mean and std ACROSS runs of each run's mean — never a single run
        (CONSTRAINTS #3).

        A run's mean is None when the metric is undefined for it: `summarise`
        reports mttd_min as None when no episode caught an incident, which is
        the honest answer rather than zero. Runs like that are dropped from the
        average and, if none survive, the metric is reported as undefined. An
        agent that catches nothing is a real outcome here — it is what a
        destabilised ablation looks like — so this path is load-bearing, not
        defensive padding.
        """
        means = [s[metric]["mean"] for s in per_run_summaries if s[metric]["mean"] is not None]
        if not means:
            return None, None
        return float(np.mean(means)), float(np.std(means))

    def fmt(value: float | None, std: float | None, width: int, places: int) -> str:
        """Never print a number for an undefined metric."""
        if value is None:
            return f"{'undefined':>{width}}"
        return f"{value:{width}.{places}f}±{std:.{places}f}"

    print(f"\n{tag} vs references on eval seeds (mean ± std across {args.repeats} runs):")
    rows: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
        tag: (
            across_runs("recall_at_deadline"),
            across_runs("total_reward"),
            across_runs("mttd_min"),
        )
    }
    for baseline in make_baselines(cfg, 12345):
        if baseline.name not in ("severity_sort", "oracle_greedy"):
            continue
        s = summarise(run_episodes(env, baseline, cfg.seeds.eval, cfg, cfg_hash), cfg)
        rows[baseline.name] = (
            (s["recall_at_deadline"]["mean"], s["recall_at_deadline"]["std"]),
            (s["total_reward"]["mean"], s["total_reward"]["std"]),
            (s["mttd_min"]["mean"], s["mttd_min"]["std"]),
        )

    for name, (recall, reward, mttd) in rows.items():
        print(f"  {name:32s} recall {fmt(recall[0], recall[1], 4, 2)}   "
              f"reward {fmt(reward[0], reward[1], 7, 1)}   "
              f"mttd {fmt(mttd[0], mttd[1], 6, 1)}")

    undefined = sum(s["mttd_undefined_episodes"] for s in per_run_summaries)
    if undefined:
        print(f"\n  {undefined} of {sum(s['n_episodes'] for s in per_run_summaries)} "
              "learner episodes caught no incident, so MTTD is undefined for them "
              "and they are excluded from the MTTD mean (not counted as zero).")

    print("\n  NOTE: the baseline rows show std across EVAL SEEDS (one deterministic")
    print("  run each); the learner row shows std across TRAINING RUNS. Different")
    print("  quantities — do not read the two spreads as comparable.")

    # A reduced run must NEVER overwrite the artefacts of a full one. This bit
    # the project once: a `--episodes 200` smoke test silently replaced a real
    # 20000-episode Q-table, and the corruption only surfaced later as an
    # unexplained drop in state coverage in scripts/compare_agents.py. Nothing
    # errored, and the stale file looked entirely valid. CONSTRAINTS #4 forbids
    # overwriting an experiment result; this makes the config-faithful path the
    # only one that can.
    is_full_run = (
        n_episodes == tcfg.common.n_episodes
        and eval_every == tcfg.common.eval_every
        and args.repeats >= 5
    )
    out_dir = ROOT / "results" if is_full_run else ROOT / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    agents[0].save(str(out_dir / f"{tag}.pt"))
    rel = out_dir.relative_to(ROOT).as_posix()
    print(f"\nrun-0 network  -> {rel}/{tag}.pt (gitignored, regenerable)")
    print(f"wall clock     -> {wall_min:.1f} min")
    if not is_full_run:
        print("  REDUCED RUN — written to results/smoke/ so it cannot be mistaken")
        print("  for, or overwrite, a full run's artefacts.")


if __name__ == "__main__":
    main()
