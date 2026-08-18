"""Combine the per-repeat JSONs from a parallel sweep into one result.

`train_dqn.py --only-repeat K` writes one JSON per run so the repeats can be
computed as parallel processes. This script is the other half: it reads them
back, checks they are actually comparable, and produces the learning curves and
the metric table that the sequential trainer would have printed.

The checking is the part that matters. A directory of JSON files looks valid
whatever is in it, so this refuses to average runs that disagree on episode
count, config hash, or ablation flags rather than silently reporting a mean
over a mixture. CONSTRAINTS #4 forbids overwriting an experiment result; a
quietly wrong aggregate is worse, because nothing is left to notice.

Usage:
    python scripts/aggregate_dqn.py                    # the control condition
    python scripts/aggregate_dqn.py --tag dqn_no_replay
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.config import load_training_config

RUNS_DIR = ROOT / "results" / "dqn_runs"


def load_runs(tag: str) -> list[dict]:
    """Every repeat for one condition, sorted by index and checked for agreement.

    Raises rather than returning a partial set: an aggregate silently computed
    over three runs when thirty were expected is a number nobody can catch by
    reading it.
    """
    directory = RUNS_DIR / tag
    if not directory.is_dir():
        raise SystemExit(f"no results for '{tag}' — expected {directory}")
    paths = sorted(directory.glob("repeat*.json"), key=lambda p: int(p.stem[6:]))
    if not paths:
        raise SystemExit(f"{directory} contains no repeat*.json")

    runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]

    # Every field here would produce a plausible-looking but meaningless mean if
    # it varied across the runs being averaged.
    for field in ("n_episodes", "config_hash", "no_replay", "no_target_network", "tag"):
        values = {r[field] for r in runs}
        if len(values) > 1:
            raise SystemExit(
                f"runs in {directory} disagree on '{field}': {sorted(values)}\n"
                f"These are not the same experiment and must not be averaged. "
                f"Delete the odd ones out or re-run them."
            )
    return runs


def curve_matrix(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(episodes, values) where values is (n_runs, n_points).

    The diagnostic curve is measured on the TRAIN-diagnostic seeds (1-10), never
    the eval seeds — see train_dqn.greedy_diagnostic.
    """
    lengths = {len(r["curve"]) for r in runs}
    if len(lengths) > 1:
        raise SystemExit(f"runs have different curve lengths: {sorted(lengths)}")
    episodes = np.array([point[0] for point in runs[0]["curve"]], dtype=np.int64)
    values = np.array([[point[1] for point in r["curve"]] for r in runs], dtype=np.float64)
    return episodes, values


def across_runs(runs: list[dict], metric: str) -> tuple[float | None, float | None, int]:
    """Mean and std ACROSS runs of each run's mean, plus the number contributing.

    A run contributes None when the metric is undefined for it — `summarise`
    reports mttd_min as None when no episode caught an incident, which is the
    honest answer rather than zero. Returning the count makes a shrunken sample
    visible instead of letting it hide inside the mean.
    """
    means = [r["summary"][metric]["mean"] for r in runs]
    present = [m for m in means if m is not None]
    if not present:
        return None, None, 0
    return float(np.mean(present)), float(np.std(present)), len(present)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate parallel DQN repeats.")
    parser.add_argument("--tag", default="dqn",
                        help="condition to aggregate: dqn, dqn_no_replay, "
                             "dqn_no_target_network")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")
    runs = load_runs(args.tag)
    n_runs = len(runs)
    n_episodes = runs[0]["n_episodes"]

    print(f"{args.tag}: {n_runs} runs x {n_episodes} episodes")
    print(f"  config hash {runs[0]['config_hash']}")
    print(f"  seed bases  {runs[0]['seed_base']} .. {runs[-1]['seed_base']}")
    print(f"  wall clock  {sum(r['wall_min'] for r in runs) / 60:.1f} core-hours "
          f"({np.mean([r['wall_min'] for r in runs]):.1f} min/run)")

    # A reduced run must never be mistaken for a full one (D-018). The trainer
    # applies this per-process; the aggregate has to apply it too, because the
    # report is what anyone actually reads.
    is_full = n_episodes == tcfg.common.n_episodes and n_runs >= 5
    out_dir = ROOT / "results" if is_full else ROOT / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not is_full:
        print(f"  REDUCED ({n_episodes} episodes, {n_runs} runs) — writing to "
              f"results/smoke/")

    episodes, values = curve_matrix(runs)
    mean_curve, std_curve = values.mean(axis=0), values.std(axis=0)

    print(f"\ngreedy diagnostic on train seeds {list(range(1, 11))} "
          f"(mean ± std across {n_runs} runs):")
    # Print a handful of points rather than all forty, chosen at fixed fractions
    # so the shape is visible without scrolling.
    for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        i = min(int(frac * (len(episodes) - 1)), len(episodes) - 1)
        print(f"  ep {episodes[i]:>6}   {mean_curve[i]:9.1f} ± {std_curve[i]:.1f}")

    # Has it plateaued? Comparing the last quarter against the preceding quarter
    # answers the question the episode budget actually turns on, and answers it
    # from data rather than from the assumption that 20000 was enough.
    quarter = max(1, len(episodes) // 4)
    late, prior = mean_curve[-quarter:].mean(), mean_curve[-2 * quarter:-quarter].mean()
    spread = std_curve[-quarter:].mean()
    print(f"\n  last quarter {late:.1f} vs previous quarter {prior:.1f} "
          f"(difference {late - prior:+.1f}, run-to-run spread ±{spread:.1f})")
    print("  " + ("PLATEAUED — the difference is inside the spread, so more "
                  "episodes would buy nothing"
                  if abs(late - prior) < spread else
                  "STILL MOVING — the difference exceeds the spread; more "
                  "episodes may still help"))

    print(f"\neval-seed metrics (mean ± std across {n_runs} runs):")
    for metric in ("recall_at_deadline", "total_reward", "mttd_min"):
        mean, std, n = across_runs(runs, metric)
        if mean is None:
            print(f"  {metric:20s} undefined in all {n_runs} runs")
        else:
            note = "" if n == n_runs else f"   [only {n}/{n_runs} runs defined]"
            print(f"  {metric:20s} {mean:9.2f} ± {std:.2f}{note}")

    # Standard error is the quantity the exit criterion turns on, and it is the
    # one E-014 found nobody had checked: five seeds met CONSTRAINTS #3 and were
    # still too few to resolve the effects being claimed.
    mean, std, n = across_runs(runs, "total_reward")
    if mean is not None and n > 1:
        sem = std / np.sqrt(n)
        print(f"\n  total_reward SEM = {sem:.2f} over {n} runs")
        print(f"  tabular q_learning is 47.6 (30-seed). A difference smaller than "
              f"~{2 * sem:.0f} is not resolvable at this sample size.")

    if not args.no_plot:
        fig, (ax, ax_loss) = plt.subplots(2, 1, figsize=(9, 7))
        ax.plot(episodes, mean_curve, linewidth=2, label=f"mean of {n_runs} runs")
        ax.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.25, label="±1 std across runs")
        ax.set_xlabel("training episode")
        ax.set_ylabel("greedy total reward (train-diag seeds)")
        ax.set_title(f"{args.tag} — {n_runs} runs x {n_episodes} episodes")
        ax.legend(fontsize=8)

        window = tcfg.common.log_smoothing_window
        losses = np.array([r["episode_losses"] for r in runs], dtype=np.float64)
        # nanmean: the loss is nan until the buffer passes learning_starts, and
        # plotting those as zero would read as "the loss was low here".
        with np.errstate(invalid="ignore"):
            mean_loss = np.nanmean(losses, axis=0)
        kernel = np.ones(window) / window
        smoothed = np.convolve(mean_loss[~np.isnan(mean_loss)], kernel, mode="valid")
        ax_loss.plot(smoothed, linewidth=1)
        ax_loss.set_yscale("log")
        ax_loss.set_xlabel(f"episode (trailing mean over {window})")
        ax_loss.set_ylabel("Huber loss")

        fig.tight_layout()
        path = out_dir / f"{args.tag}_curve.png"
        fig.savefig(path, dpi=120)
        print(f"\ncurves -> {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
