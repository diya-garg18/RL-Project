"""ROADMAP Phase 4, box 3 - sample efficiency: DQN vs REINFORCE vs actor-critic.

The question is **how much experience each algorithm needs**, so the x-axis is
ENVIRONMENT STEPS, never episodes. Those are not the same thing here and the
difference is large: measured on identical 4-episode smoke runs, the actor-critic
took 88.0 steps per shift against REINFORCE's 46.8. A 480-minute shift is a fixed
amount of wall-clock, not a fixed amount of experience - a bulk-closing policy
makes far more decisions inside it than one doing slow verifies. Plotting per
episode would hand the sample-hungry agent a free advantage, which is the exact
opposite of what this comparison is for.

Both readings of every agent are plotted, deliberately
------------------------------------------------------
Two curves per agent, and this file refuses to pick between them:

  * **sampled** - the reward the agent actually collected while training, from
    its own stochastic policy.
  * **greedy** - the periodic diagnostic, `argmax_a pi(a|s)` evaluated on the
    train-diagnostic seeds.

E-019 found these disagree violently for REINFORCE: in nine runs out of nine the
sampled policy earned POSITIVE reward while the greedy read of the same policy at
the same moment earned strongly negative reward. E-020 found the same argmax
degeneracy in the actor-critic at every entropy coefficient tested. **A
stochastic policy's argmax is not that policy.**

**That decision was taken on 2026-09-01 as D-036**, before any full Phase 4 run
existed: each algorithm is reported as the policy its own objective optimised -
SAMPLED for the policy-gradient agents, greedy for value-based ones. This script
still plots and labels both, because the gap between them is itself a Phase 4
finding, but the sampled curve is the reported one and the greedy curve is a
diagnostic.

Note the two agents differ sharply here (E-022, E-023). REINFORCE's argmax
recovers by ~episode 2000 and the two readings then agree. The actor-critic's
never does: at the full budget its greedy read is still recall 0.0022 against
the sampled policy's 0.6316. Reported greedy, that agent looks totally
collapsed; it is not.

What it reports, and the honest denominator
-------------------------------------------
"Sample efficiency" needs a target to be efficient *towards*. The target here is
`severity_sort`'s 30-seed mean reward of 40.4 (E-014), because that is the
baseline Phase 4's exit criterion names. For each agent it reports the number of
environment steps before its smoothed curve first reaches that line - and reports
**NEVER REACHED** when it does not, rather than quietly emitting the last point.
Three phases of this project have already ended below their baselines; an
efficiency table that could not express "it never got there" would be unable to
describe its own most likely outcome.

Usage:
    python scripts/compare_sample_efficiency.py
    python scripts/compare_sample_efficiency.py --window 50
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

# E-014, the 30-seed evaluation block. The number Phase 4's exit criterion is
# written against, so it is the line "sample efficiency" is measured towards.
SEVERITY_SORT_REWARD = 40.4

# (directory under results/, human label). Order is the order they were built.
AGENTS = (
    ("dqn_runs/dqn", "DQN (Phase 3)"),
    ("reinforce_runs/reinforce", "REINFORCE"),
    ("actor_critic_runs/actor_critic", "actor-critic"),
)


def load_runs(subdir: str) -> list[dict]:
    """Every repeat for one agent, from either artefact layout.

    Two layouts exist for historical reasons and both are real: `train_dqn.py`
    writes `repeat<N>.json` per process (D-027's parallel pattern), while the
    Phase 4 trainers write one `<tag>.json` holding a list per repeat, or
    `<tag>_repeat<N>.json` when driven with --only-repeat. Rather than force one,
    this normalises them into a list of {rewards, steps, curve} dicts.
    """
    directory = ROOT / "results" / subdir
    if not directory.is_dir():
        return []

    runs: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith("_aggregate.json"):
            # aggregate_phase4.py's output, not a run. It holds eval summaries
            # rather than curves and has no episode_steps by design, so the
            # "re-run the trainer" message below would be actively misleading.
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "episode_steps" not in payload:
            # An artefact from before episode_steps was recorded. Refused rather
            # than approximated: a fabricated x-axis is worse than no plot.
            print(f"  SKIPPED {path.name}: no 'episode_steps' - predates the step "
                  f"recording, re-run the trainer")
            continue
        rewards, steps = payload["episode_rewards"], payload["episode_steps"]
        curves = payload.get("curves") or ([payload["curve"]] if "curve" in payload else [])
        # Per-repeat files hold flat lists; aggregated files hold a list per repeat.
        if rewards and not isinstance(rewards[0], list):
            rewards, steps, curves = [rewards], [steps], curves or [[]]
        for i, (run_rewards, run_steps) in enumerate(zip(rewards, steps)):
            runs.append({
                "rewards": np.asarray(run_rewards, dtype=np.float64),
                "steps": np.asarray(run_steps, dtype=np.int64),
                "curve": curves[i] if i < len(curves) else [],
            })
    return runs


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average, the same smoothing the trainers' plots use."""
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def steps_to_target(cumulative_steps: np.ndarray, smoothed: np.ndarray,
                    window: int, target: float) -> int | None:
    """Environment steps before the smoothed curve first reaches `target`.

    None means never - reported as such rather than silently returning the last
    point, because "it never got there" is a likely and reportable outcome here.
    """
    reached = np.flatnonzero(smoothed >= target)
    if reached.size == 0:
        return None
    # Smoothing consumes (window - 1) leading episodes, so index i of the
    # smoothed array corresponds to episode i + window - 1 of the raw one.
    index = min(reached[0] + window - 1, len(cumulative_steps) - 1)
    return int(cumulative_steps[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 box 3: sample efficiency.")
    parser.add_argument("--window", type=int, default=100,
                        help="smoothing window in episodes (default 100)")
    parser.add_argument("--target", type=float, default=SEVERITY_SORT_REWARD,
                        help="reward level to measure steps-to-reach against")
    args = parser.parse_args()

    print("Phase 4 box 3 - sample efficiency, reward vs ENVIRONMENT STEPS")
    print(f"  target line: {args.target} (severity_sort, 30-seed mean, E-014)")
    print(f"  smoothing:   {args.window} episodes\n")

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    table: list[tuple[str, int, float, int | None]] = []
    found_any = False

    for colour_index, (subdir, label) in enumerate(AGENTS):
        runs = load_runs(subdir)
        if not runs:
            print(f"  {label}: no artefacts in results/{subdir} - not plotted")
            continue
        found_any = True
        colour = f"C{colour_index}"

        for index, run in enumerate(runs):
            cumulative = np.cumsum(run["steps"])
            smoothed = smooth(run["rewards"], args.window)
            offset = len(run["rewards"]) - len(smoothed)
            # One legend entry per agent, not per repeat, or it is unreadable.
            top.plot(cumulative[offset:], smoothed, linewidth=0.9, color=colour,
                     alpha=0.75, label=label if index == 0 else None)

            if run["curve"]:
                positions = [cumulative[min(int(e), len(cumulative)) - 1] for e, _ in run["curve"]]
                values = [g for _, g in run["curve"]]
                bottom.plot(positions, values, marker="o", markersize=3, linewidth=0.8,
                            color=colour, alpha=0.75,
                            label=label if index == 0 else None)

        total_steps = int(np.mean([run["steps"].sum() for run in runs]))
        mean_final = float(np.mean([run["rewards"][-args.window:].mean() for run in runs]))
        reach = steps_to_target(
            np.cumsum(runs[0]["steps"]), smooth(runs[0]["rewards"], args.window),
            args.window, args.target,
        )
        table.append((label, total_steps, mean_final, reach))

    if not found_any:
        raise SystemExit(
            "No artefacts found. Train the agents first:\n"
            "  python scripts/train_dqn.py --only-repeat 0\n"
            "  python scripts/train_reinforce.py\n"
            "  python scripts/train_actor_critic.py"
        )

    print(f"| agent | env steps (mean/run) | final sampled reward | steps to reach {args.target} |")
    print("|---|---|---|---|")
    for label, total_steps, mean_final, reach in table:
        reached = f"{reach:,}" if reach is not None else "**NEVER REACHED**"
        print(f"| {label} | {total_steps:,} | {mean_final:.1f} | {reached} |")

    print("\nBoth readings are plotted. The SAMPLED curve is the reported one")
    print("(D-036): what the agent collected from its own stochastic policy. The")
    print("greedy points are argmax(pi) on the train-diagnostic seeds, kept as a")
    print("diagnostic. They disagree violently for the actor-critic even at the full")
    print("budget - greedy recall 0.0022 against sampled 0.6316 (E-023) - so the")
    print("greedy curve must never be quoted as that agent's result.")

    top.axhline(args.target, linestyle="--", linewidth=0.9, color="grey",
                label=f"severity_sort = {args.target}")
    top.set_ylabel(f"sampled reward ({args.window}-episode mean)")
    top.set_title("Sample efficiency - reward collected while training")
    top.legend(fontsize=8)

    bottom.axhline(args.target, linestyle="--", linewidth=0.9, color="grey")
    bottom.set_ylabel("greedy reward (train-diag seeds)")
    bottom.set_xlabel("environment steps")
    bottom.set_title("The SAME policies read greedily - argmax(pi), not the agent (E-019)")
    bottom.legend(fontsize=8)

    figure.tight_layout()
    out = ROOT / "results" / "sample_efficiency"
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(out / "sample_efficiency.png", dpi=140)

    (out / "sample_efficiency.json").write_text(json.dumps({
        "target": args.target,
        "window": args.window,
        "agents": [
            {"label": label, "env_steps_mean": steps,
             "final_sampled_reward": final, "steps_to_target": reach}
            for label, steps, final, reach in table
        ],
        "evaluation_reading_decision": "OWED - sampled vs greedy, see E-019 section 3",
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'sample_efficiency.png'}")


if __name__ == "__main__":
    main()
