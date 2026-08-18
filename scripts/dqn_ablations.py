"""Phase 3's required ablations: what replay and the target network are worth.

ROADMAP Phase 3: *"train DQN with replay off, and with the target network off.
Show the instability."* This script is the "show" half — the training itself is
done by `run_dqn_sweep.py`, which puts each condition on its own runs.

What "instability" is taken to mean, stated before looking
---------------------------------------------------------
"Visibly destabilise" is not a measurement, so three are defined here and all
three are reported whichever way they come out:

  1. **Volatility** — mean |change| between consecutive points of the greedy
     diagnostic curve. A learner chasing a moving target oscillates; one that
     has converged does not.
  2. **Divergence between runs** — the across-run std at the end of training.
     An unstable learner ends up somewhere different every time.
  3. **Drawdown** — the largest fall from a running peak. Catastrophic
     forgetting shows up here and nowhere else, because a mean can hide it.

Final performance is reported too, but it is the weakest of the four: an
ablation can reach a similar score by a wildly unstable route, and that route
is the thing being demonstrated.

The honest-outcome clause
-------------------------
If the ablations do NOT destabilise training, that is a negative result and it
gets reported as one, to the standard E-012 and E-013 set. Before believing
such a result, check that the ablations were actually applied — an ablation
that did nothing and an ablation that was never wired up look identical in a
plot. Two independent checks exist: the `no_replay` / `no_target_network` flags
recorded in every run's JSON (asserted below), and the single-backup unit tests
`test_no_replay_ablation_trains_on_a_single_transition` and
`test_no_target_network_ablation_bootstraps_off_the_online_net`, which pin the
behaviour at one gradient step where a convergence plot cannot.

Usage:
    python scripts/dqn_ablations.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from soc_triage.config import load_training_config

from aggregate_dqn import across_runs, curve_matrix, load_runs

# tag -> (label for plots, the flag that must be true in every run's JSON)
CONDITIONS = {
    "dqn": ("control (replay + target net)", None),
    "dqn_no_replay": ("no replay", "no_replay"),
    "dqn_no_target_network": ("no target network", "no_target_network"),
}


def volatility(values: np.ndarray) -> float:
    """Mean |change| between consecutive diagnostic points, averaged over runs."""
    return float(np.abs(np.diff(values, axis=1)).mean())


def max_drawdown(values: np.ndarray) -> float:
    """Largest fall from a running peak, averaged over runs.

    Reported separately from volatility because they catch different failures:
    a curve can be smooth and still collapse once, which is what catastrophic
    forgetting looks like and what a mean would hide.
    """
    drawdowns = []
    for row in values:
        peak = np.maximum.accumulate(row)
        drawdowns.append(float((peak - row).max()))
    return float(np.mean(drawdowns))


def main() -> None:
    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")

    loaded: dict[str, tuple[list[dict], np.ndarray, np.ndarray]] = {}
    for tag, (_, required_flag) in CONDITIONS.items():
        try:
            runs = load_runs(tag)
        except SystemExit as exc:
            print(f"skipping {tag}: {exc}")
            continue
        # An ablation that was never applied looks exactly like an ablation that
        # did nothing. Assert the switch, do not assume it.
        if required_flag is not None and not all(r[required_flag] for r in runs):
            raise SystemExit(
                f"{tag} runs do not all have {required_flag}=True — the ablation "
                f"was not applied and any conclusion from them would be false"
            )
        if required_flag is None and any(
            r["no_replay"] or r["no_target_network"] for r in runs
        ):
            raise SystemExit("control runs are not unablated")
        episodes, values = curve_matrix(runs)
        loaded[tag] = (runs, episodes, values)

    if "dqn" not in loaded:
        raise SystemExit("no control runs — the ablations have nothing to be compared against")

    print("Phase 3 ablations — greedy diagnostic on train seeds 1-10\n")
    header = f"{'condition':<34}{'runs':>5}{'final':>12}{'volatility':>12}{'end std':>10}{'drawdown':>11}"
    print(header)
    print("-" * len(header))

    stats: dict[str, dict[str, float]] = {}
    for tag, (runs, episodes, values) in loaded.items():
        label = CONDITIONS[tag][0]
        quarter = max(1, values.shape[1] // 4)
        final = float(values[:, -quarter:].mean())
        end_std = float(values[:, -quarter:].mean(axis=1).std())
        stats[tag] = {
            "runs": len(runs),
            "final": final,
            "volatility": volatility(values),
            "end_std": end_std,
            "drawdown": max_drawdown(values),
        }
        s = stats[tag]
        print(f"{label:<34}{len(runs):>5}{final:>12.1f}{s['volatility']:>12.1f}"
              f"{end_std:>10.1f}{s['drawdown']:>11.1f}")

    control = stats["dqn"]
    print("\nablation / control ratios (>1 means the ablation is less stable):")
    verdicts = []
    for tag in ("dqn_no_replay", "dqn_no_target_network"):
        if tag not in stats:
            continue
        s = stats[tag]
        ratios = {k: s[k] / control[k] if control[k] else float("inf")
                  for k in ("volatility", "end_std", "drawdown")}
        print(f"  {CONDITIONS[tag][0]:<32}"
              + "  ".join(f"{k} x{v:.2f}" for k, v in ratios.items()))
        # A ratio near 1 on all three is a negative result, and is reported as
        # one rather than explained away.
        destabilised = any(v > 1.5 for v in ratios.values())
        verdicts.append((tag, destabilised, ratios))

    print()
    for tag, destabilised, ratios in verdicts:
        if destabilised:
            worst = max(ratios, key=ratios.get)
            print(f"  {CONDITIONS[tag][0]}: DESTABILISED "
                  f"(largest effect on {worst}, x{ratios[worst]:.2f})")
        else:
            print(f"  {CONDITIONS[tag][0]}: NO clear destabilisation "
                  f"(all three within 1.5x of control) — a NEGATIVE RESULT. "
                  f"Confirm the single-backup tests still pass before reporting it: "
                  f"pytest tests/test_dqn.py -k ablation")

    # --- plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for tag, (runs, episodes, values) in loaded.items():
        mean, std = values.mean(axis=0), values.std(axis=0)
        line, = ax.plot(episodes, mean, linewidth=2,
                        label=f"{CONDITIONS[tag][0]} (n={len(runs)})")
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.18,
                        color=line.get_color())
    ax.set_xlabel("training episode")
    ax.set_ylabel("greedy total reward (train-diag seeds)")
    ax.set_title("Phase 3 ablations — bands are ±1 std across runs")
    ax.legend(fontsize=8)
    fig.tight_layout()

    n_episodes = loaded["dqn"][0][0]["n_episodes"]
    is_full = n_episodes == tcfg.common.n_episodes and len(loaded["dqn"][0]) >= 5
    out_dir = ROOT / "results" if is_full else ROOT / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "dqn_ablations.png", dpi=120)

    lines = [
        "# Phase 3 ablations",
        "",
        f"{n_episodes} episodes per run. Bands in the PNG are ±1 std across runs.",
        "",
        "Instability is measured three ways, all defined before looking at the",
        "data: volatility (mean |change| between diagnostic points), end std",
        "(divergence between runs at the end), and drawdown (largest fall from a",
        "running peak). Final score is reported but is the weakest of the four —",
        "an ablation can reach a similar score by an unstable route, and the",
        "route is what is being demonstrated.",
        "",
        "| condition | runs | final | volatility | end std | drawdown |",
        "|---|---|---|---|---|---|",
    ]
    for tag in loaded:
        s = stats[tag]
        lines.append(
            f"| {CONDITIONS[tag][0]} | {s['runs']} | {s['final']:.1f} | "
            f"{s['volatility']:.1f} | {s['end_std']:.1f} | {s['drawdown']:.1f} |"
        )
    lines += ["", "## Verdict", ""]
    for tag, destabilised, ratios in verdicts:
        ratio_text = ", ".join(f"{k} x{v:.2f}" for k, v in ratios.items())
        lines.append(
            f"- **{CONDITIONS[tag][0]}**: "
            + ("destabilised" if destabilised else
               "**no clear destabilisation — negative result**")
            + f" ({ratio_text})"
        )
    (out_dir / "dqn_ablations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten -> {(out_dir / 'dqn_ablations.md').relative_to(ROOT).as_posix()}")
    print(f"           {(out_dir / 'dqn_ablations.png').relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
