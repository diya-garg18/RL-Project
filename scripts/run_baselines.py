"""Phase 0 exit gate: all five baselines x all metrics on the evaluation seeds.

Expectation (ROADMAP): oracle strictly best on recall, random clearly worst.
If not, the environment is broken — fix before Phase 1.

Prints the table and writes results/baselines.md + raw EpisodeRecords under
results/runs/. Uses EVALUATION seeds — baselines learn nothing, so there is no
tuning risk; learned agents evaluated later on these same seeds are then
directly comparable, on identical alert streams (paired comparison, brief §8).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.agents.baselines import make_baselines
from soc_triage.config import load_env_config
from soc_triage.env import SOCTriageEnv
from soc_triage.evaluation.metrics import METRIC_NAMES, summarise
from soc_triage.runner import config_hash, run_episodes, save_records

RANDOM_AGENT_ACTION_SEED = 12345  # seeds the random baseline's action choices only


def main() -> None:
    cfg_path = ROOT / "config" / "env_default.yaml"
    cfg = load_env_config(cfg_path)
    cfg_hash = config_hash(cfg_path)
    env = SOCTriageEnv(cfg)

    summaries: dict[str, dict] = {}
    for agent in make_baselines(cfg, RANDOM_AGENT_ACTION_SEED):
        records = run_episodes(env, agent, cfg.seeds.eval, cfg, cfg_hash)
        save_records(records, ROOT / "results" / "runs")
        summaries[agent.name] = summarise(records, cfg)

    lines = []
    lines.append(f"# Baseline comparison — eval seeds {list(cfg.seeds.eval)} — config {cfg_hash}")
    lines.append("")
    header = "| agent | " + " | ".join(METRIC_NAMES) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(METRIC_NAMES) + 1))
    for name, summary in summaries.items():
        cells = [name]
        for metric in METRIC_NAMES:
            mean = summary[metric]["mean"]
            std = summary[metric]["std"]
            cells.append("n/a" if mean is None else f"{mean:.2f} ± {std:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    print(table)

    # The exit-criterion checks, stated loudly rather than silently assumed.
    recalls = {n: s["recall_at_deadline"]["mean"] for n, s in summaries.items()}
    oracle_best = all(recalls["oracle_greedy"] > v for n, v in recalls.items() if n != "oracle_greedy")
    random_worst = all(recalls["random"] <= v for n, v in recalls.items() if n != "random")
    print()
    print(f"oracle strictly best on recall : {'PASS' if oracle_best else 'FAIL'}  ({recalls})")
    print(f"random worst on recall         : {'PASS' if random_worst else 'FAIL'}")

    out = ROOT / "results" / "baselines.md"
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
