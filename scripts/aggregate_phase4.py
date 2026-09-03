"""Combine per-repeat Phase 4 run files into one reportable result.

Why this exists
---------------
The actor-critic costs ~3.3 h per repeat, so a five-repeat run is driven as five
`--only-repeat` processes in parallel (D-027, the pattern `run_dqn_sweep.py`
already uses for the DQN). Each process writes its own
`<tag>_repeat<N>.json` holding a single run, and each one honestly refuses to
present itself as a result:

    WARNING: 1 run(s) only - NOT a reportable result (CONSTRAINTS #3 ...)

Something has to do the combining, and `aggregate_dqn.py` only understands the
Phase 3 payload. This is the Phase 4 equivalent.

What it does NOT do
-------------------
It does not re-evaluate anything. Every number here was computed by the trainer
against `cfg.seeds.eval` at the end of its own run; this only takes the per-run
summaries and applies `across_runs_summary` to them. There is no path in this
file that touches the environment, so it cannot accidentally become a second,
differently-configured evaluation of the same policies.

D-036 is respected literally: **sampled is the reported number, greedy is a
diagnostic**, and both are printed with that labelling so a reader cannot take
one for the other.

Usage:
    python scripts/aggregate_phase4.py --agent actor_critic
    python scripts/aggregate_phase4.py --agent reinforce
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.evaluation.metrics import MIN_RUNS_TO_REPORT, across_runs_summary

# (agent name -> results subdirectory). The two Phase 4 learners; the DQN has
# its own aggregator because its payload predates this one.
AGENT_DIRS = {
    "reinforce": "reinforce_runs/reinforce",
    "reinforce_no_baseline": "reinforce_runs/reinforce_no_baseline",
    "actor_critic": "actor_critic_runs/actor_critic",
}


def collect(directory: Path) -> tuple[list[dict], list[dict], list[str]]:
    """Every per-run eval summary in one directory, sampled and greedy.

    Handles both layouts the trainers produce: `<tag>.json` from an in-process
    multi-repeat run (its lists already hold one entry per repeat) and
    `<tag>_repeat<N>.json` from `--only-repeat`. Files that predate D-036 carry
    the old single `eval_summary` key and are refused rather than guessed at —
    an aggregate that silently mixed a greedy-only file into a sampled headline
    is exactly the confusion D-036 exists to prevent.
    """
    sampled: list[dict] = []
    greedy: list[dict] = []
    sources: list[str] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith("_aggregate.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "eval_sampled_per_run" not in payload:
            print(f"  SKIPPED {path.name}: predates D-036 (no 'eval_sampled_per_run'). "
                  f"Re-run the trainer; this file has only a greedy reading.")
            continue
        sampled.extend(payload["eval_sampled_per_run"])
        greedy.extend(payload["eval_greedy_per_run"])
        sources.append(path.name)
    return sampled, greedy, sources


def report(heading: str, aggregated: dict[str, dict], n_runs: int) -> None:
    print(f"\n  {heading}")
    for metric, stats in aggregated.items():
        if stats["mean"] is None:
            print(f"    {metric}: undefined on all {n_runs} runs")
        else:
            print(f"    {metric}: {stats['mean']:.4f} +- {stats['std']:.4f}"
                  f"  (over {stats['n_runs']} run(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=sorted(AGENT_DIRS), required=True)
    args = parser.parse_args()

    directory = ROOT / "results" / AGENT_DIRS[args.agent]
    if not directory.is_dir():
        raise SystemExit(f"no run directory at {directory} - has the trainer been run?")

    sampled, greedy, sources = collect(directory)
    if not sampled:
        raise SystemExit(f"no D-036-era run files in {directory}")

    print(f"aggregating {len(sampled)} run(s) of {args.agent} "
          f"from {len(sources)} file(s): {', '.join(sources)}")

    eval_sampled = across_runs_summary(sampled)
    eval_greedy = across_runs_summary(greedy)

    report(f"SAMPLED, ACROSS {len(sampled)} RUNS -- the reported number (D-036):",
           eval_sampled, len(sampled))
    report(f"greedy (argmax), ACROSS {len(greedy)} RUNS -- DIAGNOSTIC ONLY (D-036):",
           eval_greedy, len(greedy))

    if len(sampled) < MIN_RUNS_TO_REPORT:
        print(f"\n  WARNING: {len(sampled)} run(s) only - NOT a reportable result "
              f"(CONSTRAINTS #3 wants at least {MIN_RUNS_TO_REPORT}).")

    out = directory / f"{args.agent}_aggregate.json"
    out.write_text(json.dumps({
        "agent": args.agent,
        "n_runs": len(sampled),
        "source_files": sources,
        "eval_policy_convention": "sampled is the reported number (D-036); greedy is diagnostic",
        "eval_sampled_across_runs": eval_sampled,
        "eval_greedy_across_runs": eval_greedy,
    }, indent=1), encoding="utf-8")
    print(f"\naggregate -> {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
