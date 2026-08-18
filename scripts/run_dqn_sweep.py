"""Launch the Phase 3 sweep as parallel single-repeat processes.

Why this script exists
----------------------
`train_dqn.py --only-repeat K` runs one repeat in one single-threaded process
(~301 MB, one core, measured). This script is the scheduler that keeps a fixed
number of those in flight until the whole sweep is done, unattended.

Sequentially, 60 runs x 20000 episodes would take about 68 hours. At ten at a
time it is about eight. That is the entire reason the sweep is affordable, and
it works only because nothing is shared between repeats: each has its own seed
block slice, its own agent seed, and its own output file.

Run counts are NOT uniform across conditions, deliberately. The control is
compared against tabular Q-learning, whose total reward is 47.6 +/- 52.0 — a
spread wider than the effect being looked for, and the exact situation E-014
found had invalidated every headline comparison in Phases 0-2. Precision is
worth buying there. The ablations are compared against the control and are
expected to fail obviously; if an ablation's effect only appears at 30 runs,
that is a negative result worth reporting rather than something to spend
compute hiding.

Restartability: repeats are seeded by index alone, so this can be re-run later
with a higher --control-runs to ADD repeats without recomputing existing ones.
Already-complete repeats are skipped unless --force is given.

Usage:
    python scripts/run_dqn_sweep.py                      # the full Phase 3 sweep
    python scripts/run_dqn_sweep.py --control-runs 2 --ablation-runs 1 \
        --episodes 40 --max-parallel 2                   # smoke test
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The venv interpreter explicitly: the global 3.13 lacks this project's
# dependencies, and a detached process does not inherit a shell's PATH edits.
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAINER = ROOT / "scripts" / "train_dqn.py"

# tag -> the CLI flags that produce it. Must match train_dqn.run_tag().
CONDITIONS: dict[str, list[str]] = {
    "dqn": [],
    "dqn_no_replay": ["--no-replay"],
    "dqn_no_target_network": ["--no-target-network"],
}


def planned_jobs(
    control_runs: int, ablation_runs: int
) -> list[tuple[str, int, list[str]]]:
    """(tag, repeat_index, flags) for every run in the sweep.

    Control first, so the earliest finished wave is the one that answers
    "does this agent learn at all" before eight hours are spent on ablations
    of an agent that does not.
    """
    jobs: list[tuple[str, int, list[str]]] = []
    for tag, flags in CONDITIONS.items():
        n = control_runs if tag == "dqn" else ablation_runs
        for repeat in range(n):
            jobs.append((tag, repeat, flags))
    return jobs


def result_path(tag: str, repeat: int) -> Path:
    return ROOT / "results" / "dqn_runs" / tag / f"repeat{repeat}.json"


def already_done(tag: str, repeat: int, episodes: int) -> bool:
    """True only if a result exists AND was produced at THIS episode count.

    Matching on existence alone is not enough, and this is not hypothetical: a
    40-episode smoke test of this very script left four JSONs behind that an
    existence check happily skipped, which would have silently mixed 40-episode
    runs into a 20000-episode sweep and shrunk n without any error. Same class
    of failure as D-018, arriving through a different door.
    """
    path = result_path(tag, repeat)
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8"))["n_episodes"] == episodes
    except (json.JSONDecodeError, KeyError, OSError):
        return False  # unreadable or truncated: treat as not done, and redo it


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 3 DQN sweep in parallel.")
    parser.add_argument("--control-runs", type=int, default=30,
                        help="repeats of the unablated DQN")
    parser.add_argument("--ablation-runs", type=int, default=15,
                        help="repeats of EACH ablation")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--max-parallel", type=int, default=10,
                        help="concurrent processes; 10 x 301 MB keeps total RAM "
                             "under the 75%% ceiling on a 15.7 GB machine")
    parser.add_argument("--force", action="store_true",
                        help="re-run repeats whose JSON already exists")
    args = parser.parse_args()

    if not PYTHON.exists():
        sys.exit(f"interpreter not found: {PYTHON}")

    jobs = planned_jobs(args.control_runs, args.ablation_runs)
    if not args.force:
        skipped = [j for j in jobs if already_done(j[0], j[1], args.episodes)]
        jobs = [j for j in jobs if not already_done(j[0], j[1], args.episodes)]
        if skipped:
            print(f"skipping {len(skipped)} repeats that already have results "
                  f"(use --force to redo them)")

    total = len(jobs)
    if not total:
        print("nothing to do — every planned repeat already has a result")
        return

    waves = -(-total // args.max_parallel)  # ceiling division
    print(f"Phase 3 sweep: {total} runs x {args.episodes} episodes, "
          f"{args.max_parallel} at a time (~{waves} waves)")
    for tag in CONDITIONS:
        n = sum(1 for t, _, _ in jobs if t == tag)
        if n:
            print(f"  {tag:24s} {n} runs")
    print()

    queue = list(jobs)
    running: list[tuple[subprocess.Popen, str, int, object, float]] = []
    done = 0
    failed: list[tuple[str, int, int]] = []
    t_start = time.perf_counter()

    while queue or running:
        while queue and len(running) < args.max_parallel:
            tag, repeat, flags = queue.pop(0)
            log_dir = ROOT / "results" / "dqn_runs" / tag
            log_dir.mkdir(parents=True, exist_ok=True)
            log = open(log_dir / f"repeat{repeat}.log", "w", encoding="utf-8")
            cmd = [
                str(PYTHON), str(TRAINER),
                "--episodes", str(args.episodes),
                "--eval-every", str(args.eval_every),
                "--only-repeat", str(repeat),
                "--no-plot",
            ] + flags
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT)
            )
            running.append((proc, tag, repeat, log, time.perf_counter()))
            print(f"[{time.strftime('%H:%M:%S')}] start  {tag} repeat {repeat}")

        time.sleep(5)

        still_running = []
        for proc, tag, repeat, log, t0 in running:
            if proc.poll() is None:
                still_running.append((proc, tag, repeat, log, t0))
                continue
            log.close()
            done += 1
            mins = (time.perf_counter() - t0) / 60
            if proc.returncode == 0:
                status = "done "
            else:
                status = "FAIL "
                failed.append((tag, repeat, proc.returncode))
            elapsed = (time.perf_counter() - t_start) / 60
            print(f"[{time.strftime('%H:%M:%S')}] {status} {tag} repeat {repeat} "
                  f"in {mins:.1f} min  ({done}/{total} complete, "
                  f"{elapsed:.0f} min elapsed)")
        running = still_running

    elapsed = (time.perf_counter() - t_start) / 60
    print(f"\nsweep finished in {elapsed / 60:.1f} h ({done} runs)")
    if failed:
        # Loud, because a silently missing repeat would just shrink n and
        # nothing downstream would notice.
        print(f"{len(failed)} FAILED — see the .log beside each .json:")
        for tag, repeat, code in failed:
            print(f"  {tag} repeat {repeat} exit {code}")
        sys.exit(1)
    print("combine them with: python scripts/aggregate_dqn.py")


if __name__ == "__main__":
    main()
