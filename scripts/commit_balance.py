"""Report the commit balance between teammates, and whose turn it is (D-021).

This project is built by two students who alternate machines. The git history is
part of what gets evaluated, so it has to reflect that both people actually
worked — and it will not do that by accident, because whoever happens to be at
the keyboard during a long session accumulates commits fast.

This script is the check. Run it at the start and end of every session:

    python scripts/commit_balance.py

It reports per-author totals (honouring `.mailmap`, so split identities are
collapsed), a per-phase breakdown, and an explicit recommendation about whether
to keep going or hand over.

**What this script is not.** It does not, and must not, be used to attribute
work to someone who did not do it. The point is to keep the *real* split even by
handing over at the right time — so that every commit under a name is work that
person genuinely did and can explain in a viva. An examiner who asks either
student to walk through any commit bearing their name must get a real answer.
See CONSTRAINTS.md #24 and D-021.
"""

import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Hand over when one person is this many commits ahead. Small enough that the
# split never drifts far, large enough that nobody hands over mid-feature —
# CONSTRAINTS #25 forbids splitting a broken state across a handover.
HANDOVER_THRESHOLD = 3

PHASES = ("phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def author_counts() -> Counter:
    """Commits per canonical author. `log --use-mailmap` collapses identities."""
    out = git("log", "--use-mailmap", "--pretty=format:%aN")
    return Counter(line for line in out.splitlines() if line)


def main() -> None:
    counts = author_counts()
    total = sum(counts.values())
    if not counts:
        raise SystemExit("no commits found")

    print(f"Total commits: {total}\n")
    print("Per author (mailmap-collapsed):")
    for name, n in counts.most_common():
        share = 100 * n / total
        bar = "#" * round(share / 2)
        print(f"  {name:22s} {n:3d}  {share:5.1f}%  {bar}")

    print("\nPer phase:")
    for phase in PHASES:
        out = git("log", "--use-mailmap", f"--grep=^{phase}", "--pretty=format:%aN")
        phase_counts = Counter(line for line in out.splitlines() if line)
        if not phase_counts:
            continue
        detail = ", ".join(f"{k} {v}" for k, v in phase_counts.most_common())
        print(f"  {phase}: {sum(phase_counts.values())} commits  ({detail})")

    # --- the recommendation
    print()
    if len(counts) < 2:
        print("Only one author so far. The other teammate has no commits at all —")
        print("hand over before the next phase.")
        sys.exit(0)

    (lead_name, lead_n), (trail_name, trail_n) = counts.most_common()[0], counts.most_common()[-1]
    gap = lead_n - trail_n

    print(f"Gap: {lead_name} is {gap} commit(s) ahead of {trail_name}.")
    if gap <= HANDOVER_THRESHOLD:
        print(f"BALANCED (threshold {HANDOVER_THRESHOLD}). Either person may take the next block.")
    else:
        needed = gap - HANDOVER_THRESHOLD
        print(f"IMBALANCED. **{trail_name} should take the next block** — roughly "
              f"{needed}-{gap} commits to get back inside the threshold.")
        print("Finish the current logical change first: CONSTRAINTS #25 forbids handing")
        print("over a broken or half-documented state.")


if __name__ == "__main__":
    main()
