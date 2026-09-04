"""Cohen's kappa over the double-labelled pairs (ROADMAP 5a, final box).

Reads the label database, works out which labellers overlap, and reports kappa
for every overlapping pair of people. Prints the confusion matrix alongside,
because PROJECT_BRIEF §6.2 says a low kappa is itself a finding and you cannot
write that finding up from a coefficient alone — you need to see *how* people
disagreed.

    python scripts/report_kappa.py
    python scripts/report_kappa.py --db results/rlhf/labels.db --csv-backup

Prints "undefined" where kappa does not exist rather than substituting a
number. The case that tempts a substitution is two labellers who both used a
single category throughout: p_o and p_e are both 1, the coefficient is 0/0, and
calling it 1.0 would be the most flattering possible misreading.
"""

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.config import load_training_config
from soc_triage.rlhf.agreement import agreement_between, describe
from soc_triage.rlhf.store import CHOICES, LabelStore

DEFAULT_DB = ROOT / "results" / "rlhf" / "labels.db"


def _confusion_table(result) -> str:
    """The 3x3 table, rendered so both margins are readable."""
    label = max(len(c) for c in CHOICES)
    cell = label + 2
    stub = len("A chose ") + label            # width of the row-label column
    lines = [" " * stub + "  B chose",
             " " * stub + "".join(f"{c:>{cell}}" for c in CHOICES)]
    for a_choice in CHOICES:
        cells = "".join(
            f"{result.confusion[(a_choice, b_choice)]:>{cell}}" for b_choice in CHOICES
        )
        lines.append(f"A chose {a_choice:>{label}}{cells}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="label database (default: results/rlhf/labels.db)")
    parser.add_argument("--csv-backup", action="store_true",
                        help="also write labels.csv beside the database — the "
                             "database is gitignored and irreplaceable")
    args = parser.parse_args()

    if not args.db.exists():
        # Not an error worth a traceback: on a fresh clone this file simply has
        # not been created yet, because no one has labelled anything.
        print(f"no label database at {args.db}")
        print("Nothing has been labelled yet. Build pairs with "
              "scripts/generate_pairs.py, then label them.")
        return

    tcfg = load_training_config(ROOT / "config" / "training_default.yaml")

    with LabelStore(args.db) as store:
        total = store.count()
        labellers = store.labellers()

        print(f"labels collected : {total} (target {tcfg.rlhf.target_pairs})")
        print(f"labellers        : {', '.join(labellers) if labellers else 'none'}")
        print(f"pairs answered   : {len({r['pair_id'] for r in store.all_labels()})}")
        print()

        if len(labellers) < 2:
            print("Cohen's kappa needs two labellers; only "
                  f"{len(labellers)} has contributed so far.")
            print(f"ROADMAP 5a wants {tcfg.rlhf.double_labelled_pairs} pairs "
                  "labelled by both Pranav and Diya.")
            return

        for a, b in itertools.combinations(labellers, 2):
            result = agreement_between(store, a, b)
            print(f"--- {a} vs {b} ---")
            print(describe(result))
            if result.n_shared:
                print()
                print(_confusion_table(result))
            if result.n_shared < tcfg.rlhf.double_labelled_pairs:
                print(f"  note: {result.n_shared} shared pairs, below the "
                      f"{tcfg.rlhf.double_labelled_pairs} the roadmap asks for — "
                      "this kappa is provisional")
            print()

        if args.csv_backup:
            out = args.db.with_suffix(".csv")
            n = store.export_csv(out)
            print(f"wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
