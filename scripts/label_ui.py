"""Serve the preference-labelling page for one labeller (ROADMAP 5a, box 5).

    python scripts/label_ui.py --labeller L1
    python scripts/label_ui.py --labeller L2 --pairs results/rlhf/pairs.json

The labeller id is given here and nowhere else. The page never asks for it, and
the POST body cannot override it (D-041): a text box on screen invites a real
name into a database that has no column for one (CONSTRAINTS #23), and — worse,
because it is silent — a stale value in that box would record one person's
judgements under the other person's id, which is precisely the mistake Cohen's
kappa cannot survive.

Everything else comes from `config/training_default.yaml`: the permitted ids, the
timer cap, the host and the port. The only overrides here are paths and the port,
so a second labeller can run alongside the first without editing config.

This script is the one part of FEATURE_012 that no test covers, and that is
recorded rather than hidden — see FEATURE_012 §9. It is deliberately thin for
that reason: it parses arguments, reads config, loads pairs, and hands all four
to `create_app`. Every rule worth testing lives behind that call.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn

from soc_triage.config import load_training_config
from soc_triage.labelling.app import create_app
from soc_triage.labelling.queue import PairFileError, load_pairs

DEFAULT_CONFIG = ROOT / "config" / "training_default.yaml"
DEFAULT_PAIRS = ROOT / "results" / "rlhf" / "pairs.json"
DEFAULT_DB = ROOT / "results" / "rlhf" / "labels.db"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--labeller",
        required=True,
        help="opaque labeller id, and it must be one of config rlhf.labellers",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="the label database. Created if absent; never overwritten",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="overrides rlhf.ui_port, so both labellers can serve at once",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rlhf = load_training_config(args.config).rlhf

    # Checked before anything is opened. A typo here would otherwise start a
    # session that writes 175 rows under an id nobody recognises, and the labels
    # are the one thing in this project that cannot be regenerated.
    if args.labeller not in rlhf.labellers:
        print(
            f"unknown labeller {args.labeller!r}; "
            f"config/training_default.yaml lists {', '.join(rlhf.labellers)}",
            file=sys.stderr,
        )
        return 2

    try:
        pairs = load_pairs(args.pairs)
    except PairFileError as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        if args.pairs == DEFAULT_PAIRS and not args.pairs.exists():
            # The likeliest reason on a fresh clone, worth naming rather than
            # leaving as a bare missing-file error.
            print(
                "\nNothing has generated the pair set yet. That is "
                "scripts/generate_pairs.py (Pranav's box, ROADMAP 5a) and it "
                "needs the nine trained policies present.",
                file=sys.stderr,
            )
        return 1

    app = create_app(
        pairs=pairs,
        labeller_id=args.labeller,
        labellers=rlhf.labellers,
        db_path=args.db,
        max_seconds=rlhf.max_seconds_per_pair,
    )

    port = args.port if args.port is not None else rlhf.ui_port
    print(
        f"labelling as {args.labeller} — {len(pairs)} pairs loaded, "
        f"answers going to {args.db}\n"
        f"open http://{rlhf.ui_host}:{port}/ — ctrl-c to stop"
    )
    uvicorn.run(app, host=rlhf.ui_host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
