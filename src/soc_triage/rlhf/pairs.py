"""Building blinded preference pairs from EpisodeRecords (FEATURE_011 §3, §6, §8).

Each pair is two policies working **the same shift** — same seed, same alert
stream, same config — so that whatever a labeller prefers is a property of the
policies rather than of the luck of the draw. `rlhf.pair_must_share_seed` says
this is asserted rather than assumed, and it is.

This module imports no agent, no environment and no torch. It reads
EpisodeRecord dicts, which `runner.run_episode` already emits and whose
docstring already names "the RLHF pair builder" as a consumer. The consequence
worth protecting: pair construction works on a clone with no `results/` and no
`torch`, and nothing here can be dragged into an earlier phase's dependency
graph (CONSTRAINTS #11).

Two files come out, and the split is the blinding (FEATURE_011 §6):

    pairs.json      what the labelling UI reads. No policy names, anywhere.
    pairs_key.json  pair_id to policy names and side assignment. Analysis only.

The whole build is a pure function of (records, sampling_seed, sizes). Running
it twice produces byte-identical output. That is not tidiness: collected labels
reference `pair_id`, so a rebuild that renumbered the pairs would silently
repoint every label already gathered, and nothing would raise.
"""

import itertools
import json
import random
from pathlib import Path
from typing import Sequence

from soc_triage.rlhf.summary import summarise_episode


class PairBuildError(Exception):
    """Base for every refusal this module makes."""


class MixedConfigError(PairBuildError):
    """Records were produced under more than one config.

    Two episodes run against different `env_default.yaml` contents are not the
    same shift even on the same seed — the generator, the costs or the shift
    length may all differ. `runner.config_hash` exists so this is checkable.
    """


class InsufficientRecordsError(PairBuildError):
    """Not enough records to build the requested pair set.

    Raised with the arithmetic, because the fix is always either 'run more
    seeds' or 'ask for fewer pairs' and the caller needs to know which.
    """


def load_records(directory: str | Path) -> list[dict]:
    """Read every EpisodeRecord JSON in a directory, in sorted filename order.

    Sorted so that a caller who skips `build_pairs` and does something ad hoc
    still gets a reproducible ordering.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"no such records directory: {directory}")

    records = []
    for path in sorted(directory.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _index_records(records: Sequence[dict]) -> tuple[dict, str]:
    """Index records by (agent_name, seed), and check they share one config.

    Returns the index and the single config hash. Records are indexed rather
    than iterated so that the build depends on their content and not on the
    order the filesystem handed them over.
    """
    hashes = {record["config_hash"] for record in records}
    if len(hashes) > 1:
        raise MixedConfigError(
            f"records span {len(hashes)} different configs ({sorted(hashes)}); "
            "a pair's two sides must have run the same environment"
        )

    index: dict[tuple[str, int], dict] = {}
    for record in records:
        index[(record["agent_name"], int(record["seed"]))] = record
    return index, hashes.pop() if hashes else "unhashed"


def _eligible_seeds(index: dict, policies: Sequence[str]) -> list[int]:
    """Seeds on which *every* policy ran.

    A seed that one policy is missing cannot host every pairing, so allowing it
    would quietly unbalance the allocation — some pairings would draw from a
    larger pool of seeds than others.
    """
    all_seeds = sorted({seed for _, seed in index})
    return [
        seed for seed in all_seeds
        if all((policy, seed) in index for policy in policies)
    ]


def _blind(summary: dict) -> dict:
    """Strip the one field in a summary that names the policy.

    `run_id` reads `sarsa-seed3000004`. It is useful provenance and it is
    exactly what must not reach a labeller, so it lives in `pairs_key.json`
    instead. Everything else in a summary is already policy-agnostic.
    """
    return {key: value for key, value in summary.items() if key != "run_id"}


def _allocate(
    pairings: list[tuple[str, str]], target: int, rng: random.Random
) -> dict[tuple[str, str], int]:
    """Split `target` pairs as evenly as possible across the policy pairings.

    Every pairing gets `target // n`, and the leftover goes to a randomly
    chosen subset rather than to whichever pairings happen to sort first —
    otherwise `alpha`-vs-anything would be systematically over-represented by
    an accident of the alphabet.

    Balance matters to the Bradley-Terry fit in Phase 5b: a pairing that never
    appears leaves the model no direct evidence about those two policies and
    forces it to infer the comparison transitively.
    """
    base, remainder = divmod(target, len(pairings))
    counts = {pairing: base for pairing in pairings}
    for pairing in rng.sample(pairings, remainder):
        counts[pairing] += 1
    return counts


def _choose_double_labelled(
    pairings_by_index: list[tuple[str, str]], wanted: int, rng: random.Random
) -> set[int]:
    """Pick which pairs get labelled twice, spread across policy pairings.

    Taking the first N in file order — or a flat random sample — would let the
    double-labelled set concentrate on a few kinds of comparison, and Cohen's
    kappa would then describe agreement on those rather than on the task. So
    the selection goes round-robin over the pairings, one from each in turn,
    which guarantees the widest spread the size allows.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for i, pairing in enumerate(pairings_by_index):
        groups.setdefault(pairing, []).append(i)

    order = sorted(groups)
    rng.shuffle(order)
    for pairing in order:
        rng.shuffle(groups[pairing])

    chosen: set[int] = set()
    round_number = 0
    while len(chosen) < wanted:
        took_one = False
        for pairing in order:
            if len(chosen) == wanted:
                break
            if round_number < len(groups[pairing]):
                chosen.add(groups[pairing][round_number])
                took_one = True
        if not took_one:          # every group exhausted; caller checked sizes
            break
        round_number += 1
    return chosen


def build_pairs(
    records: Sequence[dict],
    *,
    policies: Sequence[str],
    target_pairs: int,
    double_labelled_pairs: int,
    sampling_seed: int,
) -> tuple[list[dict], dict[str, dict]]:
    """Build the pair set and its key.

    Returns `(pairs, key)`. `pairs` is what the labelling UI reads and contains
    no policy names; `key` maps pair_id to the names and the side assignment,
    for analysis only.
    """
    if double_labelled_pairs > target_pairs:
        raise ValueError(
            f"cannot double-label {double_labelled_pairs} of {target_pairs} pairs"
        )

    policies = sorted(policies)
    index, config_hash = _index_records(records)

    missing = [p for p in policies if not any(a == p for a, _ in index)]
    if missing:
        raise InsufficientRecordsError(
            f"no EpisodeRecords for {missing}; run those policies on the pair "
            "seed block first"
        )

    seeds = _eligible_seeds(index, policies)
    pairings = [tuple(pair) for pair in itertools.combinations(policies, 2)]

    capacity = len(pairings) * len(seeds)
    if target_pairs > capacity:
        raise InsufficientRecordsError(
            f"asked for {target_pairs} pairs but only {capacity} exist: "
            f"{len(pairings)} policy pairings x {len(seeds)} seeds on which "
            "every policy ran. Run more pair seeds, or lower target_pairs."
        )

    rng = random.Random(sampling_seed)
    counts = _allocate(pairings, target_pairs, rng)

    # Draw seeds per pairing, in sorted pairing order so the sequence of rng
    # calls is fixed regardless of how the records arrived.
    draft: list[tuple[tuple[str, str], int]] = []
    for pairing in pairings:
        for seed in rng.sample(seeds, counts[pairing]):
            draft.append((pairing, seed))

    # Shuffle before numbering so a labeller working through the file in order
    # does not meet every alpha-vs-beta comparison in one block, which would
    # let fatigue and boredom load onto particular pairings.
    rng.shuffle(draft)

    double_indices = _choose_double_labelled(
        [pairing for pairing, _ in draft], double_labelled_pairs, rng
    )

    pairs: list[dict] = []
    key: dict[str, dict] = {}
    for i, (pairing, seed) in enumerate(draft):
        first, second = pairing
        swapped = rng.random() < 0.5
        left_policy, right_policy = (second, first) if swapped else (first, second)

        left = summarise_episode(index[(left_policy, seed)])
        right = summarise_episode(index[(right_policy, seed)])

        # The property the whole pair set rests on, checked per pair rather
        # than trusted from the indexing above (`rlhf.pair_must_share_seed`).
        if left["seed"] != right["seed"]:
            raise PairBuildError(
                f"pair {i} has seed {left['seed']} on the left and "
                f"{right['seed']} on the right; the two sides must be the "
                "same alert stream"
            )

        pair_id = f"p{i:04d}"
        pairs.append({
            "pair_id": pair_id,
            "seed": seed,
            "config_hash": config_hash,
            "double_labelled": i in double_indices,
            "left": _blind(left),
            "right": _blind(right),
        })
        key[pair_id] = {
            "left_policy": left_policy,
            "right_policy": right_policy,
            "left_run_id": left["run_id"],
            "right_run_id": right["run_id"],
            "swapped": swapped,
            "seed": seed,
        }

    return pairs, key


def write_pairs(
    pairs: Sequence[dict], key: dict[str, dict], directory: str | Path
) -> tuple[Path, Path]:
    """Write `pairs.json` and `pairs_key.json`. Returns both paths.

    They are written side by side deliberately: separating them across
    directories would invite someone to copy the wrong one to wherever the UI
    is served from. The rule is about which file the UI *reads*, and it is
    enforced by `pairs.json` simply not containing the names.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    pairs_path = directory / "pairs.json"
    key_path = directory / "pairs_key.json"
    pairs_path.write_text(json.dumps(pairs, indent=1), encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=1, sort_keys=True), encoding="utf-8")
    return pairs_path, key_path
