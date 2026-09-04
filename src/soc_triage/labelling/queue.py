"""Which pairs a labeller sees, in what order, and where they left off.

FEATURE_012 §4.1 and D-040. Three things happen here and nothing else does:

1. **Assignment.** The 50 double-labelled pairs go to everyone, because Cohen's
   kappa compares two independent judgements of the *same* pair and is otherwise
   not computable at all. The remaining 250 are dealt round-robin, one each, so
   two labellers get 125 apiece and do 175 judgements each.

2. **Resume.** Labelling happens in short sittings across days, so the queue is
   told which pair ids this person has already answered and serves the first one
   they have not. Re-serving an answered pair would collide with
   `UNIQUE (pair_id, labeller_id)` in `store.py`; skipping an unanswered one
   would quietly leave the set short.

3. **Loading, with the blinding guard.** `load_pairs` reads `pairs.json` and
   refuses `pairs_key.json` by name, because the realistic version of that
   mistake is copying the wrong file to wherever the page is served from (D-038).

**The answered set is passed in, not queried.** This module never opens the
database. That is what lets every rule above be tested without SQLite, and it is
the same separation FEATURE_011 used to keep its data layer free of torch.

**Assignment is computed from a sorted copy of the pairs**, so the result cannot
depend on the order the file happened to be read in. Sorting by `pair_id` is not
a loss of randomness: `pairs.build_pairs` already shuffled the draft before
numbering it, precisely so a labeller working through the file in order does not
meet every alpha-vs-beta comparison in one block. That shuffle does double duty
here — file order is already random with respect to which policies are being
compared, so dealing alternate pairs gives each labeller a balanced spread
without this module needing to know what any pair contains. Which it must not:
the pairs are blinded.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# What `pairs.build_pairs` guarantees to put in every element of `pairs.json`.
# Checked at load time so a malformed file fails before a labelling session
# starts, rather than three screens into one.
_REQUIRED_PAIR_FIELDS = ("pair_id", "left", "right", "double_labelled")

# The file the page must never read (D-038). Named as a constant so the guard
# and its error message cannot drift apart.
_KEY_FILENAME = "pairs_key.json"


class QueueError(Exception):
    """Base for every refusal this module makes."""


class PairFileError(QueueError):
    """The pair file is missing, malformed, or the wrong file entirely."""


class UnknownLabellerError(QueueError):
    """The labeller id is not one of the configured ids."""


class UnassignedPairError(QueueError):
    """This labeller was never assigned that pair."""


@dataclass(frozen=True)
class Progress:
    """How far through their own share one labeller is.

    `total` is this labeller's assignment, not the size of the pair set — 175,
    not 300. Showing 300 to someone who will only ever see 175 of them would
    misreport the job as nearly twice its real length.
    """

    labeller_id: str
    done: int
    total: int

    @property
    def remaining(self) -> int:
        return self.total - self.done

    @property
    def position(self) -> int:
        """The 1-based number of the pair now on screen: "pair 4 of 175".

        Clamped at `total` so a finished labeller sees "175 of 175" rather than
        a 176th pair that does not exist.
        """
        return min(self.done + 1, self.total)


def _validate_labellers(labellers: Sequence[str]) -> tuple[str, ...]:
    """Check the configured labeller list and return it as a tuple.

    The minimum of two is not a tunable: kappa is a measure of agreement
    *between* annotators, so a single-labeller configuration cannot produce the
    number Phase 5a exists for. Better to refuse at startup than to collect 300
    labels and discover it at analysis time.
    """
    names = tuple(labellers)

    if len(names) < 2:
        raise ValueError(
            f"need at least two labellers to measure agreement, got {list(names)}"
        )

    duplicates = [name for name in set(names) if names.count(name) > 1]
    if duplicates:
        raise ValueError(
            f"duplicate labeller id(s) {sorted(duplicates)}: one person cannot "
            "hold two places in the split"
        )

    if any(not name.strip() for name in names):
        raise ValueError(f"labeller ids must be non-empty, got {list(names)}")

    return names


def assign(
    pairs: Sequence[dict], labellers: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Work out which pair ids each labeller is responsible for.

    Deterministic and independent of the order `pairs` arrives in. Written as an
    explicit loop with a running counter rather than a slice-and-zip, because the
    round-robin and the doubles-go-to-everyone rule are the decision under
    discussion in D-040 and should be readable as one (CONSTRAINTS #14).
    """
    names = _validate_labellers(labellers)
    allocation: dict[str, list[str]] = {name: [] for name in names}

    # Sorted so two machines reading the same file in different orders agree.
    ordered = sorted(pairs, key=lambda pair: pair["pair_id"])

    # Counts only the single-label pairs, so the doubles interleaved among them
    # do not push the round-robin out of step and skew the 125/125 split.
    single_index = 0

    for pair in ordered:
        pair_id = pair["pair_id"]
        if pair["double_labelled"]:
            for name in names:
                allocation[name].append(pair_id)
        else:
            allocation[names[single_index % len(names)]].append(pair_id)
            single_index += 1

    return {name: tuple(ids) for name, ids in allocation.items()}


class LabelQueue:
    """One labeller's view of the pair set: their share, and their place in it."""

    def __init__(
        self,
        pairs: Sequence[dict],
        labeller_id: str,
        labellers: Sequence[str],
        answered: Iterable[str] = (),
    ) -> None:
        names = _validate_labellers(labellers)
        if labeller_id not in names:
            raise UnknownLabellerError(
                f"unknown labeller {labeller_id!r}; the configured ids are "
                f"{', '.join(names)}"
            )

        self.labeller_id = labeller_id
        self.assigned: tuple[str, ...] = assign(pairs, names)[labeller_id]
        self._by_id = {pair["pair_id"]: pair for pair in pairs}
        self._answered = frozenset(answered)

    def next_pair(self) -> dict | None:
        """The first assigned pair this labeller has not answered, or None.

        `None` means finished, and the caller shows the done page. It does not
        mean "error" — reaching the end is the goal.
        """
        for pair_id in self.assigned:
            if pair_id not in self._answered:
                return self._by_id[pair_id]
        return None

    def pair(self, pair_id: str) -> dict:
        """Look up one assigned pair by id.

        Refuses ids belonging to the other labeller as firmly as ids that do not
        exist, and for the same reason: a POST naming either one is a bug or a
        hand-edited request, and the only safe response to both is to decline.
        """
        if pair_id not in self.assigned:
            raise UnassignedPairError(
                f"pair {pair_id!r} is not assigned to {self.labeller_id!r}"
            )
        return self._by_id[pair_id]

    def progress(self) -> Progress:
        """Counts this labeller's own answers only.

        The other person finishing a shared pair does not move this person along,
        which is why the double-labelled pairs appear in both assignments and are
        counted separately in each.
        """
        done = sum(1 for pair_id in self.assigned if pair_id in self._answered)
        return Progress(self.labeller_id, done, len(self.assigned))


def load_pairs(path: str | Path) -> list[dict]:
    """Read `pairs.json`, refusing the key file and anything malformed.

    Every check here fails at load rather than mid-session. A labelling session
    is human time; discovering a broken pair file at pair 87 wastes the 86
    answers' worth of momentum, even though the answers themselves are safe.
    """
    path = Path(path)

    if path.name == _KEY_FILENAME:
        raise PairFileError(
            f"refusing to read {_KEY_FILENAME}: it maps pair_id to policy names, "
            "and the labelling page must only ever read pairs.json (D-038)"
        )

    if not path.is_file():
        raise PairFileError(f"no pair file at {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PairFileError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise PairFileError(
            f"{path} should hold a list of pairs, found {type(data).__name__}"
        )

    seen: set[str] = set()
    for position, pair in enumerate(data):
        if not isinstance(pair, dict):
            raise PairFileError(
                f"entry {position} of {path} is {type(pair).__name__}, not a pair"
            )

        missing = [field for field in _REQUIRED_PAIR_FIELDS if field not in pair]
        if missing:
            raise PairFileError(
                f"entry {position} of {path} is missing {', '.join(missing)}"
            )

        pair_id = pair["pair_id"]
        if pair_id in seen:
            # Labels reference pair_id. Two entries sharing one would make a
            # label ambiguous about which comparison it judged.
            raise PairFileError(f"duplicate pair_id {pair_id!r} in {path}")
        seen.add(pair_id)

    return data
