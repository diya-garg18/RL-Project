"""`labelling.queue` — who sees which pair, and where they left off.

Three properties carry the weight here, and each protects something that is
expensive or impossible to notice later:

1. **The 50 double-labelled pairs reach both labellers.** If they do not, Cohen's
   kappa is not merely wrong, it is uncomputable — and that is the one number
   Phase 5a exists to produce (`ROADMAP.md` 5a, D-038).
2. **The assignment is deterministic and order-independent.** It is computed from
   `pairs.json` and must come out the same on both machines, in both sittings,
   whatever order the pairs arrived in. Otherwise "who labelled what" survives
   only as timestamps in the database (D-040).
3. **Resume is exact.** Labelling happens in short sittings across days. A queue
   that re-served an answered pair would hit the `UNIQUE (pair_id, labeller_id)`
   constraint in `store.py`; one that skipped an unanswered pair would quietly
   leave the set short.

The queue is deliberately given the set of already-answered pair ids rather than
a database handle, so none of this needs SQLite to test — the same reason
FEATURE_011 kept its data layer free of torch.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Reused rather than duplicated: `_record` builds a minimal EpisodeRecord in the
# shape `runner.run_episode` emits, and a second copy of it here would drift out
# of step with the first the moment the record shape changes (FEATURE_012 §9).
from test_rlhf_pairs import _record  # noqa: E402

from soc_triage.rlhf.pairs import build_pairs, write_pairs  # noqa: E402
from soc_triage.labelling.queue import (  # noqa: E402
    LabelQueue,
    PairFileError,
    UnassignedPairError,
    UnknownLabellerError,
    assign,
    load_pairs,
)

POLICIES = ("alpha", "beta", "gamma", "delta")   # 4 policies -> 6 pairings
SEEDS = (3000000, 3000001, 3000002, 3000003, 3000004)
LABELLERS = ("L1", "L2")


@pytest.fixture
def pairs() -> list[dict]:
    """20 pairs, 6 of them double-labelled, from 4 policies over 5 seeds."""
    records = [_record(p, s) for p in POLICIES for s in SEEDS]
    built, _key = build_pairs(
        records,
        policies=POLICIES,
        target_pairs=20,
        double_labelled_pairs=6,
        sampling_seed=20260904,
    )
    return built


@pytest.fixture
def real_shape_pairs() -> list[dict]:
    """The production configuration exactly: 300 pairs, 50 double-labelled.

    9 policies give 36 pairings and 12 seeds give 432 candidates, which is the
    real `rlhf:` block in `config/training_default.yaml`. Worth one slower
    fixture: the 125/125 split is the decision under test and it should be
    checked at the size it will actually run at, not only in miniature.
    """
    policies = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9")
    seeds = range(3000000, 3000012)
    records = [_record(p, s) for p in policies for s in seeds]
    built, _key = build_pairs(
        records,
        policies=policies,
        target_pairs=300,
        double_labelled_pairs=50,
        sampling_seed=20260904,
    )
    return built


def _singles(pairs: list[dict]) -> list[str]:
    return [p["pair_id"] for p in pairs if not p["double_labelled"]]


def _doubles(pairs: list[dict]) -> list[str]:
    return [p["pair_id"] for p in pairs if p["double_labelled"]]


# --------------------------------------------------------------------------
# Assignment — the double-labelled pairs
# --------------------------------------------------------------------------

def test_every_double_labelled_pair_goes_to_both_labellers(pairs):
    """Kappa compares two independent judgements of the SAME pair."""
    allocation = assign(pairs, LABELLERS)
    for pair_id in _doubles(pairs):
        for labeller in LABELLERS:
            assert pair_id in allocation[labeller], (
                f"{pair_id} is double-labelled but {labeller} never sees it"
            )


def test_every_single_label_pair_goes_to_exactly_one_labeller(pairs):
    allocation = assign(pairs, LABELLERS)
    for pair_id in _singles(pairs):
        holders = [name for name in LABELLERS if pair_id in allocation[name]]
        assert len(holders) == 1, f"{pair_id} went to {holders}"


def test_the_union_of_the_assignments_covers_every_pair(pairs):
    """A pair assigned to nobody is a pair that never gets labelled."""
    allocation = assign(pairs, LABELLERS)
    covered = set().union(*(set(v) for v in allocation.values()))
    assert covered == {p["pair_id"] for p in pairs}


# --------------------------------------------------------------------------
# Assignment — the split
# --------------------------------------------------------------------------

def test_the_single_label_pairs_split_evenly(pairs):
    allocation = assign(pairs, LABELLERS)
    singles = set(_singles(pairs))
    counts = [len(singles & set(allocation[name])) for name in LABELLERS]
    assert counts[0] == counts[1] == len(singles) // 2


def test_the_production_shape_gives_175_judgements_each(real_shape_pairs):
    """300 pairs, 50 doubles, 2 labellers: 50 + 125 each, 350 in total (D-040)."""
    allocation = assign(real_shape_pairs, LABELLERS)
    assert len(allocation["L1"]) == 175
    assert len(allocation["L2"]) == 175
    assert sum(len(v) for v in allocation.values()) == 350


def test_an_odd_number_of_singles_gives_the_extra_to_the_first_labeller(pairs):
    """The remainder rule is fixed, so a rebuild cannot move one pair silently."""
    singles_now = _singles(pairs)
    if len(singles_now) % 2 == 0:
        dropped = singles_now[-1]
        odd = [p for p in pairs if p["pair_id"] != dropped]
    else:
        odd = list(pairs)

    singles = set(_singles(odd))
    assert len(singles) % 2 == 1, "fixture did not produce an odd count"

    allocation = assign(odd, LABELLERS)
    first = len(singles & set(allocation["L1"]))
    second = len(singles & set(allocation["L2"]))
    assert first == second + 1


def test_three_labellers_are_supported(pairs):
    """Nothing in the rule assumes exactly two, even though we will run two."""
    labellers = ("L1", "L2", "L3")
    allocation = assign(pairs, labellers)
    assert set(allocation) == set(labellers)
    for pair_id in _doubles(pairs):
        assert all(pair_id in allocation[name] for name in labellers)


# --------------------------------------------------------------------------
# Assignment — determinism
# --------------------------------------------------------------------------

def test_two_assignments_from_the_same_input_are_identical(pairs):
    assert assign(pairs, LABELLERS) == assign(pairs, LABELLERS)


def test_the_order_pairs_arrive_in_does_not_change_the_assignment(pairs):
    """`load_pairs` reads a file; nothing should depend on how it was ordered."""
    shuffled = list(reversed(pairs))
    assert assign(shuffled, LABELLERS) == assign(pairs, LABELLERS)


def test_the_order_labellers_are_listed_in_does_change_who_gets_what(pairs):
    """Stated so it is a choice and not a surprise: `rlhf.labellers` is ordered.

    Reordering the config list reshuffles the split, so it must not be reordered
    once labelling has started. The test exists to make that visible.
    """
    forward = assign(pairs, ("L1", "L2"))
    backward = assign(pairs, ("L2", "L1"))
    assert forward != backward


# --------------------------------------------------------------------------
# Assignment — refusals
# --------------------------------------------------------------------------

def test_a_single_labeller_is_refused(pairs):
    """One labeller cannot produce a kappa, so the config is wrong, not the run."""
    with pytest.raises(ValueError, match="at least two"):
        assign(pairs, ("L1",))


def test_duplicate_labeller_ids_are_refused(pairs):
    with pytest.raises(ValueError, match="L1"):
        assign(pairs, ("L1", "L1"))


def test_an_empty_labeller_id_is_refused(pairs):
    with pytest.raises(ValueError):
        assign(pairs, ("L1", ""))


# --------------------------------------------------------------------------
# The queue — serving and resuming
# --------------------------------------------------------------------------

def test_next_pair_returns_the_first_assigned_pair_when_nothing_is_answered(pairs):
    queue = LabelQueue(pairs, "L1", LABELLERS, answered=())
    assert queue.next_pair()["pair_id"] == queue.assigned[0]


def test_next_pair_skips_pairs_this_labeller_has_already_answered(pairs):
    queue = LabelQueue(pairs, "L1", LABELLERS, answered=())
    first = queue.assigned[0]

    resumed = LabelQueue(pairs, "L1", LABELLERS, answered=(first,))
    assert resumed.next_pair()["pair_id"] == resumed.assigned[1]


def test_a_pair_the_other_labeller_answered_is_still_served(pairs):
    """`answered` is one person's history. L2 finishing a pair is not L1's answer."""
    allocation = assign(pairs, LABELLERS)
    only_l2 = [p for p in allocation["L2"] if p not in allocation["L1"]]
    queue = LabelQueue(pairs, "L1", LABELLERS, answered=tuple(only_l2))
    assert queue.next_pair()["pair_id"] == queue.assigned[0]


def test_next_pair_is_none_when_the_labeller_has_finished(pairs):
    queue = LabelQueue(pairs, "L1", LABELLERS, answered=())
    finished = LabelQueue(pairs, "L1", LABELLERS, answered=queue.assigned)
    assert finished.next_pair() is None


def test_next_pair_returns_the_whole_pair_not_just_its_id(pairs):
    """The page needs both summaries, so the queue hands back the pair itself."""
    pair = LabelQueue(pairs, "L1", LABELLERS, answered=()).next_pair()
    assert set(pair) >= {"pair_id", "left", "right", "seed"}


# --------------------------------------------------------------------------
# The queue — progress
# --------------------------------------------------------------------------

def test_progress_counts_only_this_labellers_pairs(pairs):
    queue = LabelQueue(pairs, "L1", LABELLERS, answered=())
    assert queue.progress().total == len(queue.assigned)
    assert queue.progress().total < len(pairs)


def test_progress_done_counts_answers_and_ignores_the_other_labellers(pairs):
    allocation = assign(pairs, LABELLERS)
    mine = allocation["L1"]
    not_mine = [p for p in allocation["L2"] if p not in mine]

    queue = LabelQueue(pairs, "L1", LABELLERS, answered=(mine[0], *not_mine))
    assert queue.progress().done == 1


def test_progress_remaining_is_total_minus_done(pairs):
    queue = LabelQueue(pairs, "L1", LABELLERS, answered=())
    mine = queue.assigned
    part_way = LabelQueue(pairs, "L1", LABELLERS, answered=mine[:3])
    progress = part_way.progress()
    assert progress.remaining == progress.total - progress.done == len(mine) - 3


def test_progress_position_is_the_one_based_number_of_the_pair_being_shown(pairs):
    """Shows "pair 4 of 175". A labeller who cannot see the end stops labelling."""
    queue = LabelQueue(pairs, "L1", LABELLERS, answered=())
    mine = queue.assigned
    part_way = LabelQueue(pairs, "L1", LABELLERS, answered=mine[:3])
    assert part_way.progress().position == 4


def test_position_does_not_run_past_the_total_when_finished(pairs):
    queue = LabelQueue(pairs, "L1", LABELLERS, answered=())
    finished = LabelQueue(pairs, "L1", LABELLERS, answered=queue.assigned)
    progress = finished.progress()
    assert progress.position == progress.total
    assert progress.remaining == 0


# --------------------------------------------------------------------------
# The queue — refusals
# --------------------------------------------------------------------------

def test_an_unknown_labeller_is_refused_and_the_known_ids_are_named(pairs):
    """A typo at launch must not open a session that writes 175 rows under it."""
    with pytest.raises(UnknownLabellerError) as excinfo:
        LabelQueue(pairs, "L!", LABELLERS, answered=())
    message = str(excinfo.value)
    assert "L!" in message and "L1" in message and "L2" in message


def test_asking_for_a_pair_not_assigned_to_this_labeller_is_refused(pairs):
    allocation = assign(pairs, LABELLERS)
    not_mine = next(p for p in allocation["L2"] if p not in allocation["L1"])
    queue = LabelQueue(pairs, "L1", LABELLERS, answered=())
    with pytest.raises(UnassignedPairError, match=not_mine):
        queue.pair(not_mine)


def test_asking_for_a_pair_id_that_does_not_exist_is_refused(pairs):
    queue = LabelQueue(pairs, "L1", LABELLERS, answered=())
    with pytest.raises(UnassignedPairError, match="p9999"):
        queue.pair("p9999")


def test_a_pair_assigned_to_this_labeller_is_returned_by_id(pairs):
    queue = LabelQueue(pairs, "L1", LABELLERS, answered=())
    wanted = queue.assigned[2]
    assert queue.pair(wanted)["pair_id"] == wanted


# --------------------------------------------------------------------------
# Loading — and the blinding guard (D-038)
# --------------------------------------------------------------------------

def test_load_pairs_reads_a_file_written_by_write_pairs(tmp_path, pairs):
    pairs_path, _key_path = write_pairs(pairs, {}, tmp_path)
    assert load_pairs(pairs_path) == pairs


def test_load_pairs_refuses_the_key_file_by_name(tmp_path, pairs):
    """The realistic mistake is copying the wrong file, so refuse it by name.

    `pairs_key.json` maps pair_id to policy names. Reading it here would destroy
    the blinding the whole pair set is built on (D-038).
    """
    _pairs_path, key_path = write_pairs(pairs, {"p0000": {}}, tmp_path)
    with pytest.raises(PairFileError, match="pairs_key.json"):
        load_pairs(key_path)


def test_load_pairs_on_a_missing_file_says_which_one(tmp_path):
    missing = tmp_path / "nope" / "pairs.json"
    with pytest.raises(PairFileError, match="pairs.json"):
        load_pairs(missing)


def test_load_pairs_refuses_a_file_that_is_not_a_list_of_pairs(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"pair_id": "p0000"}), encoding="utf-8")
    with pytest.raises(PairFileError):
        load_pairs(path)


def test_load_pairs_refuses_a_pair_missing_a_required_field(tmp_path):
    """Fail at load, not three screens into a labelling session."""
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([{"pair_id": "p0000", "left": {}}]), encoding="utf-8")
    with pytest.raises(PairFileError, match="right"):
        load_pairs(path)


def test_load_pairs_refuses_duplicate_pair_ids(tmp_path, pairs):
    """Labels reference pair_id; two pairs sharing one would corrupt the set."""
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([pairs[0], pairs[0]]), encoding="utf-8")
    with pytest.raises(PairFileError, match=pairs[0]["pair_id"]):
        load_pairs(path)
