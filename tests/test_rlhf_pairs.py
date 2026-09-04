"""`rlhf.pairs` — turning EpisodeRecords into blinded preference pairs.

Three properties carry most of the weight here, and each one protects something
that would be expensive or impossible to notice later:

1. **Determinism.** Labels reference `pair_id`. If a rebuild renumbered the
   pairs, every label already collected would silently point at a different
   comparison and nothing would raise.
2. **Same seed on both sides.** A pair whose two sides ran different alert
   streams is a comparison of luck, not of policies (`rlhf.pair_must_share_seed`).
3. **Blinding.** `pairs.json` is what the labelling UI reads. If a policy name
   reaches it — including inside a `run_id` like `sarsa-seed3000004` — the
   blinding is gone and the labeller is judging reputations.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.rlhf.pairs import (  # noqa: E402
    InsufficientRecordsError,
    MixedConfigError,
    build_pairs,
    load_records,
    write_pairs,
)

POLICIES = ("alpha", "beta", "gamma", "delta")   # 4 policies -> 6 pairings
SEEDS = (3000000, 3000001, 3000002, 3000003, 3000004)


def _record(agent: str, seed: int, config_hash: str = "cfg123") -> dict:
    """A minimal EpisodeRecord in the shape `runner.run_episode` emits.

    One step, so the summary has something to render; the numbers themselves do
    not matter to pair construction, only the identity fields do.
    """
    return {
        "run_id": f"{agent}-seed{seed}",
        "agent_name": agent,
        "seed": seed,
        "config_hash": config_hash,
        "steps": [{
            "state_disc": 3,
            "action": 0,
            "reward": -10.0,
            "info": {
                "action_name": "PULL_HIGHEST_SEVERITY",
                "alert_investigated": {
                    "id": 1, "arrival_time": 1.0, "severity": 2,
                    "asset_criticality": 1, "verify_cost_min": 10,
                    "alert_type": "phishing_click", "is_true_incident": False,
                    "deadline_min": 0.0,
                },
                "was_true_incident": False,
                "delay_min": None,
                "n_bulk_closed": 0,
                "bulk_closed_ids": [],
                "time_consumed": 10.0,
                "reward_breakdown": {"false_positive_cost": -10.0},
            },
        }],
        "outcome": {
            "incidents_total": 4, "incidents_caught": 0,
            "incidents_caught_in_time": 0, "incidents_missed": 4,
            "incidents_buried_by_bulk_close": 0, "critical_missed": 1,
            "missed_by_criticality": [1, 2, 1], "mttd_min": None,
            "wasted_minutes": 10.0, "total_reward": -10.0,
        },
    }


@pytest.fixture
def records() -> list[dict]:
    """Every policy on every seed — 4 x 5 = 20 records."""
    return [_record(p, s) for p in POLICIES for s in SEEDS]


def _build(records, target=20, double=6, seed=20260904):
    return build_pairs(
        records,
        policies=POLICIES,
        target_pairs=target,
        double_labelled_pairs=double,
        sampling_seed=seed,
    )


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_two_builds_from_the_same_inputs_are_identical(records):
    """Byte-identical, not merely equivalent.

    Labels reference pair_id. A rebuild that renumbered the pairs would
    repoint every label already collected, and no test anywhere would fail.
    """
    first_pairs, first_key = _build(records)
    second_pairs, second_key = _build(records)
    assert json.dumps(first_pairs) == json.dumps(second_pairs)
    assert json.dumps(first_key, sort_keys=True) == json.dumps(second_key, sort_keys=True)


def test_record_order_does_not_change_the_result(records):
    """The build must depend on the records' content, not on how the filesystem
    happened to hand them over."""
    forward, _ = _build(records)
    backward, _ = _build(list(reversed(records)))
    assert json.dumps(forward) == json.dumps(backward)


def test_a_different_sampling_seed_gives_a_different_set(records):
    """Otherwise the seed is decorative and the sampling is not really random."""
    a, _ = _build(records, seed=1)
    b, _ = _build(records, seed=2)
    assert json.dumps(a) != json.dumps(b)


# --------------------------------------------------------------------------
# Pair validity
# --------------------------------------------------------------------------

def test_every_pair_has_the_same_seed_on_both_sides(records):
    """`rlhf.pair_must_share_seed` — asserted, never assumed."""
    pairs, key = _build(records)
    for pair in pairs:
        assert pair["left"]["seed"] == pair["right"]["seed"] == pair["seed"]
        assert key[pair["pair_id"]]["seed"] == pair["seed"]


def test_no_pair_compares_a_policy_with_itself(records):
    pairs, key = _build(records)
    for pair in pairs:
        entry = key[pair["pair_id"]]
        assert entry["left_policy"] != entry["right_policy"]


def test_no_two_pairs_are_the_same_comparison(records):
    """The same two policies on the same seed twice is a wasted 20 seconds of
    someone's attention and a duplicated row in the reward model's data."""
    _, key = _build(records)
    seen = {
        (frozenset({e["left_policy"], e["right_policy"]}), e["seed"])
        for e in key.values()
    }
    assert len(seen) == len(key)


def test_pair_ids_are_unique_and_zero_padded(records):
    pairs, _ = _build(records)
    ids = [p["pair_id"] for p in pairs]
    assert len(set(ids)) == len(ids)
    assert all(pid.startswith("p") and pid[1:].isdigit() for pid in ids)
    assert len(ids[0]) == len(ids[-1]), "ids must sort lexicographically"


# --------------------------------------------------------------------------
# Balanced allocation
# --------------------------------------------------------------------------

def test_the_total_is_exactly_the_target(records):
    pairs, _ = _build(records, target=20)
    assert len(pairs) == 20


def test_every_policy_pairing_gets_a_near_equal_share(records):
    """4 policies is 6 unordered pairings. 20 pairs is 3 each with 2 left over,
    so four pairings get 3 and two get 4 — and nothing gets 0 or 6.

    Balance matters for the Bradley-Terry fit: a pairing that never appears
    leaves the model with no direct evidence about those two policies'
    relative worth, and it has to infer it transitively.
    """
    _, key = _build(records, target=20)
    counts: dict[frozenset, int] = {}
    for entry in key.values():
        pairing = frozenset({entry["left_policy"], entry["right_policy"]})
        counts[pairing] = counts.get(pairing, 0) + 1

    assert len(counts) == 6, "all six pairings must appear"
    assert sorted(counts.values()) == [3, 3, 3, 3, 4, 4]


def test_an_exactly_divisible_target_gives_an_exactly_equal_split(records):
    _, key = _build(records, target=18, double=4)
    counts: dict[frozenset, int] = {}
    for entry in key.values():
        counts[frozenset({entry["left_policy"], entry["right_policy"]})] = \
            counts.get(frozenset({entry["left_policy"], entry["right_policy"]}), 0) + 1
    assert sorted(counts.values()) == [3] * 6


# --------------------------------------------------------------------------
# Blinding — FEATURE_011 §6
# --------------------------------------------------------------------------

def test_the_pairs_file_contains_no_policy_name_anywhere(records, tmp_path):
    """Checked against the serialised file, not the in-memory object.

    `run_id` is the trap: a summary carries `sarsa-seed3000004`, which names
    the policy in passing. Substring-searching the written JSON is the only
    check that catches a name arriving through a field nobody thought about.
    """
    pairs, key = _build(records)
    write_pairs(pairs, key, tmp_path)

    text = (tmp_path / "pairs.json").read_text(encoding="utf-8")
    for policy in POLICIES:
        assert policy not in text, f"policy name {policy!r} leaked into pairs.json"
    assert "run_id" not in text


def test_the_key_file_does_carry_the_names_and_covers_every_pair(records, tmp_path):
    pairs, key = _build(records)
    write_pairs(pairs, key, tmp_path)

    written_key = json.loads((tmp_path / "pairs_key.json").read_text(encoding="utf-8"))
    assert set(written_key) == {p["pair_id"] for p in pairs}
    entry = written_key[pairs[0]["pair_id"]]
    assert set(entry) == {"left_policy", "right_policy", "left_run_id",
                          "right_run_id", "swapped", "seed"}


def test_swapped_records_which_side_the_first_policy_landed_on(records):
    """The analysis has to be able to undo the side assignment exactly, and the
    position bias itself is worth measuring rather than assuming away."""
    _, key = _build(records)
    for entry in key.values():
        first, second = sorted((entry["left_policy"], entry["right_policy"]))
        if entry["swapped"]:
            assert entry["right_policy"] == first
        else:
            assert entry["left_policy"] == first


def test_both_sides_are_used_for_the_first_policy_across_the_set(records):
    """If `swapped` were always False, position bias would be perfectly
    confounded with whichever policy sorts first alphabetically."""
    _, key = _build(records)
    swaps = [e["swapped"] for e in key.values()]
    assert any(swaps) and not all(swaps)


def test_summaries_in_pairs_still_carry_no_reward(records):
    """The §7 rule, re-checked at the surface the labeller actually sees."""
    pairs, _ = _build(records)
    text = json.dumps(pairs)
    assert "reward" not in text.lower()
    assert "total_reward" not in text


# --------------------------------------------------------------------------
# Double labelling
# --------------------------------------------------------------------------

def test_exactly_the_requested_number_of_pairs_is_double_labelled(records):
    pairs, _ = _build(records, target=20, double=6)
    flagged = [p for p in pairs if p["double_labelled"]]
    assert len(flagged) == 6


def test_double_labelled_pairs_are_spread_across_policy_pairings(records):
    """Taking the first 50 in file order would concentrate them on whichever
    pairings happen to be built first, and kappa would then describe agreement
    on one narrow kind of comparison."""
    pairs, key = _build(records, target=20, double=6)
    pairings = {
        frozenset({key[p["pair_id"]]["left_policy"], key[p["pair_id"]]["right_policy"]})
        for p in pairs if p["double_labelled"]
    }
    assert len(pairings) >= 4, f"6 double-labelled pairs spread over only {len(pairings)} pairings"


def test_asking_for_more_double_labels_than_pairs_is_refused(records):
    with pytest.raises(ValueError):
        _build(records, target=10, double=11)


# --------------------------------------------------------------------------
# Refusing inputs that would produce invalid data
# --------------------------------------------------------------------------

def test_records_from_two_different_configs_are_refused(records):
    """Two sides run under different `env_default.yaml` contents are not the
    same shift, even on the same seed. `runner.config_hash` exists precisely so
    this is checkable rather than hoped for."""
    records.append(_record("alpha", 3000009, config_hash="OTHER"))
    with pytest.raises(MixedConfigError):
        _build(records)


def test_a_missing_policy_is_refused_with_its_name(records):
    """Silently dropping a policy would produce a valid-looking pair set that
    quietly under-covers the behaviour space the Phase 6 audit depends on."""
    thinned = [r for r in records if r["agent_name"] != "delta"]
    with pytest.raises(InsufficientRecordsError) as excinfo:
        _build(thinned)
    assert "delta" in str(excinfo.value)


def test_too_few_seeds_for_the_target_is_refused_with_the_arithmetic(records):
    """6 pairings x 5 seeds is 30 candidate pairs; asking for 40 cannot be met
    without repeating a comparison, so it fails loudly instead."""
    with pytest.raises(InsufficientRecordsError) as excinfo:
        _build(records, target=40, double=5)
    message = str(excinfo.value)
    assert "40" in message and "30" in message


def test_only_seeds_where_every_policy_ran_are_eligible(records):
    """A seed one policy is missing cannot host all six pairings, so allowing it
    would quietly unbalance the allocation."""
    thinned = [r for r in records
               if not (r["agent_name"] == "delta" and r["seed"] == 3000004)]
    pairs, _ = _build(thinned, target=18, double=4)
    assert all(p["seed"] != 3000004 for p in pairs)


# --------------------------------------------------------------------------
# Reading records off disk
# --------------------------------------------------------------------------

def test_load_records_reads_a_directory_of_json(tmp_path, records):
    for record in records[:3]:
        (tmp_path / f"{record['run_id']}.json").write_text(
            json.dumps(record), encoding="utf-8")
    loaded = load_records(tmp_path)
    assert len(loaded) == 3
    assert {r["agent_name"] for r in loaded} == {"alpha"}


def test_load_records_on_a_missing_directory_says_which_one(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_records(tmp_path / "nope")
    assert "nope" in str(excinfo.value)


def test_write_pairs_creates_the_directory_and_both_files(tmp_path, records):
    pairs, key = _build(records)
    out = tmp_path / "deep" / "rlhf"
    write_pairs(pairs, key, out)
    assert (out / "pairs.json").exists()
    assert (out / "pairs_key.json").exists()
