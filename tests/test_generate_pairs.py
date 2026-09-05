"""`scripts/generate_pairs.py` — the pair-seed survey's comparison arithmetic.

Phase 5a owes one decision before the pair set can be built: which training
repeat each multi-run learner contributes. Session 12 found that three of
REINFORCE's five repeats reproduce `severity_sort` exactly on the EVALUATION
seeds, which would let a labeller compare `severity_sort` against itself under
two blinded names and call the difference a preference.

Eval-seed identity is not pair-seed identity — different alert streams, and a
policy that ties on one block may separate on another. So the survey measures it
on the block that matters (`rlhf.pair_seed_start`, 3000000..3000011) and these
tests pin the arithmetic that reads the answer off the records.

Synthetic records throughout. Nothing here loads a policy or imports torch: two
policies are indistinguishable to a labeller exactly when they take the same
actions, so the comparison is a property of action sequences and is testable
with no `results/` present at all.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_pairs import (  # noqa: E402
    action_trace,
    collapsed_pairs,
    identical_groups,
    policy_name,
    seeds_matching,
    variant_traces,
)


def _record(agent: str, seed: int, actions) -> dict:
    """A record carrying only what the comparison reads.

    The survey compares behaviour, so `steps[*]["action"]` is the whole payload.
    Rewards and outcomes are deliberately absent: if the comparison ever starts
    depending on them, these tests stop constructing a valid input and say so.
    """
    return {
        "run_id": f"{agent}-seed{seed}",
        "agent_name": agent,
        "seed": seed,
        "steps": [{"action": int(a)} for a in actions],
    }


def test_action_trace_is_the_step_actions_in_order():
    record = _record("q_learning", 3000000, [2, 0, 1, 1])
    assert action_trace(record) == (2, 0, 1, 1)


def test_action_trace_of_an_episode_with_no_steps_is_empty():
    # Not a hypothetical: an episode whose shift ends before any action is an
    # empty trace, and it must compare equal to another empty trace rather than
    # raising halfway through a 250-episode survey.
    assert action_trace(_record("random", 3000000, [])) == ()


def test_variant_traces_are_keyed_by_integer_seed():
    records = [
        _record("dqn", 3000000, [1, 2]),
        _record("dqn", 3000001, [0, 0]),
    ]
    assert variant_traces(records) == {3000000: (1, 2), 3000001: (0, 0)}


def test_seeds_matching_returns_the_seeds_where_both_took_the_same_actions():
    a = {3000000: (1, 2), 3000001: (0, 0), 3000002: (3,)}
    b = {3000000: (1, 2), 3000001: (0, 0), 3000002: (3,)}
    assert seeds_matching(a, b) == [3000000, 3000001, 3000002]


def test_seeds_matching_excludes_a_seed_where_a_single_action_differs():
    # One differing action is a different shift for the person reading it, so
    # near-identity is not identity and must not be reported as a match.
    a = {3000000: (1, 2, 3), 3000001: (0, 0)}
    b = {3000000: (1, 2, 4), 3000001: (0, 0)}
    assert seeds_matching(a, b) == [3000001]


def test_seeds_matching_ignores_seeds_only_one_side_ran():
    # A seed one side is missing is not evidence either way. Counting it as a
    # match would inflate the collapse count; counting it as a mismatch would
    # hide one.
    a = {3000000: (1,), 3000001: (2,)}
    b = {3000000: (1,)}
    assert seeds_matching(a, b) == [3000000]


def test_identical_groups_puts_indistinguishable_variants_together():
    traces = {
        "severity_sort": {3000000: (1, 1), 3000001: (2, 2)},
        "reinforce@2": {3000000: (1, 1), 3000001: (2, 2)},
        "reinforce@0": {3000000: (0, 3), 3000001: (4, 4)},
    }
    assert identical_groups(traces) == [
        ("reinforce@2", "severity_sort"),
        ("reinforce@0",),
    ]


def test_identical_groups_separates_variants_differing_on_one_seed_of_many():
    traces = {
        "a": {3000000: (1,), 3000001: (1,), 3000002: (1,)},
        "b": {3000000: (1,), 3000001: (1,), 3000002: (9,)},
    }
    assert identical_groups(traces) == [("a",), ("b",)]


def test_identical_groups_orders_deterministically():
    # The survey's output is read once and then quoted in a DECISIONS entry, so
    # two runs over the same records must print the same table. Insertion order
    # of the input dict must not reach the output.
    forward = identical_groups({
        "zeta": {3000000: (1,)},
        "alpha": {3000000: (1,)},
        "beta": {3000000: (2,)},
    })
    backward = identical_groups({
        "beta": {3000000: (2,)},
        "alpha": {3000000: (1,)},
        "zeta": {3000000: (1,)},
    })
    assert forward == backward == [("alpha", "zeta"), ("beta",)]


# --- write mode: the guard that stops a self-pair reaching a labeller --------
#
# The survey reports collapses; it cannot enforce anything, because a human
# reads its table and then chooses. These tests cover the check that runs
# unconditionally in write mode, so a bad choice -- made now, or made in six
# months by someone who never read the survey -- cannot silently produce a pair
# set in which severity_sort argues with itself.


def test_policy_name_strips_the_repeat_suffix():
    # Records must carry the bare policy name: `rlhf.policies` in the config
    # lists `dqn`, and `build_pairs` matches records on `agent_name`. A record
    # written as "dqn@0" would be invisible to it and the build would refuse
    # with "no EpisodeRecords for ['dqn']".
    assert policy_name("dqn@0") == "dqn"
    assert policy_name("actor_critic@4") == "actor_critic"


def test_policy_name_leaves_single_artefact_policies_alone():
    assert policy_name("severity_sort") == "severity_sort"
    assert policy_name("monte_carlo") == "monte_carlo"


def test_collapsed_pairs_is_empty_when_every_chosen_variant_differs():
    traces = {
        "severity_sort": {3000000: (1, 1)},
        "reinforce@0": {3000000: (0, 3)},
        "dqn@0": {3000000: (2, 2)},
    }
    assert collapsed_pairs(traces, ["severity_sort", "reinforce@0", "dqn@0"]) == []


def test_collapsed_pairs_names_two_chosen_variants_that_are_identical():
    traces = {
        "severity_sort": {3000000: (1, 1), 3000001: (2, 2)},
        "reinforce@2": {3000000: (1, 1), 3000001: (2, 2)},
        "dqn@0": {3000000: (0, 0), 3000001: (0, 0)},
    }
    assert collapsed_pairs(traces, ["severity_sort", "reinforce@2", "dqn@0"]) == [
        ("reinforce@2", "severity_sort")
    ]


def test_collapsed_pairs_ignores_a_collapse_among_variants_not_chosen():
    # reinforce@2 and reinforce@4 are identical, but neither is going into the
    # pair set. An unchosen collapse is a fact about training, not a defect in
    # the pair set, and the guard must not refuse a build over it.
    traces = {
        "severity_sort": {3000000: (1, 1)},
        "reinforce@0": {3000000: (0, 3)},
        "reinforce@2": {3000000: (1, 1)},
        "reinforce@4": {3000000: (1, 1)},
    }
    assert collapsed_pairs(traces, ["severity_sort", "reinforce@0"]) == []


def test_collapsed_pairs_reports_every_collision_in_a_fixed_order():
    # Reported in full rather than failing on the first: the operator should see
    # all of them in one run instead of rediscovering them one build at a time.
    traces = {
        "a": {3000000: (1,)},
        "b": {3000000: (1,)},
        "c": {3000000: (1,)},
        "d": {3000000: (9,)},
    }
    assert collapsed_pairs(traces, ["d", "c", "b", "a"]) == [
        ("a", "b"), ("a", "c"), ("b", "c")
    ]
