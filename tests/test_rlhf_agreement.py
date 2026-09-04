"""`rlhf.agreement` — Cohen's kappa, checked against arithmetic done on paper.

Every expected value in this file was worked out by hand and the working is in
the docstring. That is the same discipline `tests/test_tiny_mdp.py` applies to
the learners (D-014): a number produced by running the function and then written
into the test proves only that the function is deterministic.

kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e is the
agreement expected from the two labellers' own marginal frequencies.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.rlhf.agreement import (  # noqa: E402
    agreement_between,
    cohens_kappa,
    describe,
)
from soc_triage.rlhf.store import LabelStore  # noqa: E402


def _labels(*choices: str) -> dict[str, str]:
    """pair_id -> choice, for pairs p0000, p0001, ... in the order given."""
    return {f"p{i:04d}": choice for i, choice in enumerate(choices)}


# --------------------------------------------------------------------------
# The anchor: a 10-pair example solved on paper
# --------------------------------------------------------------------------

def test_kappa_matches_a_hand_worked_ten_pair_example():
    """Ten shared pairs, three categories.

        pair : 0 1 2 3 4 5 6 7 8 9
        A    : L L L L R R R T T T
        B    : L L L R R R T T T L
        agree: y y y n y y n y y n   -> 7 agreements

    p_o = 7/10 = 0.7

    Marginals are the same for both by construction: L 4, R 3, T 3 for A;
    L 4, R 3, T 3 for B (B has L at pairs 0,1,2,9).

    p_e = (4/10)(4/10) + (3/10)(3/10) + (3/10)(3/10)
        = 0.16 + 0.09 + 0.09
        = 0.34

    kappa = (0.70 - 0.34) / (1 - 0.34) = 0.36 / 0.66 = 6/11 = 0.545454...
    """
    a = _labels("left", "left", "left", "left", "right",
                "right", "right", "tie", "tie", "tie")
    b = _labels("left", "left", "left", "right", "right",
                "right", "tie", "tie", "tie", "left")

    result = cohens_kappa(a, b)

    assert result.n_shared == 10
    assert result.observed_agreement == pytest.approx(0.7)
    assert result.expected_agreement == pytest.approx(0.34)
    assert result.kappa == pytest.approx(6 / 11)
    assert result.undefined_reason is None


def test_perfect_agreement_is_exactly_one():
    """A and B answer identically across all three categories.

    p_o = 1. p_e = (1/3)^2 * 3 = 1/3. kappa = (1 - 1/3)/(1 - 1/3) = 1.0
    """
    a = _labels("left", "right", "tie")
    b = _labels("left", "right", "tie")
    result = cohens_kappa(a, b)
    assert result.kappa == pytest.approx(1.0)
    assert result.expected_agreement == pytest.approx(1 / 3)


def test_agreement_exactly_at_chance_is_exactly_zero():
    """Two categories, marginals 50/50 each, half the answers agreeing.

        A : L L R R
        B : L R L R      -> agreements at pairs 0 and 3, so p_o = 0.5
        p_e = (2/4)(2/4) + (2/4)(2/4) = 0.25 + 0.25 = 0.5
        kappa = (0.5 - 0.5) / (1 - 0.5) = 0.0

    This is the case that makes kappa worth computing at all: raw agreement of
    50% sounds like partial agreement and is in fact none.
    """
    a = _labels("left", "left", "right", "right")
    b = _labels("left", "right", "left", "right")
    result = cohens_kappa(a, b)
    assert result.observed_agreement == pytest.approx(0.5)
    assert result.kappa == pytest.approx(0.0)


def test_systematic_disagreement_is_negative():
    """A and B answer oppositely every time.

        A : L L R R
        B : R R L L      -> p_o = 0
        p_e = 0.5 as above, so kappa = (0 - 0.5)/0.5 = -1.0

    Reported as -1.0 rather than clamped to 0. A negative kappa is a real and
    informative result: it says the two labellers are not merely unaligned but
    anti-aligned, which points at a misread interface far more often than at a
    genuine difference of opinion.
    """
    a = _labels("left", "left", "right", "right")
    b = _labels("right", "right", "left", "left")
    assert cohens_kappa(a, b).kappa == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# Only the shared pairs count
# --------------------------------------------------------------------------

def test_only_pairs_both_labellers_answered_are_used():
    """A answered five, B answered four, three overlap.

        shared: p0001 (L/L), p0002 (R/R), p0003 (T/L)
        p_o = 2/3
    """
    a = {"p0000": "left", "p0001": "left", "p0002": "right",
         "p0003": "tie", "p0004": "tie"}
    b = {"p0001": "left", "p0002": "right", "p0003": "left", "p0009": "right"}

    result = cohens_kappa(a, b)
    assert result.n_shared == 3
    assert result.observed_agreement == pytest.approx(2 / 3)


def test_marginals_are_computed_over_the_shared_pairs_only():
    """The classic way to get kappa wrong.

    A's marginals must be counted over the three shared pairs (L, R, T), not
    over all five they answered (L, L, R, T, T). Using the full set would give
    A a 'tie' rate of 2/5 where the shared subset says 1/3, and p_e would be a
    number describing a comparison that was never made.
    """
    a = {"p0000": "left", "p0001": "left", "p0002": "right",
         "p0003": "tie", "p0004": "tie"}
    b = {"p0001": "left", "p0002": "right", "p0003": "left"}

    result = cohens_kappa(a, b)
    # Shared A: left, right, tie -> 1/3 each. Shared B: left, right, left -> L 2/3, R 1/3.
    # p_e = (1/3)(2/3) + (1/3)(1/3) + (1/3)(0) = 2/9 + 1/9 = 3/9 = 1/3
    assert result.expected_agreement == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# The cases where kappa does not exist — and must not be faked
# --------------------------------------------------------------------------

def test_kappa_is_undefined_when_both_labellers_used_one_category():
    """Everyone always said 'left'.

    p_o = 1 and p_e = 1, so kappa = 0/0. The temptation is to call this perfect
    agreement and return 1.0. It is not: two people who always press the same
    button agree by construction, and there is no evidence in the data about
    whether they would agree on anything else. Returning None with a reason is
    the honest answer, and the caller must print it as 'undefined', not as a
    number.
    """
    a = _labels("left", "left", "left", "left")
    b = _labels("left", "left", "left", "left")

    result = cohens_kappa(a, b)
    assert result.kappa is None
    assert result.undefined_reason is not None
    assert result.observed_agreement == pytest.approx(1.0)


def test_kappa_is_undefined_below_two_shared_pairs():
    """One agreement is not a measurement (CONSTRAINTS #3 in miniature)."""
    for shared in ({}, {"p0000": "left"}):
        result = cohens_kappa({"p0000": "left"}, shared)
        assert result.kappa is None
        assert result.undefined_reason is not None


def test_an_unknown_choice_is_refused_rather_than_silently_counted():
    """A choice outside left/right/tie means the data came from somewhere the
    store did not validate. Counting it would produce a kappa over a category
    set nobody designed."""
    with pytest.raises(ValueError):
        cohens_kappa(_labels("left", "maybe"), _labels("left", "left"))


# --------------------------------------------------------------------------
# What the result carries for the report
# --------------------------------------------------------------------------

def test_the_result_carries_the_confusion_matrix():
    """PROJECT_BRIEF §6.2 says a low kappa is itself a finding. To write that
    up you need to see *how* people disagreed, not only that they did."""
    a = _labels("left", "left", "right")
    b = _labels("left", "tie", "right")
    confusion = cohens_kappa(a, b).confusion
    assert confusion[("left", "left")] == 1
    assert confusion[("left", "tie")] == 1
    assert confusion[("right", "right")] == 1
    assert confusion[("tie", "left")] == 0


# --------------------------------------------------------------------------
# Reading the two labellers straight out of the store
# --------------------------------------------------------------------------

def test_agreement_between_reads_two_labellers_from_a_store(tmp_path):
    """The 50 double-labelled pairs are the reason this convenience exists."""
    with LabelStore(tmp_path / "labels.db") as store:
        store.add_label("p0000", "A1", "left")
        store.add_label("p0000", "A2", "left")
        store.add_label("p0001", "A1", "right")
        store.add_label("p0001", "A2", "right")
        store.add_label("p0002", "A1", "tie")
        store.add_label("p0002", "A2", "left")
        store.add_label("p0003", "A1", "left")   # A2 never answered this one

        result = agreement_between(store, "A1", "A2")

    assert result.n_shared == 3
    assert result.observed_agreement == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# The sentence the report will quote
# --------------------------------------------------------------------------

def test_describe_states_the_undefined_case_in_words_not_as_a_number():
    """The failure this guards against is a report that prints kappa as 0.000
    when the truth is that it does not exist."""
    result = cohens_kappa(_labels("left", "left"), _labels("left", "left"))
    text = describe(result)
    assert "undefined" in text
    assert "0.000" not in text


def test_describe_reports_the_coefficient_and_the_sample_size():
    """A kappa without an n is not quotable (CONSTRAINTS #3's habit)."""
    a = _labels("left", "left", "left", "left", "right",
                "right", "right", "tie", "tie", "tie")
    b = _labels("left", "left", "left", "right", "right",
                "right", "tie", "tie", "tie", "left")
    text = describe(cohens_kappa(a, b))
    assert "0.545" in text          # 6/11, the hand-worked value above
    assert "10 pairs" in text
    assert "moderate" in text       # 0.545 falls in the 0.4-0.6 band
