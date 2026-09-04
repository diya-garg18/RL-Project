"""Cohen's kappa over the double-labelled pairs (FEATURE_011 §10).

    kappa = (p_o - p_e) / (1 - p_e)

`p_o` is how often the two labellers actually agreed. `p_e` is how often they
would have agreed by chance alone, given each one's own habit of choosing left,
right or tie. Subtracting the second from the first is the whole idea: two
people who both press "left" 90% of the time will agree ~82% of the time while
sharing no judgement at all.

Written by hand rather than imported from `sklearn`. It is six lines of
arithmetic, it is on the syllabus, and CONSTRAINTS #7's principle is that
anything a viva might ask us to derive is written out where we can read it.

PROJECT_BRIEF §6.2: "If kappa is low, that is itself a finding about how
ill-defined 'good triage' is." This module therefore reports the ingredients —
n, p_o, p_e and the confusion matrix — alongside the coefficient, because a
kappa quoted on its own cannot be written up.
"""

import itertools
from dataclasses import dataclass, field

from soc_triage.rlhf.store import CHOICES, LabelStore


@dataclass(frozen=True)
class Agreement:
    """The result of comparing two labellers, with the working shown.

    `kappa` is None when the coefficient does not exist; `undefined_reason`
    then says why. A caller printing this must print "undefined" and the
    reason, never a substituted number.
    """

    kappa: float | None
    n_shared: int
    observed_agreement: float
    expected_agreement: float
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    undefined_reason: str | None = None


def _check_choices(labels: dict[str, str], who: str) -> None:
    """Refuse anything outside the three answers the store allows.

    A stray category means the data came from a path that skipped
    `LabelStore`'s CHECK constraint. Counting it would yield a kappa over a
    category set nobody designed, and it would look perfectly plausible.
    """
    for pair_id, choice in labels.items():
        if choice not in CHOICES:
            raise ValueError(
                f"{who} gave choice {choice!r} on pair {pair_id!r}; "
                f"expected one of {CHOICES}"
            )


def cohens_kappa(labels_a: dict[str, str], labels_b: dict[str, str]) -> Agreement:
    """Cohen's kappa between two labellers, over the pairs they both answered.

    Both arguments map pair_id to one of CHOICES. Pairs only one of them
    answered are ignored — and, importantly, so are they when computing the
    marginals. Counting a labeller's habits over their whole workload rather
    than over the shared subset produces a `p_e` describing a comparison that
    was never made.
    """
    _check_choices(labels_a, "labeller A")
    _check_choices(labels_b, "labeller B")

    shared = sorted(set(labels_a) & set(labels_b))
    n = len(shared)

    # Every cell exists, including the empty ones — a confusion matrix with
    # missing keys forces every reader to write `.get(k, 0)`.
    confusion = {combo: 0 for combo in itertools.product(CHOICES, CHOICES)}
    for pair_id in shared:
        confusion[(labels_a[pair_id], labels_b[pair_id])] += 1

    if n < 2:
        return Agreement(
            kappa=None,
            n_shared=n,
            observed_agreement=0.0,
            expected_agreement=0.0,
            confusion=confusion,
            undefined_reason=(
                f"only {n} pair(s) labelled by both; kappa needs at least 2"
            ),
        )

    # Observed agreement: the diagonal of the confusion matrix.
    agreements = sum(confusion[(choice, choice)] for choice in CHOICES)
    p_o = agreements / n

    # Expected agreement: each labeller's own rate of choosing each category,
    # multiplied together and summed. Explicit loop over the three categories
    # rather than a comprehension, per CONSTRAINTS #14.
    p_e = 0.0
    for choice in CHOICES:
        rate_a = sum(1 for pair_id in shared if labels_a[pair_id] == choice) / n
        rate_b = sum(1 for pair_id in shared if labels_b[pair_id] == choice) / n
        p_e += rate_a * rate_b

    if p_e == 1.0:
        # Both labellers used exactly one category, and the same one. p_o is
        # also 1, so the formula is 0/0. This is not perfect agreement: two
        # people who always press the same button agree by construction, and
        # the data says nothing about whether they would agree elsewhere.
        return Agreement(
            kappa=None,
            n_shared=n,
            observed_agreement=p_o,
            expected_agreement=p_e,
            confusion=confusion,
            undefined_reason=(
                "both labellers used a single category throughout, so chance "
                "agreement is 1.0 and kappa is 0/0 — undefined, not 1.0"
            ),
        )

    return Agreement(
        kappa=(p_o - p_e) / (1.0 - p_e),
        n_shared=n,
        observed_agreement=p_o,
        expected_agreement=p_e,
        confusion=confusion,
    )


def agreement_between(store: LabelStore, labeller_a: str, labeller_b: str) -> Agreement:
    """Cohen's kappa between two labellers, read straight from the database.

    This is the form the 50 double-labelled pairs are actually measured with
    (ROADMAP 5a, PROJECT_BRIEF §6.2).
    """
    labels_a = {row["pair_id"]: row["choice"] for row in store.labels_by(labeller_a)}
    labels_b = {row["pair_id"]: row["choice"] for row in store.labels_by(labeller_b)}
    return cohens_kappa(labels_a, labels_b)


def describe(result: Agreement) -> str:
    """One paragraph a report can quote, including the undefined case.

    Landis & Koch's bands (<0 poor, 0-0.2 slight, 0.2-0.4 fair, 0.4-0.6
    moderate, 0.6-0.8 substantial, >0.8 almost perfect) are quoted as the
    conventional reading, and labelled as a convention rather than a fact —
    they are a rule of thumb from a 1977 biometrics paper, not a property of
    this data.
    """
    if result.kappa is None:
        return (f"kappa undefined over {result.n_shared} shared pair(s): "
                f"{result.undefined_reason}")

    bands = [(0.0, "poor (worse than chance)"), (0.2, "slight"), (0.4, "fair"),
             (0.6, "moderate"), (0.8, "substantial"), (1.01, "almost perfect")]
    reading = next(name for edge, name in bands if result.kappa < edge)

    return (
        f"kappa = {result.kappa:.3f} over {result.n_shared} pairs labelled by "
        f"both ({reading}, on the conventional Landis & Koch bands). "
        f"Raw agreement {result.observed_agreement:.1%}, chance agreement "
        f"{result.expected_agreement:.1%}."
    )
