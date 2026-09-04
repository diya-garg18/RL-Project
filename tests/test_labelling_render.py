"""`labelling.render` — the summary a labeller actually reads, as HTML.

Most of this file is guards rather than happy-path rendering, because the two
things that can go wrong here are both silent and both expensive:

1. **A reward number reaching the screen** would make the labels a partial
   read-back of `config/env_default.yaml` rather than an independent judgement,
   and a Bradley-Terry model fitted to them would look like a success while
   establishing nothing (D-039). `summary.py` already strips them; these tests
   check that this module does not reintroduce one by computing it.
2. **A policy name reaching the screen** would replace judgement of outcomes with
   judgement of reputations (D-038). Following D-038's own method, the check is a
   substring search of the *rendered output* rather than an inspection of the
   object, because that is the only form that catches a name arriving through a
   field nobody thought about.

The panes are checked for content, not for layout. A test that asserted on markup
structure would fail every time the page was restyled, which would teach us
nothing and would eventually be deleted for crying wolf.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_rlhf_pairs import _record  # noqa: E402

from soc_triage.rlhf.pairs import build_pairs  # noqa: E402
from soc_triage.labelling.queue import Progress  # noqa: E402
from soc_triage.labelling.render import (  # noqa: E402
    render_done_page,
    render_pair_page,
    render_summary,
)

# Named after the real policies on purpose: these are the strings that must not
# survive into the HTML (D-038).
REAL_POLICY_NAMES = ("sarsa", "dqn", "q_learning", "monte_carlo", "reinforce")
MAX_SECONDS = 300


@pytest.fixture
def pair() -> dict:
    records = [
        _record(policy, seed)
        for policy in REAL_POLICY_NAMES
        for seed in (3000000, 3000001)
    ]
    built, _key = build_pairs(
        records,
        policies=REAL_POLICY_NAMES,
        target_pairs=10,
        double_labelled_pairs=2,
        sampling_seed=20260904,
    )
    return built[0]


@pytest.fixture
def progress() -> Progress:
    return Progress("L1", done=3, total=175)


def _page(pair: dict, progress: Progress) -> str:
    return render_pair_page(pair, progress, max_seconds=MAX_SECONDS)


# --------------------------------------------------------------------------
# Guard 1 — no reward, anywhere (D-039)
# --------------------------------------------------------------------------

def test_the_page_never_shows_the_word_reward(pair, progress):
    assert "reward" not in _page(pair, progress).lower()


def test_a_pane_never_shows_the_word_reward(pair):
    assert "reward" not in render_summary(pair["left"]).lower()


def test_a_reward_smuggled_into_a_summary_still_does_not_reach_the_page(pair, progress):
    """Defence in depth: `summary.py` strips rewards, and this does not print them.

    If a future change to the record shape let a reward field through upstream,
    the page should still not display it — so the renderer names the fields it
    shows rather than iterating over whatever it was handed.
    """
    poisoned = dict(pair)
    poisoned["left"] = dict(pair["left"], total_reward=-515.4)
    poisoned["left"]["outcome"] = dict(pair["left"]["outcome"], total_reward=-515.4)

    html = render_pair_page(poisoned, progress, max_seconds=MAX_SECONDS)
    assert "515" not in html
    assert "reward" not in html.lower()


# --------------------------------------------------------------------------
# Guard 2 — no policy names, and no route to the key file (D-038)
# --------------------------------------------------------------------------

def test_no_policy_name_appears_in_the_rendered_page(pair, progress):
    html = _page(pair, progress).lower()
    for name in REAL_POLICY_NAMES:
        assert name not in html, f"policy name {name!r} reached the labeller"


def test_no_run_id_appears_in_the_rendered_page(pair, progress):
    """`run_id` reads `sarsa-seed3000004`, so it names the policy in passing."""
    assert "seed3000000" not in _page(pair, progress)


def test_no_module_in_the_labelling_package_mentions_the_key_file():
    """A grep, deliberately. The rule is "never read pairs_key.json", and the
    cheapest durable enforcement is that the string does not appear in the code
    that serves pages — with the single exception of the guard that refuses it.
    """
    package = ROOT / "src" / "soc_triage" / "labelling"
    offenders = []
    for module in sorted(package.glob("*.py")):
        text = module.read_text(encoding="utf-8")
        if "pairs_key" in text and module.name != "queue.py":
            offenders.append(module.name)
    assert offenders == [], f"{offenders} mention pairs_key"


# --------------------------------------------------------------------------
# What the labeller needs in order to answer
# --------------------------------------------------------------------------

def test_both_panes_are_rendered(pair, progress):
    html = _page(pair, progress)
    assert html.count("labelling-pane") == 2


def test_the_three_answers_are_all_offered(pair, progress):
    """left, right and tie — `tie` is a real answer, not a missing one."""
    html = _page(pair, progress)
    for choice in ("left", "right", "tie"):
        assert f'data-choice="{choice}"' in html


def test_the_page_carries_the_pair_id_so_the_answer_can_be_posted(pair, progress):
    assert pair["pair_id"] in _page(pair, progress)


def test_the_page_carries_the_timer_cap_for_the_browser_to_apply(pair, progress):
    """The cap lives in config and is handed to the page, not hardcoded in JS."""
    assert str(MAX_SECONDS) in _page(pair, progress)


def test_the_page_shows_the_position_and_the_total(pair, progress):
    html = _page(pair, progress)
    assert "4" in html and "175" in html


# --------------------------------------------------------------------------
# The pane contents — the evidence a judgement is made from
# --------------------------------------------------------------------------

def test_a_pane_shows_every_timeline_step(pair):
    html = render_summary(pair["left"])
    assert html.count("timeline-row") == len(pair["left"]["timeline"])


def test_a_pane_names_the_action_taken(pair):
    assert "PULL_HIGHEST_SEVERITY" in render_summary(pair["left"])


def test_a_pane_distinguishes_a_real_incident_from_a_false_positive(pair):
    """The whole judgement rests on this distinction being unmissable."""
    html = render_summary(pair["left"]).lower()
    assert "false positive" in html


def test_a_pane_shows_the_outcome_counts(pair):
    outcome = pair["left"]["outcome"]
    html = render_summary(pair["left"])
    assert str(outcome["incidents_total"]) in html
    assert str(outcome["incidents_missed"]) in html


def test_a_pane_shows_minutes_wasted_on_false_positives(pair):
    assert "wasted" in render_summary(pair["left"]).lower()


def test_an_unmeasurable_mttd_reads_as_not_applicable_and_never_as_none(pair):
    """`mttd_min` is None when nothing was caught. "None" on screen is a bug."""
    summary = pair["left"]
    assert summary["outcome"]["mttd_min"] is None, "fixture assumption changed"
    html = render_summary(summary)
    assert ">None<" not in html
    assert "n/a" in html.lower()


def test_a_caught_incident_is_shown_with_how_late_it_was(pair):
    caught = dict(pair["left"])
    caught["caught"] = [{
        "minute": 42.0, "delay_min": 17.0, "severity": 3,
        "asset_criticality": 2, "alert_type": "c2_beacon", "verify_cost_min": 20,
    }]
    html = render_summary(caught)
    assert "17" in html and "c2_beacon" in html


# --------------------------------------------------------------------------
# Escaping, and the finished page
# --------------------------------------------------------------------------

def test_text_from_the_record_is_html_escaped(pair):
    """Not a live threat — the strings come from our own config — but a page that
    interpolates untrusted-shaped text without escaping is one refactor away from
    being one, and escaping costs nothing.
    """
    nasty = dict(pair["left"])
    nasty["timeline"] = [dict(
        pair["left"]["timeline"][0],
        action="<script>alert(1)</script>",
    )]
    html = render_summary(nasty)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_the_done_page_says_the_labeller_has_finished(progress):
    finished = Progress("L1", done=175, total=175)
    html = render_done_page(finished)
    assert "175" in html
    assert "L1" in html


def test_the_done_page_offers_no_answer_buttons(progress):
    finished = Progress("L1", done=175, total=175)
    html = render_done_page(finished)
    assert 'data-choice="left"' not in html
