"""One pair, rendered as a page a person can judge from (FEATURE_012 §6).

This module reformats. It does not compute, and that is the design rather than a
simplification: every number on screen comes from `summary.summarise_episode`,
which built it from the environment's own outcome record. If a figure looks
wrong, the bug is upstream in `summary.py` and belongs there.

**It prints named fields, never "whatever it was handed."** That is what makes
D-039 hold under change: `summary.py` strips every reward field today, and if
some future change to the record shape let one through, a renderer that iterated
over keys would put it on screen. This one would not, because `total_reward` is
not in the list of things it knows how to draw.

Plain string building with `html.escape`, no template engine. The page is two
panes, a progress line and three buttons; a template dependency would add a file
format, a search path and a loader to a job that fits in one screen of Python
(CONSTRAINTS #13).
"""

import html
from typing import Any, Sequence

from .queue import Progress

# Kept out of the markup below so the page body stays readable. No colours
# beyond a light/dark neutral: the two panes must look identical, or the styling
# itself becomes a bias in the comparison.
_STYLE = """
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; padding: 1.5rem;
         max-width: 1200px; }
  h1 { font-size: 1.1rem; font-weight: 600; margin: 0 0 1rem; }
  .progress { color: #666; font-variant-numeric: tabular-nums; }
  .panes { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .pane { border: 1px solid #8884; border-radius: 8px; padding: 1rem;
          min-width: 0; }
  .pane h2 { font-size: 0.95rem; margin: 0 0 0.75rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem;
          font-variant-numeric: tabular-nums; }
  td { padding: 0.15rem 0.4rem 0.15rem 0; vertical-align: top;
       border-bottom: 1px solid #8882; }
  .real { font-weight: 600; }
  .miss { color: #b00; }
  ul { padding-left: 1.1rem; margin: 0.5rem 0; }
  .answers { display: flex; gap: 0.75rem; justify-content: center;
             margin: 1.5rem 0; }
  button { font: inherit; padding: 0.6rem 1.4rem; border-radius: 6px;
           border: 1px solid #8886; cursor: pointer; }
  #problem { color: #b00; text-align: center; min-height: 1.5rem; }
"""

# The browser half of D-042. `performance.now()` is monotonic, so it does not
# jump if the machine clock is corrected mid-sitting. Over the cap the page
# sends null rather than a number, because "we do not know how long this took"
# is true and "1200 seconds of deliberation" is not. The server applies the same
# cap again -- a page can always be edited, and this file cannot be the only
# thing standing between a stray value and the one artefact we cannot regenerate.
_SCRIPT = """
  const PAIR_ID = "__PAIR_ID__";
  const CAP_SECONDS = __CAP__;
  const shownAt = performance.now();

  document.querySelectorAll("[data-choice]").forEach(function (button) {
    button.addEventListener("click", function () {
      const elapsed = (performance.now() - shownAt) / 1000;
      button.disabled = true;
      fetch("/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pair_id: PAIR_ID,
          choice: button.dataset.choice,
          seconds: elapsed > CAP_SECONDS ? null : elapsed
        })
      }).then(function (response) {
        if (response.ok) { window.location = "/"; return; }
        button.disabled = false;
        response.text().then(function (text) {
          document.getElementById("problem").textContent = text;
        });
      });
    });
  });
"""


def _esc(value: Any) -> str:
    """Escape anything on its way into the page.

    The strings here come from our own config, so this is not defending against
    an attacker. It defends against the day someone points this renderer at text
    from somewhere else and nobody remembers to add escaping then.
    """
    return html.escape(str(value), quote=True)


def _minutes(value: float) -> str:
    return f"{float(value):.0f}"


def _alert_text(alert: dict | None) -> str:
    """The four fields a labeller needs to judge whether an alert was worth time."""
    if alert is None:
        return ""
    return (
        f"sev {_esc(alert['severity'])} &middot; "
        f"crit {_esc(alert['asset_criticality'])} &middot; "
        f"{_esc(alert['alert_type'])} &middot; "
        f"{_esc(alert['verify_cost_min'])} min"
    )


def _timeline_rows(timeline: Sequence[dict]) -> str:
    """One row per action, in the order the analyst took them.

    An explicit loop building a list of strings, rather than a nested
    comprehension: this is the part of the page a viva question is most likely to
    land on (CONSTRAINTS #14).
    """
    rows: list[str] = []
    for step in timeline:
        alert = step["alert"]
        if alert is None:
            verdict = ""
        elif step["was_true_incident"]:
            verdict = '<span class="real">REAL INCIDENT</span>'
        else:
            verdict = "false positive"

        detail = _alert_text(alert)
        if step["n_bulk_closed"]:
            closed = f"closed {_esc(step['n_bulk_closed'])} alerts unread"
            detail = f"{detail} &middot; {closed}" if detail else closed

        rows.append(
            '<tr data-role="timeline-row">'
            f"<td>{_minutes(step['minute'])}</td>"
            f"<td>{_esc(step['action'])}</td>"
            f"<td>{detail}</td>"
            f"<td>{verdict}</td>"
            "</tr>"
        )
    return "".join(rows)


def _caught_cards(caught: Sequence[dict]) -> str:
    """The incidents this shift actually found, and how late each one was."""
    if not caught:
        return "<li>nothing caught</li>"

    cards: list[str] = []
    for card in caught:
        cards.append(
            "<li>"
            f"caught at {_minutes(card['minute'])} min, "
            f"{_minutes(card['delay_min'])} min after it arrived &mdash; "
            f"{_alert_text(card)}"
            "</li>"
        )
    return "".join(cards)


def _outcome_block(outcome: dict) -> str:
    """The scoreboard. Every figure is the environment's, none is computed here."""
    mttd = outcome["mttd_min"]
    mttd_text = "n/a &mdash; nothing caught" if mttd is None else f"{float(mttd):.1f} min"

    missed = _esc(outcome["incidents_missed"])
    critical = _esc(outcome["critical_missed"])

    return (
        "<ul>"
        f"<li>incidents caught: {_esc(outcome['incidents_caught'])} of "
        f"{_esc(outcome['incidents_total'])} "
        f"({_esc(outcome['incidents_caught_in_time'])} before their deadline)</li>"
        f'<li class="miss">incidents missed: {missed} '
        f"({critical} on crown-jewel assets)</li>"
        f"<li>buried unread: {_esc(outcome['incidents_buried_by_bulk_close'])}</li>"
        f"<li>analyst minutes wasted on false positives: "
        f"{_esc(outcome['wasted_minutes'])}</li>"
        f"<li>mean time to detect: {mttd_text}</li>"
        "</ul>"
    )


def render_summary(summary: dict) -> str:
    """One shift, as one pane. The unit a judgement is made from."""
    return (
        '<section class="pane" data-role="labelling-pane">'
        f"<h2>{_minutes(summary['shift_minutes'])} analyst-minutes, "
        f"{_esc(summary['n_steps'])} actions</h2>"
        f"<table>{_timeline_rows(summary['timeline'])}</table>"
        f"<h2>Outcome</h2>"
        f"{_outcome_block(summary['outcome'])}"
        f"<ul>{_caught_cards(summary['caught'])}</ul>"
        "</section>"
    )


def _document(title: str, body: str, script: str = "") -> str:
    script_tag = f"<script>{script}</script>" if script else ""
    return (
        "<!doctype html>"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title>"
        f"<style>{_STYLE}</style></head>"
        f"<body>{body}{script_tag}</body></html>"
    )


def render_pair_page(pair: dict, progress: Progress, *, max_seconds: int) -> str:
    """The labelling screen: two shifts, three answers, and where you are.

    `max_seconds` is handed in from `rlhf.max_seconds_per_pair` rather than being
    a constant here, so the cap is visible in config where it can be defended
    (CONSTRAINTS #9).
    """
    header = (
        f"<h1>Which shift was handled better? "
        f'<span class="progress">pair {progress.position} of {progress.total} '
        f"&middot; {_esc(progress.labeller_id)}</span></h1>"
    )

    panes = (
        '<div class="panes">'
        f"{render_summary(pair['left'])}"
        f"{render_summary(pair['right'])}"
        "</div>"
    )

    answers = (
        '<div class="answers">'
        '<button data-choice="left">Left was better</button>'
        '<button data-choice="tie">Can&rsquo;t tell</button>'
        '<button data-choice="right">Right was better</button>'
        "</div>"
        '<p id="problem"></p>'
    )

    script = (
        _SCRIPT
        .replace("__PAIR_ID__", _esc(pair["pair_id"]))
        .replace("__CAP__", str(int(max_seconds)))
    )

    return _document("Labelling", header + panes + answers, script)


def render_done_page(progress: Progress) -> str:
    """Shown when this labeller has answered every pair assigned to them.

    Deliberately a dead end with no buttons: there is no "change my last answer"
    path, because `store.py` has no delete by design.
    """
    body = (
        "<h1>Finished</h1>"
        f"<p>{_esc(progress.labeller_id)} has answered all "
        f"{progress.total} assigned pairs. Nothing further to do &mdash; "
        "you can close this tab.</p>"
    )
    return _document("Finished", body)
