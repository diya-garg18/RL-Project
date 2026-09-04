"""The two routes: show the next pair, take the answer (FEATURE_012 §7).

`GET /` renders whatever pair this labeller has not yet answered. `POST /label`
stores one judgement. There is nothing else, deliberately — no login, no edit, no
delete, no export. Export already exists in `store.export_csv`, and there is no
delete path anywhere in `store.py` by design.

Three things about the shape of this file are decisions rather than habits.

**The labeller id is a closure variable, not a request field.** It is fixed when
the app is created and the request cannot influence it (D-041). A body that
includes `labeller_id` is ignored: the `Answer` model has no such field, so the
value is discarded before any handler sees it. If a request could name its own
labeller, one person's judgements could be recorded under the other's id, and
Cohen's kappa would silently be comparing a person with themselves.

**The queue is rebuilt from the database on every request**, rather than held in
memory and updated. It makes each request a little more expensive and it makes
resume correct for free: close the tab, come back tomorrow, and the page is
already at the right pair, because "where I got to" is a fact about the database
and never a fact about the process.

**The store is opened per request and closed again.** A `sqlite3` connection
belongs to the thread that made it, and FastAPI runs synchronous handlers in a
threadpool, so a shared connection would be a threading bug waiting for a second
click. Opening a local SQLite file is cheap; a corrupted set of irreplaceable
labels is not.
"""

from pathlib import Path
from typing import Sequence

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from ..rlhf.store import DuplicateLabelError, InvalidChoiceError, LabelStore
from .queue import LabelQueue, UnassignedPairError
from .render import render_done_page, render_pair_page


class Answer(BaseModel):
    """One judgement, as the page sends it.

    No `labeller_id` field, and that absence is the enforcement of D-041 rather
    than an oversight: unknown keys in the body are dropped, so a request cannot
    claim to be somebody else.

    `seconds` is optional because the page sends `null` when its own cap has
    already tripped. Both routes to "unknown" end in the same stored `None`.
    """

    pair_id: str
    choice: str
    seconds: float | None = None


def _clean_seconds(value: float | None, max_seconds: int) -> float | None:
    """Apply the D-042 cap server-side, and reject the impossible.

    The browser applies the same cap first. This is not redundancy for its own
    sake: a page can be edited or replayed by hand, and the database it writes to
    is the one artefact in the project that cannot be regenerated. A negative
    elapsed time is impossible rather than merely large, so it is treated as not
    measured — storing it would corrupt any mean computed later just as surely as
    a fabricated 1200 would.

    `NaN` is checked explicitly rather than left to the range comparison below.
    `nan < 0` and `nan > max_seconds` are both False in IEEE 754, so a plain range
    check lets it straight through — and Python's `json` module accepts the
    non-spec `NaN` token that strict JSON forbids, so a client can send one. It
    happens to read back as `None` today only because SQLite's C driver silently
    converts NaN to NULL on storage; that is an accident of one storage engine,
    not something this function should depend on to keep its own contract.
    """
    if value is None:
        return None
    if value != value:  # NaN is the only float that is not equal to itself.
        return None
    if value < 0 or value > max_seconds:
        return None
    return float(value)


def create_app(
    *,
    pairs: Sequence[dict],
    labeller_id: str,
    labellers: Sequence[str],
    db_path: str | Path,
    max_seconds: int,
) -> FastAPI:
    """Build the labelling app for one labeller and one pair set.

    `pairs` arrives already loaded rather than as a path, so the routes can be
    tested without a file on disk and the launcher keeps sole responsibility for
    reading — and for refusing — `pairs.json` (`queue.load_pairs`).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Built once here purely to fail loudly at startup on an unknown labeller id
    # or a bad labeller list. A typo should stop the launch, not surface as an
    # error page after somebody has settled in to label.
    LabelQueue(pairs, labeller_id, labellers, answered=())

    app = FastAPI(title="Preference labelling", docs_url=None, redoc_url=None)

    def current_queue() -> LabelQueue:
        """This labeller's queue as the database currently sees it."""
        with LabelStore(db_path) as store:
            answered = [row["pair_id"] for row in store.labels_by(labeller_id)]
        return LabelQueue(pairs, labeller_id, labellers, answered=answered)

    @app.get("/", response_class=HTMLResponse)
    def next_pair() -> HTMLResponse:
        queue = current_queue()
        pair = queue.next_pair()
        if pair is None:
            return HTMLResponse(render_done_page(queue.progress()))
        return HTMLResponse(
            render_pair_page(pair, queue.progress(), max_seconds=max_seconds)
        )

    @app.post("/label")
    def record_answer(answer: Answer):
        queue = current_queue()

        # Checked before anything is written: an id belonging to the other
        # labeller, or to no pair at all, is a bug or a hand-edited request, and
        # the only safe answer to both is to decline.
        try:
            queue.pair(answer.pair_id)
        except UnassignedPairError as exc:
            return PlainTextResponse(str(exc), status_code=400)

        seconds = _clean_seconds(answer.seconds, max_seconds)

        try:
            with LabelStore(db_path) as store:
                store.add_label(
                    answer.pair_id,
                    labeller_id,
                    answer.choice,
                    seconds_taken=seconds,
                )
        except InvalidChoiceError as exc:
            return PlainTextResponse(str(exc), status_code=400)
        except DuplicateLabelError:
            # A refresh replays the POST. The judgement is already stored and the
            # first answer stands, so this is success from the labeller's point
            # of view: the page moves on. Returning 500 here would interrupt a
            # sitting over work that was never at risk.
            pass

        return JSONResponse({"ok": True})

    return app
