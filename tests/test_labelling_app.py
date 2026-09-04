"""`labelling.app` — the two routes, and what they refuse to write.

The happy path here is three lines long: serve a pair, take an answer, store it.
Almost every test below is about a way the answer could be wrong, because the
database this writes to is the one artefact in the project that cannot be
regenerated (`store.py`, and the note in `.gitignore`).

The four that matter most:

- **A refresh is not a second opinion.** Submitting an answer and pressing reload
  replays the POST. `UNIQUE (pair_id, labeller_id)` stops the duplicate row, but
  the page must not show a 500 in the middle of a sitting either.
- **The labeller id comes from the launch, never from the request.** A body field
  claiming to be someone else must be ignored, or one person's judgements land
  under the other person's name and Cohen's kappa is quietly measuring one person
  against themselves (D-041).
- **The timer cap is applied server-side too.** The browser applies it first, but
  a page can be edited and this file cannot be the last line of defence (D-042).
- **A pair belonging to the other labeller is refused**, not silently stored.

These use FastAPI's `TestClient`, which is why `httpx2` is in `requirements.txt`
(D-043). The launcher itself is not covered here — see FEATURE_012 §9 for that
gap, stated rather than papered over.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from test_rlhf_pairs import _record  # noqa: E402

from soc_triage.rlhf.pairs import build_pairs  # noqa: E402
from soc_triage.rlhf.store import LabelStore  # noqa: E402
from soc_triage.labelling.app import create_app  # noqa: E402
from soc_triage.labelling.queue import assign  # noqa: E402

POLICIES = ("sarsa", "dqn", "q_learning", "monte_carlo")
SEEDS = (3000000, 3000001, 3000002)
LABELLERS = ("L1", "L2")
MAX_SECONDS = 300


@pytest.fixture
def pairs() -> list[dict]:
    records = [_record(p, s) for p in POLICIES for s in SEEDS]
    built, _key = build_pairs(
        records,
        policies=POLICIES,
        target_pairs=12,
        double_labelled_pairs=4,
        sampling_seed=20260904,
    )
    return built


@pytest.fixture
def db_path(tmp_path) -> Path:
    """Never the real `results/rlhf/labels.db`, in any test in this file."""
    return tmp_path / "labels.db"


@pytest.fixture
def client(pairs, db_path) -> TestClient:
    app = create_app(
        pairs=pairs,
        labeller_id="L1",
        labellers=LABELLERS,
        db_path=db_path,
        max_seconds=MAX_SECONDS,
    )
    return TestClient(app)


def _rows(db_path: Path) -> list[dict]:
    with LabelStore(db_path) as store:
        return store.all_labels()


def _mine(pairs: list[dict]) -> tuple[str, ...]:
    return assign(pairs, LABELLERS)["L1"]


def _answer(client: TestClient, pair_id: str, choice: str = "left", seconds=12.5):
    return client.post(
        "/label",
        json={"pair_id": pair_id, "choice": choice, "seconds": seconds},
    )


# --------------------------------------------------------------------------
# Serving pairs
# --------------------------------------------------------------------------

def test_the_first_page_serves_the_first_assigned_pair(client, pairs):
    response = client.get("/")
    assert response.status_code == 200
    assert _mine(pairs)[0] in response.text


def test_the_page_is_html(client):
    assert "text/html" in client.get("/").headers["content-type"]


def test_after_answering_the_next_pair_is_served(client, pairs):
    mine = _mine(pairs)
    _answer(client, mine[0])
    assert mine[1] in client.get("/").text


def test_the_done_page_is_served_once_every_assigned_pair_is_answered(client, pairs):
    for pair_id in _mine(pairs):
        assert _answer(client, pair_id).status_code == 200
    assert "Finished" in client.get("/").text


def test_the_served_page_carries_no_policy_name(client):
    """The blinding guard, end to end through the real route (D-038)."""
    body = client.get("/").text.lower()
    for name in POLICIES:
        assert name not in body


def test_the_served_page_shows_this_labellers_own_total(client, pairs):
    """175 in production; here, however many L1 was assigned — never all 12."""
    assert str(len(_mine(pairs))) in client.get("/").text


# --------------------------------------------------------------------------
# Storing an answer
# --------------------------------------------------------------------------

def test_an_answer_is_written_with_the_pair_choice_and_labeller(client, pairs, db_path):
    pair_id = _mine(pairs)[0]
    assert _answer(client, pair_id, choice="right").status_code == 200

    rows = _rows(db_path)
    assert len(rows) == 1
    assert rows[0]["pair_id"] == pair_id
    assert rows[0]["choice"] == "right"
    assert rows[0]["labeller_id"] == "L1"


def test_a_tie_is_stored_as_an_answer_and_not_as_a_blank(client, pairs, db_path):
    """`tie` is PROJECT_BRIEF §6.2's "can't tell" — real data, not a missing value."""
    _answer(client, _mine(pairs)[0], choice="tie")
    assert _rows(db_path)[0]["choice"] == "tie"


def test_seconds_under_the_cap_are_stored_as_given(client, pairs, db_path):
    _answer(client, _mine(pairs)[0], seconds=18.25)
    assert _rows(db_path)[0]["seconds_taken"] == pytest.approx(18.25)


def test_seconds_over_the_cap_are_stored_as_unknown(client, pairs, db_path):
    """Left open over lunch. None is true; 1200 seconds of thought is not (D-042)."""
    _answer(client, _mine(pairs)[0], seconds=MAX_SECONDS + 1)
    assert _rows(db_path)[0]["seconds_taken"] is None


def test_seconds_exactly_at_the_cap_are_kept(client, pairs, db_path):
    """The boundary is stated in a test so it cannot drift silently."""
    _answer(client, _mine(pairs)[0], seconds=MAX_SECONDS)
    assert _rows(db_path)[0]["seconds_taken"] == pytest.approx(MAX_SECONDS)


def test_a_null_from_the_browser_is_stored_as_unknown(client, pairs, db_path):
    _answer(client, _mine(pairs)[0], seconds=None)
    assert _rows(db_path)[0]["seconds_taken"] is None


def test_a_negative_elapsed_time_is_stored_as_unknown(client, pairs, db_path):
    """Impossible, therefore not measured. Storing it would corrupt any mean."""
    _answer(client, _mine(pairs)[0], seconds=-3.0)
    assert _rows(db_path)[0]["seconds_taken"] is None


def test_a_missing_seconds_field_is_accepted_as_unknown(client, pairs, db_path):
    pair_id = _mine(pairs)[0]
    response = client.post("/label", json={"pair_id": pair_id, "choice": "left"})
    assert response.status_code == 200
    assert _rows(db_path)[0]["seconds_taken"] is None


# --------------------------------------------------------------------------
# What it refuses
# --------------------------------------------------------------------------

def test_a_fourth_choice_is_refused_and_nothing_is_written(client, pairs, db_path):
    response = _answer(client, _mine(pairs)[0], choice="maybe")
    assert response.status_code == 400
    assert _rows(db_path) == []


def test_a_pair_assigned_to_the_other_labeller_is_refused(client, pairs, db_path):
    allocation = assign(pairs, LABELLERS)
    not_mine = next(p for p in allocation["L2"] if p not in allocation["L1"])

    response = _answer(client, not_mine)
    assert response.status_code == 400
    assert not_mine in response.text
    assert _rows(db_path) == []


def test_an_unknown_pair_id_is_refused(client, db_path):
    assert _answer(client, "p9999").status_code == 400
    assert _rows(db_path) == []


def test_the_labeller_id_cannot_be_overridden_by_the_request(client, pairs, db_path):
    """The whole point of binding the id at launch (D-041).

    If a request could name its own labeller, one person's answers could land
    under the other's id and kappa would compare a person with themselves.
    """
    pair_id = _mine(pairs)[0]
    client.post(
        "/label",
        json={
            "pair_id": pair_id,
            "choice": "left",
            "seconds": 5.0,
            "labeller_id": "L2",
        },
    )
    assert [row["labeller_id"] for row in _rows(db_path)] == ["L1"]


# --------------------------------------------------------------------------
# The refresh path
# --------------------------------------------------------------------------

def test_answering_the_same_pair_twice_does_not_add_a_second_row(client, pairs, db_path):
    pair_id = _mine(pairs)[0]
    _answer(client, pair_id, choice="left")
    _answer(client, pair_id, choice="right")
    assert len(_rows(db_path)) == 1


def test_the_first_answer_wins_a_replayed_submission(client, pairs, db_path):
    """A reload replays the POST. It must not overwrite the judgement either."""
    pair_id = _mine(pairs)[0]
    _answer(client, pair_id, choice="left")
    _answer(client, pair_id, choice="right")
    assert _rows(db_path)[0]["choice"] == "left"


def test_a_replayed_submission_is_not_an_error(client, pairs):
    """Mid-sitting, a 500 costs momentum on work that is already safely stored."""
    pair_id = _mine(pairs)[0]
    _answer(client, pair_id)
    assert _answer(client, pair_id).status_code == 200


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

def test_the_database_is_created_if_it_does_not_exist(pairs, tmp_path):
    fresh = tmp_path / "nested" / "labels.db"
    app = create_app(
        pairs=pairs,
        labeller_id="L1",
        labellers=LABELLERS,
        db_path=fresh,
        max_seconds=MAX_SECONDS,
    )
    client = TestClient(app)
    _answer(client, _mine(pairs)[0])
    assert fresh.is_file()


def test_an_unknown_labeller_id_is_refused_at_creation(pairs, db_path):
    """A typo at launch must not open a session that writes 175 rows under it."""
    with pytest.raises(Exception, match="L1"):
        create_app(
            pairs=pairs,
            labeller_id="L!",
            labellers=LABELLERS,
            db_path=db_path,
            max_seconds=MAX_SECONDS,
        )
