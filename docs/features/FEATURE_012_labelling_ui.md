# FEATURE_012 — The preference-labelling web page (Phase 5a, human interface)

**Status:** design agreed 2026-09-05, session 13. Implementation follows in this
document as it lands.
**Phase:** 5a (`ROADMAP.md` → "Phase 5 — RLHF", sub-block 5a, box 5)
**Owner:** Diya (`PROJECT_BRIEF.md` §9, line 231)
**Started:** 2026-09-05 · **Finished:** —
**Model that helped design it:** Claude Opus 5

---

## 1. What and why

FEATURE_011 built everything needed to *ask* two humans which of two shifts was
handled better, and stopped one step short of asking them. This feature is that
step: a local web page that shows one pair at a time, takes a left / right / tie
answer, and writes it through `rlhf/store.py`.

It is the only part of Phase 5 that a person sits in front of, and it is the last
thing standing between the project and its first real preference data. Everything
downstream — the Bradley–Terry reward model in 5b, the re-trained policies in 5c,
the κ number, the reward-hacking audit in Phase 6 — is a function of the 350
judgements this page collects.

That gives it an unusual property for a piece of UI: **its output cannot be
regenerated.** Every other artefact under `results/` is a function of the code and
a seed. These 350 answers are roughly 100 minutes of two people's attention, and
if the page records them wrongly the only fix is to spend the 100 minutes again.
So the design below is biased throughout towards refusing bad data over accepting
it, in the same spirit as `store.py`.

---

## 2. Scope — and what is deliberately outside it

| in scope (this feature) | out of scope |
|---|---|
| `labelling/queue.py` — which pairs a given labeller sees, and where they left off | `scripts/generate_pairs.py` (**Pranav**, FEATURE_011 §2) |
| `labelling/render.py` — a summary rendered as readable HTML | `rlhf/reward_model.py` (Phase 5b) |
| `labelling/app.py` — the FastAPI routes | policy re-training (Phase 5c) |
| `scripts/label_ui.py` — the launcher | the React dashboard (`web/`, Phase 6) |
| config keys for the labellers, the timer cap, and the bind address | the Groq/Llama justification layer |
| | collecting the actual 350 judgements (human time, both of us) |

The page is a **consumer** of FEATURE_011 and adds nothing to it. It imports
`rlhf.store` and reads the JSON that `rlhf.pairs` writes. It does not import
`rlhf.pairs` itself, and it never recomputes a summary — if a rendering looks
wrong, the bug is upstream in `summary.py` and belongs there.

---

## 3. Roadmap link

`ROADMAP.md` → Phase 5 → 5a, box 5:

> - [ ] Labelling UI (simplest thing that works — a local FastAPI page, or even a
>   CLI with rendered text) *(**Diya's box** — `PROJECT_BRIEF.md` §9. It reads
>   `results/rlhf/pairs.json` and writes through `rlhf/store.py`. It must never
>   read `pairs_key.json`.)*

The web page is chosen over the CLI fallback. `summary.render_text` already exists
and stays as the fallback if the page ever breaks mid-session; it is not deleted.

---

## 4. The three decisions taken before implementation

All three were put to Diya on 2026-09-05 with the alternatives spelled out, and
all three are recorded in `DECISIONS.md` (D-040, D-041, D-042). Summarised here
because they shape every module below.

### 4.1 The 250 single-label pairs are split 125/125 by a fixed rule (D-040)

300 pairs, of which `rlhf.double_labelled_pairs: 50` go to **both** labellers so
that Cohen's κ is computable. That leaves 250 needing one opinion each, and the
question is whose.

**Chosen:** a deterministic assignment computed from `pairs.json` before anyone
starts. Doubles go to everyone; singles are dealt round-robin in file order.
Each labeller therefore does 50 + 125 = **175 judgements**, and the total is 350.

**Rejected:** a shared queue serving whichever pair nobody has answered yet.
Simpler and self-balancing, but the split then becomes an artefact of who had a
free evening, it cannot be reconstructed afterwards except by reading timestamps
out of the database, and two people labelling at the same time race each other.
A project whose thesis is reproducibility should not have to explain that in a
viva.

**Why round-robin in file order is unbiased.** `pairs.build_pairs` already
shuffles the draft before numbering, specifically so that a labeller working
through the file does not meet every alpha-vs-beta comparison in one block. That
shuffle is doing double duty here: because file order is already random with
respect to which policies are being compared, dealing alternate pairs gives each
labeller a balanced spread of policy pairings without the queue needing to know
what any pair contains — which it must not, since the pairs are blinded.

### 4.2 The labeller id is set at launch, never typed into the page (D-041)

`store.add_label` needs a `labeller_id`, and CONSTRAINTS #23 requires it to be
opaque — the schema deliberately has nowhere to put a name.

**Chosen:** `python scripts/label_ui.py --labeller L1`. The page never asks.

**Rejected:** a text box on the page. Two failure modes, both landing on the one
file in the project that cannot be regenerated. First, a text box is an
invitation to type a real name or an email address, and once that is in the
SQLite the only fixes are editing the irreplaceable file or losing labels.
Second — and worse, because it is silent — a stale value in the box lets one
person's judgements be recorded under the other person's id. That does not merely
lose data; it corrupts κ *specifically*, because κ's entire premise is that the
two sets of judgements came from two independent people.

The launcher validates the id against `rlhf.labellers` and refuses an unknown
one, naming the permitted values. A typo should not open a session that writes
175 rows under `L!`.

### 4.3 The timer is measured in the browser, with a cap, and `None` past it (D-042)

`store.add_label` takes `seconds_taken: float | None`, and Pranav's docstring
gives the reasoning for the `None`:

> A defaulted 0.0 would read as an instant decision and would silently skew any
> later analysis of how long people spent thinking.

**Chosen:** the page notes when the pair was displayed and when the answer was
clicked, and sends the elapsed time with the answer. If it exceeds
`rlhf.max_seconds_per_pair` (300), the stored value is **`None`**, not the
number.

**Why the cap.** Open a pair, walk away, come back twenty minutes later and
click: without a cap the row claims 1200 seconds of deliberation. That is not a
rounding error, it is a fabricated number sitting in the data and dragging any
average computed from it. `None` states "we do not know how long this took",
which is true. 1200 states something false.

This is not a new rule — it is Pranav's `None`-over-`0.0` reasoning applied at
the other end of the range. Being the same rule twice is the argument for it.

**Rejected:** server-side timing (serve → POST). Measures the same quantity
slightly worse, adds network noise, and has the identical coffee-break flaw, so
it buys nothing for the JavaScript it saves.

---

## 5. Where the code lives, and why not `web/`

| path | what |
|---|---|
| `src/soc_triage/labelling/` | the package: `queue.py`, `render.py`, `app.py` |
| `scripts/label_ui.py` | the launcher — parses args, loads config, runs uvicorn |
| `tests/test_labelling_*.py` | one test file per module |

`web/` was the first instinct and was rejected on a practical ground:
`tests/conftest.py` inserts exactly one path, `src/`, and all twelve existing test
files import via `from soc_triage.…`. A package under `web/label/` would need its
own `sys.path` insert in every test file that touched it — friction, and a
deviation from how every other test in the repo is written for no gain. `web/`
stays reserved for the Phase 6 React dashboard, which `ROADMAP.md` already names.

It is a **sibling of `rlhf/`, not a module inside it.** The dependency runs one
way — `labelling` imports `rlhf.store`, and `rlhf` imports nothing from
`labelling` — which keeps FEATURE_011's best property intact: the whole data
layer stays testable with no web framework installed at all.

---

## 6. What the page shows

One pair per screen, two panes side by side, built from the `left` and `right`
summaries already in `pairs.json`. Each pane carries exactly what
`summary.render_text` renders, in HTML rather than plain text:

- a header line — analyst-minutes spent, number of actions
- the timeline — minute, action, the alert's severity / criticality / type /
  verify cost, and whether it turned out to be a real incident or a false
  positive; plus how many alerts a bulk-close buried unread
- the outcome — incidents caught out of total (and how many before deadline),
  a card per caught incident with its delay, incidents missed, how many on
  crown-jewel assets, how many buried unread, minutes wasted on false positives,
  and MTTD

Three answers: **left better**, **right better**, **can't tell**. The third is
`tie` in the store, and it is a real answer rather than a missing one
(`PROJECT_BRIEF.md` §6.2) — 5b has to treat it as data.

Progress is shown as "pair *n* of 175 for L1", because a labeller who cannot see
how much is left is a labeller who stops.

**Deliberately absent: every reward number.** `summary.py` already strips them and
two of its tests fail if they reappear, so the page cannot show what it is never
given. The page adds no reward of its own and computes nothing from the summaries
— it reformats them. See D-039 for why this matters more than it looks.

---

## 7. The two guards, and what they are guarding

**Guard 1 — the page never reads `pairs_key.json` (D-038).** That file maps
`pair_id` to policy names, and reading it would destroy the blinding the whole
pair set is built around. Three things enforce it:

1. `load_pairs` refuses a path whose filename is `pairs_key.json`, naming the
   mistake. This catches the realistic version of the error — copying the wrong
   file to wherever the page is served from — rather than the theoretical one.
2. A test greps every module under `labelling/` for the string `pairs_key` and
   fails if it appears.
3. A test builds pairs from records named after real policies (`sarsa`, `dqn`, …),
   renders a page, and asserts none of those names appears in the HTML. This
   follows D-038's own method: search the *rendered artefact* by substring, not
   the object, because that is the only form that catches a name arriving through
   a field nobody thought about.

**Guard 2 — one opinion per person per pair, and a refresh is not an opinion.**
`store.py` enforces `UNIQUE (pair_id, labeller_id)` in the schema and raises
`DuplicateLabelError`. The page will meet this constantly and routinely: submit
an answer, hit browser-refresh, and the same POST is replayed. That must not
become a 500 error in the middle of a labelling session, and it must not become a
second row. It advances to the next pair, because the answer the labeller gave is
already safely recorded.

---

## 8. Config additions

New keys under `rlhf:` in `config/training_default.yaml`, validated in
`config/training_validation.py` alongside the FEATURE_011 rules. No magic numbers
in the modules (CONSTRAINTS #9).

| key | value | why it is a key and not a constant |
|---|---|---|
| `labellers` | `[L1, L2]` | the assignment split is computed from this list, and κ needs at least two entries. Opaque ids per CONSTRAINTS #23 |
| `max_seconds_per_pair` | `300` | the timer cap of §4.3. A tunable that changes what lands in the database, so it belongs where it can be seen and defended |
| `ui_host` | `127.0.0.1` | localhost on purpose. Binding `0.0.0.0` would put unlabelled shift data and a writable database on the local network for no benefit |
| `ui_port` | `8000` | |

Validation rules to add: at least two labellers, no duplicates among them, ids
non-empty, `max_seconds_per_pair` strictly positive, `ui_port` in range.

---

## 9. Testing approach

Test-first throughout, per `CLAUDE.md` and the way FEATURE_003 and FEATURE_011
were built. One test file per module.

`queue.py` and `render.py` are pure functions over dicts and need nothing beyond
pytest — no HTTP, no database, no `results/` directory. Fixtures are built by
running the real `pairs.build_pairs` over synthetic records, reusing the
`_record` factory already in `tests/test_rlhf_pairs.py` rather than writing a
second one. This preserves the property FEATURE_011 §13 protects: everything is
runnable on a fresh clone with no artefacts at all.

**The routes are tested through FastAPI's `TestClient`, which needed a new
dependency (D-043).** `TestClient` requires `httpx2`, which was not installed, so
CONSTRAINTS #8 applied and Diya approved it on 2026-09-05. It is pinned in
`requirements.txt` alongside step 4 of the build order, and it is the first entry
there that exists purely for tests. Installing it pulled two transitive
dependencies — `httpcore2 2.12.0` and `truststore 0.10.4` — which is worth
knowing on the other machine: `pip install httpx2` is not a single-package
install.

**Rejected:** starting `uvicorn` in a subprocess and driving it with
`urllib.request` from the standard library — zero new dependencies, and it would
have tested the actual server the humans run rather than an in-process
substitute, catching launcher bugs `TestClient` cannot see. It lost on cost and
reliability: a second or two of startup per test file on a suite that already
takes eight minutes on this machine, plus port-binding flakiness that would
produce failures unrelated to the code under test. A test that fails for reasons
of its own teaches nothing, and the launcher is thin enough (§10 step 5) to be
verified by running it once and showing the output.

The gap this leaves is stated rather than papered over: **nothing in the suite
exercises `scripts/label_ui.py` end to end.** It gets a manual verification with
real output pasted into §13, which is what CONSTRAINTS #16 asks for.

Separately: **`python_multipart` is also absent**, which FastAPI needs to parse a
plain HTML `<form>` POST. This one is designed around rather than installed — the
page already needs JavaScript for the timer of §4.3, so the answer is sent as
JSON via `fetch`, which FastAPI parses with no extra package. A dependency
avoided by a choice already being made for another reason is the cheapest kind.

---

## 10. Build order

1. `docs/features/FEATURE_012_labelling_ui.md` — this file, before the code
2. `labelling/queue.py` + tests — assignment, resume, progress. No HTTP, no DB
3. `labelling/render.py` + tests — HTML, plus both blinding guards
4. `labelling/app.py` + tests — routes, and the duplicate-submit path
5. `scripts/label_ui.py` + config keys and their validation
6. docs — `DECISIONS.md` (D-040 … D-042), `EXPLAIN.md`, `FLOW.md`,
   `ARCHITECTURE.md`, `TEST_CHECKLIST.md`, `HANDOVER.md`, and ticking the
   `ROADMAP.md` box

Steps 2 and 3 are independent of the pending decision in §9.

---

## 11. Files touched

| File | New/Modified | What changed |
|---|---|---|
| `docs/features/FEATURE_012_labelling_ui.md` | New | this design |
| `src/soc_triage/labelling/__init__.py` | New | package marker; states the one-way dependency on `rlhf` |
| `src/soc_triage/labelling/queue.py` | New | assignment, resume, progress, `load_pairs` + the key-file guard |
| `src/soc_triage/labelling/render.py` | New | the two panes, the three answers, the timer script |
| `src/soc_triage/labelling/app.py` | New | `GET /` and `POST /label` |
| `scripts/label_ui.py` | New | the launcher |
| `tests/test_labelling_queue.py` | New | 33 tests |
| `tests/test_labelling_render.py` | New | 21 tests |
| `tests/test_labelling_app.py` | New | 23 tests |
| `tests/test_rlhf_config.py` | Modified | 15 tests → 26; the four new keys and their refusals |
| `src/soc_triage/config/training.py` | Modified | `RLHFConfig` gains `labellers`, `max_seconds_per_pair`, `ui_host`, `ui_port` |
| `src/soc_triage/config/training_validation.py` | Modified | seven new rules for those keys |
| `config/training_default.yaml` | Modified | the four keys, with their reasoning in comments |
| `requirements.txt` | Modified | `httpx2==2.12.0`, test-only (D-043) |

---

## 12. What was tried that didn't work

*The most valuable section in this file. Filled in as it happens, not afterwards.*

- **`web/label/` as the location** — abandoned before any code, on the import-path
  ground in §5. Recorded because it was the obvious first instinct and the reason
  it loses is not obvious until you read `tests/conftest.py`.
- **A plain HTML `<form>` POST** — abandoned because FastAPI needs
  `python_multipart` for it and the JSON path needs nothing. See §9.

---

## 13. How it was verified

*Actual commands and actual output, per habit #8. Filled in as each step lands.*

Baseline before any of this work, on Diya's machine, 2026-09-05:

```
$ .\.venv\Scripts\python.exe -m pytest tests/ -q
286 passed in 486.73s (0:08:06)
```

Worth recording for the next session: the same 286 tests on the same commit took
**126.49 s** on Pranav's machine the day before. Nearly 4x, and it is the machine,
not the code — the spread `HANDOVER.md` warns about is still live.

**Each module, watched failing before it existed.** Every one of these three RED
runs was the missing-module error rather than a typo, which is the point of
looking:

```
$ pytest tests/test_labelling_queue.py -q
E   ModuleNotFoundError: No module named 'soc_triage.labelling'
$ pytest tests/test_labelling_render.py -q
E   ModuleNotFoundError: No module named 'soc_triage.labelling.render'
$ pytest tests/test_labelling_app.py -q
E   ModuleNotFoundError: No module named 'soc_triage.labelling.app'
```

Then green, each immediately after its module was written:

```
$ pytest tests/test_labelling_queue.py -q
33 passed in 0.20s
$ pytest tests/test_labelling_render.py -q
21 passed in 0.12s
$ pytest tests/test_labelling_app.py -q
23 passed in 1.05s
$ pytest tests/test_rlhf_config.py -q          # 11 new, watched fail first
26 passed in 0.91s
```

The queue and render suites run in a fifth of a second between them because
neither imports torch, the environment, or SQLite — the property FEATURE_011 §13
protects, extended one layer up.

**The config change was the risky one and was treated as such.** Adding a field to
a frozen config dataclass broke sixteen tests in session 9, so every construction
site of `RLHFConfig` was found before editing:

```
$ grep -rn "RLHFConfig(" --include=*.py .
src\soc_triage\config\training.py:322:    rlhf = RLHFConfig(
```

Exactly one, the loader itself. That is why no other file needed touching, and it
is a fact worth having rather than assuming in either direction.

**The launcher, which no test covers, verified by running it.** Both refusals:

```
$ python scripts/label_ui.py --labeller L9
unknown labeller 'L9'; config/training_default.yaml lists L1, L2
exit=2

$ python scripts/label_ui.py --labeller L1
cannot start: no pair file at C:\...\results\rlhf\pairs.json

Nothing has generated the pair set yet. That is scripts/generate_pairs.py
(Pranav's box, ROADMAP 5a) and it needs the nine trained policies present.
exit=1
```

And a real serve against a fixture pair set — actual uvicorn, actual HTTP, actual
row, with the two guards checked against the page the server really sent:

```
fixture: 12 pairs at ...\scratchpad\fixture\pairs.json
GET / -> 3707 bytes of HTML
policy names leaked into the page: none
the word 'reward' present: False
pair on screen: ['p0000']
POST /label -> 200 {"ok":true}
POST /label again (refresh) -> 200
page advanced to the next pair: True
rows in the database: 1
  {'id': 1, 'pair_id': 'p0000', 'labeller_id': 'L1', 'choice': 'left',
   'created_at': '2026-09-04T19:16:56Z', 'seconds_taken': 21.5}
```

The refresh is the line worth reading twice: the POST was replayed and the
database still holds **one** row, with the first answer intact. `created_at` reads
the 4th because it is UTC and the local date was the 5th — correct, not a bug.

---

## 14. Follow-ups left open

- The 350 judgements themselves. Human time, both of us, and the page is only
  the instrument.
- `scripts/generate_pairs.py` is Pranav's and unbuilt, so no real `pairs.json`
  exists yet. The page is built and tested against fixtures and meets the real
  file when his generator runs. If the two disagree about the JSON shape, the
  contract in FEATURE_011 §6 is the arbiter, not either implementation.
- A "revisit my last answer" affordance is **not** built. It would need a delete
  path through `store.py`, which currently has none by design.

---

## 15. Plain-English summary

*(For `EXPLAIN.md` Part 7 when this is done.)*

To learn what "good" triage means from people rather than from a formula, we have
to show people two versions of the same shift and ask which went better. This is
the page that asks. It shows two summaries side by side — what the agent did
minute by minute, which real incidents it caught and how late, what it missed,
how much time it burned on false alarms — and offers three buttons: left, right,
or can't tell. It never shows the policy names, so nobody can judge by
reputation, and it never shows the hand-written reward score, so nobody can
simply read our own formula back to us. It remembers where each person got to,
so labelling can be done in short sittings. It records how long each answer took,
because a three-second answer and a forty-second answer are different kinds of
judgement — and if someone leaves the page open over lunch it records "unknown"
rather than pretending they thought for twenty minutes.
