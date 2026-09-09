# CLAUDE.md — Operating rules for this project

**Read this file first, every session, before doing anything else.**

This project follows the *AI Collaboration Field Guide* ("Don't just trust the AI. Trace it."). The documentation here is not decoration — it is the mechanism by which two students stay able to explain, in an interview, code that an AI helped write. If the humans can't explain it, the project has failed regardless of what the metrics say.

---

## Session start protocol

Do these eight things before writing any code, **in this order**:

1. **`git fetch`, then `git pull --ff-only` — before reading anything else.** This is step 1 because on a project with two people alternating machines, *every other step in this list is a lie on a stale checkout*. A clean tree and a `## master...origin/master` line with no `ahead` marker look identical whether you are current or nine commits behind — nothing local can tell you, because nothing local has asked the remote. If the tree is dirty or the pull will not fast-forward, **stop and say so**; do not create a merge nobody asked for. *(Added 2026-09-05, session 14. Session 14's preflight read `HANDOVER.md` and ran `commit_balance.py` on a checkout 9 commits behind. Both agreed and both were wrong — "IMBALANCED, Pranav 8 ahead, hand over to Diya" was really BALANCED 55/56 — and the plan about to be proposed routed work around Diya's labelling UI, which she had built, tested and pushed that morning.)*
2. Read `HANDOVER.md` — where things actually stand. **Read it after the pull, not before**; the pull frequently rewrites it.
3. Read `ROADMAP.md` — find the current phase and the next unchecked task.
4. Read `CONSTRAINTS.md` — the hard boundaries.
5. **Confirm which machine and which teammate this is** — `git config user.name`. The two students alternate machines, and the answer changes who should be committing.
6. **Run `python scripts/commit_balance.py` and report the result** (CONSTRAINTS #26). If the person at this keyboard is the one *ahead*, say so immediately and recommend handing over before starting new work. Its answer depends on history, so a pre-pull run of it is not evidence — it must run **after** step 1.
7. State the plan in prose and **wait for approval before implementing** (Field Guide habit #11: ask *why* before *what*).
8. **Know now how this session must end.** Two things are owed at the end of *every*
   session, and both are easy to run out of room for if you discover them late: the
   `docs/onboarding/` update, and a plain-English explanation of the session written in
   the chat. Read "The teaching-back rule" below **at the start**, not at the end, and
   budget for it — the work stops early if it has to.

Skim `ARCHITECTURE.md` if the task touches more than one module.

## Session end protocol

Do all of these before the session closes. This is not optional and it is not "if there's time".

1. Update `HANDOVER.md` — done / in progress / broken / next / watch out for.
2. Append to `DECISIONS.md` — any meaningful choice made, with reasoning **and the model version that made it** (habit #14).
3. Update `EXPLAIN.md` — plain-English description of anything new. **This is the most important document in the repo.**
4. Update `FLOW.md` if execution paths changed.
5. Update `ARCHITECTURE.md` if a module was added or its responsibility changed.
6. Log any training run in `docs/experiments/EXPERIMENT_LOG.md`.
7. **Update `docs/onboarding/` for anything that changed.** See "The teaching-back rule" below — this is a hard requirement, not a nice-to-have.
8. **Commit in meaningful pieces, not one lump** (CONSTRAINTS #24). One logical change per commit, message explaining what and why.
9. **Run `python scripts/commit_balance.py` again and report it.** State plainly whether the work should move to the other teammate (CONSTRAINTS #26).
10. **Leave the repo transfer-ready** — see the machine-transfer checklist below. The next session may be on the other person's laptop.
11. **Teach the session back in the chat, in plain English.** See below. Nothing is finished until this is done.

---

## The teaching-back rule — mandatory at the end of every session

> Added 2026-09-09 at Pranav's request. The reason is the project's founding constraint:
> **if the humans cannot explain the code, the project has failed regardless of what the
> metrics say.** Documentation that is written but never read does not discharge that,
> and re-deriving a session's work weeks later from a diff is the expensive way to learn
> it. So the explanation happens *while the context is still live*, in the conversation,
> where it can be questioned.

**Two obligations, both hard, both at the end of every session.**

### 1. The onboarding docs get updated — every session, without being asked

`docs/onboarding/` is a plain-English guide to the whole project, written for someone who
knows nothing about it. It is read alongside the code, so it goes stale the moment the
code moves.

> **Before writing or editing anything in that directory, read
> `docs/onboarding/_SPEC_AND_STATUS.md`.** It is the contract for the set: what is
> written and what is not, the scoping decisions already taken, the house style the
> existing documents establish, and a per-section plan for each remaining document. The
> seven documents must read as one voice; a document written in a different register is
> worse than no document, because the reader stops trusting the set.

| Document | Update it when |
|---|---|
| `00_labelling_handbook.md` | anything about collecting preferences changes |
| `01_the_project_in_plain_english.md` | the MDP, a phase status, or a headline result changes |
| `02_the_world.md` | env, generator, state, alerts or config changes |
| `03_the_agents.md` | any file under `agents/` changes |
| `04_running_and_measuring.md` | `runner.py`, `evaluation/`, or any script changes |
| `05_rlhf_and_labelling.md` | anything under `rlhf/` or `labelling/` changes |
| `06_decisions_experiments_results.md` | a decision, experiment or negative result is added |

**A code change without its doc update is an unfinished change.** If a session touched a
file, the document covering that file is checked in the same session — and if it needed
no edit, say so explicitly rather than leaving it ambiguous.

The Markdown is the source of truth; the `.docx` exports are regenerated from it.

### 2. Claude explains the session in the chat, in plain terms

Before handing back, write a plain-English account of the session **in the conversation
itself**, not only in a file. It must cover:

| Section | What it answers |
|---|---|
| **What I did** | Every change, in plain language. No jargon that has not been unpacked. |
| **Why** | The reasoning behind each choice, including options rejected. |
| **How it works** | Enough that a reader could re-derive the change themselves. |
| **What I verified** | The exact commands run and the exact output. Claims without evidence do not count. |
| **What I did NOT do** | Anything skipped, deferred, or left broken — stated plainly. |
| **What to check yourself** | The parts a human should not take on trust. |

**Write it for someone learning, not someone auditing.** The goal is that Pranav and Diya
finish the session understanding the work well enough to defend it in a viva — not that
they have a record they could reconstruct it from later.

Pranav's own words when he asked for this, kept here because the phrasing is the spec:

> *"in the session chat only it will explain everything it did that session in clear,
> plain simple terms so that the users can understand and learn alongside it rather than
> wasting time later."*

**"Learn alongside it" is the bar.** Not a summary of what changed — a teach-back that
leaves the reader able to do it themselves. In practice that means:

- **Explain the idea before the diff.** Why this approach, what was rejected, and what
  the reader would have had to know to arrive at it.
- **Unpack every term.** If a sentence contains "off-policy", "bootstrapping", "Huber
  delta" or "kappa", the sentence before it defines the term. No exceptions for terms
  used earlier in the project — the person reading may not have been at that session.
- **Show the actual output.** "Tests pass" is not evidence; `391 passed in 173.74s` is.
  Quote the number you saw, not the number you expected.
- **Say what you did not do, plainly and first-person.** Skipped, deferred, broken,
  half-done, or not started. A gap discovered later costs more than one admitted now.
- **Point at what to double-check.** Name the specific claims a human should verify
  rather than take on trust, and say why those and not others.
- **Never bury bad news.** A failing test, a worse number, a wrong earlier claim — that
  goes near the top, in plain language, not softened.

This is not optional and it is not "if there is room in the context". If a session is
running long, stop the work early and leave room for the explanation. **An unexplained
change is worth less to this project than no change at all.**

---

## Machine transfer — the handover checklist

The two students alternate machines continuously. **Assume every session is the last one on this machine.** Before the session closes, all of these must be true (CONSTRAINTS #25):

| Check | Command |
|---|---|
| Tests pass | `.\.venv\Scripts\python.exe -m pytest tests/ -q` |
| Nothing uncommitted | `git status --porcelain` returns nothing |
| Nothing unpushed | `git status -sb` shows no `ahead` |
| No stray zero-byte files staged | see `docs/bugs/BUG_001` |
| `HANDOVER.md` describes the true current state | read it back and check it |
| Commit balance reported | `python scripts/commit_balance.py` |

**What the other machine needs to reproduce the work:** `results/` is gitignored and every artefact in it is regenerable — the commands are listed in `HANDOVER.md` → "Reproduce on this device". Nothing in `results/` should ever be needed to continue; if it is, that is a bug in the scripts, not a reason to commit binaries.

**Anything that only exists on one machine must be written down, not remembered.** A path, a tool version, an install workaround, a `pip` failure and its fix — those go in `HANDOVER.md` under "Watch out for", because the other person cannot see this terminal.

---

## The nine Field Guide documents, and who owns them

| File | Purpose | Update cadence |
|---|---|---|
| `HANDOVER.md` | Where things stand right now | Every session |
| `DECISIONS.md` | Why, not just what. Append-only. | Every decision |
| *Explicit comments* | Inline intent in the code itself | Continuously |
| `FLOW.md` | How execution travels between files | When paths change |
| `docs/features/`, `docs/bugs/` | One file per feature/bug, start to finish | Per feature/bug |
| `ARCHITECTURE.md` | The system map | When structure changes |
| `CONSTRAINTS.md` | What must never happen | Rarely — ask before editing |
| `TEST_CHECKLIST.md` | What "done" means, with real commands | When new checks are added |
| `ROLLBACK.md` | How to undo | When risky work starts |

Plus two not in the Field Guide, added for this project:

| `EXPLAIN.md` | Everything the project does, in plain English, for a reader who knows nothing | **Every session** |
| `docs/onboarding/` | The full guided tour — seven documents that walk a newcomer from "what is a SOC" to every file in the repo, read alongside the code. **`_SPEC_AND_STATUS.md` is the contract: read it before writing or editing anything in that directory.** Markdown is the source of truth; `.docx` exports are generated from it. | **Every session** — see the teaching-back rule |

---

## Working rules

### Scope
- **One logical change per request** (habit #12). If asked for something large, decompose it, show the decomposition, and do the first piece.
- Do not build ahead. If the roadmap says Phase 2, do not write Phase 4 code because it seems convenient.
- Do not refactor code you were not asked to touch.

### Code
- **Comment intent, not syntax** (habit #3). `# increment i` is noise. `# Ties broken by alert id so runs are reproducible across seeds` is the point.
- Every RL algorithm gets a docstring naming the update rule it implements and the textbook section it comes from (Sutton & Barto 2nd ed.).
- Keep files under 500 lines. Split when they grow.
- No magic numbers in code. Every tunable number lives in `config/*.yaml`.
- Type hints on all public functions.

### The teaching constraint — this project's defining rule
Both students must be able to **write the core functions from memory in an interview**. Therefore:

- Prefer the clear implementation over the clever one, every single time.
- Prefer explicit loops over dense vectorised one-liners in the *learning* code, even at a speed cost. Vectorise the simulator if needed; keep the algorithms readable.
- No library shortcuts for anything on the syllabus. Q-learning, SARSA, Monte Carlo, DQN, REINFORCE, actor–critic and the reward model are all **written by hand**.
- If an implementation cannot be explained to a fourth-year student in five minutes, it is the wrong implementation.

### Verification
- **Never claim something works without running it** and showing the output. AI claiming success and code actually working are two different facts.
- Run `TEST_CHECKLIST.md` before declaring any phase complete.
- Report failures plainly, with the output. A failing test reported honestly is worth more than a passing test that was never run.

### Honesty (this project's whole thesis)
- If a result looks too good, treat that as a bug report and investigate before celebrating.
- Negative results get documented, not deleted. Pranav has already shipped two documented negative results (ChurnLens leakage retraction, EHCV fusion ablation). That is the standard here.
- Never tune on the evaluation seeds. Training seeds and evaluation seeds are disjoint and that separation is enforced in code.
- If a metric improves, check *why* before writing it down.

---

## Git

- `git init` on first session; commit at the end of every session.
- Commit messages: `phase<N>: <what changed>` — e.g. `phase2: add tabular Q-learning agent + epsilon decay`.
- **Do not add a `Co-Authored-By` trailer** (per the user's global instruction).
- Never commit: `results/`, `*.pt`, `*.npy`, `.env`, the label database.

---

## Ask the humans before

- Adding any new dependency to `requirements.txt`
- Changing anything in `CONSTRAINTS.md`
- Changing the MDP definition — state buckets, action set, or reward structure (these are in `PROJECT_BRIEF.md` §3 and the students must be able to defend them from memory)
- Deleting or rewriting an existing experiment result
- Starting a training run expected to take more than ~10 minutes
