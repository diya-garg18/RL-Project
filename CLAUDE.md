# CLAUDE.md — Operating rules for this project

**Read this file first, every session, before doing anything else.**

This project follows the *AI Collaboration Field Guide* ("Don't just trust the AI. Trace it."). The documentation here is not decoration — it is the mechanism by which two students stay able to explain, in an interview, code that an AI helped write. If the humans can't explain it, the project has failed regardless of what the metrics say.

---

## Session start protocol

Do these four things before writing any code:

1. Read `HANDOVER.md` — where things actually stand.
2. Read `ROADMAP.md` — find the current phase and the next unchecked task.
3. Read `CONSTRAINTS.md` — the hard boundaries.
4. State the plan in prose and **wait for approval before implementing** (Field Guide habit #11: ask *why* before *what*).

Skim `ARCHITECTURE.md` if the task touches more than one module.

## Session end protocol

Do all of these before the session closes. This is not optional and it is not "if there's time".

1. Update `HANDOVER.md` — done / in progress / broken / next / watch out for.
2. Append to `DECISIONS.md` — any meaningful choice made, with reasoning **and the model version that made it** (habit #14).
3. Update `EXPLAIN.md` — plain-English description of anything new. **This is the most important document in the repo.**
4. Update `FLOW.md` if execution paths changed.
5. Update `ARCHITECTURE.md` if a module was added or its responsibility changed.
6. Log any training run in `docs/experiments/EXPERIMENT_LOG.md`.

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

Plus one document not in the Field Guide, added for this project:

| `EXPLAIN.md` | Everything the project does, in plain English, for a reader who knows nothing | **Every session** |

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
