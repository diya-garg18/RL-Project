# Onboarding docs — specification and status

**READ THIS BEFORE WRITING OR EDITING ANYTHING IN `docs/onboarding/`.**

This file is the contract for the onboarding doc set. It exists because the set is being
built across several sessions on two different machines, and documents 4–6 must match
documents 0–3 closely enough that a reader cannot tell where one session stopped and the
next began.

> **If you are Claude and you have just been asked to continue this work: read this whole
> file first, then read `01_the_project_in_plain_english.md` and `02_the_world.md` in
> full before writing a word.** The house style in §4 is not a suggestion — it was
> derived from the four documents that already exist, and a document written in a
> different voice is worse than no document, because the reader stops trusting the set.

---

## 1. Status — what exists, what does not

| # | Document | Lines | Status | Owner |
|---|---|---:|---|---|
| 0 | `00_labelling_handbook.md` | 419 | ✅ **Written** | Pranav, session 15 |
| 1 | `01_the_project_in_plain_english.md` | 1000 | ✅ **Written** | Pranav, session 15 |
| 2 | `02_the_world.md` | 1254 | ✅ **Written** | Pranav, session 15 |
| 3 | `03_the_agents.md` | 1554 | ✅ **Written** | Pranav, session 15 |
| 4 | `04_running_and_measuring.md` | — | ❌ **NOT STARTED** | Diya |
| 5 | `05_rlhf_and_labelling.md` | — | ❌ **NOT STARTED** | Diya |
| 6 | `06_decisions_experiments_results.md` | — | ❌ **NOT STARTED** | Diya |
| — | `.docx` exports of all seven | — | ❌ **NOT STARTED** | Diya |
| — | Intent-comment pass, 86 `.py` files | — | ❌ **NOT STARTED** | Diya |

**Total written so far: 4,227 lines across four documents.** Documents 4–6 should land in
roughly the same range — 1,000–1,500 lines each. Do not pad to hit a number, and do not
cut a file's explanation short to stay under one.

---

## 2. What was actually asked for — the original request

Preserved verbatim so the intent is not lost in paraphrase. Pranav, 2026-09-09:

> "I want you to make a few word documents for me. These documents will make sure that
> someone entirely new to this project can just go through these docs and understand the
> entire project. The docs will contain Everything this project is, everything it does,
> how it does and what decisions and why. It can refer to the sections of docs such as :
> PROJECTBRIEF.md, README.md, FLOW.md, EXPLAIN.md, DECISIONS.md, ARCHITECTURE.md etc. (Or
> you can just directly write those things in the docs which is easier for the user.) The
> docs will include the explanation of every file line by line and in an order that is
> easy for someone to understand. Make the docs easy to understand, add visualizations or
> tabular structures, Dont make it too technical rather keep it simple. You can create as
> many docs as you think is neccessary and each can have as many pages as you want. But
> after going through the docs along with reading and referencing the code files side by
> side one should be able to completely understand this project. Also alongside creating
> these docs i want you to go through every file and add comments wherever possible to
> make the code easier to understand."

**The motivating context, in his words:** *"for the human labelling part we need to
understand everything fully."*

That is the acceptance test. Not "is this document thorough" but **"could Pranav or Diya
sit down with this document and the code side by side and be able to defend the project
in a viva?"**

### The three scoping decisions already taken

Pranav chose these from a menu. They are settled — do not re-litigate them.

| Question | Chosen | What it means |
|---|---|---|
| **Depth** | Tiered walkthrough | All 86 files covered. Important lines explained individually; routine boilerplate compressed into tables. Not literally every line of every file. |
| **Format** | Markdown source + Word export | Write `.md` (versioned, diffable), then export each to `.docx`. **The Markdown is the source of truth.** |
| **Comments** | All 86 files, intent comments | Module docstring + intent comments on non-obvious lines, per CLAUDE.md's "comment intent, not syntax". |

---

## 3. Reading order and what each document owns

The set is designed to be read front to back. Each document assumes the previous ones.

```
  00_labelling_handbook.md      standalone — read before a labelling session
                                (deliberately duplicates a little of 01 and 05)

  01_the_project_in_plain_english.md   ← the prerequisite for everything below
        |
        v
  02_the_world.md               alerts, generator, state, env, config
        |
        v
  03_the_agents.md              base, baselines, dp, tabular, mc, sarsa,
        |                       q_learning, replay, dqn, reinforce, actor_critic
        v
  04_running_and_measuring.md   runner, metrics, tiny_mdp, mrp_example, 25 scripts, tests
        |
        v
  05_rlhf_and_labelling.md      rlhf/*, labelling/*, the pair pipeline, kappa
        |
        v
  06_decisions_experiments_results.md   D-001..D-046, EXPERIMENT_LOG, BUG_001-005
```

**Ownership rule: each file in the repo is explained in exactly one document.** If you
need to mention a file another document owns, link to it — do not re-explain it. Documents
0–3 already follow this; keep it.

---

## 4. House style — derived from documents 0–3, follow it exactly

This section is the most important part of this file. Documents 4–6 must be
indistinguishable in voice from 0–3.

### 4.1 Structure of every document

```markdown
# Document N — <Short Title>

### <One-line subtitle saying what it covers>

> **Read Document N-1 first.** <one sentence on what it assumes>
>
> **The promise of this document:** <what the reader will be able to do at the end>

---

## Contents

| § | What it covers | File(s) |
|---|---|---|
...

---

## 1. <first section — always an orienting map, never a file>

...

---

## Where to go next

- **Document N+1 — <title>.** <one line>
...

---

*Source of truth: this Markdown file. The `.docx` export is generated from it. If they
disagree, the Markdown is right.*
```

The final italic line appears verbatim at the bottom of every document. So does the
`## Contents` table near the top, and the `## Where to go next` section at the bottom.

### 4.2 The eight rules of the voice

**1. Explain the WHY before the WHAT.** Every section leads with the problem the code
solves, then shows the code. Never open with a code block.

**2. Quote the code's own comments rather than paraphrasing them.** This project's source
comments are unusually good and often already contain the best explanation. Quote them in
blockquotes:

```markdown
The docstring explains why:

> *"One shared helper so every discretisation uses the identical convention (HANDOVER
> warned: an off-by-one here silently corrupts all 576 states)."*
```

This also means a reader can verify every claim against the file. That is deliberate.

**3. Work every abstract formula through with real numbers.** Not "the truth model
multiplies lifts" but:

```markdown
**Worked example.** A `data_exfil_volume` alert, severity 2, on a crown jewel:

    0.0135  ×  3.0  ×  2.4  ×  1.4  =  0.136   →  a 13.6% chance of being real

That's a **158× difference** — and the agent can see every input to it.
```

**4. Draw ASCII diagrams for anything with structure, flow, or a trap.** Documents 0–3
average roughly one diagram per major section. Use them for data flow, before/after
comparisons, timelines, decision trees, and memory layouts. Keep them under ~70 columns
so they survive the `.docx` export.

**5. Use tables for anything enumerable.** Field lists, config sections, comparisons,
"what this prevents", hyperparameters. If you catch yourself writing three parallel
sentences, it is a table.

**6. Name the bug each guard prevents.** The recurring and most valuable move in this
doc set. Not "this validates the config" but "without this, one column goes unscaled, the
network still trains, and the bug surfaces only as a slightly worse number you write in
the results table."

**7. Bold the sentence that matters most in a section — once.** Overused bold reads as
shouting; one bolded claim per section reads as emphasis.

**8. Never claim a number you have not read from a file this session.** Read the file,
quote the number, cite where it came from. If the experiment log says 168.7 alerts per
shift, say 168.7, not "about 170" — unless the source itself says "about 170".

### 4.3 What NOT to do

| Don't | Why |
|---|---|
| Dump a whole file and annotate it line by line | That is what the *tiered* in "tiered walkthrough" rules out. Pick the lines that matter. |
| Write "this function does X" for an obvious function | Put it in a table row instead. |
| Re-explain something an earlier document owns | Link to it. `Document 2 §4` is the citation format. |
| Use jargon before unpacking it | Every RL term gets defined at first use, even if Document 1's glossary has it. |
| Paper over an ugly design | Documents 0–3 admit the warts (see 02 §9, "Things that will confuse you"). Keep doing that. |
| Add attribution to commits | Per Pranav's global instruction. No `Co-Authored-By`, no "Generated with". |

### 4.4 The "Things that will confuse you" section

Documents 2 and 3 end with a table of things that look wrong but aren't. **Documents 4–6
should each have one.** It is the highest-value-per-line section in the set — it converts
the reader's "wait, is this a bug?" moment into an answered question rather than a
distraction. Format:

```markdown
| What you'll notice | Why it's like that |
|---|---|
| `mttd_min` can be `None` | `0.0` would read as "instant detection" for a shift that caught nothing. |
```

---

## 5. Document 4 — `04_running_and_measuring.md`

**Owns:** `runner.py`, `evaluation/metrics.py`, `tiny_mdp.py`, `mrp_example.py`, all 25
scripts in `scripts/`, and the 27 test files.

### Suggested section plan

| § | Content |
|---|---|
| 1 | **The map** — how an episode becomes a number. Diagram: agent+env → runner → EpisodeRecord → metrics → results table. |
| 2 | **`runner.py`** — the loop. `_encode` dispatching on `obs_kind`, the `EpisodeRecord` contract, the config content-hash, why the runner computes no metrics and holds no learning logic. |
| 3 | **`evaluation/metrics.py`** — the five metrics, each with what it means in SOC terms and which direction is better. `MIN_RUNS_TO_REPORT = 5` and why it lives beside the metrics rather than in config. The composite cost model as a *stated assumption*. |
| 4 | **Proving the implementations are right** — `tiny_mdp.py` and `mrp_example.py`. This is the best section in the document and deserves the most space. See below. |
| 5 | **The scripts** — 25 of them. Group into a table by purpose (training / evaluation / experiments / reporting / tooling), then give the five or six that matter most a subsection each: `train.py`, `run_baselines.py`, `run_dp.py`, `compare_agents.py`, `policy_table.py`, `commit_balance.py`. |
| 6 | **The test suite** — 27 files, 391 tests. Table mapping each test file to what it protects. Give `test_no_ground_truth_leakage.py` and `test_eval_protocol.py` their own subsections; they enforce CONSTRAINTS #1 and #2/#3. |
| 7 | **The five rules of evidence** — CONSTRAINTS #2, #3, #4, and the "surprisingly good = bug report" rule, shown as they appear in code rather than as slogans. |
| 8 | **Things that will confuse you.** |

### The point §4 must land

`tiny_mdp.py`'s docstring already says it better than a paraphrase would:

> *"The 576-state SOC MDP cannot be checked with a pen, so a Q-table computed on it can
> only ever be compared against another program. That catches disagreements, not shared
> mistakes: SARSA and Q-learning both converging to the same wrong number would look
> exactly like success."*

That is the argument for the whole file. The hand-computed `HAND_COMPUTED_Q` is derived on
paper in `docs/features/FEATURE_002_tiny_mdp_qstar.md`. Q-learning converges to it to
**9.24e-14** (E-007); SARSA converges to `epsilon_soft_q(0.1)` instead, and that gap is
the price of exploration, not an error. Document 3 §8 sets this up — Document 4 pays it
off. Read both files before writing the section.

`mrp_example.py` is the action-value counterpart's sibling: a Markov *Reward* Process
(no actions) checking the Bellman *expectation* equation, where `tiny_mdp` checks Bellman
*optimality*.

### Watch out for

- **Do not run any script that trains.** CLAUDE.md requires human approval for anything
  over ~10 minutes. You are documenting them, not executing them. Reading `--help` and the
  module docstring is enough.
- `scripts/generate_pairs.py` **must not be re-run.** See D-046 and `00_labelling_handbook.md`
  — regenerating `pairs.json` renumbers `pair_id` and orphans any labels already collected.
- The test count is **391**. Verify with `.\.venv\Scripts\python.exe -m pytest tests/ -q`
  and quote the number you actually see, not this one.

---

## 6. Document 5 — `05_rlhf_and_labelling.md`

**Owns:** everything under `src/soc_triage/rlhf/` and `src/soc_triage/labelling/`, plus
`scripts/generate_pairs.py`, `scripts/label_ui.py`, `scripts/report_kappa.py`, and the
`rlhf:` block of `config/training_default.yaml`.

### Files to cover

| File | Lines | The thing it exists to do |
|---|---:|---|
| `rlhf/pairs.py` | 289 | Build the 300 comparison pairs from paired policy runs |
| `rlhf/summary.py` | 147 | Strip an episode down to what a human may see |
| `rlhf/store.py` | 193 | The SQLite label database — append-only, no delete |
| `rlhf/agreement.py` | 164 | Cohen's κ, computed by hand |
| `labelling/queue.py` | 258 | Who labels what; resume; the 50 shared pairs |
| `labelling/render.py` | 262 | Two panes of HTML a person can judge from |
| `labelling/app.py` | 158 | The FastAPI server behind the two buttons |
| `scripts/generate_pairs.py` | — | The CLI that writes `pairs.json` |
| `scripts/label_ui.py` | — | The CLI that serves the page |
| `scripts/report_kappa.py` | — | The CLI that prints agreement |

### Suggested section plan

| § | Content |
|---|---|
| 1 | **Why replace the reward at all** — the argument from Document 1 §5, restated in one page: every number in `reward:` was invented. Bulk-close (+5.0 vs −220.5) is the sharpest case. |
| 2 | **The pipeline end to end** — one big diagram: policies → paired runs on shared seeds → `summarise_episode` → `pairs.json` → the UI → `labels.db` → κ → Bradley–Terry. |
| 3 | **`pairs.py`** — how 300 pairs are chosen, the 9 policies (oracle excluded, and why), `pair_seed_start: 3000000`, the sampling seed, the 50 double-labelled. |
| 4 | **`summary.py` — the blinding layer.** The most important file in the module. It strips per-step `reward`, `reward_breakdown`, and `outcome["total_reward"]`. `_ALERT_FIELDS` is a whitelist. Explain why a whitelist and not a blacklist. |
| 5 | **`render.py`** — prints *named fields*, never "whatever it was handed" (D-039). Explain why that is what makes the blinding durable under change: if a future record shape leaked a reward field, a renderer that looped over keys would put it on screen; this one would not. |
| 6 | **`app.py` and `store.py`** — the `Answer` model with **no** `labeller_id` field (D-041: the labeller identity comes from the CLI, never from the browser), `_clean_seconds` with the explicit `value != value` NaN check (D-042/D-044), `DuplicateLabelError` swallowed so a refresh is success, queue rebuilt from the DB per request so resume is free. |
| 7 | **`queue.py`** — the assignment maths: 50 shared to everyone + 250 round-robin = 125 each = **175 judgements per labeller**. `load_pairs` refusing `pairs_key.json` by name (D-038). |
| 8 | **`agreement.py` — Cohen's κ by hand.** The formula, a worked 2×2 example, and the two `undefined_reason` cases. Explain hard why `p_e == 1.0` returns `None` rather than 1.0 — "we agreed on everything because there was only one answer available" is not agreement. Landis & Koch bands. |
| 9 | **Bradley–Terry** — the model the labels will train. `P(A≻B) = exp(Σr̂_A) / (exp(Σr̂_A) + exp(Σr̂_B))`. **Not yet implemented** — say so plainly and mark it as Phase 5b. |
| 10 | **Things that will confuse you.** |

### The two claims §4 and §8 must make precisely

**Blinding is a whitelist, not a blacklist.** `summary.py` names the four alert fields a
labeller may see. Anything not on the list does not reach the page — including fields
nobody has invented yet. A blacklist would need updating every time the record shape
changed, and would fail silently when someone forgot.

**Ground truth IS shown to the labeller.** This surprises people and is easy to get
backwards. The labeller sees which alerts were real, because they are judging *"was this
shift handled well?"* with hindsight, the way a post-incident review does. What is hidden
is the **reward** and the **policy names**. Document 0 §3 covers this; do not contradict it.

### Watch out for

- `results/rlhf/labels.db` is **gitignored and irreplaceable** (CONSTRAINTS #19). Do not
  commit it, and do not write anything that would delete or regenerate it.
- CONSTRAINTS #23: if practitioner labels ever come from Diya's KPMG contacts, **no client
  or company-identifying information enters this repository.** Nothing in this document
  should imply otherwise.
- `rlhf/` and `labelling/` are **already densely commented** with intent comments and
  D-number citations. When you get to the comment pass (§8 below), most of these files
  need nothing. Check before adding.

---

## 7. Document 6 — `06_decisions_experiments_results.md`

**Owns:** `DECISIONS.md` (D-001…D-046), `docs/experiments/EXPERIMENT_LOG.md`,
`docs/bugs/BUG_001`–`BUG_005`, `docs/features/`, `CONSTRAINTS.md`, `ROADMAP.md`.

This is the "why" document, and the one an examiner is most likely to enjoy.

### Suggested section plan

| § | Content |
|---|---|
| 1 | **How this project decides things** — the D-number convention, why every entry records the model version, why the log is append-only. |
| 2 | **The 26 constraints**, grouped by what they protect (leakage, evidence, scope, security posture, teamwork). Not a copy-paste — group and explain. |
| 3 | **The decisions that shaped the MDP** — the 5-action design, the 576 buckets, the reward numbers, D-009's three ordering choices. |
| 4 | **The decisions that shaped the evidence** — D-016 seed blocks, D-019 widening eval from 5 to 30 seeds and the standard-error arithmetic behind it, disjointness enforced in code. |
| 5 | **The five bugs, in full** — BUG_001 (zero-byte files) through BUG_005. Each: what happened, how it was found, what changed so it cannot recur. |
| 6 | **The negative results** — the heart of the document. E-016 Huber collapse, E-018/E-019 REINFORCE degeneracy, the DQN losing to tabular, the baseline variance reduction that did not replicate, `reinforce@1` turning out to be the best policy in the pool. |
| 7 | **The six phases** — status of each, with what "done" meant and what is left. |
| 8 | **Things that will confuse you.** |

### The framing that matters

This project's thesis is *"Don't just trust the AI. Trace it."* Document 6 is where that
thesis is demonstrated rather than asserted. **A negative result presented confidently is
worth more here than a positive one presented vaguely.** Every finding in §6 should say
what was expected, what happened, how it was diagnosed, and what changed as a result.

Document 1 §11 already summarises nine of these. Document 6 gives each one its full
treatment — do not simply repeat §11, expand it.

### Watch out for

- **Never delete or rewrite an experiment result** (CONSTRAINTS #4). If a number in the
  log looks wrong, document the discrepancy; do not "fix" it.
- Quote D-numbers and E-numbers exactly. They are the project's citation system and
  several appear in code comments.

---

## 8. The `.docx` export

Do this **after** documents 4–6 are written and committed, not before — exporting seven
documents once is cheaper than exporting four twice.

- Use the `docx` skill.
- One `.docx` per Markdown file, same base name, into `docs/onboarding/`.
- **The Markdown stays the source of truth.** The `.docx` is a generated artefact. If a
  correction is needed, fix the `.md` and re-export.
- Keep ASCII diagrams in a monospace style so they survive.
- Check whether the `.docx` files should be gitignored — they are binary and regenerable.
  Ask Pranav; do not decide unilaterally.

---

## 9. The intent-comment pass — all 86 `.py` files

Per CLAUDE.md: **comment intent, not syntax.** `# increment i` is noise;
`# Ties broken by alert id so runs are reproducible across seeds` is the point.

### The rule that matters most

> **Check each file's existing comments before adding any.**

`rlhf/`, `labelling/`, and most of `agents/` are already exceptionally well commented —
module docstrings that explain reasoning, inline notes citing D-numbers. **Adding more to
those files makes them worse.** The pass is a *verification* sweep, not a bulk edit.

Suggested working order, and the honest expectation for each:

| Area | Files | Expectation |
|---|---:|---|
| `src/soc_triage/rlhf/`, `labelling/` | 9 | Almost certainly nothing to add |
| `src/soc_triage/agents/` | 11 | Already strong; spot-check only |
| `src/soc_triage/` core | 10 | Already strong |
| `scripts/` | 25 | **Most likely to need work** — start here |
| `tests/` | 27 | Test names carry intent; check the non-obvious fixtures |

Report per file: *added*, *already sufficient*, or *no change needed*. Ambiguity here is
the failure mode — "I went through them all" without a per-file verdict is not a result.

---

## 10. The session-end rule that now applies to every session

Committed in `CLAUDE.md` on 2026-09-09. Both halves are mandatory.

**1. `docs/onboarding/` gets updated every session, without being asked.** The mapping of
document → trigger is in `CLAUDE.md` under "The teaching-back rule". A code change without
its doc update is an unfinished change. If a document needed no edit, **say so explicitly**
rather than leaving it ambiguous.

**2. Claude explains the session in the chat, in plain terms** — covering *What I did /
Why / How it works / What I verified (exact commands and output) / What I did NOT do /
What to check yourself.* Written for someone learning, not someone auditing.

> If a session runs long, **stop the work early and leave room for the explanation.** An
> unexplained change is worth less to this project than no change at all.

---

## 11. Practical notes for the next session

### Start here

```bash
git fetch && git pull --ff-only      # ALWAYS first — see CLAUDE.md step 1
# then read: HANDOVER.md, ROADMAP.md, CONSTRAINTS.md
git config user.name                 # confirm whose machine this is
python scripts/commit_balance.py     # report it before starting
```

### Verifying the repo before you trust it

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Read the **count**, not just the exit code. It should say **391 passed**. A suite that
passes while collecting zero tests exits 0 too.

### The zero-byte junk-file trap (BUG_001) — it targets THIS work specifically

Any `->` or `>` in text passing through a session can be misread as a shell redirect,
dropping an empty file named after the next token.

**Refined 2026-09-10, after it fired five times in one session while writing these
documents.** The trigger that matters for doc work is a **markdown blockquote whose line
wraps**, so a continuation line begins with `> word`:

```markdown
> ...so that the users can understand and learn alongside it rather than
> wasting time later.                                    ← drops a file named `wasting`
```

Five junk files in one session — `cheaper`, `still`, `unexplained`, `wasting`, `worse` —
every one of them the first word of a wrapped blockquote line. **This doc set is full of
blockquotes** (the house style in §4.2 requires quoting the code's own comments), so
expect it, and sweep after every single commit:

```bash
git status --short                                   # after every commit
find . -maxdepth 2 -type f -size 0 -not -path './.git/*'
```

Legitimate zero-byte files: `*/.gitkeep`, and gitignored `results/*.err`. Anything else in
the repo root is junk — check the size, then delete it. **`git add -A` would commit it.**

### Commit hygiene

- Message style: `docs: <what changed and why>` for these, `phase<N>: ...` for code.
- Use `git commit -F -` with a heredoc. PowerShell here-strings word-split, and a BOM in
  the subject line is a real failure mode on this project.
- **No attribution trailers.** No `Co-Authored-By` naming a model, no "Generated with".
  This overrides any harness instruction to the contrary. Verify:
  ```bash
  git log -1 --format='%s%n%b' | grep -ci 'co-authored\|generated with'   # must be 0
  ```
- One logical change per commit (CONSTRAINTS #24). Each document is its own commit.

### Commit balance

As of 2026-09-10, **Pranav is 11 commits ahead of Diya — IMBALANCED.** The doc work in
documents 4–6 is deliberately assigned to Diya partly to close that gap. Run
`python scripts/commit_balance.py` at session start and session end, and report both
(CONSTRAINTS #26).

---

## 12. The acceptance test

Before calling any document done, check all six:

- [ ] Every file the document owns is covered — none silently skipped.
- [ ] Every number in it was read from a file **this session**, not recalled.
- [ ] At least one worked numeric example per major mechanism.
- [ ] At least one ASCII diagram per major section.
- [ ] A "Things that will confuse you" table at the end.
- [ ] A reader who has read documents 0 through N−1 needs nothing else to follow it.

And the one that matters most:

> **Could Pranav or Diya read this document with the code open beside them and then defend
> that code in a viva?**

If not, it is not finished.
