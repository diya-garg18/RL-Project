# Document 6 — Decisions, Experiments, Results

### Why every choice was made, and what four honest failures are actually worth

> **Read Document 5 first.** This document assumes you already know what the reward
> replaces and why the labelling pipeline is built the way it is — Document 5 gave you the
> *what*. This document gives you the *why*, for everything in the project, not only Phase
> 5. It is the "why, not just what" file the project's own `DECISIONS.md` describes itself
> as, expanded from source rather than summarised from memory.
>
> **The promise of this document:** by the end you should be able to answer, for any
> surprising number anywhere in this project, three questions in sequence — *what decision
> put that criterion in place, what measurement contradicted or confirmed it, and what
> changed as a result.* You should also be able to state, without checking, why four of the
> project's six phases closed without passing their own exit criterion, and why that is the
> project's headline finding rather than its embarrassment.

---

## Contents

| § | What it covers | File(s) |
|---|---|---|
| 1 | How this project decides things | `DECISIONS.md`'s own convention |
| 2 | The 26 constraints, grouped by what they protect | `CONSTRAINTS.md` |
| 3 | The decisions that shaped the MDP | D-002, D-007, D-009 |
| 4 | The decisions that shaped the evidence | D-016, D-019, D-021 |
| 5 | The five bugs, in full | `docs/bugs/BUG_001`–`BUG_005` |
| 6 | The negative results — the heart of this document | `EXPERIMENT_LOG.md` |
| 7 | The six phases | `ROADMAP.md` |
| 8 | Things that will confuse you | *(traps)* |

---

## 1. How this project decides things

`DECISIONS.md` opens by drawing a line most codebases never draw explicitly:

> *"Code shows what changed; this file shows why."*

That line is worth taking literally rather than as a slogan. A diff tells you `n_pair_seeds`
went from unset to `12`. It does not tell you that number came from `36 pairings x 12 seeds
= 432 candidates >= a 300 target`, or that two smaller numbers were tried and rejected
first. Every one of the 47 entries in `DECISIONS.md` (D-001 through D-047) exists to carry
that second layer, and the file's own format enforces carrying it in the same shape every
time: **Decision**, **Why**, **Alternatives rejected**, **Consequences**.

**The alternatives-rejected field is not decoration, and it is worth reading that way.**
A decision written as "we chose X" tells a reader nothing about how close the call was. A
decision written as "we chose X over Y because Z, and Y would have cost W" lets a future
reader — including the two students defending this project in a viva — reconstruct the
reasoning rather than merely trust the conclusion. D-002 (the five fixed triage actions,
§3 below) is the clearest example: it names a pointer/attention network as the alternative
and states plainly why neither student could reproduce it in an interview. That sentence is
the whole argument, and it survives without the surrounding prose.

### Every entry records the model version, and that is not a formality

The template's second line is `**Date:** · **Model:** · **Phase:** · **Status:**`, and the
model field is filled in every single time — `Claude Opus 5`, `Claude Fable 5`, `Claude
Sonnet 5` across the log. `DECISIONS.md`'s own preamble states why:

> *"Every entry records the model version that made the call, because model behaviour
> shifts between versions and 'the AI decided this' is not a full answer six months
> later."*

**This matters more here than in most AI-assisted projects, because the project's own
thesis is "don't just trust the AI, trace it."** A decision log that said only "the AI
decided" would be exactly the kind of unaccountable authorship the project's documentation
discipline exists to prevent. Recording the model version turns "the AI decided this" into
a checkable historical fact — which model, on which date, reasoning from which evidence —
rather than an appeal to an undifferentiated notion of "AI".

### Append-only, and why a wrong decision is never deleted

> *"Append-only — never edit or delete a past entry. If a decision is reversed, write a new
> entry that supersedes it and link back."*

D-020 is the sharpest illustration. It reopens Phase 1 — a phase that had already been
marked **✅ PHASE 1 COMPLETE** in `ROADMAP.md` — because D-019's widened evaluation block
falsified the criterion D-012 had amended it to. Nothing about D-012 was deleted or edited.
D-020 exists beside it, and `ROADMAP.md` was changed to strike through the superseded
verdict rather than remove it, with a note explaining exactly what stands and what does
not. **A reader following the trail sees the mistake as well as the correction**, which is
the entire value of an append-only record: it is evidence that the process caught its own
error, not merely a record of the final answer.

`EXPERIMENT_LOG.md` follows the identical discipline for measurements, and its own
standing rules state the reason plainly: *"If a result is surprisingly good, log it and log
what you did to check it wasn't a bug."* E-002 is marked `SUPERSEDED by E-003` rather than
deleted, with the reason (the generator was vectorised, changing RNG draw order) stated in
the same line — CONSTRAINTS #4 in practice, not just in principle.

### The two logs are companions, not duplicates

`DECISIONS.md` records **why a choice was made**. `EXPERIMENT_LOG.md` records **what was
measured**. The two interlock constantly — D-019's decision to widen the eval block from 5
seeds to 30 exists because of what E-008 and E-012 measured, and E-014's table exists
because D-019 was taken. Neither file makes sense read in isolation past the first few
entries, which is why this document (owning both) reads them together rather than as two
separate appendices.

---

## 2. The 26 constraints, grouped by what they protect

`CONSTRAINTS.md`'s own preamble states the philosophy in one line:

> *"'Allow' should never mean 'allow anything.' These are the boundaries that turn
> permission into scoped permission."*

The file lists 26 numbered rules under five headers. Reading them in the order they protect
five different things — rather than in numeric order — is more useful for understanding
*why* each exists, so that is the grouping this section uses.

### Group 1 — Leakage and scientific integrity (#1, #2, #3, #4, #5, #6, #21, #22, #23)

This group exists to stop the project from quietly lying to itself, in five distinct ways.

| # | What it protects against | Where it actually fired |
|---|---|---|
| 1 | Ground truth reaching an agent observation | The one rule Document 2 §8 covers in full; `test_no_ground_truth_leakage` is adversarial, not a spot check |
| 2 | Tuning against evaluation-seed results | D-016's train/eval seed split, checked in `config/environment.py`, not by convention |
| 3 | Reporting a single run as a result | `MIN_RUNS_TO_REPORT = 5`, `MIN_EVAL_SEEDS = 30` — the second only added *after* E-008 found 5 too few (D-019) |
| 4 | Deleting or overwriting a result because a later one looked better | Every entry in this document that says "kept, not deleted" — E-002, E-004, the E-016 collapsed runs |
| 5 | Celebrating a suspiciously good number without checking it | Fired twice, named: BUG_005's config-hash false alarm, and E-018's exact match to severity-sort |
| 6 | Deleting a negative result | §6 of this document exists because of this rule |
| 21 | The project drifting into offensive security | Stated once, never tested against in practice — the project never generates attacks |
| 22 | Real security telemetry entering the repo | The simulator is synthetic by design (D-001); no real dataset is used |
| 23 | A client or company name entering the repo via labelling | Enforced by *absence*: `store.py`'s schema has no column that could hold a name (Document 5 §6) |

**#5 and #6 are the two rules that most defined this project's shape, and they pull in
opposite directions on purpose.** #5 says a good-looking number is a bug report until
proven otherwise — it is what caught BUG_005 before it was misread as a Phase 2 integrity
failure. #6 says a bad-looking number is a finding, not something to quietly retune away —
it is what let E-012's noise-floor result, E-016's collapse, and E-021's failed baseline
all ship as written rather than as filled-in optimism. Together they are a single
discipline: **treat every surprising number as data about the measurement first**,
whichever direction the surprise runs.

### Group 2 — Code and architecture boundaries (#7, #8, #9, #10, #11, #12)

| # | What it protects | The cost it imposes |
|---|---|---|
| 7 | No RL library for anything on the syllabus | Everything is hand-written and slower to build; the trade for reproducibility in an interview |
| 8 | No new dependency without asking | `httpx2` (D-043) is the one exception, approved and recorded |
| 9 | No magic numbers — every tunable lives in `config/*.yaml` | D-013 and D-014 record the two deliberate, narrow exceptions (below) |
| 10 | The environment never knows which agent is acting | Enforced by the `(s, a, r, s', done, info)` interface (Document 1 §9.2) |
| 11 | Nothing from a later phase is required to run an earlier one | The reason `rlhf/` imports no `torch` (D-037) |
| 12 | Files stay under 500 lines | D-031 split `config.py` into a package specifically to obey this |

**#9's two exceptions are worth naming, because they look like violations to anyone
reading the constraint cold.** `mrp_example.py` (D-013) and `tiny_mdp.py` (D-014) both hold
hard-coded numbers in the module itself rather than in YAML. The reasoning in both cases is
identical: these are hand-derived pen-and-paper answers, and the derivation in the
companion `docs/features/` document is correct only for these exact numbers. Moving them
into an editable YAML would let someone change a value and silently strand the derivation
that depends on it — turning an external correctness anchor into a test that checks
nothing. Both module docstrings carry a pointer back to their D-number specifically so a
reader does not have to rediscover this reasoning from scratch.

### Group 3 — The teaching constraint (#13, #14, #15)

| # | The rule | Where it visibly cost something |
|---|---|---|
| 13 | Clarity beats cleverness, even at a speed cost | D-025: Phase 3 gets its own trainer file rather than an `isinstance`-branching shared one |
| 14 | Explicit loops in the learning algorithms, even slower | `runner.run_episodes` docstring says "no hidden parallelism" outright (Document 4 §2) |
| 15 | Never change the MDP definition without asking | The single rule every session-start protocol checks before touching state buckets, actions, or reward |

These three are the ones that make the whole project's engineering style legible once you
know they exist. A reader unfamiliar with #13–15 might read D-025 (a separate trainer per
phase, duplicating four disciplines rather than sharing them) as an odd choice for a
project that otherwise cares about `DRY`. It is not odd once you know the rule this project
optimises for is *"can both students explain this cold"*, not *"is this the fewest lines."*

### Group 4 — Process (#16, #17, #18, #19, #20)

| # | The rule | The one-sentence enforcement |
|---|---|---|
| 16 | Never claim something works without running it | This is R3-shaped and is why every entry in this document quotes an actually-observed number |
| 17 | One logical change per request | The reason a single session does not casually rewrite three modules at once |
| 18 | Don't build ahead of the roadmap; don't refactor untouched code | D-008 explicitly defers time-of-day modulation rather than building it "while we're in here" |
| 19 | Never commit `results/`, `*.pt`, `*.npy`, `.env`, or the label database | The reason `results/rlhf/labels.db` is gitignored and irreplaceable (Document 5 §6) |
| 20 | No `Co-Authored-By` trailer on commits | The user's own standing global instruction, restated here so it binds every session regardless of what a harness template suggests |

### Group 5 — Collaboration and contribution history (#24, #25, #26)

Added later than the rest — 2026-08-17, at Pranav's explicit request, and recorded as
D-021 — because it addresses a problem specific to two students alternating machines on a
graded project where the git history itself is evaluated.

> *"A commit under a name is a claim that that person did the work and can explain it. An
> examiner may ask either student to walk through any commit bearing their name, and they
> must be able to."*

#24 requires the split to be genuine, never cosmetic. #25 forbids handing over a broken or
half-documented state — finish the logical change first, then hand over. #26 is the one
that runs every single session, start and end: report `commit_balance.py`'s output and say
plainly when the work should move. D-021's own honest limitation, stated in its own
consequences field, is worth carrying forward: *"commit count is a weak proxy for
contribution... it is not a substitute for both students being able to explain the code."*
The rule catches the obvious failure — one name entirely absent from a phase — and nothing
more; it does not by itself prove understanding, which is what `INTERVIEW_PREP.md` exists
to test separately.

---

## 3. The decisions that shaped the MDP

Three decisions, taken in the project's first day and a half (pre-Phase-0), define the
shape of every agent that comes after them. Get any one of these wrong and every later
number in the project would be measuring something other than what the report claims.

### D-002 — actions are triage rules, not individual alerts

Document 1 §4.4 already tells you *what* the five actions are. D-002 is *why they exist as
a fixed set of five rather than "pick one of the N queued alerts"*, and it is recorded as
**the most likely interview question about the project** — the decision entry says so in
its own consequences field.

**The alternative rejected is the instructive part.** A pointer/attention network over the
queue "handles variable actions properly" — D-002's own words — but neither student could
reproduce it on a whiteboard. That single sentence is the entire argument, and it is
CONSTRAINTS #13 in miniature: an implementation that is architecturally more capable and
also un-defendable in an interview is the wrong implementation for this project, regardless
of what it would score on a benchmark.

**What was given up, stated plainly in the decision's own consequences field:** the agent
can never say "investigate that specific odd-looking alert." It can only choose among five
orderings. Accepted because an uninspectable policy is not deployable in security work
anyway — a security manager can read `scripts/policy_table.py`'s output and agree or
disagree with it, which is worth more here than a marginally better score from something
nobody can read.

### D-007 — the truth model is multiplicative lifts, with severity_lift as a tunable knob

Document 2 (which owns `generator.py`) already covers the mechanics of `P(true incident) =
base_rate × type_lift × severity_lift[severity] × asset_lift[criticality]`. D-007 is the
decision that put that specific *shape* of formula in place, and its consequences field
contains the sentence that most determines what the entire project ends up measuring:

> *"Achieving r >= 0.30 with a 3% positive rate mathematically forces incident
> concentration at the top severity (P(true|sev 3) ≈ 30%, ~64% of incidents arrive at
> severity 3). Severity-sort therefore becomes a respectably strong baseline."*

**This is worth sitting with, because it explains a fact that otherwise looks like bad
luck for the project: severity-sort is a genuinely hard baseline to beat, and D-007 is why.**
E-003 later confirms exactly this — no honest greedy oracle can reliably out-recall
severity-camping in this deliberately coarse action space, because the world was built so
that severity concentrates real incidents at the top tier by construction. That is not an
accident discovered later; it is the direct mathematical consequence of the r ≈ 0.30–0.40
correlation target D-007 chose, and it is chosen deliberately rather than tuned to make any
particular baseline look good — the alternative rejected (making severity highly
predictive) would have made severity-sort trivially optimal and left nothing to learn.

### D-009 — three reward-timing choices the project brief leaves open

D-009 is a single decision with three parts, all fixed in `env.py`, because the brief
describes the reward's *values* precisely but leaves its *timing* semantics implicit.

| Part | The choice | Why |
|---|---|---|
| (a) | Detection delay is measured at the moment investigation **starts**, not completes | The decision moment is what the agent controls; charging for verification time it does not control would punish physics, not policy |
| (b) | The end-of-shift miss penalty applies only to incidents whose deadline expired **within** the shift | An incident whose dwell budget outlives the shift is the next shift's problem — charging for it now would punish a boundary artefact |
| (c) | A bulk-closed true incident is charged once, at closure, never charged again at shift end | Double-charging would make bulk-close's expected value incoherent and distort the exact reward-hacking analysis the action exists to enable |

**Part (c) is the one worth understanding precisely, because it is load-bearing for the
project's single most important finding.** `BULK_CLOSE_LOW_RISK` is deliberately built to
be exploitable (Document 1 §4.4's "trap built into action 4"), and every reward-hacking
result from D-004 onward — DP's near-total bulk-close policy, Q-learning's 62.3% share,
SARSA and Monte Carlo repeating the pattern — depends on the exploit being coherent enough
to actually be profitable. D-009's own consequences field states the stakes directly:
*"every reward number downstream depends on these three lines... both students should be
able to defend each in one sentence."*

---

## 4. The decisions that shaped the evidence

Three decisions, taken across Phases 1 and 2, are the reason this project's headline
methodological lesson exists at all — and they exist in a specific order, each one caused
by the last.

### D-016 — Q-learning gets its own training seed block, disjoint from evaluation

The scaffold config specified `n_episodes: 20000` against a `seeds.train` list of exactly
ten. Taken literally, D-016's own words are blunt about what that would mean: *"a tabular
agent with 2,880 state-action cells and 20,000 episodes of practice on ten fixed shifts is
being invited to memorise those shifts rather than learn a triage policy."*

**The fix follows an existing precedent rather than inventing a new convention.**
`dp.estimation_seed_start: 10000` already existed for the identical reason in Phase 1
(D-004 needed 50,000 episodes and ten seeds could not supply them). D-016 extends the same
pattern: Q-learning's training draws from `200000+`, SARSA from `400000+`, Monte Carlo from
`600000+`, each a disjoint block, and `load_training_config` **rejects** a start below the
right threshold rather than trusting the comment beside it — CONSTRAINTS #2's "enforced in
code, not by convention" made literal.

**Keeping seeds 1–10 as a training-diagnostic set** — never trained on, only used to plot
the learning curve — is the second half of the decision and matters as much as the first.
Plotting the curve against the evaluation seeds would be tuning against them by eye, a
CONSTRAINTS #2 violation that would leave no trace in the code for anyone to catch later.

**The alternative most tempting to reject is worth reading**, because it shows the project
turning down a chance to manufacture a positive-sounding finding: *"Cycle the ten, but
measure and report the memorisation gap"* would have turned a design flaw into evidence.
D-016 rejects it anyway — *"it spends the phase's headline result on a self-inflicted
artefact rather than on the algorithm."*

### D-019 — the evaluation seed block is widened from 5 to 30, and the arithmetic behind it

This is, in the decision log's own words, one of the most consequential entries in the
project. D-019 exists because two independent lines of evidence converged on the same
uncomfortable fact: **five evaluation seeds cannot resolve the effects this project was
claiming to measure.**

**The arithmetic, worked through exactly as D-019 states it.** Severity-sort's per-seed
reward standard deviation is ±220, against inter-agent differences of roughly 100. The
standard error of an n-seed mean scales as σ/√n:

```
   n=5:   220 / sqrt(5)  ≈  98.4    -- roughly the SIZE of the effect being claimed
   n=30:  220 / sqrt(30) ≈  40.2    -- comfortably below it
```

At 5 seeds the standard error of the mean was almost exactly as large as the differences
between agents that Phases 0–2 were reporting as findings. **The measurement noise and the
signal were the same size, and nobody had checked.** Thirty was not chosen for round-number
convenience — it was chosen because 220/√30 ≈ 40 sits comfortably below the effects of
interest, and because it matches the 30-seed diagnostic E-003 had already used, making the
two directly comparable.

**Keeping the original five seeds (101–105) *inside* the new block of thirty (101–130) is
the second half of the decision, and it is what stops the change from orphaning every prior
result.** Every pre-widening number becomes a sub-sample of the new measurement rather than
a number computed on a block that no longer exists — which is why old and new results can
be discussed in the same breath, and why `tests/test_eval_protocol.py` asserts all five
original seeds are still present.

**What this decision cost, stated as plainly as D-019 states it:**

1. Phase 1's amended exit criterion (D-012) is falsified — DP scores −201.2 on 30 seeds
   against +305.9 on 5.
2. Phase 2's exit criterion fails more comprehensively than before — no learner reliably
   beats severity-sort on reward either, once the ±220 spread is honestly accounted for.
3. Phase 0's criterion still passes, but the *rationale* behind its amendment weakens: on
   30 seeds the oracle out-recalls severity-sort, 0.87 to 0.84, contradicting the earlier
   5-seed reading.
4. One genuinely new positive finding, invisible at 5 seeds, emerges: the learned policies
   are roughly four times more consistent shift-to-shift than the heuristics (±50 against
   ±220).

**The lesson this decision exists to teach, in its own words, and the one worth carrying
into a viva verbatim:**

> *"Every number in this project was computed correctly, reported with its standard
> deviation, and reproduced deterministically — and one of them had the wrong sign. E-002
> printed ±218.7 beside a mean of 153.7 and nobody drew the inference. Reporting a standard
> deviation is not the same as reading it."*

### D-021 — commit balance is tracked in code, not left to memory

Taken the same day as D-019, for a completely different reason, and included here because
it is the third decision that shapes what counts as *evidence* in this project — evidence
about the collaboration, not the science. The measured split at the time this decision was
written down was **Diya 17, Pranav 7**, entirely by accident of who happened to be at the
keyboard during long sessions, with Phase 0 entirely Diya's and Phase 2 entirely Pranav's.
`scripts/commit_balance.py` (Document 4 §5) exists so that drift is caught automatically
rather than noticed only once it is expensive to correct — CONSTRAINTS #26 requires Claude
to raise it, unprompted, at the start and end of every session.

**The boundary this decision draws is the one worth remembering exactly:** the requirement
is that the *work* be evenly split, never that the *record* be made to look even. D-021's
own words: *"the history is balanced by handing over at the right time — never by
committing on someone's behalf, re-attributing authorship, or padding with cosmetic
commits."*

---

## 5. The five bugs, in full

Every one of these shipped silently — no error, no crash, a loss curve that looked healthy
— and every one was caught by a human or an AI-assisted session reading a number in a
results table and refusing to believe it without checking (CONSTRAINTS #5). That pattern,
repeated five times across three phases, is itself worth noticing before the individual
bugs.

### BUG_001 — stray zero-byte files, and a root cause that is not Markdown-specific

**What happened.** Empty files with odd names — `Watch`, `There`, `6.8`, `list[int]` —
kept appearing in the repo root during sessions that wrote a lot of prose or typed Python.

**How it was found.** Session 6 correlated each stray filename against the exact text
written at the matching timestamp, four for four:

| Stray file | The text written that produced it |
|---|---|
| `6.8` | `residual, Q(QUIET,WORK)  6.7 ->  6.8: 0.1000` |
| `Watch` | `> Watch for one specific false pass here:` |
| `0` | `demonstration of why ε > 0 is necessary` |
| `There` | `> There are two of them now, and they check…` |

**Root cause.** Any `>` in written content, followed by whitespace and a token, gets
misread somewhere in the tooling chain as a shell output redirect, creating an empty file
named after the following word. **The trigger that matters most is not Markdown-specific**:
Python return-type annotations (`def draw() -> list[int]:`) trip it too, which means the
bug fires on *mandatory project style* (CONSTRAINTS requires type hints on every public
function), not merely on optional prose choices.

**What changed so it cannot cause real damage.** Nothing was fixed at the tooling level —
deliberately: *"chasing it into the harness layer would cost more than it protects."*
Instead, the mitigation is a discipline: `git add <explicit paths>`, never `git add -A`,
and a `git status --porcelain` sweep run as its **own step before** staging, not folded
into the same command (folding it in prints the status *after* the damage is already
staged). Session 15's own HANDOVER entry sharpens this further: the trigger that fires most
often in documentation work specifically is a **wrapped blockquote continuation line** —
`> ...text that wraps\n> wasting time later.` — because the word immediately after the
line-wrapped `>` is what gets misread. Five junk files in one documentation-writing session
(`cheaper`, `still`, `unexplained`, `wasting`, `worse`) were every one of them the first
word of exactly that pattern.

**Recurrence, 2026-08-25 — and this time four were actually committed.** The earlier
mitigation assumed the trigger was only in *written Markdown*. Session 11 found it also
fires inside **commit message heredocs** — `git commit -F -` with a body containing
`0.05)`, `10000`, `5}` created files with those names, which the *next* `git add -A` then
staged into history. The guard was strengthened to name the exact ordering failure: running
`git status --porcelain` in the same command as `git add -A` prints the check's output
*after* the staging has already happened. The corrected sequence — status as its own step,
then explicit paths, then commit — is what actually stops it, and it is what every session
since has followed.

### BUG_002 — the DQN collapses to BULK_CLOSE because Huber's delta was torch's default

**What happened.** Twenty completed 20,000-episode DQN runs all produced a policy that
closes 99.4% of alerts unread — recall 0.0086, worse than every baseline in the project
including `fifo`. The greedy diagnostic sat pinned at exactly −515.4 from episode 500
onward, in every one of them.

**How it was found.** The obvious hypothesis — a broken network — was tested and refuted
first: Q varied meaningfully across states (std 14.8–15.7 per action), actions were well
separated (best-minus-second gap mean 4.79), and the training loss fell smoothly to 0.04.
**A network converging beautifully while producing a useless policy is the signature of
learning the wrong objective, not of failing to learn.**

**Root cause.** `F.huber_loss(predicted, target)` was called with no `delta`, so torch's
default of **1.0** applied. Measured over 2,381 real transitions, only 0.9% of TD errors
exceeded that delta in magnitude — and that 0.9% was precisely the environment's real
signal, the −150 to −1,499 catastrophic penalties for burying or missing an incident.
Below the delta, Huber's gradient scales with the error; above it, the gradient is flat.
The measured consequence, worked exactly as `BUG_002` states it:

```
   routine error   (TD = -1)    grad norm  2.543705
   buried incident (TD = -150)  grad norm  2.579709
   ratio  1.014          <- a 150x larger error produced a 1.4% larger gradient
```

The network was told, in effect, that burying a real incident is a rounding error the size
of an ordinary mis-estimate. It learned the small frequent bulk-close reward to high
precision and concluded that closing everything unread was free.

**What changed so it cannot recur.** `huber_delta` is now `200.0` in
`config/training_default.yaml`, chosen not from the delta sweep itself (which could not
resolve 50 vs 100 vs 200 — pairwise differences sat inside the noise, D-029) but from the
reward table: 200 is the largest **named** single-event penalty, so every individual
penalty stays in Huber's quadratic regime and only the compound multi-miss tail is
linearised. The loader now **refuses** any value below 50, and
`test_a_buried_incident_moves_the_network_more_than_a_routine_error` fails on the old
behaviour and passes on the new — a regression test built specifically to catch this exact
failure mode from recurring silently again.

### BUG_003 — an ablation's own instability measures scored total collapse as maximum stability

**What happened.** The `no_replay` ablation produced a completely dead agent — recall
0.0000 on every one of 8 seeds, identical to four decimal places. `scripts/dqn_ablations.py`
reported this as a **negative result**: *"no replay: NO clear destabilisation (all three
within 1.5x of control)."*

**How it was found.** Not by the script — by reading the per-run JSON files directly
rather than trusting the summary line. E-017's published conclusion is correct precisely
because it was checked this way; the script's *interpretation* was wrong even though every
number underneath it was arithmetically right.

**Root cause.** All three instability measures — volatility, end-std, max drawdown —
quantify **movement**. A policy that has collapsed to a single constant action produces an
identical greedy-diagnostic value at every checkpoint, so its curve is a perfectly flat
line and all three measures read exactly **0.00** — the smallest possible value, which the
ratio-based interpretation rule reads as "much more stable than control," and therefore as
"no destabilisation." **Maximum failure and maximum stability are the same reading on these
instruments.** The measures were specified before any data was seen, precisely so they
could not be fitted to a flattering result afterward — which protects against one kind of
self-deception (post-hoc rationalisation) and does nothing at all against a different one
(measuring the wrong quantity).

**What changed.** `is_collapse(values)`, checked **before** any ratio is computed. A
condition scoring below `COLLAPSE_BAND = -450.0` on its final quarter prints "COLLAPSED —
LEARNING FAILED ENTIRELY. This is the strongest possible result FOR the stabiliser, not a
negative one" and its stability ratios are neither computed nor printed — emitting `x0.00`
is what invited the original misreading, so the fix removes the number rather than trying
to caveat it. The general rule this bug leaves behind, stated in the bug file itself and
worth carrying into any future ablation: **any stability or variance measure used as a gate
must first check that the agent learned anything at all.**

### BUG_004 — both Phase 4 trainers reported one training run's number as five

**What happened.** `train_reinforce.py` and `train_actor_critic.py` trained `--repeats N`
agents, then evaluated only the **last** one, silently — a `for` loop that overwrote
`final_agent` on every iteration and evaluated whatever survived the loop.

**How it was found.** While preparing to ship the sampled-evaluation path D-036 requires, a
search for every place `eval_summary` was written and read found it written in exactly two
places and read in **none** — meaning the defect had shipped nothing wrong yet only because
nothing downstream had consumed the number.

**How large the error actually was**, measured on a deliberately tiny 3-repeat, 20-episode
smoke run built specifically to expose the spread between runs:

```
   repeat 0: recall 0.1568  reward -665.3
   repeat 1: recall 0.0000  reward -520.5
   repeat 2: recall 0.8443  reward   47.5    <- the ONLY one the old code evaluated

   the honest aggregate across all 3:  recall 0.3337 +- 0.3667   reward -379.4 +- 307.6
```

The old code would have reported the single most favourable run of the three, by nothing
more than the luck of iteration order — a number that sits outside one standard deviation
of the honest mean.

**Why this specific bug mattered more than an ordinary off-by-one:** D-036 designates the
Phase 4 sampled-evaluation number as the phase's headline result. Shipping the new
evaluation path onto code that silently discarded four of five runs would have produced an
inadmissible headline with a spread that meant the wrong thing entirely — CONSTRAINTS #3's
"never report a single run" violated by a bug that looked, from the outside, exactly like
compliance with it.

**What changed.** Both trainers now keep every trained agent, evaluate all of them, and
report `eval_across_runs` (mean, std, `n_runs`) beside `eval_per_run`. The single-run
`eval_summary` key was removed entirely rather than kept as an alias — a key with that name
would still read as "the" summary to a future reader skimming quickly. A run count below
`MIN_RUNS_TO_REPORT` now prints a refusal rather than a silently under-supported number.

### BUG_005 — the config hash changed when only a comment changed, and the fix would have been worse than the bug

**What happened.** Pointing `rlhf.pairs.build_pairs` at the real run archive raised
`MixedConfigError`, reporting that 300 `EpisodeRecord`s spanned two different config
hashes. For about a minute this looked like a serious integrity problem — it would have
meant the Phase 2 comparison table (E-014) compared agents measured on two genuinely
different environments.

**How it was found not to be a real problem.** The two hashes were traced to two git
revisions, and `git diff` between them showed exactly one changed line, entirely inside a
YAML comment:

```diff
-  # Widened from [101..105] to [101..130] on 2026-08-16 (D-019). Five seeds
+  # Widened from [101..105] to [101..130] on 2026-08-17 (D-019). Five seeds
```

Both config files were 4,991 bytes. No environment parameter changed at all — only a date
correction inside a comment describing an unrelated decision.

**Root cause.** `runner.config_hash()` hashes the raw bytes of the YAML file, not the
parsed and normalised configuration. Comments, whitespace, and key order all move the hash,
because the hash is of **the file**, not of **the settings**.

**Why this is documented and deliberately not fixed.** The obvious fix — hash the parsed,
normalised config instead — was considered and rejected on three grounds: it would change
the hash of every future record while every existing record kept its old hash, so the two
would never compare equal again, turning a one-time cosmetic false positive into a
permanent one; `config_hash` is Phase-0 code that every later phase's records already
depend on, and changing it to make a Phase 5 module happier is building backwards
(CONSTRAINTS #18); and the guard in `pairs.py` was, in fact, correct twice over — it caught
a genuine (if harmless) config split, and separately it refused a record set
(`results/runs`, the eval-seed archive) that Document 5's own rules forbid building pairs
from in the first place.

**The asymmetry that makes this an acceptable bug to leave alone**, stated in the bug file's
own words: a hash that only ever errs toward *"these might differ, go check"* is the safe
direction for a scientific-integrity guard. A **false "same"** — two genuinely different
configs hashing identically — cannot happen, because identical bytes really are an
identical config. Erring toward over-caution costs five minutes of investigation; erring
the other way could not be checked at all.

---

## 6. The negative results — the heart of this document

The project's founding thesis is *"Don't just trust the AI. Trace it."* This section is
where that thesis is demonstrated with real numbers rather than merely stated as a
principle. Every finding below states what was expected, what happened, how it was
diagnosed, and what changed as a result — Document 1 §11 already lists nine of these in
one page; this section gives each its full treatment.

### The Huber delta collapsed twenty runs, and the loss curve never showed it (E-016)

Already covered as BUG_002's root cause above — included here because it is the single
clearest instance in the whole project of a result that looked like a training failure and
was actually a measurement failure. **The lesson worth carrying into a viva verbatim:**
"the loss curve looked excellent — converging to 0.04. The network was healthy by every
structural check. The only visible symptom was a number in a results table being bad."

### REINFORCE's greedy policy degenerates before it converges, and clears again by the full budget (E-018, E-019, E-022)

**What was expected.** A policy-gradient method training for 300 episodes should be
partway toward something interesting, not finished.

**What happened.** At 300 episodes, REINFORCE's greedy policy matched severity-sort **to
every digit reported** — recall 0.8443, reward 40.44, both exact. `baselines.py` defines
severity-sort as a constant policy (`PULL_HIGHEST_SEVERITY`, unconditionally); the trained
agent's greedy policy chose that same action 1,131 times out of 1,131 steps across all 30
eval episodes.

**How it was diagnosed.** Checked against CONSTRAINTS #5 before being written down: not a
bug — two policies emitting the same action in every state produce identical trajectories
on identical seeds by construction, so identical metrics are the correct outcome, not a
coincidence. What was genuinely open was whether 300 episodes was a way-point or a
terminus.

**What the full run then found (E-022), and why the earlier reading was incomplete.**
**Sixty-seven times more training produces the same answer** — three of five full 20,000-
episode runs matched severity-sort to four decimal places, two of them with the policy
gradient having fallen to exactly **0.00** (a saturated, deterministic policy with nothing
left to push). But E-022 also traced *when* this happens inside a single run: repeat 0's
greedy diagnostic reads −515.4 (the BULK_CLOSE collapse signature) at episodes 1000 and
1500, then **+37.5 by episode 2000**, after which it never returns to collapse. **The
evidence E-019 measured its greedy-vs-sampled divergence on (1500 episodes) was measured
inside exactly this transient window.** Stated as E-022 states it: *"the evidence D-036
was decided on was a transient, at least for REINFORCE."* The decision itself was still
taken correctly — before any full run existed, so it cannot have been chosen to favour a
number — and it happens to cost nothing here because both readings converge by 20,000
episodes.

**What changed.** D-033's prediction — *"policy gradient rediscovers the human heuristic
and stops there"* — became a measured outcome rather than a hypothesis. Phase 4 closed
built-but-not-passed for REINFORCE specifically because of this pattern.

### The actor-critic's greedy/sampled divergence is permanent, not a transient — the opposite of REINFORCE (E-023)

**What was expected**, following E-022's finding: that a longer run would resolve the
actor-critic's early greedy/sampled gap the same way it resolved REINFORCE's.

**What happened.** It did not. Across all five full 20,000-episode runs, sampled recall
was **0.6316 ± 0.0405**; greedy recall, read from the identical five agents at the identical
moment, was **0.0022 ± 0.0027** — indistinguishable from the Phase 3 BULK_CLOSE collapse.
The greedy diagnostic sat at exactly −515.4 at every progress print from episode 500 to
20,000, on every one of the five repeats, for the entire run.

**How it was diagnosed.** The mechanism is the entropy bonus doing precisely what it was
built to do (D-034): a policy paid to stay spread out never sharpens, so its argmax never
stops being an arbitrary tie-break among near-equal probabilities. This is the opposite
outcome from REINFORCE, which has no entropy bonus and is free to converge.

**What changed, and why it matters more than either finding alone.** *"Had Phase 4 been
reported through the `_GreedyView` the code shipped with, the actor-critic would have been
written up as a total training failure. It is not one."* Read together, E-022 and E-023
sharpen D-036 into a claim narrower than either experiment alone could support: the
sampled/greedy divergence is **transient** for a policy-gradient method left to sharpen on
its own, and **permanent** for one held open deliberately by an entropy bonus. Neither
E-019 (1,500 episodes) nor E-020 (80 episodes) could have told these two cases apart —
both were measured inside windows too short to distinguish "still converging" from "held
open on purpose."

### The textbook baseline variance reduction did not replicate; bootstrapping's did (E-021)

**What was expected.** Sutton & Barto §13.4: subtracting a learned baseline from the
policy-gradient return reduces the estimator's variance.

**What happened, measured at a budget deliberately large enough to trust (30 episodes per
condition, ~4,000+ samples of the coefficient per condition):**

| condition | coefficient std | ratio |
|---|---|---|
| REINFORCE, no baseline | 146.94 | — |
| REINFORCE, with baseline | **147.68** | **1.00×** — no reduction at all |
| Actor-critic (bootstrapped) | **30.89** | **4.78×** — the textbook reduction, but from a different mechanism |

**How it was diagnosed.** The baseline's variance reduction is only guaranteed when
`b(s)` is close to `E[G_t | s]` — the identity that makes it unbiased holds for *any*
function of state, including a useless one, but a baseline merely uncorrelated with the
return reduces nothing. At 30 episodes the value head has seen 30 fresh shifts and has not
learned an accurate `v(s)` yet. **The theory is not contradicted; its precondition is not
yet met.** A companion unit test, `test_a_trained_baseline_centres_the_update_coefficients`,
feeds the identical episode 40 times and shows the coefficients collapsing by more than
half — proving the *code* implements the baseline correctly, while E-021 shows the real
*environment* does not let it pay off inside this budget. Both are true simultaneously, and
the log is explicit that neither should be quoted without the other.

**Why bootstrapping's reduction is different in kind, not merely in degree.** `G_t` sums
roughly 50 noisy rewards; a TD error is one reward plus one estimate. Even a poor critic
produces a narrower coefficient, because the summation that creates the spread never
happens at all. The 4.78× is structural — it does not have to be *earned* through training
the way the baseline's reduction would have to be.

**What changed.** Nothing in the code — this is a reported negative result, not a bug.
`ROADMAP.md` box 4 records it as measured rather than filled in with the flattering
prediction, and it is explicitly called out as **"a negative result on the half of the
theory everybody quotes, and the better interview answer for it."**

### The DQN loses to tabular Q-learning, and the phase's own premise was still confirmed (E-017)

**What was expected.** Phase 3's exit criterion: DQN matches or beats tabular Q-learning on
the same evaluation seeds.

**What happened.** Recall 0.48 ± 0.19 against tabular's 0.73 — a gap of 6–8 standard
errors, clearly resolvable. The learning curve had genuinely **plateaued** (last quarter
18.5 against the previous quarter −14.6, a difference smaller than the run-to-run spread of
±150.0): more episodes would buy nothing. **This is a converged agent that is worse than
the lookup table it was meant to improve on, not an undertrained one.**

**How it was diagnosed as a real result rather than an artefact.** The paired-per-seed
protocol (D-028) found the reward comparison itself unresolvable — |mean|/SEM of 1.42,
below the project's own resolvability bar — so the gate fails specifically on recall and
MTTD, not on reward, and the log states that distinction rather than picking whichever
framing sounded more damning.

**What the phase established anyway, and why it is not simply a failure.** In **21 of 42
visited buckets**, the DQN chose a different action for situations the 576-bucket
discretisation merges into one state. The continuous representation genuinely distinguishes
cases the buckets cannot — the phase's founding hypothesis was correct. It simply did not
convert into a better policy at this training budget. Stated in the log as the entry's most
important sentence: *"the hypothesis behind the phase was correct and the phase still
failed its gate."*

### Two DQN ablations reversed the textbook expectation, in opposite directions (E-017)

Replay switched off produced **total** collapse — recall 0.0000 on all 8 seeds, identical
to four decimal places, indistinguishable from the E-016 BULK_CLOSE signature. **Replay is
not a refinement in this environment; it is load-bearing.** Removing the target network,
meanwhile, made the agent **better** — recall 0.588 against the control's 0.481, reward
+43.5 against −46.9, both differences clearing the project's ratio-2 resolvability bar.
Textbook DQN teaching treats the target network as a stabiliser; here it is measurably
counterproductive, and the log states the mechanism only as an unverified hypothesis
(a frozen target may be stale enough on a small, 19,461-parameter network to hold the
estimate back without the divergence risk the target network exists to prevent) rather than
as an established explanation — per E-015's lesson, a plausible story is not the same as a
tested one.

### `reinforce@1` was nearly discarded as a duplicate, and turned out to be the best policy in the pool (D-045, E-024)

**What was expected.** Three of REINFORCE's five training repeats matched severity-sort's
recall to four decimal places on the *evaluation* seeds (E-022), which made "discard the
three clones, keep the other two" look like the obvious rule for choosing which repeat
enters the RLHF pair set.

**What happened when the check was actually run, on the *pair* seeds rather than the eval
seeds.** Only **two** of the three reproduce severity-sort's actions exactly —
`reinforce@2` and `reinforce@4`. `reinforce@1` is not a clone at all: it scores recall
**0.8947**, the highest in the entire 21-variant pool, above even severity-sort's 0.8530.
Equal *eval-seed* recall had been silently read as equal *behaviour*, and the two are not
the same claim.

**How it was diagnosed.** By measuring action traces — every decision, in order — on a
seed block the eval-seed comparison had never touched, rather than trusting a metric
computed on a different set of shifts to describe behaviour on this one.

**What this changes, and what it does not.** It does not overturn E-022's finding that
policy gradient can converge onto a hand-written heuristic — that finding is arguably
strengthened, since the collapse reproduces on a second, independent seed block. **What it
corrects is the count and the consequence:** a rule chosen in advance — either "median
repeat by recall" (which lands on 0.8530, a clone) or "discard everything that looks like
severity-sort" (which would have discarded the best policy in the pool) — would have gotten
this wrong in both directions, and neither error would have raised any alarm. The
methodological point, stated plainly in the log: **a rule decided before measuring is not
a substitute for measuring**, and here the measurement cost two minutes of inference against
what would have been fifty hours of human labelling spent on a degenerate pair set.

### The evaluation protocol itself was the biggest finding of the whole project (E-008, E-014)

Already covered in detail in §4 above as D-019's motivating evidence — restated here
because it belongs in this section on its own terms: **the discovery that the project's own
measurement instrument was too weak to support its conclusions is a negative result about
the project's methodology, and it is the negative result with the largest downstream
consequences of any in this log.** Every number in Phases 0–2 that predates 2026-08-17 was
computed correctly and reported with its own standard deviation — the process CONSTRAINTS
#3 asks for was followed exactly — and one of those numbers still had the wrong sign.

---

## 7. The six phases

`ROADMAP.md`'s own status line, current as of the last update, states the pattern plainly:
four of six phases closed **built-but-not-passed** — every work item complete, tested, and
independently verified; the stated exit criterion measured honestly and found unmet; the
gate left standing rather than rewritten to match the result.

| Phase | Status | What was built | Why the gate did or did not pass |
|---|---|---|---|
| **0 — Foundation** | ✅ Passed | Simulator, config, env, baselines, metrics | Oracle strictly best on total reward, 168.0 vs 40.4 (30-seed). One amendment's *rationale* weakened by E-014, the criterion itself holds. |
| **1 — DP** | ⛔ Built, criterion falsified | Model estimation, VI/PI, the hand-worked MRP | Amended once legitimately (D-012, a category error in the original wording); falsified on better measurement (D-019/E-014, DP scores −201.2 not +305.9); **not** amended a second time (D-022) |
| **2 — Tabular model-free** | ⛔ Built-but-not-passed | Monte Carlo, SARSA, Q-learning, the tiny-MDP anchor | All three learners lose to severity-sort on recall (0.66–0.72 vs 0.84); the "interpretable strategy shift" claim withdrawn (E-013) when SARSA reversed it |
| **3 — DQN** | ⛔ Built-but-not-passed | The Q-network, replay buffer, target network, the two required ablations | Recall 0.48 vs tabular's 0.73, a resolvable 6–8 SEM gap; the phase's premise (continuous state distinguishes merged buckets) held even as its gate did not |
| **4 — Policy gradient** | ⛔ Built-but-not-passed | REINFORCE, actor-critic, the variance demonstration | Fourth consecutive phase to close this way (D-033 governs); sample-efficiency ordering clear (9,273 / 294,545 / 1,567,392 steps to 40.4), but none of the three learners beats severity-sort's reward by more than its own spread |
| **5a — Collect preferences** | ✅ Code complete, result pending | The full RLHF data layer, `generate_pairs.py`, the labelling UI (Document 5, in full) | Nothing failed — the phase's result is 350 human judgements, and as of this document none exist yet. This is the only phase whose remaining work is human time, not code. |
| **5b/5c — Reward model, re-train** | ⬜ Not started | — | Blocked on 5a's labels |
| **6 — Audit and delivery** | ⬜ Not started | — | Blocked on 5b/5c |

**The pattern across Phases 1–4 is itself the reportable finding**, and D-033 is the
decision that settles how the project reports it. Rather than restate each failed criterion
after seeing its result — which D-033's own reasoning calls "the exact failure this project
exists to avoid" — the criteria stay exactly as written, and the honest failures become the
report's spine. D-033's consequences field draws the distinction that matters most: this
does **not** mean the phases failed as engineering. Every work item in Phases 1–4 is
genuinely complete, independently verified, and each one produced a real finding (Phase
1's reward exploit two phases early; Phase 2's discovery of its own evaluation protocol's
weakness; Phase 3's replay-is-load-bearing / target-network-is-harmful pair; Phase 4's
severity-sort rediscovery and the transient-vs-permanent greedy/sampled divergence). **Four
consecutive unmet criteria is a finding about how the criteria were written, not a verdict
on the code that was built to meet them.**

**Why D-012's amendment was legitimate and D-020/D-022's refusal to repeat it was also
correct — the distinction that makes the whole sequence defensible rather than
inconsistent.** D-012 fixed a genuine category error: the original Phase 1 criterion asked
a reward-maximising algorithm (DP) to top a metric (recall) it does not optimise. That is a
flaw in the *criterion*, correctable once, without it being goalpost-moving. No such flaw
exists in Phases 2, 3, or 4 — their criteria ask the learners to beat severity-sort on
metrics those learners were, in fact, trying to optimise, and on an honest measurement they
simply did not. Amending any of those gates after the result was known would have been
tuning the criterion to the outcome, which is precisely the failure this project's entire
documentation discipline exists to prevent — and D-020, taking that decision for Phase 2
first, explicitly names and rejects the single most tempting version of it: restating the
gate on reward *consistency*, where the learners genuinely do win (±50 vs ±220), because
"nobody set out to optimise variance" before seeing that number.

---

## 8. Things that will confuse you

| What you'll notice | Why it's like that |
|---|---|
| Four of six phases are marked "built-but-not-passed", not "failed" | Every work item in those phases is genuinely complete and independently verified; only the stated exit criterion was not met. D-033 makes this distinction explicit and permanent rather than letting "phase closed unpassed" read as "phase abandoned." |
| Phase 1's exit criterion was amended once (D-012) but a near-identical situation in Phase 2 was **not** amended (D-020) | D-012 fixed a genuine category error in the original wording (asking a reward-maximiser to top a metric it doesn't optimise). Phases 2–4 have no such error — the learners simply didn't do the thing their criteria describe, so amending would be tuning the criterion to the result. |
| `mrp_example.py` and `tiny_mdp.py` have hard-coded numbers, which looks like a CONSTRAINTS #9 violation | D-013/D-014's narrow, named exception: these are pen-and-paper derivations whose entire value depends on the exact numbers matching a companion document. Both docstrings point back to their D-number so this isn't rediscovered as a bug every session. |
| The DQN's target network made the agent **worse**, and removing it improved every measured metric | E-017's genuinely measured, resolvable result (ratio 2.25–2.51 against the project's own bar). The mechanism is stated only as an untested hypothesis — a stale frozen target may be holding back a small, low-divergence-risk network — because a plausible story is not a tested one (E-015's lesson). |
| REINFORCE's greedy policy looks broken at 1,500 episodes and healthy at 20,000 | E-022 found the collapse-to-severity-sort window is a transient, not a limit — the greedy diagnostic passes through the BULK_CLOSE value around episodes 1000–1500 and clears by 2000. Any number quoted from inside that window (E-019, E-020) describes an undertrained policy, not a property of policy gradient in this environment. |
| The actor-critic's sampled and greedy readings disagree **permanently**, not just early on | The opposite mechanism from REINFORCE: an entropy bonus deliberately keeps the policy spread, so its argmax never stops being an arbitrary tie-break. D-036's sampled-is-the-headline rule is what makes this readable rather than reported as a training failure. |
| `config_hash` fired a false alarm over a one-character comment edit, and the fix was rejected | BUG_005: hashing the raw file bytes is deliberately over-sensitive rather than under-sensitive — a false "these differ, go check" costs five minutes; a false "these are the same" cannot happen at all and would be far worse to miss. |
| The commit-balance script (`commit_balance.py`) explicitly says it must never be used to attribute one person's work to the other | CONSTRAINTS #24: the git history is part of what gets evaluated, and a commit under a name is a claim that person can explain it. The balance is corrected by *handing over the work*, never by re-labelling who did it. |
| Some decisions are marked "Diya countersign: pending" months after being taken | D-012, D-019, D-020, and D-022 all change what the report is allowed to claim, and the project's own convention (both Phase 0 amendments carried explicit dual sign-off) sets that as the bar these larger changes should also clear — recorded honestly as outstanding rather than quietly assumed. |
| Two ablations in Phase 3 both "disproved" what they were built to demonstrate | `no_replay` was supposed to show instability and instead showed total collapse (BUG_003 — a flatline reads as maximum stability to a variance statistic); `no_target_network` was supposed to show the target network helps and instead showed it hurts. Both are reported as measured, not as the textbook expectation. |

---

## Where to go next

- **`DECISIONS.md` and `docs/experiments/EXPERIMENT_LOG.md`, in full.** This document
  expands the entries most load-bearing for understanding the project as a whole; roughly
  half the log's 47 decisions and 24 experiments are not named here at all, and every claim
  above is traceable back to its D-number or E-number in those files directly.
- **`docs/bugs/` and `docs/features/`**, the two folders this document draws its bug
  histories and feature derivations from — each holds the full write-up a summary here
  necessarily compresses.
- **Document 1 §11**, the one-page version of §6 above, useful as a quick-reference index
  into the same nine findings before returning here for the full treatment of any one of
  them.
- **`ROADMAP.md`**, read alongside §7 above, to see the phase statuses exactly as the
  project itself tracks them — including the checkboxes, the strikethrough superseded
  verdicts, and the cut-order list for what would go first if time ran out.
- **This is the last document in the reading order.** A reader who has worked through
  Documents 0–6 has now covered every file the onboarding set was built to explain; the
  code itself, read alongside these six documents, is the rest of the tour.

---

*Source of truth: this Markdown file. The `.docx` export is generated from it. If they
disagree, the Markdown is right.*
