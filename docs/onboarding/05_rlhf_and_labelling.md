# Document 5 — RLHF and Labelling

### What replaces the hand-written reward, and how it stays honest while doing it

> **Read Document 4 first.** This document assumes you know what an `EpisodeRecord`
> is, who writes one (`runner.run_episode`), and that `evaluation/metrics.py` is one
> of two consumers of it. This is the other one.
>
> **The promise of this document:** by the end you should be able to say, for any
> field a human labeller sees or does not see, *which file removed or kept it, and
> why removing it — or keeping it — is the thing that makes the 300 preference
> judgements mean something.* That is the whole of Phase 5a's engineering, and it is
> almost entirely about subtraction: less reaches the screen than the environment
> actually knows, and the file that decides what stays is the one worth reading twice.

---

## Contents

| § | What it covers | File(s) |
|---|---|---|
| 1 | Why replace the reward at all | *(motivation)* |
| 2 | The pipeline end to end | *(map)* |
| 3 | `pairs.py` — building 300 blinded comparisons | `src/soc_triage/rlhf/pairs.py` |
| 4 | `summary.py` — the blinding layer | `src/soc_triage/rlhf/summary.py` |
| 5 | `render.py` — the page a labeller actually reads | `src/soc_triage/labelling/render.py` |
| 6 | `app.py` and `store.py` — the two routes and the one irreplaceable file | `src/soc_triage/labelling/app.py`, `src/soc_triage/rlhf/store.py` |
| 7 | `queue.py` — who sees what, and where they left off | `src/soc_triage/labelling/queue.py` |
| 8 | `agreement.py` — Cohen's kappa, by hand | `src/soc_triage/rlhf/agreement.py` |
| 9 | Bradley–Terry — the model the labels will train (not yet built) | *(Phase 5b)* |
| 10 | Things that will confuse you | *(traps)* |

---

## 1. Why replace the reward at all

Document 1 §5 already made this case once; this section restates it in one page
because everything below only makes sense once the motive is clear.

Every number in `config/env_default.yaml`'s reward block was **invented**. Not
estimated, not calibrated against a real SOC, not derived from any external source —
typed in by two students who had to pick *something*. Catching a real incident is
worth +100. Missing one costs −200. Bulk-closing a batch of low-severity alerts
earns +5.0 per alert; letting a real incident get buried inside a bulk close is
charged separately and far more harshly. Nobody knows if those ratios are right, and
Document 3 §4's DP result is the sharpest demonstration of why that matters: the
policy that maximises this reward **best** — nearly perfectly, in fact, using
`agents/dp.py`'s exact planning — turned out to bulk-close ~97% of the time and
abandon 57% of real incidents while earning the highest total reward of any agent
measured in the whole project. The reward is exploitable, and every agent from
Phase 1 onward found some version of the same exploit, because they are all doing
exactly what they were told: maximise this number.

**So Phase 5 stops asking the reward function for its opinion and starts asking
people for theirs.** Not "how many wasted minutes equal one missed breach" — nobody
can answer that with a number — but the much easier question a security manager
answers every day without thinking about it twice:

> *"Here are two shifts. Same alerts, handled two different ways. Which one went
> better?"*

`docs/onboarding/00_labelling_handbook.md` §1 puts it the same way, because this is
the one idea the whole onboarding set repeats from two directions: the handbook says
it to the person about to click a button, and this document says it to the person
about to read the code that makes clicking that button meaningful.

**The reward numbers are invented; the 300 comparisons this document builds are the
replacement, and everything else in Phase 5a exists only to make those comparisons
trustworthy.** Two threats to that trust recur through every file below and are
worth naming once, up front, because they explain almost every design decision that
follows:

1. **If a labeller can see the reward, they will just agree with it.** The judgement
   collected would not be independent; it would be a read-back of the very numbers
   Phase 5 exists to replace, and a model fitted to it would rediscover
   `env_default.yaml` at considerable expense.
2. **If a labeller can see which policy is which, they will judge the policy's
   reputation instead of the outcome.** "Oh, this is the neural network" is a bias
   that has nothing to do with whether the shift was actually handled well.

Every guard in `rlhf/summary.py`, `rlhf/pairs.py`, `labelling/render.py` and
`labelling/queue.py` traces back to one of these two threats. Reading this document
is largely reading the same two-sentence worry, engineered five different ways.

---

## 2. The pipeline end to end

Document 4 §1 left one arrow unexplained — the branch out of `EpisodeRecord` into
"rlhf/ + labelling/ (a DIFFERENT consumer of the same records)". This is that branch,
drawn in full:

```
  9 trained policies, run on the PAIR seed block (12 seeds, own D-016 block)
         |
         |  runner.run_episode()  — same function Document 4 §2 covers,
         |  called once per (policy, seed) — 108 EpisodeRecords total
         v
   results/rlhf/records/*.json        (bare policy name, e.g. "dqn" not "dqn@0")
         |
         |  rlhf.pairs.build_pairs()   (§3 below)
         |    - picks 300 (policy, policy, seed) triples, every pair sharing
         |      one seed
         |    - rlhf.summary.summarise_episode()  strips every reward field  (§4)
         |    - strips the one remaining name-bearing field, run_id
         v
   +------------------+          +----------------------+
   | pairs.json       |          | pairs_key.json       |
   | NO policy names  |          | pair_id -> policy     |
   | what the UI reads|          | names + side + seed   |
   +------------------+          | ANALYSIS ONLY (D-038) |
         |                       +----------------------+
         |
         |  labelling.queue.load_pairs()   refuses to open the key file by name
         v
   labelling.queue.assign()      50 shared + 250 round-robin -> 175 each (§7)
         |
         v
   labelling.app  (FastAPI, two routes)         (§6)
         |
         |  GET /   -> labelling.render.render_pair_page()   (§5)
         |  POST /label -> rlhf.store.LabelStore.add_label()
         v
   results/rlhf/labels.db          (SQLite, gitignored, IRREPLACEABLE)
         |
         |  rlhf.agreement.cohens_kappa()   over the 50 shared pairs   (§8)
         v
   Cohen's kappa                    a number the report cites, or an honest
                                     "undefined" — see §8
         |
         v
   (Phase 5b, NOT YET BUILT)   Bradley-Terry reward model r-hat(state, action)
                               trained on all 300 labelled pairs             (§9)
```

**The one-sentence version of this document:** every file from `pairs.py` down to
`app.py` removes something — a reward number, a policy name, a name-bearing field
nobody thought to check — and the file that removes the *most important* thing
(`summary.py`, stripping every reward) is also the file every other file in the
pipeline trusts to have done it correctly, which is why §4 gets the most space of
any section here.

**Nothing above this diagram belongs to Phase 5 at all.** The nine policies, the
environment, the runner — all of it is Documents 2, 3 and 4's territory, unmodified.
CONSTRAINTS #11 ("nothing from a later phase may be required to run an earlier one")
cuts both ways here: Phase 5 depends on Phases 0–4 having produced trained policies,
but nothing in Phases 0–4 depends on Phase 5 existing, and `rlhf/pairs.py`'s own
module docstring makes the same claim from the data side — *"this module imports no
agent, no environment and no torch... pair construction works on a clone with no
`results/` and no `torch`."* A grader could delete every file this document covers
and every earlier phase would still reproduce exactly.

---

## 3. `pairs.py` — building 300 blinded comparisons

289 lines. The problem this file solves, stated by its own docstring before anything
else is:

> *"Each pair is two policies working the same shift — same seed, same alert
> stream, same config — so that whatever a labeller prefers is a property of the
> policies rather than of the luck of the draw."*

**Why the same seed matters, worked through concretely.** If the left pane ran on a
quiet 20-alert shift and the right pane ran on a brutal 90-alert shift with three
simultaneous crown-jewel incidents, a labeller who preferred the left pane would be
telling you nothing about the two policies — only that quiet nights are easier than
busy ones. Holding the seed fixed removes that confound completely: both panes see
the identical Poisson-generated alert stream, the identical severities, the
identical ground truth. The only thing that can differ between the two panes is the
*decision* made at each step. `rlhf.pair_must_share_seed` names this convention in
config (`training_default.yaml` line 205), and it is checked, not merely assumed —
`build_pairs` raises `PairBuildError` if it is ever violated:

```python
if left["seed"] != right["seed"]:
    raise PairBuildError(
        f"pair {i} has seed {left['seed']} on the left and "
        f"{right['seed']} on the right; the two sides must be the "
        "same alert stream"
    )
```

**This check runs per pair, inside the loop that builds each one — not once,
trusting the indexing above it.** The indexing (`_index_records`, keyed by
`(agent_name, seed)`) should make a mismatch impossible by construction, and the
assertion is there anyway. That is the project's recurring habit — Document 4 §5's
`run_dp.py` does the same thing with its VI/PI cross-check — applied to a place where
"impossible by construction" has never once been good enough on its own to skip
checking.

### Two files come out, and the split is the whole point

```
   pairs.json       what labelling.queue.load_pairs() reads.
                    NO policy names, anywhere, checked by
                    test_the_pairs_file_contains_no_policy_name_anywhere.

   pairs_key.json   pair_id -> {left_policy, right_policy, run_ids, swapped, seed}
                    ANALYSIS ONLY. Needed later to compute which policy a
                    labeller actually preferred; never read by the labelling page.
```

Splitting into two files rather than one file with a "hide this part" flag is a
deliberate choice, and it is the same shape of decision `runner.py`'s
`_alert_to_dict` makes for the opposite reason (Document 4 §2): a field that is
*present but supposed to be ignored* depends on every future reader remembering the
rule. A field that **does not exist in the file the UI reads** cannot leak no matter
what a future change to the UI does with it.

### The nine policies, and one deliberate absence

```yaml
policies:    # 9 policies, 36 unordered pairings. oracle_greedy is excluded
             # on purpose: it reads ground truth, so its pairs are foregone
             # conclusions and the Bradley-Terry gradient there is ~0.
  [random, severity_sort, dp, q_learning, sarsa, monte_carlo, dqn, reinforce,
   actor_critic]
```

The oracle from Document 3 §3 — the one baseline sanctioned to cheat, "never
presented as a result" — is absent here for a reason specific to this phase.
`tests/test_rlhf_config.py::test_including_the_oracle_is_refused` pins it: including
it would let the config load, run, and quietly waste roughly a ninth of the labelling
budget on comparisons nobody's judgement could move, because the oracle's advantage
over every other policy is structural rather than a matter of taste.

**The arithmetic behind `n_pair_seeds: 12` and `target_pairs: 300`**, worked through:

```
   9 policies  ->  C(9,2) = 36 unordered pairings   (itertools.combinations)
   36 pairings x 12 seeds = 432 candidate (pairing, seed) combinations
   target_pairs = 300  <=  432   -- comfortably inside capacity
```

`test_the_shipped_target_fits_in_the_shipped_capacity` checks exactly this
arithmetic against the shipped config, and `test_too_few_seeds_for_the_target_is_
refused_with_the_arithmetic` checks the refusal on the other side: ask for 40 pairs
out of a 6-pairing x 5-seed = 30-candidate pool and `InsufficientRecordsError` names
both numbers in its message, because — as the docstring on `_eligible_seeds` puts
it — the fix is always either "run more seeds" or "ask for fewer pairs," and the
caller needs to know which without re-deriving the arithmetic themselves.

### `_allocate()` — spreading 300 pairs across 36 pairings without favouring the alphabet

```python
base, remainder = divmod(target, len(pairings))
counts = {pairing: base for pairing in pairings}
for pairing in rng.sample(pairings, remainder):
    counts[pairing] += 1
```

300 pairs over 36 pairings is 8 each with 12 left over (`300 = 36*8 + 12`). Handing
those 12 to whichever pairings happen to sort first alphabetically would
systematically favour comparisons involving `actor_critic` — the docstring names
this exact failure mode, *"otherwise `alpha`-vs-anything would be systematically
over-represented by an accident of the alphabet"* — so the remainder is drawn as a
random subset instead. **Why balance matters at all, and it is not cosmetic:** Phase
5b's Bradley–Terry fit (§9) needs direct evidence for every pairing; a pairing that
never appears leaves the model to *infer* the comparison transitively through
whatever chain of other comparisons connects the two policies, which is strictly
weaker evidence than a direct one.

### `_choose_double_labelled()` — spreading the 50 shared pairs the same way

The 50 pairs Cohen's kappa (§8) will be computed on cannot simply be "the first 50 in
file order," for the same reason the allocation above cannot favour the alphabet:

> *"Taking the first N in file order — or a flat random sample — would let the
> double-labelled set concentrate on a few kinds of comparison, and Cohen's kappa
> would then describe agreement on those rather than on the task."*

The method is round-robin across pairings — one from `(dqn, sarsa)`, one from
`(random, dp)`, and so on, cycling through every pairing before taking a second from
any of them — which is what guarantees the widest possible spread the requested
count allows. `tests/test_rlhf_pairs.py::test_double_labelled_pairs_are_spread_
across_policy_pairings` checks this with a synthetic fixture (6 double-labelled pairs
spread across at least 4 of 6 pairings) rather than trusting the description.

### The swap — and why `run_id` is the one field that has to be stripped separately

```python
swapped = rng.random() < 0.5
left_policy, right_policy = (second, first) if swapped else (first, second)
```

Without this, whichever policy sorts first alphabetically — `actor_critic` before
`dqn` before `sarsa` — would always land on the left pane, and a labeller who
developed even a mild "left is usually decent" habit across 175 pairs would be
measuring a position bias, not a policy preference.
`test_both_sides_are_used_for_the_first_policy_across_the_set` pins that the swap
actually fires both ways across a real build, not merely that the code path exists.

**`_blind()` is four lines and it exists for one field:**

```python
def _blind(summary: dict) -> dict:
    """Strip the one field in a summary that names the policy.

    `run_id` reads `sarsa-seed3000004`. It is useful provenance and it is
    exactly what must not reach a labeller, so it lives in `pairs_key.json`
    instead. Everything else in a summary is already policy-agnostic.
    """
    return {key: value for key, value in summary.items() if key != "run_id"}
```

**`run_id` is the trap in this file, and it is worth sitting with why.** Every other
field `summarise_episode` (§4) produces is already policy-agnostic by construction —
a timeline of actions, an outcome block, nothing that names who chose those actions.
`run_id` is the one exception, because it was built as a debugging convenience
(`"sarsa-seed3000004"`, so a human staring at raw JSON on disk can tell at a glance
which run produced it) and nobody designed it to travel to a labelling page.
`tests/test_rlhf_pairs.py::test_the_pairs_file_contains_no_policy_name_anywhere`
checks this the hard way — a substring search of the *serialised* `pairs.json` text
for every one of the four synthetic policy names, plus the literal string `"run_id"`
— rather than inspecting the in-memory dict, because a substring search of the
written file is the only check that would catch a name leaking through some field
nobody had thought to name in advance.

### Determinism — why this file is a pure function and must stay one

> *"The whole build is a pure function of (records, sampling_seed, sizes). Running it
> twice produces byte-identical output. That is not tidiness: collected labels
> reference `pair_id`, so a rebuild that renumbered the pairs would silently repoint
> every label already gathered, and nothing would raise."*

**This is the single most consequential guarantee in the whole labelling pipeline,
and it is the reason `scripts/generate_pairs.py --write` refuses to overwrite an
existing `pairs.json` without `--force` (§3 below and D-046).** Nothing checks at
label-collection time whether the pair a label references still means the same
comparison it meant when the label was recorded — the label just stores a bare
`pair_id` string. If a rebuild ever renumbered `p0142` from "dqn vs sarsa on seed
3000007" to "random vs actor_critic on seed 3000002," every label already collected
against `p0142` would silently become a judgement about a different comparison than
the one it actually recorded, and nothing in the system would notice or complain.
`test_two_builds_from_the_same_inputs_are_identical` and
`test_record_order_does_not_change_the_result` check byte-identical JSON output, not
merely equivalent content, specifically because the labels reference the exact
serialised `pair_id` strings.

---

## 4. `summary.py` — the blinding layer

147 lines, and this is the single most important file in the module. Everything
else in this document exists to serve, protect or display what this file decides a
labeller is allowed to know.

### The two things it does, and the asymmetry between them

> *"A summary is a report of an episode, not a recomputation of it: the counts come
> straight from `record["outcome"]`... The one thing this module removes is the
> reward."*

**Reporting rather than recomputing matters for a reason worth stating plainly:**
`record["outcome"]` was built by the environment (`env.py:239`) with full knowledge
of ground truth. If `summarise_episode` instead recomputed, say, `incidents_missed`
by counting the timeline it renders, it could disagree with the environment's own
figure — because the timeline does not record which alerts were left sitting
unhandled in the queue at the end of the shift (a documented limitation, noted below
in §10). Any such disagreement would put two different numbers in front of a
labeller with no way for them to know which one was authoritative.

### The whitelist, and why it is a whitelist and not a blacklist

```python
# The alert fields a labeller needs in order to judge whether an investigation
# was a good use of the shift. `id` is an opaque number that only adds noise on
# screen, and `deadline_min` is a parameter of the reward we are replacing.
_ALERT_FIELDS = ("severity", "asset_criticality", "alert_type", "verify_cost_min")


def _alert_view(alert: dict | None) -> dict | None:
    """Project an investigated alert down to the fields worth showing."""
    if alert is None:
        return None
    return {field: alert[field] for field in _ALERT_FIELDS}
```

**This is a whitelist — four named fields, and nothing else survives — not a
blacklist of the fields being removed, and the distinction is the single most
important design decision in this file.** A blacklist ("show everything except
`reward` and `reward_breakdown` and `total_reward`") would need updating every time
the `Alert` dataclass or the record shape grew a new field, and it would fail
*silently* the moment someone forgot: a new field would simply pass through, because
nothing on the blacklist named it. A whitelist has the opposite failure mode, and it
is the safe one — a field invented next month, one nobody involved in this document
has thought about yet, does not reach the page **by default**, because it was never
added to `_ALERT_FIELDS` in the first place. Getting this wrong requires a positive
act (adding the field to the tuple); getting a blacklist wrong requires only
forgetting, which is the failure mode every guard in this whole codebase is built to
assume will eventually happen.

Two fields are deliberately left off even though they exist on every `Alert`: `id`
(an opaque integer, pure noise on a screen a human is trying to read quickly) and
`deadline_min` — which is not noise, it is a parameter of the exact reward function
this phase exists to replace, and showing it would let a labeller reason in terms of
the reward's own deadline logic rather than their own judgement of urgency.

### The reward strip — the rule this entire feature turns on

```python
outcome = {k: v for k, v in record["outcome"].items() if k != "total_reward"}
```

One line, and `test_total_reward_is_the_only_outcome_key_dropped` pins that it is
*exactly* one key — not zero, not two — dropped from the outcome block. But the
per-step timeline needs its own separate treatment, because a step's `info` dict
carries `reward` and `reward_breakdown` fields that `summarise_episode` never even
reads into the timeline row it builds:

```python
timeline.append({
    "minute": elapsed,
    "action": info["action_name"],
    "alert": alert,
    "was_true_incident": info["was_true_incident"],
    "n_bulk_closed": info["n_bulk_closed"],
})
```

Five keys, and `reward` is conspicuously not one of them — it is not stripped after
the fact, it is simply never picked up in the first place. Document 4 §2 already
showed why `_alert_to_dict` puts the reward *into* the `EpisodeRecord` — because
`metrics.py` needs it and the runner has no way to know which downstream consumer
will need which field. Here, one layer later, is the file that decides a human never
sees it. `test_summary_never_shows_a_reward` checks both directions of the failure
this line-count elides:

```python
def _walk(node, path="summary"):
    """Yield every (path, key, value) in a nested dict/list structure."""
```

**Two separate checks, and both matter, because each catches a different mistake.**
Checking only key names would miss a future renderer that passed the reward number
through under some innocent-sounding key. Checking only values would miss an *empty*
`reward_breakdown` dict today that some later change quietly starts populating —
the key would exist, contain nothing now, and slip past a check that only looked for
the number `235.0` itself. Walking the entire nested structure and asserting neither
failure mode occurs, on a fixture built from the real record shape observed in
`results/runs/severity_sort-seed101.json` (not an invented shape), is what makes
this test worth trusting.

### Ground truth is shown — the claim that is easy to get backwards

> *"Showing ground truth, by contrast, is required here... That is not a
> CONSTRAINTS #1 problem: #1 forbids ground truth in an agent observation, and a
> summary is environment-side output for a person."*

**Say this precisely, because it is the fact in this document most likely to be
misstated in a viva.** The labeller sees exactly which alerts were real incidents and
which were false alarms — `was_true_incident` travels through the timeline unaltered,
and the `caught` list (incidents actually caught) is built directly from it. **What
is hidden is the reward, and the policy's identity — never the outcome.** The
labeller is judging *"did this shift go well?"* with hindsight, exactly the way a
human post-incident review works: you know, after the fact, which alerts turned out
to matter. CONSTRAINTS #1's firewall governs what an **agent** may observe while
acting — Document 2 §8 covers that wall in full — and a summary produced after an
episode has already ended, read by a person rather than fed back into `agent.act()`,
is a completely different kind of object. `test_ground_truth_is_shown_because_
labellers_judge_outcomes` pins the claim directly against the fixture, and
`docs/onboarding/00_labelling_handbook.md` §3.2 states the same rule for the person
about to sit down and label: *"This is intentional and it is not cheating."*

### `render_text()` — a fallback that exists because the roadmap allows it

A second, plain-text rendering lives in this same file, and its docstring is honest
about why it is not the labelling UI:

> *"This is not the labelling UI — it exists so a pair can be eyeballed from a
> terminal while building the pair set, and so a CLI fallback is possible if the web
> page slips."*

Worth noting for what it implies about risk management in this project:
`ROADMAP.md` 5a explicitly allows "even a CLI with rendered text" as a fallback if
Diya's FastAPI page had not landed in time. It did land (FEATURE_012, session 13),
so `render_text` never became the primary interface — but it stayed, because it is
useful for a fast sanity check on a pair set before serving it, and
`test_rendered_text_never_shows_a_reward` holds it to the identical blinding standard
as everything else, because a fallback that leaked what the primary path correctly
hides would be worse than no fallback at all.

---

## 5. `render.py` — the page a labeller actually reads

262 lines, almost all of them guards rather than happy-path markup-building. Its own
docstring states the design principle that makes the second half of this section
possible:

> *"It prints named fields, never 'whatever it was handed.' That is what makes D-039
> hold under change: `summary.py` strips every reward field today, and if some future
> change to the record shape let one through, a renderer that iterated over keys
> would put it on screen. This one would not, because `total_reward` is not in the
> list of things it knows how to draw."*

**This is defence in depth, and it is worth being precise about what it defends
against, because it is a different failure mode than §4's.** `summary.py` is
*supposed* to strip every reward field before this module ever sees a summary. If
that guarantee ever silently broke — a future edit to `summarise_episode` that
accidentally let `total_reward` back into the `outcome` dict — a renderer built the
naive way, looping over `summary["outcome"].items()` and printing each key-value
pair, would put the leaked number on screen with zero additional code required to do
so. This renderer cannot, because every field it draws is named explicitly in the
function that draws it:

```python
def _outcome_block(outcome: dict) -> str:
    """The scoreboard. Every figure is the environment's, none is computed here."""
    return (
        "<ul>"
        f"<li>incidents caught: {_esc(outcome['incidents_caught'])} of "
        f"{_esc(outcome['incidents_total'])} ...</li>"
        ...
    )
```

`tests/test_labelling_render.py::test_a_reward_smuggled_into_a_summary_still_does_
not_reach_the_page` proves this by deliberately poisoning a summary with a
`total_reward` field and asserting the string `"515"` never appears in the rendered
HTML. That test would fail if `render.py` were ever refactored into the "iterate over
whatever keys exist" shape that is more common in quick HTML-templating code, and
that is precisely the point of writing it.

### The second guard — no policy names, and a grep for the file itself

`test_no_module_in_the_labelling_package_mentions_the_key_file` is unusual among the
tests in this project — it is a literal text search over the source files
themselves, not over any runtime behaviour:

```python
for module in sorted(package.glob("*.py")):
    text = module.read_text(encoding="utf-8")
    if "pairs_key" in text and module.name != "queue.py":
        offenders.append(module.name)
```

**The rule being enforced is "never read `pairs_key.json`," and the cheapest durable
way to enforce a rule about what code *does not do* is to check that the string
naming the forbidden file does not appear anywhere it should not.** A behavioural
test can only catch a violation that some code path actually exercises; a grep over
the source catches a violation the instant someone types it, before it is ever run.
The single named exception is `queue.py`, because that is the one place the rule
lives as a positive refusal (§7) rather than an absence.

### Two guards for two threats, checked at the page level too

`test_no_policy_name_appears_in_the_rendered_page` re-checks §3 and §4's blinding one
level further downstream, against the actual served HTML rather than the JSON that
feeds it — and `test_no_run_id_appears_in_the_rendered_page` checks specifically for
the string `"seed3000000"`, because `run_id` is the trap named in §3, and a defence
worth having is worth checking twice at two different layers of the stack.

### The style rule that is easy to miss and matters for the same reason blinding does

```python
_STYLE = """
  ...no colours beyond a light/dark neutral: the two panes must look identical,
  or the styling itself becomes a bias in the comparison.
"""
```

Not a cosmetic note. If the left pane were rendered with a subtly different visual
weight than the right — a slightly bolder border, a warmer background — a labeller's
attention would be drawn unevenly before they had read a single word of either
outcome. The panes are styled identically on purpose, for the same reason
`_blind()` exists: **anything that differs between the two panes other than the
policy's actual decisions is a confound, whether it is a data field or a pixel.**

### `_esc()` — escaping data that is not, today, a threat

```python
def _esc(value: Any) -> str:
    """... this is not defending against an attacker. It defends against the day
    someone points this renderer at text from somewhere else and nobody remembers
    to add escaping then."""
```

Every string on the page today comes from this project's own generator and config,
not from anyone outside it. `html.escape` is applied anyway, because the cost of
escaping is zero and the alternative — adding escaping later, under time pressure,
the day this renderer gets pointed at a different data source — is a real XSS risk
that this file simply never has to become. `test_text_from_the_record_is_html_
escaped` checks it against a deliberately hostile `action` field
(`"<script>alert(1)</script>"`) even though nothing in the current pipeline could
produce one.

---

## 6. `app.py` and `store.py` — the two routes and the one irreplaceable file

`app.py` is 158 lines and its own docstring is blunt about the shape of the file:
*"There is nothing else, deliberately — no login, no edit, no delete, no export."*
`store.py` is 193 lines and its docstring explains why that matters more here than
almost anywhere else in the project:

> *"This file holds the only thing in the project that cannot be regenerated by
> re-running a script. Every other artefact under `results/` is a function of the
> code and a seed; 300 preference judgements are ~100 minutes of somebody's
> attention. That asymmetry is why this module is deliberately strict: it would
> rather raise than accept a row that quietly corrupts the set."*

**Everything else in this section is a consequence of that one asymmetry.** A bug in
`runner.py` costs a re-run. A bug that corrupts `results/rlhf/labels.db` costs 100
minutes of two specific named people's time that can never be gotten back — which is
why this pair of files is written the way it is, and why almost every test against
them is about a way an answer *could* be wrong rather than about the happy path.

### The `Answer` model — an absence is the enforcement

```python
class Answer(BaseModel):
    """One judgement, as the page sends it.

    No `labeller_id` field, and that absence is the enforcement of D-041
    rather than an oversight: unknown keys in the body are dropped, so a
    request cannot claim to be somebody else.
    """
    pair_id: str
    choice: str
    seconds: float | None = None
```

**The labeller id is bound once, when `create_app` is called from the launch
script, and there is no route through which a request can change it.** This is the
same trap `SendMessage`-style multi-tenant systems fall into constantly: a text box
on a form is the *obvious* place to put an identity, and it is exactly wrong here.
`00_labelling_handbook.md` §2.3 states the operational consequence in plain terms —
"If Diya sat down at a browser still showing `L1`, her 175 judgements would be
recorded as Pranav's" — and `test_the_labeller_id_cannot_be_overridden_by_the_
request` proves the code refuses it: a POST body that includes
`"labeller_id": "L2"` is silently dropped by Pydantic before the handler even sees
it, because `Answer` has no field to hold it.

**Why this matters more than an ordinary auth bug would.** Cohen's kappa (§8)
measures agreement between two *independent* people. If one person's answers could
be recorded under the other's name — through a stale browser tab, a copy-pasted
curl command, anything — the double-labelled set would silently contain rows where
one person is being compared against themselves, and kappa would come out
flatteringly high while measuring nothing real. There would be no error, no crash,
no log line — just a number that looks fine and is wrong.

### `_clean_seconds()` — the NaN bug, and why the fix does not rely on luck

```python
def _clean_seconds(value: float | None, max_seconds: int) -> float | None:
    if value is None:
        return None
    if value != value:  # NaN is the only float that is not equal to itself.
        return None
    if value < 0 or value > max_seconds:
        return None
    return float(value)
```

**Name the bug this catches, because it is a real one that shipped and was found
mid-session (D-044).** `nan < 0` and `nan > max_seconds` are both `False` in IEEE
754 — NaN compares false against everything, including itself — so a naive range
check that skipped straight to the third `if` would let a `NaN` value pass straight
through untouched. The docstring is explicit that a client *can* send one: Python's
`json` module accepts the non-spec `NaN` token that strict JSON forbids, so a hand-
crafted request, or a browser bug, can produce it. **The original version of this
function happened to work anyway**, because SQLite's C driver silently converts a
`NaN` float to `NULL` on storage — but that is an undocumented, driver-specific
accident, not a contract this function should be allowed to depend on. The explicit
`value != value` check makes the refusal happen for a *reason* rather than by luck of
which database engine happens to be underneath it, and
`test_nan_is_rejected_by_clean_seconds_itself_not_by_sqlite_luck` checks the pure
function directly, decoupled from any storage engine at all.

### The refresh path — a duplicate is success, not an error

```python
except DuplicateLabelError:
    # A refresh replays the POST. The judgement is already stored and the
    # first answer stands, so this is success from the labeller's point
    # of view: the page moves on. Returning 500 here would interrupt a
    # sitting over work that was never at risk.
    pass
```

**The bug this avoids is subtle: treating an idempotency violation as a server
error.** `UNIQUE (pair_id, labeller_id)` in the schema (below) is exactly correct —
one opinion per person per pair, forever — but the *behaviour* on hitting it matters
just as much as the constraint itself. A double-click, or a browser replaying a POST
on refresh, produces the identical request twice. The first insert succeeds; the
second raises `sqlite3.IntegrityError`, which `store.py` re-raises as the typed
`DuplicateLabelError` specifically so `app.py` can catch it and treat it as "already
done" rather than "something went wrong" — returning `{"ok": True}` either way, so
the labeller's browser simply moves on to the next pair with no visible interruption.
`test_a_replayed_submission_is_not_an_error` and `test_the_first_answer_wins_a_
replayed_submission` pin both halves: the second POST does not crash, and it does
not silently overwrite the first answer either.

### The schema — two constraints that live in SQL, not only in Python

```sql
CREATE TABLE IF NOT EXISTS labels (
    ...
    choice        TEXT    NOT NULL CHECK (choice IN ('left', 'right', 'tie')),
    ...
    UNIQUE (pair_id, labeller_id)
);
```

**Why these two guards are written into the schema itself, and not left to Python
validation alone:** the docstring is explicit that Diya's labelling page writes to
this exact file and *does not go through the `LabelStore` class to do it in every
respect* — the `CHECK` and `UNIQUE` constraints are the ones the database itself
enforces, so a bug in some other future writer, one that never imports
`store.LabelStoreError`, still cannot corrupt the table. `test_the_check_constraint_
is_in_the_schema_not_only_in_python` and its `UNIQUE` counterpart both connect to the
database with a raw `sqlite3.connect` — bypassing `LabelStore` entirely — and assert
the *database itself* refuses a bad row. That is a materially different guarantee
than "the Python wrapper validates it," and this project is careful to test the
stronger claim rather than the weaker one that happens to be easier to write.

**CONSTRAINTS #23, enforced by absence rather than by a rule anyone has to
remember:**

```python
assert columns == {
    "id", "pair_id", "labeller_id", "choice", "created_at", "seconds_taken",
}
```

An opaque labeller id, a choice, a timestamp, how long it took. **There is nowhere
in this schema to put a name, an employer, or a client**, which is the enforcement
CONSTRAINTS #23 asks for — if practitioner labels are ever collected through Diya's
KPMG contacts, no client- or company-identifying information can enter this
repository, because the table this data lands in has no column that could hold it.

### `export_csv()` — a backup path for the one file that cannot be regenerated

The `.gitignore` comment beside `results/rlhf/labels.db` tells the next reader to
keep a CSV export beside it, and this function is why that instruction is
satisfiable. Written even for an empty store, so a backup taken before any labelling
has happened is a valid file with just a header rather than a zero-byte surprise —
`test_export_csv_of_an_empty_store_still_writes_the_header` pins exactly that.
`00_labelling_handbook.md` §5.4 puts the operational instruction plainly: *"Back it
up manually — copy it somewhere safe after each sitting."*

---

## 7. `queue.py` — who sees what, and where they left off

258 lines doing exactly three things, per its own docstring: **assignment**,
**resume**, and **loading, with the blinding guard.**

### The assignment maths, worked through with the real numbers

```
   300 pairs total
   |
   +-- 50 double-labelled  ------->  BOTH labellers see all 50
   |                                 (this is what Cohen's kappa is computed on)
   |
   +-- 250 single-labelled -------->  dealt round-robin, one at a time
                                       -> 125 to L1, 125 to L2

   L1's total: 50 + 125 = 175 judgements
   L2's total: 50 + 125 = 175 judgements
   Combined:   350 judgements over 300 distinct pairs
```

`tests/test_labelling_queue.py::test_the_production_shape_gives_175_judgements_each`
checks exactly this arithmetic at the real production shape — 9 policies, 12 seeds,
300 target, 50 double — rather than only at the smaller fixture size most of the
other tests use, because *"the 125/125 split is the decision under test and it
should be checked at the size it will actually run at, not only in miniature."*

### `assign()` — why it is a plain loop with a counter, not a slice-and-zip

```python
single_index = 0
for pair in ordered:
    if pair["double_labelled"]:
        for name in names:
            allocation[name].append(pair["pair_id"])
    else:
        allocation[names[single_index % len(names)]].append(pair["pair_id"])
        single_index += 1
```

The comment names the reason directly: *"the round-robin and the doubles-go-to-
everyone rule are the decision under discussion in D-040 and should be readable as
one"* — CONSTRAINTS #14's readability rule applied here for the same reason it is
applied to Q-learning's update rule in Document 3: this is a place an examiner might
ask to see, and a clever one-liner would obscure the exact decision being made.
**The counter only advances on single-label pairs**, so the doubles interleaved
among them in file order do not throw off the alternation — without that separation
the split could silently drift from 125/125 depending on where the doubles happened
to land in the sorted file.

### Determinism — sorted by `pair_id`, not trusted to file order

```python
ordered = sorted(pairs, key=lambda pair: pair["pair_id"])
```

`test_the_order_pairs_arrive_in_does_not_change_the_assignment` checks this the same
way §3's determinism tests do — reversing the input and confirming the output is
identical. **The sort does not throw away randomness that mattered**, because
`build_pairs` (§3) already shuffled the draft before numbering it, specifically so a
labeller working through pairs in `pair_id` order would not meet every
`alpha`-vs-`beta` comparison as one block. That shuffle does double duty: it protects
against fatigue-by-clustering for the person labelling, *and* it means this module
can sort deterministically without needing to know anything about which policies any
pair contains — which it must not know, because the pairs are blinded before they
ever reach this file.

### `load_pairs()` — the blinding guard, stated as a named constant so it cannot drift

```python
_KEY_FILENAME = "pairs_key.json"
...
if path.name == _KEY_FILENAME:
    raise PairFileError(
        f"refusing to read {_KEY_FILENAME}: it maps pair_id to policy names, "
        "and the labelling page must only ever read pairs.json (D-038)"
    )
```

**Name the realistic version of this mistake, because it is not malice — it is
copying the wrong file to the wrong folder.** `00_labelling_handbook.md` §3.4 says
the same thing from the other side: *"That guard exists because the realistic
version of this mistake is not malice, it is copying the wrong file into the wrong
folder."* Checked by filename before the file is even opened, so the refusal fires
before any JSON parsing, any field inspection, any chance for the key file's contents
to reach anything downstream at all.

Every other check in `load_pairs` fails at load time rather than mid-session,
deliberately: a malformed pair file discovered at pair 87 of 175 would waste the 86
answers of momentum leading up to it (the answers themselves are safe in the
database — it is the labeller's concentration that is not recoverable). Missing
required fields, duplicate `pair_id`s, a file that is not a JSON list at all — all
three are checked in one pass over the file, before any labeller ever sees a screen.

### `Progress` — one detail that keeps the "of" honest

```python
@dataclass(frozen=True)
class Progress:
    """`total` is this labeller's assignment, not the size of the pair set —
    175, not 300. Showing 300 to someone who will only ever see 175 of them
    would misreport the job as nearly twice its real length."""
```

The page says "pair 4 of 175," never "pair 4 of 300" — a small thing, but showing
the wrong denominator would make a labeller believe they are less than a third of
the way through work they are actually most of the way through, which is exactly
the kind of small honesty this whole phase depends on getting right everywhere at
once.

---

## 8. `agreement.py` — Cohen's kappa, by hand

164 lines, written by hand rather than imported from `sklearn` — CONSTRAINTS #7's
principle that anything a viva might ask us to derive should exist somewhere we can
actually read it, applied to a formula rather than an RL algorithm this time.

### The formula, and the intuition it encodes

```
        p_o - p_e
  kappa = ---------
        1 - p_e
```

`p_o` is how often the two labellers actually agreed — the diagonal of the confusion
matrix, divided by the number of shared pairs. `p_e` is how often they would agree
by pure chance, given each person's own individual habit of saying "left," "right,"
or "tie." **The reason to subtract `p_e` at all**, and the reason raw agreement on
its own would be a bad measure:

> *"Two people who both press 'left' 90% of the time will agree ~82% of the time
> while sharing no judgement at all."*

Raw agreement of 82% sounds like strong consensus. It is nothing of the sort here —
it is two people independently doing whatever their personal button-mashing habit
does, and Cohen's kappa exists precisely to strip that coincidence out and report
only the agreement that exceeds it.

### The worked example — the exact one this project checks its code against

`tests/test_rlhf_agreement.py::test_kappa_matches_a_hand_worked_ten_pair_example`
carries the full derivation in its own docstring, and it is worth reproducing here
because §12 of `_SPEC_AND_STATUS.md` requires a worked numeric example for this
mechanism and there is no better one than the one the test suite itself was built
against:

```
    pair : 0 1 2 3 4 5 6 7 8 9
    A    : L L L L R R R T T T
    B    : L L L R R R T T T L
    agree: y y y n y y n y y n   ->  7 agreements out of 10

    p_o = 7 / 10 = 0.70

    Marginals (by construction, identical for both labellers):
       A:  L 4/10,  R 3/10,  T 3/10
       B:  L 4/10,  R 3/10,  T 3/10

    p_e = (4/10)(4/10) + (3/10)(3/10) + (3/10)(3/10)
        = 0.16 + 0.09 + 0.09
        = 0.34

    kappa = (0.70 - 0.34) / (1 - 0.34) = 0.36 / 0.66 = 6/11 = 0.545454...
```

That number — 0.545, in the "moderate" Landis & Koch band — is what
`describe(cohens_kappa(a, b))` actually prints for this fixture, checked in
`test_describe_reports_the_coefficient_and_the_sample_size`. **Nothing about this
example is a real measurement**; it is a hand-solved anchor built to pin the
arithmetic, the same discipline `tiny_mdp.py` (Document 4 §4) applies to the
learners. The real kappa — over the real 50 double-labelled pairs, once both of you
have finished labelling — does not exist yet, and this document must not be read as
implying it does.

### `p_e` is computed over the *shared* pairs only — the classic way to get this wrong

```python
rate_a = sum(1 for pair_id in shared if labels_a[pair_id] == choice) / n
```

**Name the mistake this line exists to prevent.** If a labeller's marginal rate were
computed over *every* pair they answered — 175, not the 50 they share with the other
labeller — the `p_e` produced would describe a comparison that was never actually
made. `test_marginals_are_computed_over_the_shared_pairs_only` pins a worked case
where the two denominators genuinely disagree (a labeller's "tie" rate is 2/5 over
their full workload but 1/3 over the shared subset) and checks that the code uses
the narrower, correct one.

### The two cases where kappa does not exist, and the discipline of saying so

```python
if p_e == 1.0:
    # Both labellers used exactly one category, and the same one. p_o is
    # also 1, so the formula is 0/0. This is not perfect agreement...
    return Agreement(kappa=None, ..., undefined_reason=(
        "both labellers used a single category throughout, so chance "
        "agreement is 1.0 and kappa is 0/0 — undefined, not 1.0"
    ))
```

**The temptation this code refuses is real, and it is worth stating exactly what it
would mean to give in to it.** If both labellers pressed "left" on every single
pair they shared, `p_o = 1.0` and `p_e = 1.0`, and the formula is `0/0` —
mathematically undefined. Returning `1.0` in that case would be the single most
flattering possible misreading available: it would report "perfect agreement" for a
situation that provides *zero evidence at all* about whether the two people would
agree on anything, because two people who always press the same button agree by
construction, independent of any actual judgement. `Agreement.kappa` is typed
`float | None` specifically so a caller cannot accidentally treat "undefined" as a
number, and `describe()` prints the words *"kappa undefined ... "* rather than a
number in this case — `test_describe_states_the_undefined_case_in_words_not_as_a_
number` checks the literal string `"0.000"` never appears in that output. The
second undefined case, fewer than two shared pairs, is the same discipline applied
to sample size: CONSTRAINTS #3's "never report a single run" rule, restated here for
agreement statistics rather than for episode rewards.

### The confusion matrix — why it travels alongside the coefficient, always

```python
confusion = {combo: 0 for combo in itertools.product(CHOICES, CHOICES)}
```

Every cell exists from the start, including the empty ones — a matrix with missing
keys would force every downstream reader to write `.get(key, 0)` defensively. And
the reason this ships alongside the number at all, rather than the number standing
alone: `PROJECT_BRIEF.md` §6.2 says a low kappa is itself a finding, and
`agreement.py`'s own module docstring names the consequence — *"a kappa quoted on its
own cannot be written up."* A number without the pattern of disagreement behind it
tells the reader *that* two people disagreed, not *how* — whether it was systematic
(one person consistently more generous) or scattered (genuine, unresolvable
difference of judgement), and only the confusion matrix can distinguish those two
very different findings.

### Landis & Koch — a convention, labelled as one

```python
bands = [(0.0, "poor (worse than chance)"), (0.2, "slight"), (0.4, "fair"),
         (0.6, "moderate"), (0.8, "substantial"), (1.01, "almost perfect")]
```

| kappa | Conventional label |
|---|---|
| < 0 | poor — worse than chance |
| 0.0 – 0.2 | slight |
| 0.2 – 0.4 | fair |
| 0.4 – 0.6 | moderate |
| 0.6 – 0.8 | substantial |
| > 0.8 | almost perfect |

`describe()`'s docstring is careful to call these *"the conventional reading, and
labelled as a convention rather than a fact — they are a rule of thumb from a 1977
biometrics paper, not a property of this data."* Worth remembering when the real
number arrives: a κ of, say, 0.35 reading as "fair" is a fact about Landis & Koch's
1977 naming choices, not a verdict this project's data has independently earned.

**And if the real κ comes out low — that is a result, not a failure.**
`00_labelling_handbook.md` §6.1 states the commitment this project made in advance,
before any label existed to be disappointed by: *"A low κ would mean that 'good
triage' is genuinely ill-defined even between the two people who built the system.
That is a real, reportable finding about the problem domain."* Deciding what counts
as an acceptable result *before* seeing the number is what stops the number from
being quietly tuned after the fact — the same discipline CONSTRAINTS #5 states for
every other result in this project ("a result that looks surprisingly good is a bug
report until proven otherwise") turned around to cover a result that looks
disappointing instead.

---

## 9. Bradley–Terry — the model the labels will train

**Not yet implemented.** `ROADMAP.md` lists it under Phase 5b, and every box under
that heading is unticked. This section exists so a reader of this document knows
what the 300 labels are *for*, without this document overstating what currently
exists in the repository — CLAUDE.md's evidence discipline applies to documentation
exactly as much as it applies to a results table.

The model `rlhf/reward_model.py` will eventually implement is a small MLP
`r̂(state, action)` fitted with the Bradley–Terry choice model, as
`00_labelling_handbook.md` §6.2 previews it for the person about to generate the
data it will consume:

```
                          exp( sum r-hat(s,a) over shift A )
   P(A preferred to B) = ------------------------------------------------
                          exp( sum over A ) + exp( sum over B )
```

**In words, worked through:** score every step of both shifts with the same small
network, add up each side's total, and the side with the higher total should be the
one a human actually picked. Training pushes the model's *predicted* probability of
preferring A towards the *actual* click a labeller made, over all 300 pairs — 80% of
them used to fit the model, 20% held out to check it did not merely memorise the
training pairs. Where `r̂` ends up *disagreeing* with the hand-written reward this
document has spent its whole first section arguing against is explicitly flagged, in
`00_labelling_handbook.md` §6.2's framing, as the interesting part worth
investigating and writing up — not a bug to be quietly fixed until the two agree.

Phase 5c then re-trains Q-learning and DQN against `r̂` in place of the hand reward
and collects **fresh** held-out preference labels comparing the RLHF-trained policy
against the hand-reward-trained one. That comparison — plotted as one graph, per
`ROADMAP.md`'s Phase 5 exit criterion — is the headline result the entire project
exists to produce, and as of this session it does not exist, because it depends on
300 labels that do not exist yet either.

---

## 10. Things that will confuse you

| What you'll notice | Why it's like that |
|---|---|
| `pairs.json` and `pairs_key.json` are written side by side in the same directory | Deliberate — separating them across directories would invite someone to copy the wrong one to wherever the UI is served from. The rule is enforced by content (`pairs.json` simply does not contain names), not by keeping the files physically apart. |
| A summary shows which alerts were *real* but never shows the *reward* | Ground truth is required so a labeller can judge outcomes with hindsight (§4); the reward is the exact thing Phase 5 exists to replace, so showing it would let a labeller simply agree with the number under scrutiny. |
| `run_id` looks like harmless provenance but has to be stripped separately from everything else in a summary | Every other field `summary.py` produces is already policy-agnostic; `run_id` (`"sarsa-seed3000004"`) is the one exception, built for a human staring at raw JSON on disk, never designed to reach a labelling page. |
| `_clean_seconds` checks `value != value` instead of a plain range comparison | `nan < 0` and `nan > max` are both `False` in IEEE 754, so a naive range check lets `NaN` straight through. The explicit self-inequality check is the only reliable way to catch it (D-044). |
| A duplicate label submission returns `{"ok": true}`, not an error | A refresh replays the POST on an answer that is already safely stored. Returning an error would interrupt a labelling sitting over work that was never actually at risk. |
| There is no way to edit or delete a stored label, anywhere in the code | Deliberate. `store.py` has no delete path by design, and `render_done_page` is a genuine dead end with no buttons — if you misclick, the fix is to note the pair number and mention it, not to patch the database. |
| Cohen's kappa can come back as `None` even after real labelling has happened | It is mathematically undefined, not merely unmeasured, whenever both labellers use exactly one category throughout their shared pairs (`p_e == 1.0` makes the formula `0/0`). Reported as "undefined" with a reason, never substituted with a flattering `1.0`. |
| The labeller id is never a field anywhere in the browser | It is bound once, at the command line, when the server launches (D-041) — putting it on the page would let a stale browser tab silently record one person's answers under the other's name. |
| `test_no_module_in_the_labelling_package_mentions_the_key_file` is a grep, not a behavioural test | Some rules ("never read this filename") are cheaper and more durably enforced by checking the source text directly than by exercising every code path that might violate them. |
| The 50 double-labelled pairs are not marked on screen | Deliberately invisible to the labeller — knowing a pair is "one of the special 50" would change how carefully it gets judged relative to the other 250, and that difference would contaminate the very statistic those 50 pairs exist to produce. |

---

## Where to go next

- **Document 6 — Decisions, Experiments, Results.** Every D-number cited in this
  document (D-038 through D-046), in full, with the reasoning behind each — and the
  five bugs, including D-044's NaN fix, given their own complete treatment.
- **Document 4 §1.** The other consumer of `EpisodeRecord` this document's §2
  diagram completes — `evaluation/metrics.py`'s branch of that same fork.
- **`docs/onboarding/00_labelling_handbook.md`.** The same machinery from the other
  side: what a labeller actually does, sitting at the keyboard, rather than what the
  code underneath them does to make that sitting meaningful.

---

*Source of truth: this Markdown file. The `.docx` export is generated from it. If
they disagree, the Markdown is right.*
