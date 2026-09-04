# FEATURE_011 — RLHF preference-pair collection (Phase 5a, data layer)

**Status:** design agreed 2026-09-04, session 12. Implementation follows in this
document as it lands.
**Phase:** 5a (`ROADMAP.md` → "Phase 5 — RLHF", sub-block 5a)
**Owner:** Pranav
**Model that helped design it:** Claude Opus 5

---

## 1. What this is, in one paragraph

Phase 5 replaces the hand-written reward with one learned from human judgement.
Before any of that can happen, somebody has to *ask* the humans. This feature is
the plumbing for that question: it takes episodes the project has already run,
pairs them up so that each pair is two different policies working **the same
shift**, renders each side as something a person can actually read, stores the
answers in a small database, and measures whether two people agree with each
other. It does not train anything and it does not display anything — the reward
model is 5b, and the labelling page is Diya's (`PROJECT_BRIEF.md` §9, line 231).

---

## 2. Scope — and what is deliberately outside it

| in scope (this feature) | out of scope |
|---|---|
| `rlhf/summary.py` — EpisodeRecord → a readable summary | the labelling web page (**Diya**, brief §9) |
| `rlhf/pairs.py` — deterministic pair construction | `rlhf/reward_model.py` (Phase 5b) |
| `rlhf/store.py` — SQLite for the collected labels | policy re-training (Phase 5c) |
| `rlhf/agreement.py` — Cohen's κ, written by hand | the Groq/Llama justification layer |
| `scripts/generate_pairs.py` — the driver | the React dashboard |
| `scripts/report_kappa.py` — the κ report | collecting the actual 300 labels (human time) |

The split is not arbitrary. `PROJECT_BRIEF.md:231` assigns the preference-labelling
UI to Diya and `:235` assigns the Bradley–Terry reward model to Pranav. Building
the UI here would take her box and skew a commit balance that is currently even
(91 commits, gap 3, `scripts/commit_balance.py` 2026-09-04).

---

## 3. The decision that shapes everything else: pairs are built from files, not agents

`rlhf/pairs.py` **never imports an agent and never imports torch.** It reads
EpisodeRecord JSON files off disk and pairs them.

Three reasons, in order of how much they matter:

1. **CONSTRAINTS #11** — nothing from a later phase may be required to run an
   earlier one. The inverse discipline is just as useful: the RLHF data layer
   must not drag Phase 3/4 machinery behind it. Someone should be able to build
   pairs on a laptop with no `torch` installed.
2. **Seven of the nine policies need trained parameters**, and those live in
   gitignored `results/` (`*.pt`, `*.npy`). Coupling pair construction to agent
   loading would make the pair set unbuildable on a fresh clone. Coupling it to
   *records* means the records are the artefact to regenerate, and there is
   already a function that writes them (`runner.save_records`).
3. `runner.py`'s own docstring already promises this: *"emits EpisodeRecord dicts
   … the interchange format that evaluation, **the RLHF pair builder**, and the
   dashboard all consume."* The contract was designed in Phase 0. This feature
   is the first thing to actually use it.

So the pipeline has a seam in the middle:

```
scripts/generate_pairs.py          rlhf/pairs.py        rlhf/store.py
  (needs agents + torch)             (pure data)          (pure data)
        |                                 |                    |
   run each policy on           read records, group        collect labels
   the pair-seed block          by seed, emit pairs        from labellers
        |                                 |                    |
        v                                 v                    v
 results/rlhf/records/*.json  -->  results/rlhf/pairs.json --> results/rlhf/labels.db
                                   results/rlhf/pairs_key.json
```

---

## 4. Seeds — a new block, and why it is forced

**New config key: `rlhf.pair_seed_start: 3000000`**, following the D-016
per-consumer block convention. Verified free: the existing blocks are 10000–59999
(model estimation), 200000, 400000, 600000, 1000000, 1200000, 1400000, 1600000,
1800000, 2000000 and 2200000.

The labelled episodes may **not** come from the eval block `[101..130]`. This is
not a stylistic preference, it is CONSTRAINTS #2 applied one level up:

> The reward model is *fitted* to the preferences humans express about these
> episodes. Phase 5c then re-trains policies on that reward model, and Phase 5/6
> evaluate those policies on the eval seeds. If the labelled episodes were eval
> episodes, human judgement of eval-seed outcomes would have been baked into the
> reward the policy maximises — evaluation-set information flowing into training,
> laundered through a person. Every downstream eval number would be contaminated
> and no test would catch it.

The train block `[1..10]` is technically permitted but rejected on a different
ground: ten alert streams spread over 300 pairs means a labeller sees the same
shift roughly thirty times, and boredom and memory both become confounds. A
dedicated block sized by `rlhf.n_pair_seeds: 12` fixes that.

`rlhf.pair_must_share_seed: true` already exists in the config and is asserted,
not assumed — a pair whose two sides ran different alert streams is not a
comparison of policies, it is a comparison of luck.

---

## 5. The policy pool

Nine policies: `random`, `severity_sort`, `dp`, `q_learning`, `sarsa`,
`monte_carlo`, `dqn`, `reinforce`, `actor_critic`.

`oracle_greedy` is **excluded**. It reads `is_true_incident` directly
(`baselines.py:89`, `obs_kind="snapshot"` — the sanctioned exception), so it
would win nearly every pair it appeared in. A pair whose answer is a foregone
conclusion costs a labeller twenty seconds and teaches the Bradley–Terry model
almost nothing, because the gradient of the logistic loss vanishes where the
prediction is already confident and correct. `fifo` and `cheapest_first` are
excluded for a duller reason: nine policies is already 36 unordered pairings,
and eleven would be 55, which does not divide 300 usefully.

Wide rather than narrow is a deliberate choice for the Phase 6 audit. Brief §7
asks for **state-visitation overlap between the RLHF policy and the policies that
generated the labelled pairs**. That question is only answerable if the pair
policies covered a wide behavioural range in the first place — a pool of three
similar learners would make any later policy look out-of-distribution.

---

## 6. Blinding — two artefacts, not one

Pair construction writes **two** files:

- `results/rlhf/pairs.json` — what the labelling UI reads. Contains `pair_id`,
  `seed`, `config_hash`, `left`, `right` (two rendered summaries), and
  `double_labelled`. **It does not contain policy names.**
- `results/rlhf/pairs_key.json` — `pair_id → {left_policy, right_policy, swapped}`.
  Read only by analysis code. Never sent to the UI.

Two biases are being defended against, and they are different:

1. **Name bias.** A labeller who can see that the left side is `random` and the
   right is `dqn` is no longer judging outcomes. Splitting the key out makes the
   blinding structural rather than a promise about how the UI is written.
2. **Position bias.** People pick the left option more often than chance. Which
   policy lands on which side is decided by a seeded RNG
   (`rlhf.pair_sampling_seed`) and recorded as `swapped` in the key, so the
   analysis can undo it exactly and, if we want, *measure* the position bias
   rather than merely hoping it cancelled.

---

## 7. What a labeller actually sees — and the one number they must not

`rlhf/summary.py` turns an EpisodeRecord into an `EpisodeSummary`:

**Timeline** — one row per step: minute, action name, and for an investigation
the alert's severity, type and verify cost. For a bulk-close, how many alerts it
swallowed.

**Outcome cards** — from `record["outcome"]`, which the environment computed with
full ground truth (`env.py:239`, allowed: it is evaluation output, never an agent
observation):

- incidents caught, and the delay on each (reconstructed from
  `steps[].info.delay_min` where `was_true_incident` is true)
- incidents missed, and how many were on crown-jewel assets
  (`missed_by_criticality[2]`)
- incidents buried by bulk-close
- wasted analyst minutes
- mean time to detect

**The summary must never show `total_reward`, or any per-step reward.** This is
the point of the whole phase. The hand-written reward is the thing whose numbers
we admit are invented (brief §6.1); showing it to a labeller would anchor them to
the very quantity RLHF exists to replace, and the resulting reward model would be
an expensive re-derivation of `config/env_default.yaml`. There is a test for this.

### Known limitation, stated rather than hidden

Missed incidents are reported as **counts, not per-incident cards**. The
EpisodeRecord does not carry the alerts left unhandled in the queue at the end of
the shift — only the ones that were investigated or bulk-closed appear in
`steps[]`. Rendering a card per missed incident would require adding the final
queue to the record, which changes `runner.py`, a Phase 0 module that all 499
existing record files were written by. That trade is not worth it: the outcome
block already gives the count, the criticality breakdown and the buried count,
which is enough for a person to judge "this shift went badly and here is roughly
how". If per-incident missed cards later prove necessary, the upgrade is an
*additive* key on the record and old files stay readable.

---

## 8. Pair construction — deterministic, balanced, and reproducible

36 unordered policy pairings × 12 seeds = 432 candidate pairs; we need 300.

- **Balanced allocation:** every policy pairing gets `300 // 36 = 8` pairs, and
  the remaining `300 - 288 = 12` pairings get a 9th. Which twelve is decided by
  the seeded RNG, not by dict order, so no pairing is systematically favoured by
  an accident of how Python happened to sort a set.
- **Determinism:** the whole construction is a pure function of
  `(records, rlhf.pair_sampling_seed, config)`. Running it twice produces
  byte-identical `pairs.json`. This matters because labels reference `pair_id`;
  if a rebuild renumbered the pairs, every label already collected would silently
  point at a different comparison.
- **Double-labelling:** `rlhf.double_labelled_pairs: 50` pairs are flagged
  `double_labelled: true`, chosen by the same seeded RNG after a deterministic
  shuffle, and spread across policy pairings rather than taken as the first 50 in
  file order.

---

## 9. Storage

SQLite at `results/rlhf/labels.db`. Already gitignored twice over — `.gitignore`
has both `results/*` and `*.db`, and `CONSTRAINTS.md` #19 names the label database
explicitly.

```sql
CREATE TABLE IF NOT EXISTS labels (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id       TEXT    NOT NULL,
    labeller_id   TEXT    NOT NULL,
    choice        TEXT    NOT NULL CHECK (choice IN ('left', 'right', 'tie')),
    created_at    TEXT    NOT NULL,          -- ISO-8601, UTC
    seconds_taken REAL,
    UNIQUE (pair_id, labeller_id)
);
```

`labeller_id` is **opaque** — `A1`, `A2`, `P1`. CONSTRAINTS #23 permits an opaque
id, a choice and a timestamp and nothing else; no names, no employer, nothing
that could identify a KPMG practitioner. The `UNIQUE (pair_id, labeller_id)`
constraint is what makes double-labelling meaningful: one judgment per person per
pair, so κ is computed over genuinely independent opinions rather than over
someone who clicked the same pair twice.

`seconds_taken` is nullable because a CLI labeller may not measure it and a
silently-invented zero would be worse than an honest null.

---

## 10. Cohen's κ, by hand

`rlhf/agreement.py` implements

```
kappa = (p_o - p_e) / (1 - p_e)
```

over the three categories `left`/`right`/`tie`, on the pairs both labellers
answered. `p_o` is the observed agreement rate; `p_e` is the agreement expected
from each labeller's own marginal frequencies. No `sklearn` — this is on the
syllabus, it is six lines, and CONSTRAINTS #7's principle is that anything a viva
might ask us to derive gets written by hand.

Two edge cases the code must handle rather than divide by zero:

- **`p_e == 1`** — both labellers used exactly one category, always the same one.
  κ is undefined here, not 1.0. Return `None` and say so.
- **Fewer than two shared pairs** — return `None`, not a κ computed from one
  agreement.

A low κ is a **finding, not a failure** (brief §6.2). If two people who built this
simulator cannot agree on what good triage looks like, that is direct evidence for
§3.5's claim that the reward is un-writable by hand, and it belongs in the report
in that form.

---

## 11. Config additions

```yaml
rlhf:
  target_pairs: 300              # already present
  double_labelled_pairs: 50      # already present
  pair_must_share_seed: true     # already present
  pair_seed_start: 3000000       # NEW — own block (D-016 convention), §4 above
  n_pair_seeds: 12               # NEW — 12 x 36 pairings = 432 candidates for 300 pairs
  pair_sampling_seed: 20260904   # NEW — RNG for allocation, side-swap, double-label choice
  policies:                      # NEW — the pool, §5 above. oracle_greedy excluded.
    [random, severity_sort, dp, q_learning, sarsa, monte_carlo, dqn, reinforce, actor_critic]
```

No new pip dependency: `sqlite3`, `json`, `random` and `itertools` are stdlib
(CONSTRAINTS #8 untouched).

---

## 12. Tests

| test | what it actually pins down |
|---|---|
| `test_summary_never_shows_reward` | no key or rendered string in an `EpisodeSummary` contains a reward. The §7 rule, enforced rather than promised. |
| `test_summary_counts_match_outcome` | caught/missed/wasted in the summary equal `record["outcome"]` — the renderer reports, it does not recompute |
| `test_pairs_share_a_seed` | every pair's two sides ran the same alert stream (`pair_must_share_seed`) |
| `test_pairs_are_deterministic` | two builds from the same records + seed are byte-identical |
| `test_pairs_never_pair_a_policy_with_itself` | |
| `test_pair_allocation_is_balanced` | each pairing gets 8 or 9; total is exactly `target_pairs` |
| `test_pairs_json_contains_no_policy_names` | the blinding in §6, checked against the serialised file, not the in-memory object |
| `test_double_labelled_count` | exactly `double_labelled_pairs` flagged |
| `test_store_rejects_bad_choice` | the CHECK constraint fires |
| `test_store_one_label_per_labeller_per_pair` | the UNIQUE constraint fires |
| `test_kappa_perfect_agreement` | κ = 1.0 |
| `test_kappa_chance_agreement` | κ ≈ 0 when labellers are independent |
| `test_kappa_undefined_cases` | returns `None` for single-category and <2 shared pairs, rather than dividing by zero |
| `test_kappa_matches_hand_worked_example` | a 3×3 table solved on paper — the anchor, built the same way `tiny_mdp` anchors the learners (D-014) |

---

## 13. Build order

1. `rlhf/summary.py` + tests — nothing depends on anything else
2. `rlhf/store.py` + tests — independent of summary
3. `rlhf/agreement.py` + tests — independent of both
4. `rlhf/pairs.py` + tests — needs summary
5. config keys + validation
6. `scripts/generate_pairs.py` — needs trained policies present; the only step
   that can be blocked by a missing artefact
7. `scripts/report_kappa.py`

Steps 1–5 are runnable and testable on a clone with no `results/` at all. That is
the property worth protecting.
