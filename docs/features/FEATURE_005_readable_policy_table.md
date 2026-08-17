# FEATURE_005 — The learned policy as a readable table

**Status:** done
**Phase:** 2 · **Owner:** Pranav · **Started:** 2026-08-17 · **Finished:** 2026-08-17
**Model(s) used:** Claude Opus 5.

---

## What and why

ROADMAP Phase 2, box 6 — and the roadmap flags it as *"a headline figure for the report and the viva."*

A 576×5 Q-table is not a result anyone can read. This turns it into the question the project actually cares about: **as the shift runs out, does the agent change its strategy, and in a way a human analyst would recognise?** That question is also half of the Phase 2 exit criterion, so the figure is load-bearing rather than decorative.

## Approach

`scripts/policy_table.py` writes `results/policy_table.md` with two views:

1. **The full grid** — three panels, one per `time_left` bucket, each 16 rows (`queue_len` × `oldest_age`) by 12 columns (`max_severity` × `asset_criticality`). 16 × 12 × 3 = 576, so nothing is aggregated away and nothing is hidden.
2. **The strategy-shift summary**, split deliberately into two sub-views:
   - **2a, the learned policy** — of the visited states in each time bucket, what fraction have each action as the greedy choice. This is what the agent would *do*, and it is the figure box 6 asks for.
   - **2b, where experience was spent** — action share weighted by visit count. This is the ε-greedy *behaviour* policy accumulated across the whole run, including the early ε = 1.0 phase, so it is partly random by construction.

Splitting 2a from 2b was not the original plan. The first version printed only the visit-weighted view, which mixes ~6,000 episodes of decaying-ε exploration into a figure captioned as the learned policy. The two tell noticeably different stories — BULK_CLOSE reads as 52.3% in the crunch under 2b and 46.2% under 2a, and the *trends* differ in direction between the two. Presenting the exploration-contaminated number as "the policy" would have been wrong in a figure explicitly intended for a viva.

## The integrity problem this feature exists around

**An unvisited state has an all-zero Q row, so `argmax` returns action 0 by the tie-break rule.** Q-learning visits 121 of 576 states. Rendered naively, **455 cells (79%) would have printed `PULL_HIGHEST_SEVERITY`** — a decision the agent never made, presented as a confident preference, in the project's headline figure.

The fix required changing the agent: `QLearningAgent` now records per-(s,a) visit counts. They play no part in learning and exist solely so the table can print `·` for "never seen". Added test-first — `test_visits_are_counted_per_state_action` and `test_unvisited_states_are_reported_as_unvisited_not_as_action_zero`, both watched to fail (`AttributeError: 'QLearningAgent' object has no attribute 'visits'`).

This is the same class of error as D-011's unvisited-pair convention in DP, reached by a different route: a defensible internal default becoming a false claim the moment it is displayed. Worth stating in the report as a general principle — the tie-break is correct code and would have produced a dishonest figure.

**`encode`/`decode` are checked against each other on every run** before anything is printed (`_self_check`, all 576 ids). A transposed mixed-radix unpack produces a table that still looks entirely plausible, so it is exactly the kind of error that survives review.

## Result

**Coverage: 121/576 states (21%).** Per time bucket: 83 early, 25 mid, **13 in the crunch**.

Learned policy, share of visited states per bucket:

| time_left | SEVERITY | OLDEST | CRITICAL | CHEAPEST | **BULK_CLOSE** | states |
|---|---|---|---|---|---|---|
| >240m (early) | **34.9%** | 12.0% | 7.2% | 20.5% | **25.3%** | 83 |
| 60–240m (mid) | 28.0% | 20.0% | 8.0% | 8.0% | **36.0%** | 25 |
| <60m (crunch) | **15.4%** | 7.7% | 15.4% | 15.4% | **46.2%** | 13 |

**Both trends are monotonic across all three buckets.** Working alerts by severity falls 34.9% → 28.0% → 15.4%; bulk-closing rises 25.3% → 36.0% → 46.2%.

So box 6's "behaviourally interpretable strategy shift as time runs out" **exists and is visible**. Whether it is the *right* shift is a separate question, and one this feature deliberately does not answer — see below.

## What was tried that didn't work, and what is honestly uncertain

**The interpretation has two readings and the data does not separate them.** The charitable one: an analyst under deadline pressure does triage more aggressively, so the shift is a recognisable human strategy. The unflattering one: the reward charges end-of-shift misses at the very end, and bulk-closing is the cheapest way to empty the queue before that charge lands — so this is the E-008 reward hack intensifying precisely where the reward makes it most profitable. Both fit. Distinguishing them needs a per-action reward decomposition within the crunch bucket, which has not been done. **The log and this file state both readings rather than choosing the flattering one.**

**The crunch column rests on 13 states.** 46.2% is six states; 15.4% is two. The three-bucket monotonicity is reassuring — a coincidence would not usually align in both directions across all three — but the most interesting column of the headline figure currently has thirteen data points behind it. Coverage falls off with `time_left` for a structural reason (the crunch is a small slice of each shift, and few queue configurations are reachable within it), which explains the thinness without excusing it.

**The re-run to capture visit counts doubled as a reproducibility check.** Training was re-run because the agent had not previously recorded visits. It reproduced E-008 to the digit — recall 0.73 ± 0.03, reward 270.9 ± 105.5, MTTD 22.0 ± 15.6 — confirming the pipeline is deterministic under its seeds.

## Files touched

| File | New/Modified | What changed |
|---|---|---|
| `scripts/policy_table.py` | **New** | The renderer, `encode`/`decode`, the self-check |
| `src/soc_triage/agents/q_learning.py` | Modified | Per-(s,a) visit counts, for display honesty only |
| `scripts/train.py` | Modified | Saves `results/q_learning_visits.npy` beside the Q-table |
| `tests/test_tabular.py` | Modified | 2 tests for visit counting and the unvisited/action-0 distinction |

`results/policy_table.md` is gitignored. Regenerate with `python scripts/train.py && python scripts/policy_table.py`.

## Follow-ups left open

- **Decide between the two interpretations** with a per-action reward decomposition inside the crunch bucket. This is the single most valuable remaining Phase 2 analysis, and it feeds the report's central argument.
- The crunch bucket's 13-state support will not improve by training longer — it is structural. If the figure needs to carry more weight, the `time_left` bucket boundaries are the thing to revisit, and those are MDP definition (CONSTRAINTS #15 — ask first).
- The same renderer should be pointed at the DP policy (`results/dp_policy.npy`) so the two can be compared cell by cell. That is ROADMAP box 5's territory.

## Plain-English summary

The agent's knowledge lives in a table of 2,880 numbers, which nobody can read. This turns it into a picture: for each situation the agent might face, which of its five options does it choose — laid out so you can see how the answer changes as the end of the shift approaches.

There is a clear pattern. Early in the shift the agent mostly works alerts by severity, worst first. As the clock runs down it does that less and less, and instead mass-dismisses batches of alerts more and more. The trend moves steadily in the same direction across all three time periods.

You could read that generously — a real analyst under time pressure also triages more aggressively. You could also read it cynically: the scoring system charges you for missed incidents only at the very end of the shift, and emptying the queue cheaply right before that is a good way to dodge the bill. Both explanations fit what we can see. We have written down both rather than picking the one that flatters the project, and noted what evidence would settle it.

One thing nearly went wrong. The agent only ever encounters about a fifth of the possible situations. For the other four fifths it has no opinion at all — but the way the code picks an action means "no opinion" silently comes out looking like "definitely work the most severe alert." Printed as-is, 79% of the picture would have shown a confident recommendation the agent never actually made. The agent now counts how often it has visited each situation purely so the picture can say "never been here" instead, and there is a test to keep it that way.
