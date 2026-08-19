# BUG_003 — `dqn_ablations.py` scores a total collapse as "no destabilisation"

**Status:** fixed
**Severity:** wrong-results
**Phase:** 3 · **Found:** 2026-08-19 · **Fixed:** 2026-08-19
**Model(s) used:** Claude Opus 5

---

## Symptom

The `no_replay` ablation produced a completely dead agent — recall 0.0000 on every one of 8 seeds — and `scripts/dqn_ablations.py` reported it as a **negative result**, i.e. as evidence that experience replay does not matter:

```
condition                          runs       final  volatility   end std   drawdown
------------------------------------------------------------------------------------
control (replay + target net)        30        18.5       167.8      44.2      635.2
no replay                             8      -515.4         0.0       0.0        0.0
no target network                     8       121.6        59.0       9.6      308.9

ablation / control ratios (>1 means the ablation is less stable):
  no replay                       volatility x0.00  end_std x0.00  drawdown x0.00

  no replay: NO clear destabilisation (all three within 1.5x of control) - a NEGATIVE RESULT
```

The per-run truth, which the summary line hides:

```
no replay, all 8 runs:  recall 0.0000   reward -520.5   final greedy -515.4
```

Identical to four decimal places across eight different seeds.

## Reproduction

Deterministic. Any condition whose agent collapses to a constant policy reproduces it:

```
$ .\.venv\Scripts\python.exe scripts/dqn_ablations.py
  no replay: NO clear destabilisation (all three within 1.5x of control) - a NEGATIVE RESULT
```

## Why this matters

**It inverts the answer to one of the two questions Phase 3 exists to ask.** The ROADMAP box reads "train DQN with replay off... Show the instability. This is the cleanest possible demonstration of *why* those two tricks exist." Taken at face value, the script's output says replay is unnecessary — the exact opposite of the truth, on a question that will be asked in the viva.

No published number is wrong: **E-017 reports the correct conclusion**, because the per-run JSONs were read directly instead of trusting the summary line. The defect is in the interpretation rule, not the arithmetic — every number in the table above is correct.

## Hypotheses

| # | Hypothesis | How to test it | Result |
|---|---|---|---|
| 1 | The ablation was never wired up, so it silently ran the control | Check `no_replay` runs differ from control | **REFUTED.** They differ enormously (recall 0.0000 vs 0.481), and `test_no_replay_ablation_trains_on_a_single_transition` pins the behaviour at a single backup. |
| 2 | The three measures are computed wrongly | Unit tests on synthetic curves | **REFUTED.** `test_a_flat_curve_has_zero_volatility_and_zero_drawdown` asserts exactly this behaviour and passes. The measures do what they were specified to do. |
| 3 | The measures are correct but measure the wrong quantity | Reason about what a collapsed policy looks like to a variance statistic | **CONFIRMED.** See root cause. |

## Root cause

All three measures — volatility (mean absolute step between checkpoints), end-std, and max drawdown — quantify **movement**. A policy that has collapsed to a single constant action produces an identical trajectory at every evaluation, so its greedy curve is a perfectly flat line at −515.4 and all three measures are exactly **0.00**.

The interpretation rule then compares ablation to control as a ratio and flags destabilisation when the ratio exceeds 1.5. A flatline gives `x0.00` — the *smallest possible* ratio — which the rule reads as "much more stable than control", and therefore as "no destabilisation".

**Cause connects to symptom exactly:** the script asks "did it wobble more?" when the question that matters is "did it learn at all?". Maximum failure and maximum stability are the same reading on these instruments. The measures were deliberately specified before any data was seen — which was the right instinct and is *why* they were trusted — but pre-registration protects against fitting a metric to the data after the fact, not against choosing a metric that cannot see the failure mode.

## Fix

| File | What changed |
|---|---|
| `scripts/dqn_ablations.py` | Added `COLLAPSE_BAND = -450.0` and `is_collapse(values)`, checked **before** any ratio is computed. A collapsed condition prints `COLLAPSED — no stability ratios` and its ratios are neither computed nor printed: emitting `x0.00` is what invited the misreading in the first place. Its verdict line now reads **"COLLAPSED — LEARNING FAILED ENTIRELY. This is the strongest possible result FOR the stabiliser, not a negative one."** The markdown writer carries the same three-way verdict. |
| `tests/test_dqn_analysis.py` | Four tests (below). The existing `test_a_flat_curve_has_zero_volatility_and_zero_drawdown` was **kept** — it is correct about the arithmetic, and its presence beside the new tests is what makes the distinction legible. |

Two design points worth keeping:

- **The check keys on the failure *value*, not on flatness.** An agent that converges and then sits still is the success case; a rule that treated any flat curve as collapse would destroy it. `test_a_flat_curve_at_a_GOOD_score_is_not_a_collapse` pins that.
- **It is judged on the final quarter, not the whole curve.** Every run starts near the collapse value before it has learned anything; only failing to *leave* it is the defect. `test_collapse_is_judged_on_the_END_of_training_not_the_start` pins that.

The general rule worth carrying into Phases 4 and 5: **any stability or variance measure used as a gate must be preceded by a liveness check.** "Did not move" and "did not destabilise" are indistinguishable to a variance statistic, and only one of them is good news.

## Verification

Re-run against the real 8 no-replay runs (unchanged on disk), which previously produced the wrong verdict:

```
ablation / control ratios (>1 means the ablation is less stable):
  no replay            COLLAPSED - no stability ratios (final -515.4, below the -450 always-BULK_CLOSE band)
  no target network    volatility x0.35  end_std x0.22  drawdown x0.49

  no replay: **COLLAPSED - LEARNING FAILED ENTIRELY.** This is the strongest possible
  result FOR the stabiliser, not a negative one: without it the agent never learns.
  Stability ratios are meaningless here and are not reported (BUG_003).
  no target network: NO clear destabilisation (all three within 1.5x of control) - a NEGATIVE RESULT
```

The `no_target_network` verdict is deliberately unchanged: that one *is* a genuine negative result, and the fix must not convert every unexpected reading into a collapse.

```
$ .\.venv\Scripts\python.exe -m pytest tests/ -q
130 passed
```

## Regression test added

Four in `tests/test_dqn_analysis.py`, all failing before the fix (`ImportError: cannot import name 'is_collapse'`):

- `test_a_flatlined_collapse_is_not_mistaken_for_stability` — a flat curve at −515.4 is a collapse
- `test_a_flat_curve_at_a_GOOD_score_is_not_a_collapse` — flatness alone is not failure
- `test_a_healthy_volatile_curve_is_not_a_collapse` — the control's own shape
- `test_collapse_is_judged_on_the_END_of_training_not_the_start` — a run that recovers is not a collapse

## Plain-English summary

We built three yardsticks to measure whether switching off part of the algorithm made training unstable. They measure how much the agent's score jumps around. Then we switched off experience replay, and the agent didn't become unstable — it stopped learning altogether and sat at exactly the same terrible score forever.

A perfectly flat line is perfectly steady. All three yardsticks read zero, the script compared that against the normal agent's jumpiness, concluded the ablated version was *calmer*, and printed that replay apparently doesn't matter. The truth is the reverse: without replay the agent learns nothing at all.

What makes this worth writing down is that we did the careful thing and it still failed. We chose those three measures in advance, precisely so we couldn't pick flattering ones after seeing the results. That guards against one kind of self-deception and does nothing about another — measuring the wrong thing entirely. The only reason we caught it is that we opened the individual run files instead of believing the summary.
