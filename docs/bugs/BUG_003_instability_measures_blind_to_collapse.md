# BUG_003 — `dqn_ablations.py` scores a total collapse as "no destabilisation"

**Status:** diagnosed — **fix not applied** (handed to the next session; see "Fix")
**Severity:** wrong-results
**Phase:** 3 · **Found:** 2026-08-19 · **Fixed:** —
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

**Not applied.** The next session should implement it; the change is small and the specification is unambiguous.

| File | What should change |
|---|---|
| `scripts/dqn_ablations.py` | Before computing any ratio, check whether each condition **learned anything**. Gate on final greedy performance against the collapse band (below −450, the always-BULK_CLOSE value from E-016) and on `recall_at_deadline` being distinguishable from 0. If a condition failed that check, report **"COLLAPSED — learning failed entirely"** and do not compute or print stability ratios for it at all, because they are meaningless. |
| `tests/test_dqn_analysis.py` | Add a test that a flatlined curve at the collapse value is classified as collapse, **not** as "no destabilisation". The existing `test_a_flat_curve_has_zero_volatility_and_zero_drawdown` should stay — it is correct about the arithmetic and its presence is what makes the omission visible. |

The general rule worth carrying into Phases 4 and 5: **any stability or variance measure used as a gate must be preceded by a liveness check.** "Did not move" and "did not destabilise" are indistinguishable to a variance statistic, and only one of them is good news.

## Verification

Not yet performed — the fix is not applied. When it is, the check is that `scripts/dqn_ablations.py` on the existing `results/dqn_runs/dqn_no_replay/` (8 runs, all recall 0.0000, kept) reports a collapse rather than a negative result, and that the full suite still passes.

## Regression test added

None yet — see "Fix". This is a `wrong-results` defect, so shipping it without a test would need a very good reason and there isn't one. The reason it is unfixed is scope and commit balance (Pranav is 10 commits ahead of Diya; CONSTRAINTS #26), not difficulty.

## Plain-English summary

We built three yardsticks to measure whether switching off part of the algorithm made training unstable. They measure how much the agent's score jumps around. Then we switched off experience replay, and the agent didn't become unstable — it stopped learning altogether and sat at exactly the same terrible score forever.

A perfectly flat line is perfectly steady. All three yardsticks read zero, the script compared that against the normal agent's jumpiness, concluded the ablated version was *calmer*, and printed that replay apparently doesn't matter. The truth is the reverse: without replay the agent learns nothing at all.

What makes this worth writing down is that we did the careful thing and it still failed. We chose those three measures in advance, precisely so we couldn't pick flattering ones after seeing the results. That guards against one kind of self-deception and does nothing about another — measuring the wrong thing entirely. The only reason we caught it is that we opened the individual run files instead of believing the summary.
