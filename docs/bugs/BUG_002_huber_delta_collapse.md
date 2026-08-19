# BUG_002 — DQN collapses to BULK_CLOSE because the Huber delta is 1.0

**Status:** fixed
**Severity:** wrong-results
**Phase:** 3 · **Found:** 2026-08-19 · **Fixed:** 2026-08-19
**Model(s) used:** Claude Opus 5

---

## Symptom

Twenty completed DQN runs (20000 episodes each) all produced an agent that closes almost every alert unread. From `results/dqn_runs/dqn_delta1_E016/repeat0.log`:

```
  repeat 0  ep   500/20000  eps 0.779  grad steps   6484  loss  12.32  greedy(train-diag)  -515.4  [0.3 min]
  repeat 0  ep  5000/20000  eps 0.082  grad steps 140984  loss   1.20  greedy(train-diag)  -515.4  [7.0 min]
  repeat 0  ep 18000/20000  eps 0.050  grad steps 787746  loss   2.32  greedy(train-diag)  -515.4  [40.7 min]
  repeat 0  ep 20000/20000  eps 0.050  grad steps 887726  loss   0.32  greedy(train-diag)  -515.9  [45.8 min]
```

The greedy diagnostic is identical to one decimal place at 36 of 40 checkpoints, from episode 500 to episode 20000. Evaluation on the 30 eval seeds:

```
  recall_at_deadline  0.0086 +- 0.0336
  total_reward        -480.7   (fifo, the worst baseline, is -702; random -271; severity_sort +51)
```

**The loss curve looked healthy throughout** — it converged to ~0.04-0.3. Nothing errored and all 20 runs completed.

## Reproduction

Deterministic, 100% of runs, and visible within 500 episodes:

```
$ .\.venv\Scripts\python.exe scripts/train_dqn.py --episodes 500 --repeats 1 --no-plot   # at commit c4e613e
greedy(train-diag) -515.4 ; final greedy action mix a4=100.0%
```

`-515.4` is the value of the constant always-BULK_CLOSE policy on the 10 train-diagnostic seeds, which is why it never varies: a constant action gives identical trajectories every time it is measured.

## Why this matters

It invalidates the whole first Phase 3 sweep. No published number is affected — the sweep was stopped and analysed before any result was written up, so nothing in `EXPERIMENT_LOG.md` needed marking superseded. The 20 runs are kept at `results/dqn_runs/dqn_delta1_E016/` and documented as **E-016** (CONSTRAINTS #4: not deleted, not overwritten).

Had the sweep been left to finish overnight as planned, the two ablations would have been ablations of a broken agent and the phase gate would have been decided on 60 runs of a policy that reads nothing.

## Hypotheses

| # | Hypothesis | How to test it | Result |
|---|---|---|---|
| 1 | The network collapsed — dead units, or Q independent of the state | Load `repeat0.pt`, compute Q over 2381 real transitions, measure spread across states and actions | **REFUTED.** std across states 14.8-15.7 per action; best-minus-second gap mean 4.79. The network is healthy and well differentiated. |
| 2 | Huber saturation — all gradients clipped to a constant, so relative errors are lost | Measure the TD-error distribution against the delta | **CONFIRMED, but not as first framed.** Only 0.9% of TD errors exceed 1.0 in magnitude — yet those 0.9% are the missed-incident penalties, i.e. the entire signal. |
| 3 | Gradient clipping (`grad_clip_norm: 10`) is destroying the signal | Probe arms A (delta 1, clip 10) vs B (delta 50, clip 10) | **REFUTED.** A and B share clip=10 and differ completely; raising clip to 100 (arm D) made things worse, not better. |
| 4 | The greedy diagnostic itself is broken and always reports the same policy | Cross-check against the independent end-of-run eval path, and against training reward vs epsilon | **REFUTED.** Eval gives recall 0.0086 independently, and training reward *degrades* from -267 to -403 as epsilon decays — the greedy policy really is worse than random. |

## Root cause

`src/soc_triage/agents/dqn.py` called `F.huber_loss(predicted, target)` with no `delta`, so torch's default of **1.0** applied.

`config/env_default.yaml` prices the outcomes that matter as `bulk_close_true_incident: -150.0` and `end_of_shift_missed: -200.0` (scaled by an asset multiplier up to 2.5), against `bulk_close_fp: +0.5` for closing a junk alert. Measured over 2381 real transitions:

```
per-step reward : mean -2.167  std 46.448  min -1499.5  max +1.5
TD error        : median 0.139   p99 0.967   max 1454.6    (absolute values)
fraction of TD errors above 1.0 in magnitude : 0.9%
```

Huber is quadratic below the delta (gradient scales with the error) and linear above it (gradient is flat). At delta 1.0 every catastrophic penalty sat in the flat region, so the network was told they were all "wrong by about 1":

```
routine error   (TD = -1)    grad norm 2.543705
buried incident (TD = -150)  grad norm 2.579709
ratio 1.014        <- 150x the error, 1.4% more gradient
```

**Cause connects to symptom exactly.** BULK_CLOSE pays a small, frequent, reliably-learnable +0.5. Its cost arrives rarely and hugely — and was flattened to nothing. So the network fitted the small rewards to a loss of 0.04 and rated bulk-closing the best action in 99.4% of states. A low loss and a useless policy are not in tension here; the agent solved the problem it was actually given.

The comment in `training_default.yaml` justified Huber as "less sensitive to the large negative outlier rewards" — backwards. Those penalties are the signal, not outliers. Tabular Q-learning uses the raw TD error and reaches recall 0.73 on the same environment, which is the contrast that should have prompted the question sooner.

## Fix

| File | What changed |
|---|---|
| `config/training_default.yaml` | added `huber_delta: 200.0` with the measurement in the comment |
| `src/soc_triage/config.py` | `DQNConfig.huber_delta` field, loader entry, and validation refusing values below 50 |
| `src/soc_triage/agents/dqn.py` | `F.huber_loss(predicted, target, delta=self.dcfg.huber_delta)` |
| `tests/test_dqn.py` | new failing-first test; the target-network test now reads the delta from config instead of hard-coding torch's default |
| `tests/test_dqn_config.py` | three tests for the bound and the shipped value |

200 is the largest *named* single-event penalty in `env_default.yaml`, so every individual penalty stays quadratic and only the compound multi-miss tail is linearised. Chosen this way rather than from the delta sweep because the sweep could not resolve 50 vs 100 vs 200 (pairwise difference over SEM was 0.05 to 0.40) — see D-029 and E-016.

## Verification

The regression test fails on the old code and passes on the new:

```
$ .\.venv\Scripts\python.exe -m pytest tests/test_dqn.py -q -k buried    # before the fix
E       AssertionError: a -150 buried-incident penalty moved the network only 1.0x as much
        as a routine -1 error; the Huber delta is compressing the signal the agent most needs

$ .\.venv\Scripts\python.exe -m pytest tests/ -q                          # after
126 passed
```

End-to-end through the real trainer and shipped config, 3 runs x 3000 episodes (15% of the training budget):

```
dqn vs references on eval seeds (mean +- std across 3 runs):
  dqn              recall 0.48+-0.21   reward  -49.4+-136.6   mttd 17.1+-8.7
  severity_sort    recall 0.84+-0.16   reward   40.4+-220.1   mttd 28.3+-18.3
  oracle_greedy    recall 0.87+-0.16   reward  168.0+-232.9   mttd 39.8+-43.9
```

recall 0.0086 -> 0.48, and the greedy curve moves freely (113.9 / 61.7 / 9.7 / -118 / 153.1 / 130.1) instead of sitting on -515.4.

**Not claimed:** that the DQN is now good. At 3000 episodes it is still below severity_sort on recall and reward, still volatile, and 16 of 90 eval episodes caught nothing. The collapse is fixed; the phase gate is unmeasured.

## Regression test added

`tests/test_dqn.py::test_a_buried_incident_moves_the_network_more_than_a_routine_error` — two identical agents each take one backup, one on a -1 reward and one on -150, with clipping disabled so the assertion is about the loss. It requires the larger error to move the network more than 10x as much; the old code scores 1.014.

Plus `tests/test_dqn_config.py`: `test_huber_delta_below_the_measured_collapse_threshold_is_refused` (rejects 1.0, 10.0, 25.0), `test_huber_delta_must_be_positive`, and `test_the_shipped_config_keeps_every_named_penalty_quadratic`.

## Plain-English summary

The agent was being marked wrong on a scale that stopped at 1. Closing a batch of alerts unread earns a small reward for tidiness, and occasionally buries a real security incident, which costs 150 to 1500. But the loss function had a ceiling of 1: burying a real incident registered as almost exactly as bad as being slightly off about something harmless — 1.4% worse, measured. So the agent learned the small tidy-up rewards very precisely, concluded that closing everything unread was free, and did it 99.4% of the time. It caught 0.9% of the incidents it was built to catch.

The trap is that nothing looked broken. The training error fell steadily to near zero, which normally means success — and it *was* success, at the wrong task. The network had correctly learned the objective it was handed; the objective was the thing that was wrong. Raising that ceiling to 200, the price of the worst single mistake the environment defines, took recall from 0.9% to 48% at a fifth of the training time.
