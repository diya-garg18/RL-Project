# EXPERIMENT_LOG.md — Every training run

> **Append-only.** Never edit or delete an entry. If a run turns out to be invalid, mark it `SUPERSEDED` with the reason and add a new entry — don't erase it (`CONSTRAINTS.md` #4).
>
> This file is the evidence base for the report. Every number that appears in the report must be traceable to an entry here.

**Entry format:**

```
## E-nnn — <agent> — <date>
**Status:** valid | SUPERSEDED (reason)
**Config:** env_default.yaml @ <git sha> · training_default.yaml @ <git sha>
**Command:**
**Seeds:** train=[...] eval=[...]
**Runtime:**
**Result:** | metric | mean | std |
**Observations:** anything surprising
**Notes:** deviations from default config, known caveats
```

---

## Standing rules

1. **5 seeds minimum**, mean ± std. Never report a single run.
2. **Eval seeds are disjoint from train seeds.** Never tune against eval results.
3. Record the **config git sha**, not just the filename — configs change.
4. If a result is surprisingly good, log it *and* log what you did to check it wasn't a bug (`CONSTRAINTS.md` #5).
5. Negative results get entries too. An ablation that showed no effect is a finding.

---

## Runs

## E-001 — generator calibration — 2026-08-13

**Status:** valid
**Config:** env_default.yaml @ 69dfc88 (final tuned values)
**Command:** `.\.venv\Scripts\python.exe scripts\calibrate_generator.py`
**Seeds:** calibration=[1000..1099] · robustness checks=[2000..2099], [3000..3099] · (train/eval seeds untouched)
**Runtime:** ~5 s

**Result:**
| metric | value | target | verdict |
|---|---|---|---|
| alerts/shift | 168.7 | ~170 | PASS |
| true-incident rate | 3.34% | 2.5–3.5% | PASS |
| incidents/shift | 5.6 ± 2.6 | — | — |
| Pearson r(severity, truth) | 0.323 | 0.30–0.40 | PASS |
| robustness (seeds 2000s) | rate 3.13%, r 0.311 | in band | PASS |
| robustness (seeds 3000s) | rate 3.27%, r 0.317 | in band | PASS |

**Observations:** P(true|severity) = 0.20% / 0.69% / 5.18% / 29.67% for severities 0–3; ~64% of true incidents carry severity 3 — the unavoidable consequence of demanding r ≥ 0.30 at a 3% base rate (see D-007). Tuning took 3 iterations: base_rate 0.03 → 0.0175 → 0.0145 → 0.0135; top severity_lift 8 → 11 → 13 → 15. An intermediate config (rate 3.30%, r 0.310) was rejected because its robustness check (r 0.298) straddled the band edge.

**Notes:** Not a training run — this is the Phase 0 calibration gate (ROADMAP). Numbers also recorded in EXPLAIN.md Part 8. Human verification: Diya reviewed and approved these numbers this session.

---

## E-002 — baseline comparison — 2026-08-14

**Status:** valid
**Config:** env_default.yaml @ 2c1d974 (hash 13dddbb15332)
**Command:** `.\.venv\Scripts\python.exe scripts\run_baselines.py`
**Seeds:** eval=[101..105] (baselines learn nothing, so no tuning risk; identical alert streams per seed = paired comparison)
**Runtime:** ~3 s

**Result (mean ± std over 5 seeds):**
| agent | MTTD min | recall@deadline | wasted min | critical misses | composite ₹ | total reward |
|---|---|---|---|---|---|---|
| random | 78.8 ± 51.5 | 0.46 ± 0.07 | 427 ± 21 | 0.0 | 311,289 | −298 |
| fifo | 246.3 ± 97.3 | 0.20 ± 0.19 | 456 ± 16 | 0.8 | 778,056 | −1,054 |
| severity_sort | 36.6 ± 23.3 | 0.85 ± 0.16 | 413 ± 17 | 0.0 | 118,756 | +68 |
| cheapest_first | 54.8 ± 48.1 | 0.47 ± 0.12 | 454 ± 4 | 0.6 | 552,043 | −610 |
| oracle_greedy | 41.3 ± 22.7 | 0.86 ± 0.12 | 356 ± 25 | 0.0 | 127,221 | +193 |

**Exit-gate checks:** oracle strictly best on mean recall — PASS (0.864 vs 0.846). Random worst — **FAIL: fifo (0.20) is far below random (0.46).**

**Observations:**
1. FIFO's collapse is mechanistically clean, not a bug: in an overloaded queue it always works the oldest alert, so by investigation time deadlines have often expired (MTTD 246 min — 6–7× severity-sort's). This is *why triage exists*; random beats it because random sometimes pulls fresh high-signal alerts. Proposed: amend the exit criterion wording to "oracle strictly best; random and FIFO clearly at the bottom" — awaiting human decision, criterion text not yet changed.
2. First oracle version scored 0.72 (below severity-sort) — two defects found and fixed (see D-010): no path-clearing for unreachable incidents, and an unconditional-bulk-close hygiene loop. Lesson: even a cheating baseline needs debugging.
3. Per-seed pairing: oracle ≥ severity-sort on 4/5 seeds; loses seed 101 by one incident (id 143, arrival min 464/480 — end-game timing). "Upper bound" holds in expectation only.
4. Severity-sort at 0.85 recall confirms D-007: it is the strong opponent, exactly as the calibration's severity-concentration predicted.
