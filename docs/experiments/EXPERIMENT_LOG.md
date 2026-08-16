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

**Status:** SUPERSEDED by E-003 (generator vectorised for Phase 1 speed — same distributions, but RNG draw order changed, so the same seeds now produce different alert streams; per CONSTRAINTS #4 this entry stays, unedited below this line)
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

---

## E-003 — vectorised generator: recalibration + baseline re-run + 30-seed diagnostic — 2026-08-14

**Status:** valid
**Config:** env_default.yaml @ 6ab8032 (values unchanged from E-002; generator internals vectorised)
**Commands:** `python scripts/calibrate_generator.py` · `python scripts/run_baselines.py` · 30-seed diagnostic (seeds 5000–5029, fresh block)
**Runtime:** seconds each; estimation projection now 1.3 min for 50k episodes (was 37.6 min — the reason for the change; profile showed generate_shift at 82% of runtime)

**Recalibration (seeds 1000–1099):** 168.7 alerts/shift · rate 3.20% · r = 0.321 — still in band, no retuning needed. Distributions unchanged by construction; only stream identities changed.

**Baseline re-run (eval seeds, new streams):** severity_sort 0.87 recall, oracle 0.77 — **the E-002 "oracle strictly best on recall" conclusion did NOT survive the stream change.**

**30-seed diagnostic (seeds 5000–5029, no tuning of anything):**
| agent | recall | total reward | MTTD |
|---|---|---|---|
| severity_sort | **0.826 ± 0.148** | 50.6 | **26.5** |
| oracle_greedy | 0.799 ± 0.199 | **145.0** | 38.7 |
| random | 0.545 | −270.9 | 98.4 |
| cheapest_first | 0.391 | −381.2 | 65.3 |
| fifo | 0.141 | −702.2 | 198.6 |

**Finding (the real content of this entry):** within the deliberately coarse 5-rule action space, perfect information does NOT yield the best recall@deadline — severity-sort's camp-on-severity-3 behaviour is near-unbeatable on that one metric because ~64% of incidents carry severity 3 (D-007). The oracle's information advantage shows decisively on **total reward** (145 vs 51 — the environment's actual objective, which also prices wasted time, asset criticality, and bulk hygiene) but not on recall. E-002's 0.86-vs-0.85 recall win was 5-seed noise.

**Implications flagged for the humans:**
1. The Phase 0 exit criterion's "oracle strictly best on recall" cannot be met robustly by any honest greedy oracle in this action space. Proposal: restate the oracle-dominance gate on **total reward** (objective-level), and record the recall finding as a feature of the design, not a failure.
2. Phase 2's exit criterion ("Q-learning beats severity-sort on recall@deadline and MTTD") may be similarly optimistic on the recall half — the learnable headroom is concentrated in reward/wasted-minutes/composite-cost. Flagging now, deciding later with real Q-learning numbers in hand.
3. FIFO-worst (E-002 obs. 1) reconfirmed at 30 seeds (0.141) — that amendment stands.

---

## E-004 — Phase 1 DP: model estimation + VI/PI + evaluation — 2026-08-14

**Status:** valid
**Config:** env_default.yaml + training_default.yaml @ ff6ecec (gamma 0.99, theta 1e-4)
**Command:** `python scripts/run_dp.py`
**Seeds:** estimation=[10000..59999] (50k random episodes) · eval=[101..105]
**Runtime:** estimation 1.2 min · VI 7.6 s (1075 sweeps) · PI 4.9 s (6 rounds)

**Convergence / correctness:** VI converged, final Δ 9.95e-05 < 1e-4 ✓ · **VI/PI policy agreement 100%** ✓ · curve in results/dp_convergence.png.
**Coverage:** 133/576 states, 589/2880 state-action pairs visited; visited-state counts min 1 / median 597 / max 595k. Unvisited pairs = absorbing self-loop, reward 0 (D-011).

**Evaluation (eval seeds, real environment):**
| agent | recall | total reward | MTTD |
|---|---|---|---|
| dp | 0.43 ± 0.17 | **305.9 ± 127.6** | **6.3** |
| oracle_greedy | 0.77 | 214.1 | 15.6 |
| severity_sort | **0.87** | 153.7 | 23.0 |

**THE FINDING — first confirmed reward hacking, found by planning, two phases early.**
DP's policy is ~97% BULK_CLOSE (used as *paid waiting*: 2 min/step, +0.5 per junk closed) plus 4–8 surgical PULL_HIGHEST_SEVERITY strikes per shift. It catches sev-3 incidents almost instantly (MTTD 6.3), abandons 57% of incidents (**recall 0.43 — below random's 0.52**), buries zero (P(real | bulk-eligible) ≈ 0.1%), and still scores highest — beating the truth-seeing oracle by 43% on the reward it optimises.
Why the reward permits this (checked, per CONSTRAINTS #5): (a) bulk-close credit makes waiting profitable; (b) misses are only charged when the deadline expires in-shift (D-009), so ignoring the queue is cheap; (c) exponential decay pays maximum for instant catches. Arithmetic verified against per-step breakdowns (seed 101: two crit-2 instant catches ≈ 492 ✓). The policy also validated in the *true* environment, so this is not an artifact of the estimated model.
**This is the brief §3.5 deliberate trap being sprung — the strongest possible motivation for Phase 5 (RLHF): the hand reward provably rewards behaviour no SOC manager would accept.**

**Flagged for the humans:** Phase 1 exit says "DP beats severity-sort on recall@deadline" — it doesn't (0.43 vs 0.87), *because DP optimises the reward, not recall, and the reward is exploitable*. Decision needed (same shape as the Phase 0 amendment): restate on total reward + document the hack as a headline finding, or treat the reward function as needing a patch (the brief says the trap is deliberate — patching it would delete the RLHF motivation).

**RESOLVED 2026-08-16 (D-012, approved by Pranav; Diya countersign pending):** gate restated on total reward, reward left unpatched, hack promoted to a headline finding and carried into Phase 5 as its primary motivation. Nothing in the measurements above changed — only the criterion they are judged against. See E-005 for the verification work that closed the phase.

---

## E-005 — Phase 1 closure: Bellman machinery verified against a hand-derived answer — 2026-08-16

**Status:** valid
**Config:** the MRP is self-contained (D-013); γ = 0.9 for the example only, *not* the project's 0.99
**Commands:** `python scripts/run_mrp_example.py` · `pytest tests/test_mrp_bellman.py -v`
**Runtime:** < 1 s
**Model:** Claude Opus 5

**Why this experiment exists.** E-004's correctness evidence was entirely *internal*: value iteration converged, and policy iteration agreed with it on 100% of states. But VI and PI share `greedy_policy` and the same Bellman expression — a wrong equation would make both converge, agree perfectly, and be wrong together. No Phase 1 result had been compared to an answer produced outside the code.

**Method.** A five-state Markov Reward Process (QUIET / BACKLOG / INVESTIGATING / CONFIRMED / MISSED), small enough to solve by hand. Rewards booked on the transition and folded into R(s) via S&B eq. 3.5. Solved four ways: by hand on paper; closed form V = (I − γP)⁻¹R; iterative policy evaluation; and — the load-bearing route — the shipped `agents/dp.value_iteration`, run on the MRP expanded into a degenerate MDP whose five actions are identical, so `max_a` collapses to the MRP Bellman equation.

**Hand-derived answer:** R = [−1, −4, +20, 0, 0], and

| state | V (by hand) |
|---|---|
| QUIET | 52/11 = 4.727272… |
| BACKLOG | −4 |
| INVESTIGATING | +20 |
| CONFIRMED | 0 (absorbing) |
| MISSED | 0 (absorbing) |

**Result:** all four routes agree. Largest disagreement with the hand-derived vector **7.11e-15** — floating-point noise at ~15 significant figures. `agents/dp.value_iteration` converged in 44 sweeps, final Δ 8.88e-15. 7/7 new tests pass; suite now 14/14.

**What this does and does not establish.** It establishes that the Bellman backup in `agents/dp.py` is the textbook equation (S&B eq. 3.14 / §4.1 / §4.4), independently of anything else in the repo. It does **not** validate `estimate_model` — the MRP supplies exact dynamics, whereas the 576-state P̂/R̂ are sampled, and their sampling error remains bounded only by the coverage figures in E-004 (133/576 states) and the D-011 unvisited-pair convention. Those are separate concerns and stay open as stated caveats.

**Negative result worth keeping.** The first design booked the incident payoff as `R(CONFIRMED) = +10` on an absorbing state. It diverges — an absorbing state with non-zero reward collects it forever, giving V = 10/(1−γ) = 100 and swamping the chain. Fixed by moving to per-transition rewards. `test_absorbing_states_have_zero_value` now guards against re-introducing it. Recorded because it is a mistake a student would plausibly make on the exam version of this question.
