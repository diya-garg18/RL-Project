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

---

## E-006 — Phase 2 anchor: a hand-derived `q_*` verified against the shipped solver — 2026-08-17

**Model:** Claude Opus 5 · **Phase:** 2 · **Feature:** FEATURE_002 · **Decision:** D-014
**Not a training run.** No agent was trained and no environment episode was executed. Logged here because it establishes a reference answer every Phase 2 result will be measured against, exactly as E-005 did for Phase 1.

**What was run**

```
.\.venv\Scripts\python.exe -m pytest tests/test_tiny_mdp.py -v     ->  13 passed in 0.36s
.\.venv\Scripts\python.exe -m pytest tests/ -q                     ->  27 passed in 0.38s
```

**The object.** A two-state, two-action, deterministic, continuing MDP with γ = 0.9 (`src/soc_triage/tiny_mdp.py`). States `QUIET` / `BUSY`; actions `WAIT` / `WORK`. Rewards: `(QUIET,WAIT) = +1`, `(QUIET,WORK) = −5`, `(BUSY,WAIT) = −1`, `(BUSY,WORK) = +4`.

**The hand-derived answer** (derivation in `docs/features/FEATURE_002_tiny_mdp_qstar.md`):

| | WAIT | WORK | V(s) | π*(s) |
|---|---|---|---|---|
| QUIET | **10.0** | 6.7 | 10 | WAIT |
| BUSY | 10.7 | **13.0** | 13 | WORK |

`V(QUIET) = 1/(1−0.9) = 10` exactly; `V(BUSY) = 4 + 0.9(10) = 13` exactly.

**Result:** three independent routes agree.

| Route | Agreement with the hand-derived answer |
|---|---|
| Bellman optimality residual on `HAND_COMPUTED_Q` (S&B eq. 3.20, no solver) | **1.78e-15** |
| `agents/dp.value_iteration` on the action-padded MDP → V | matches to < 1e-12, final Δ < 1e-14 |
| Q rebuilt from that V as `R(s,a) + γ·P[s,a]@V` | matches to < 1e-12 |

Greedy policy from all routes: `[WAIT, WORK]`, as derived.

**Mutation check — evidence the anchor can fail.** A test that cannot fail is worth nothing, so wrong values were injected deliberately:

```
residual, correct Q        : 1.776e-15
residual, Q(BUSY,WORK)  13.0 -> 13.1: 0.1000
residual, Q(QUIET,WAIT) 10.0 ->  9.9: 0.0900
residual, Q(QUIET,WORK)  6.7 ->  6.8: 0.1000
policy if Q(QUIET,WORK)=99 : [1 1] vs hand [0 1]
```

A 0.1 error sits **thirteen orders of magnitude** above the 1e-12 tolerance. No plausible wrong answer slips through.

**What this does and does not establish.** It establishes a trusted `q_*` for a fixture the Phase 2 learners can be graded against, and it ties that fixture to `value_iteration`, already independently validated in E-005. It establishes **nothing** about Monte Carlo, SARSA or Q-learning — none of them exist yet. It also says nothing about the 576-state environment: the tiny MDP has exact, deterministic dynamics, whereas the real one is sampled and stochastic. A learner passing here has a correct update rule, not a working agent.

**Negative result worth keeping.** Two earlier fixture designs were discarded for giving the optimal action a winning margin of only 0.1 and 0.3 on values near 10 — a 1–3% gap. A tabular learner within 3% of `q_*` is completely ordinary, so the greedy policy would have flipped at random and the test would have been flaky rather than strict. The margin, not the tidiness of the numbers, is what makes a fixture testable; the shipped design has margins of 3.3 and 2.3, frozen as `MIN_ACTION_GAP`. Recorded because "the test is flaky, mute it" is the wrong conclusion to reach later from a fixture that was mis-designed early.

---

## E-007 — Q-learning converges to the hand-derived q\* on the tiny MDP — 2026-08-17

**Model:** Claude Opus 5 · **Phase:** 2 · **Feature:** FEATURE_003 · **Decision:** D-015
**Not a run on the real environment.** This is the tiny 2-state fixture only. No SOC episode was simulated, no eval seed was touched, and no Phase 2 exit criterion was tested.

**What was run**

```
.\.venv\Scripts\python.exe -m pytest tests/test_tabular.py -q   ->  20 passed
.\.venv\Scripts\python.exe -m pytest tests/ -q                  ->  47 passed in 2.56s
```

**Setup.** `QLearningAgent`, α = 0.1, γ = 0.9, ε: 1.0 → 0.1 at 0.99 per episode, 500 episodes × 200 steps, episode start state alternating at random so neither state is starved.

**Result.** The learned Q-table reproduces the pen-and-paper answer of FEATURE_002:

| | WAIT | WORK |
|---|---|---|
| QUIET | 10.000000 | 6.700000 |
| BUSY | 10.700000 | 13.000000 |

`max |Q − q*| = 9.237e-14`. Greedy policy `[WAIT, WORK]` — as derived.

**Convergence speed (seed 0)**

| Episodes | max \|Q − q\*\| | Policy correct |
|---|---|---|
| 10 | 6.248e-02 | **yes** |
| 50 | 9.237e-14 | yes |
| 500 | 9.237e-14 | yes |
| 2000 | 9.237e-14 | yes |

**The policy is right after 10 episodes; the values take ~50 to reach machine precision.** Behaviour converges well before value does — worth remembering, and worth expecting again on the real environment where it will be far less clean.

**Across 5 seeds (500 episodes):** mean 9.237e-14, std **0.000e+00**, policy `[0 1]` every time.

**The zero standard deviation was investigated before being accepted (CONSTRAINTS #5).** Identical results across seeds is exactly what a broken seed looks like. It is not broken: early trajectories genuinely diverge — first 20 actions were `10011101100100011001` for seed 0 against `11000011011010000110` for seed 1, with visibly different Q-tables after 3 episodes — and they converge to the same place. A deterministic MDP has no target noise and a unique fixed point, so exploration order changes the speed of convergence, not its destination. `test_different_seeds_explore_differently_but_reach_the_same_fixed_point` now asserts both halves.

**Negative result worth keeping.** The convergence test was first written with a tolerance of `1e-2`, guessed from what a tabular learner "usually" achieves. The measured error is `9.24e-14` — twelve orders of magnitude tighter. A 1e-2 tolerance would have passed a materially wrong backup. Tightened to `1e-9`. Same lesson as E-006's action-gap finding: **measure the fixture, then set the tolerance.** A plausible-looking number chosen in advance is not a threshold, it is a guess that happens to be written in a test file.

**What this does and does not establish.** It establishes that the Q-learning update rule is the textbook one (S&B §6.5) and that it is distinguishable from SARSA at the level of a single backup. It establishes **nothing** about the SOC environment: 2 states versus 576, deterministic versus stochastic, and 4 state-action pairs versus 2,880. Expect the real thing to be slower, noisier, and — on the evidence of E-004 — quite possibly to rediscover the bulk-close reward hack.

---

## E-008 — Q-learning on the real 576-state environment — 2026-08-17

**Model:** Claude Opus 5 · **Phase:** 2 · **Feature:** FEATURE_004 · **Decisions:** D-015, D-016
**The first learning run on the real environment.** Everything before this was baselines, planning, or a 2-state fixture.

**What was run**

```
.\.venv\Scripts\python.exe scripts\train.py
  5 runs x 20000 episodes, alpha 0.1, gamma 0.99,
  epsilon 1.0 -> 0.05 at 0.9995/episode (floor reached ~episode 6000)
  training seeds 200000+ (D-016), one fresh shift per episode
  training completed in 2.8 min
```

**Headline result — eval seeds 101–105, mean ± std across the 5 runs**

| agent | recall@deadline | total reward | MTTD (min) |
|---|---|---|---|
| **q_learning** | **0.73 ± 0.03** | **270.9 ± 105.5** | **22.0 ± 15.6** |
| severity_sort | 0.87 ± 0.16 | 153.7 ± 218.7 | 23.0 ± 12.4 |
| oracle_greedy | 0.77 ± 0.13 | 214.1 ± 207.6 | 15.6 ± 6.5 |
| dp (E-004) | 0.43 | 305.9 ± 127.6 | — |

> The q_learning spread is across **training runs**; the baseline spreads are across **eval seeds** (one deterministic run each). Different quantities — not comparable as written. Fixing this presentation is a follow-up.

**The Phase 2 exit criterion is NOT met.** It requires Q-learning to beat severity-sort on recall@deadline *and* MTTD. Recall: **0.73 vs 0.87 — fails.** MTTD: 22.0 vs 23.0 — marginally better, and well inside the spread. The gate is not met and is **not** being quietly restated here; see "Decision owed" below.

### Finding 1 — the reward hack recurs, weaker, exactly as D-012 predicted

Action distribution of the run-0 greedy policy over eval episodes:

| action | steps | share |
|---|---|---|
| BULK_CLOSE_LOW_RISK | 297 | **62.3%** |
| PULL_CHEAPEST | 97 | 20.3% |
| PULL_HIGHEST_SEVERITY | 47 | 9.9% |
| PULL_MOST_CRITICAL_ASSET | 28 | 5.9% |
| PULL_OLDEST | 8 | 1.7% |

DP used BULK_CLOSE ~97% of the time and scored recall 0.43. Q-learning uses it 62.3% and scores recall 0.73. **Same pathology, less extreme** — it earns more reward than severity-sort (270.9 vs 153.7) while catching fewer real incidents (0.73 vs 0.87).

This is the outcome D-012 flagged in advance and declined to pre-empt. Two algorithms with nothing in common — exact planning on an estimated model, and model-free sampling — independently discover the same exploit. **That is direct evidence the pathology lives in the reward function, not in any one algorithm**, which is the strongest available motivation for learning a reward from human preferences in Phase 5. Recorded as a headline result, not a defect.

### Finding 2 — the evaluation seed block is not representative *(this one is bigger)*

The greedy diagnostic during training sat around 40–60 reward while the final eval-seed number was 270.9. That gap was investigated before the result was accepted (CONSTRAINTS #5). It is **not** an agent problem — the same agents score very differently on the two seed sets:

| agent | train-diag seeds 1–10 | eval seeds 101–105 | gap |
|---|---|---|---|
| random | −201.1 ± 218.9 | −155.7 ± 148.9 | +45 |
| severity_sort | **−78.7 ± 325.5** | **+153.7 ± 218.7** | **+232** |
| oracle_greedy | +94.0 ± 317.3 | +214.1 ± 207.6 | +120 |

**Every agent, including the oracle, scores far better on the eval seeds.** The five eval seeds are systematically easier than seeds 1–10, and the per-seed standard deviation is enormous — severity_sort's ±325.5 dwarfs the ~117-point difference between it and Q-learning that E-008 is otherwise reporting.

This is not a Phase 2 problem. It affects **E-002, E-003, E-004 and E-008 alike**: every headline comparison in this project so far rests on 5 evaluation shifts drawn from a distribution whose spread is several times larger than the effects being measured. CONSTRAINTS #3 requires ≥5 seeds and that was honoured — but 5 turns out to be far too few for this environment's variance, which is a property nobody had measured until now.

**No numbers were changed and no seed block was altered in response.** Widening the eval block would invalidate the comparability of every prior experiment and is a decision for the humans (see below).

### Other observations

- **The learning curve does not visibly converge.** After epsilon floors at 0.05 (~episode 6000), the greedy diagnostic keeps swinging between −262.9 and +121.3 with no trend. Given Finding 2, most of that is probably shift-to-shift variance rather than instability in the learner — but it has not been separated, and until it is, "Q-learning converged" is not a claim this run supports.
- Curve saved to `results/q_learning_curve.png` (gitignored, regenerable).
- Run-0 alone scores reward 113.6 / recall 0.68 on eval, against the 5-run mean of 270.9 — consistent with the ±105.5 spread, and a reminder that the single saved Q-table is not the reported result.

**Decision owed to the humans (not taken here).** Three questions, in order of importance: (1) is the eval block widened, and if so, does every prior experiment get re-run for comparability? (2) is the Phase 2 gate restated on total reward, as Phase 1's was in D-012 — and if so, does the recall figure travel beside it under the same rule? (3) do the α/γ/ε ablations (ROADMAP box 7) run before or after that is settled? Following D-012's precedent, the gate is left **unmet and unamended** until a human decides.

---

## E-009 — The learned policy, rendered readable — 2026-08-17

**Model:** Claude Opus 5 · **Phase:** 2 · **Feature:** FEATURE_005 · ROADMAP box 6
**Not a new training run.** `scripts/train.py` was re-run only to capture per-(s,a) visit counts, which the agent did not previously record. **The re-run reproduced E-008 exactly** — recall 0.73 ± 0.03, reward 270.9 ± 105.5, MTTD 22.0 ± 15.6 — which is itself the reproducibility check CONSTRAINTS #3 implies.

```
.\.venv\Scripts\python.exe scripts\train.py --no-plot
.\.venv\Scripts\python.exe scripts\policy_table.py    ->  results/policy_table.md
```

**Coverage: 121 of 576 states visited (21%).** The other 455 have all-zero Q rows. They are printed as `·`, never as an action — see the finding below.

### The strategy shift, in the LEARNED policy

Share of visited states in each `time_left` bucket whose greedy action is each of the five:

| time_left | PULL_HIGHEST_SEVERITY | PULL_OLDEST | PULL_MOST_CRITICAL | PULL_CHEAPEST | **BULK_CLOSE** | states |
|---|---|---|---|---|---|---|
| >240m (early) | **34.9%** | 12.0% | 7.2% | 20.5% | **25.3%** | 83 |
| 60–240m (mid) | 28.0% | 20.0% | 8.0% | 8.0% | **36.0%** | 25 |
| <60m (crunch) | **15.4%** | 7.7% | 15.4% | 15.4% | **46.2%** | 13 |

**Both trends are monotonic across all three buckets.** As the shift runs out, the agent works alerts by severity progressively less (34.9% → 28.0% → 15.4%) and bulk-closes progressively more (25.3% → 36.0% → 46.2%).

**Interpretation, and the uncomfortable part.** The roadmap asked for a "behaviourally interpretable strategy shift as time runs out", and there is one. It is even superficially plausible — a real analyst does triage more aggressively under deadline pressure. But read against E-008, the more likely reading is less flattering: the reward charges end-of-shift misses at the very end, and bulk-closing is the cheapest way to clear the queue before that charge lands. This is not obviously an analyst's judgement being learned; it is plausibly the reward hack intensifying exactly where the reward makes it most profitable. **Both readings are consistent with the data and this log does not pick between them.** Distinguishing them needs the per-action reward decomposition in the crunch bucket, which has not been done.

### Finding — the crunch bucket rests on 13 states

The `<60m` column is computed from **13 visited states**, so 46.2% is six states and 15.4% is two. The three-bucket monotonicity is reassuring — a coincidence would not usually line up in both directions across all three — but the headline figure for the report and viva currently has its most interesting column supported by thirteen data points.

Coverage falls off sharply with `time_left` (83 / 25 / 13 states), which makes sense: the crunch is a small slice of each shift and few queue configurations are reachable within it. It is nonetheless a real limitation on the claim, not a footnote.

### Finding — 455 unvisited states would have printed as a confident preference

An unvisited state has an all-zero Q row, so `argmax` returns action 0 by the tie-break rule. Rendered naively, **455 of 576 cells (79%) would have shown `PULL_HIGHEST_SEVERITY`** — a decision the agent never made, in a figure intended for a report and a viva. The agent now records per-(s,a) visit counts for no purpose other than letting the table print `·` instead, and `test_unvisited_states_are_reported_as_unvisited_not_as_action_zero` pins the distinction.

This is the same class of error as D-011's unvisited-pair convention in DP, arriving by a different route: a defensible internal default becoming a false claim the moment it is displayed.

### What this does and does not establish

It establishes what the learned policy does, per state, with data coverage made explicit. It does **not** establish *why* — the strategy shift has two competing explanations and the evidence here does not separate them. It also inherits E-008's eval-seed caveat in full.

---

*(E-008's open decisions are recorded at the end of E-008, above.)*

---

## E-010 — SARSA and Monte Carlo on the real environment — 2026-08-17

**Model:** Claude Opus 5 · **Phase:** 2 · **Feature:** FEATURE_006 · **Decision:** D-017
Same protocol as E-008: 5 runs × 20,000 episodes, own seed block per algorithm (D-016), eval seeds read once at the end. SARSA 3.4 min, Monte Carlo 3.0 min.

**All three learners, eval seeds 101–105, mean ± std across 5 training runs**

| agent | recall@deadline | total reward | MTTD (min) |
|---|---|---|---|
| **sarsa** | 0.74 ± 0.01 | **324.1 ± 81.6** | 23.3 ± 11.7 |
| q_learning | 0.73 ± 0.03 | 270.9 ± 105.5 | 22.0 ± 15.6 |
| monte_carlo | 0.71 ± 0.02 | 177.3 ± 91.7 | **18.6 ± 3.0** |
| dp (E-004) | 0.43 ± 0.17 | 305.9 ± 127.6 | 6.3 |
| severity_sort | **0.87 ± 0.16** | 153.7 ± 218.7 | 23.0 ± 12.4 |
| oracle_greedy | 0.77 ± 0.13 | 214.1 ± 207.6 | 15.6 ± 6.5 |

**SARSA earns the highest reward of any agent in the project — 324.1, above even DP's 305.9.** That is worth pausing on: DP computes the exact optimum *for its estimated model*, and a model-free learner beat it. The most likely explanation is D-004 rather than anything remarkable about SARSA — DP's optimum is only as good as a transition model estimated from 50k random rollouts covering 133 of 576 states, whereas SARSA learned directly from 100,000 real shifts. Not investigated further; recorded as the reading, not the conclusion.

**The result that matters is the agreement, not the ranking.** All three learners land at recall 0.71–0.74, and **all three fall short of severity-sort's 0.87**. Three algorithms — off-policy TD, on-policy TD, and Monte Carlo, which share no update rule between them — trained on disjoint seed blocks, converge on the same trade: substantially more reward, meaningfully less recall.

Combined with DP (E-004), that is **four independent methods finding the same exploit**. The reward function, not the algorithm, is what selects this behaviour. This was D-012's prediction and it is now the single best-supported claim in the project.

**Monte Carlo has the best MTTD (18.6 ± 3.0) and by far the tightest spread.** Unexplained. It is also the weakest on reward. Worth a look before the report, not chased here.

### What this does not establish

Nothing about ranking the three against each other. The differences between them (177 to 324 reward) are of the same order as the eval-seed variance E-008 measured (severity_sort's own per-seed std is ±218.7). **Do not report "SARSA is the best tabular method" from this table.** It is one draw from a noisy measurement, and E-012 below shows that even deliberate hyperparameter changes fail to clear that noise.

---

## E-011 — Cross-agent comparison against DP (ROADMAP box 5) — 2026-08-17

**Model:** Claude Opus 5 · **Phase:** 2 · **Feature:** FEATURE_006
`scripts/compare_agents.py`. DP's Q-table was reconstructed from the converged V and saved by `scripts/run_dp.py` (re-run; reproduced E-004 exactly — 133/576 coverage, 1075 sweeps, final Δ 9.95e-05, VI/PI 100%).

**Coverage — states each agent actually visited, of 576**

| agent | states | state-action pairs |
|---|---|---|
| dp | 133 | 589/2880 |
| q_learning | 121 | 480/2880 |
| monte_carlo | 120 | 530/2880 |
| sarsa | 115 | 517/2880 |

**Policy agreement and max-norm Q distance**

| pair | agree (both visited) | agree (all 576) | max \|ΔQ\| |
|---|---|---|---|
| sarsa vs dp | **43.9%** | 85.9% | 306.7 |
| q_learning vs dp | 36.8% | 84.4% | 320.0 |
| monte_carlo vs dp | 30.1% | 82.8% | 309.8 |
| q_learning vs sarsa | 29.5% | 85.4% | 116.2 |
| q_learning vs monte_carlo | 25.0% | 84.4% | 203.5 |
| sarsa vs monte_carlo | 21.8% | 83.5% | 172.7 |

### Finding — the "all 576" column is a manufactured number

Agreement looks like 83–86% across the board until unvisited states are excluded, at which point it collapses to **22–44%**. The difference is entirely states neither agent has ever seen, where both fall back to a convention that is not a decision: the learners to the argmax tie-break on an all-zero row (FEATURE_005), DP to the D-011 absorbing self-loop. Two agents that have never visited a state "agree" there, and with ~450 such states that artefact dominates.

Any future comparison in this project must exclude unvisited states or it is measuring conventions. This is the third time the same class of error has appeared — D-011, FEATURE_005/E-009, and now here — always as a defensible internal default that becomes a false claim when displayed or compared.

### Finding — near-identical performance from very different Q-tables

Max-norm Q distances of 116–320 between agents whose eval performance sits within 0.03 recall of each other, and policy agreement as low as 21.8% between SARSA and Monte Carlo. The agents are not converging on a common solution; they are finding **different policies of similar value**. Either the reward has a broad plateau of near-equivalent strategies, or the metrics cannot distinguish them. Both readings would matter for the report and neither has been tested.

---

## E-012 — Hyperparameter ablations (ROADMAP box 7) — 2026-08-17

**Model:** Claude Opus 5 · **Phase:** 2 · **Feature:** FEATURE_006
`scripts/ablations.py`, 3 runs × 6,000 episodes per configuration, own seed block (800000+), **measured on the train-diagnostic seeds 1–10 — never on the eval seeds.** An ablation is tuning, and CONSTRAINTS #2 forbids tuning against evaluation seeds whether a program or a human does the reading. Sweep took 4.1 min.

Reduced budget relative to a headline run (20,000 × 5). These numbers rank configurations; they are **not** comparable to E-008/E-010.

| sweep | value | greedy reward on train-diag | individual runs |
|---|---|---|---|
| **alpha** | 0.02 | 76.0 ± 50.0 | [6, 115, 107] |
| | 0.10 *(default)* | 29.2 ± 46.2 | [75, −34, 47] |
| | 0.30 | 4.0 ± 46.0 | [68, −39, −17] |
| **gamma** | 0.90 | 24.0 ± 54.4 | [100, −7, −22] |
| | 0.95 | 55.3 ± 6.0 | [47, 59, 60] |
| | 0.99 *(default)* | 29.2 ± 46.2 | [75, −34, 47] |
| **epsilon_decay** | 0.999 | 111.4 ± 41.1 | [138, 143, 53] |
| | 0.9995 *(default)* | 29.2 ± 46.2 | [75, −34, 47] |
| | 0.9999 | −105.9 ± 139.0 | [−234, −171, 87] |

### Headline: NONE of the three ablations clears the noise floor

The script computes this rather than leaving it to the reader — the spread *between* configuration means against the typical spread *within* a configuration's repeats:

| sweep | between-config spread | within-config spread | verdict |
|---|---|---|---|
| alpha | 29.9 | 47.4 | **not distinguishable from noise** |
| gamma | 13.7 | 35.5 | **not distinguishable from noise** |
| epsilon_decay | 89.6 | 75.4 | **not distinguishable from noise** |

Every sweep's between-config variation is smaller than, or comparable to, the variation between repeats of the *same* configuration. Look at the individual runs: the default configuration alone produced 75, −34 and 47 — a range of 109, wider than the gap between any two alpha settings.

**This is the roadmap box being answered honestly rather than filled in.** The tempting write-up — "alpha=0.02 is best, gamma=0.95 is best, slower epsilon decay helps" — would be reading three random draws as three findings. The epsilon-decay sweep is the only one even suggestive (0.999 at 111.4 against 0.9999 at −105.9), and its between/within ratio is 1.19, nowhere near enough.

The clearest single illustration is alpha=0.02's three runs: **6, 115 and 107**. The same configuration, differing only in seed, spans 109 reward — wider than the gap between any two alpha values in the sweep. Whatever this table appears to say about alpha, that spread says louder that it cannot say it yet.

The sweep is deterministic under its seeds and was re-run to confirm: every row reproduced exactly.

**A negative result, and a useful one.** It is independent confirmation of E-008's variance finding, arrived at from a different direction: this environment's shift-to-shift noise is large enough to swallow deliberate, order-of-magnitude hyperparameter changes. Any future tuning needs far more repeats — or a lower-variance evaluation protocol — before it can claim anything. Recorded so nobody re-runs this sweep expecting a different answer.

---

## E-013 — The strategy shift does not replicate across algorithms — 2026-08-17

**Model:** Claude Opus 5 · **Phase:** 2 · **Feature:** FEATURE_006
**This entry partially retracts an interpretation offered in E-009.** `scripts/policy_table.py --agent {sarsa,monte_carlo}`.

E-009 reported that Q-learning's policy shows a monotonic strategy shift as the shift runs out — bulk-closing rising 25.3% → 36.0% → 46.2% while severity-first falls 34.9% → 28.0% → 15.4% — and offered two readings (analyst-like escalation, or the reward hack intensifying). Running the same figure for the other two learners was the obvious check and it had not been done.

**Bulk-close share by time bucket, per algorithm**

| agent | >240m (early) | 60–240m | <60m (crunch) | direction |
|---|---|---|---|---|
| q_learning | 25.3% | 36.0% | **46.2%** | rises into the crunch |
| monte_carlo | 23.1% | 28.6% | **42.9%** | rises |
| **sarsa** | 47.4% | 51.9% | **25.0%** | **falls — the opposite** |

**Severity-first share, same buckets**

| agent | >240m | 60–240m | <60m |
|---|---|---|---|
| q_learning | 34.9% | 28.0% | 15.4% |
| monte_carlo | 26.9% | 25.0% | 7.1% |
| **sarsa** | 17.1% | 22.2% | **50.0%** |

**SARSA learned the reverse trend**, on the same environment, the same reward and the same protocol. Two of three agree; one contradicts. That is not a robust behavioural property.

**What this means for E-009's claim.** "There is a behaviourally interpretable strategy shift as time runs out" was over-read from a single algorithm. It holds for Q-learning and Monte Carlo and fails for SARSA. Given each crunch bucket rests on only 12–14 visited states — a limitation E-009 did state — a direction disagreement across algorithms is entirely consistent with the trends being noise at that sample size.

**The honest position:** the *existence* of a consistent, interpretable end-of-shift strategy is **not established**. The per-algorithm figures stand as measured; the interpretation does not. Both readings E-009 offered are now under-supported, and the per-action reward decomposition that would have separated them is a lower priority than it looked, because there may be no stable effect to explain.

**Why this was caught.** Nothing failed. The Q-learning figure was internally consistent, monotonic across three buckets, and had a plausible story attached. It took running the identical analysis on two more agents — which cost one command each once the script took an `--agent` flag — to find that the story does not replicate. Replication across algorithms should be the default for any behavioural claim in this project, not an afterthought.

---

## E-014 — Every result re-measured on 30 eval seeds. Most of them change. — 2026-08-17

**Model:** Claude Opus 5 · **Phase:** 2 (with consequences for 0 and 1) · **Decision:** D-019
**This is the most consequential entry in this log.** The evaluation seed block was widened from 5 seeds to 30 (D-019), and every agent in the project was re-measured. **No prior entry has been altered or deleted** (CONSTRAINTS #4); E-002 through E-013 stand as recorded, and the original five seeds are a subset of the new thirty, so old numbers are a sub-sample of these rather than being orphaned.

Nothing about training changed — Q-tables and the E-011 comparison reproduced byte-identically, confirming the learners never saw the eval block.

### The table

Mean ± std over eval seeds 101–130. Learner rows are across 5 training runs; baseline and DP rows are across the 30 seeds.

| agent | recall@deadline | total reward | MTTD | reward on 5 seeds (old) |
|---|---|---|---|---|
| oracle_greedy | **0.87 ± 0.16** | **168.0 ± 232.9** | 39.8 | 214.1 |
| q_learning | 0.72 ± 0.04 | 47.6 ± 52.0 | 31.8 | 270.9 |
| sarsa | 0.66 ± 0.04 | 40.5 ± 49.4 | 36.3 | 324.1 |
| severity_sort | 0.84 ± 0.16 | 40.4 ± 220.1 | 28.3 | 153.7 |
| monte_carlo | 0.70 ± 0.05 | −16.4 ± 77.0 | 42.6 | 177.3 |
| **dp** | **0.23 ± 0.24** | **−201.2 ± 438.5** | 4.8 | **+305.9** |
| random | 0.52 ± 0.25 | −304.0 ± 191.6 | 93.9 | — |
| cheapest_first | 0.47 ± 0.25 | −318.5 ± 316.2 | 45.1 | — |
| fifo | 0.16 ± 0.13 | −665.3 ± 228.8 | 196.2 | — |

### Finding 1 — DP inverts completely: +305.9 becomes −201.2

The DP policy was Phase 1's headline result and the highest reward of any agent in the project. On 30 seeds it is **the worst of every planned or learned agent**, at −201.2 ± 438.5, with recall collapsing from 0.43 to 0.23. Its standard deviation, ±438.5, is the largest of any agent — it is not merely worse, it is wildly inconsistent.

**This falsifies criterion 2 of Phase 1's amended exit criterion** (D-012: "The DP policy achieves the highest mean total reward of any agent on the evaluation seeds"). That criterion was measured on five seeds and is false on thirty.

The most likely explanation is D-004 compounded by D-011: DP's policy is optimal for a transition model estimated from 133 of 576 states, and on shifts that wander outside that estimated core it has no useful guidance and falls back to low-value actions. Five seeds happened not to expose that; thirty do. **This is a hypothesis, not a measurement** — it has not been tested by, for instance, correlating per-seed DP reward against how far each shift strays from the visited state core. That test is the obvious next step and has not been run.

### Finding 2 — no learner beats severity-sort on reward any more

On 5 seeds every learner appeared to beat severity-sort's 153.7 by a wide margin. On 30 seeds: q_learning 47.6, sarsa 40.5, severity_sort 40.4. Given severity-sort's ±220.1 spread, **these are indistinguishable.** Monte Carlo (−16.4) is worse.

So the story that survived E-010 — "the learners buy more reward at the cost of recall" — **does not survive honest measurement.** They pay the recall (0.66–0.72 against 0.84) and get no reliable reward advantage for it. On this evidence the tabular learners are simply worse at this task than a severity sort.

### Finding 3 — but the learners are far more *consistent* than the heuristics

The one result that improves under widening, and it was invisible at 5 seeds:

| agent | reward std |
|---|---|
| q_learning | **±52.0** |
| sarsa | **±49.4** |
| monte_carlo | ±77.0 |
| severity_sort | ±220.1 |
| oracle_greedy | ±232.9 |
| dp | ±438.5 |

The learned policies are **four times more consistent shift-to-shift** than severity-sort or even the oracle. A SOC lead choosing between a policy averaging 40 ± 220 and one averaging 48 ± 52 would not be indifferent, and nothing in the reward function expresses that preference. Recorded as a genuine finding, and as further evidence the hand-written reward does not capture what a human values.

### Finding 4 — the oracle now out-recalls severity-sort, contradicting a Phase 0 rationale

Oracle recall 0.87 against severity-sort 0.84. Phase 0's second gate amendment was justified partly by the claim that "in this deliberately coarse action space no honest greedy oracle can reliably out-recall severity-camping" (E-003). On 30 eval seeds the oracle does edge it out. Phase 0's gate as written still **passes** (it requires the oracle strictly best on total reward, which holds: 168.0 vs 40.4), so Phase 0 is not reopened — but the reasoning attached to the amendment is weaker than it was, and the report should not repeat that sentence as established.

### What this says about the reward-hacking narrative

The narrative is **not** destroyed, but it must be restated. What is still true: the reward is exploitable, and every agent trades recall away chasing it. What is no longer true: that the trade pays off. The exploit produced high reward on five particular shifts and does not generalise.

That is arguably a *stronger* case for Phase 5, not a weaker one. A hand-written reward that merely gets gamed is bad. A hand-written reward that gets gamed *and* fails to deliver even the reward it was gamed for is worse — it means the objective is not just misaligned with what we want, it is unstable. Learning a reward from human preferences is the response to exactly that.

### The methodological lesson, which is the real result

Five seeds satisfied CONSTRAINTS #3's "at least 5 seeds". Every headline number in Phases 0, 1 and 2 was computed correctly, reported with a standard deviation, and reproduced deterministically — **and one of them had the wrong sign.** The constraint was met and the measurement was still misleading, because nobody had checked whether five samples could resolve the effects being claimed.

The std was visible the whole time. E-002 reported severity-sort at ±218.7 next to a mean of 153.7 and nobody drew the inference. **Reporting a standard deviation is not the same as reading it.** The check that was missing is trivial: compare the spread to the size of the effect being claimed before believing the effect.

`tests/test_eval_protocol.py` now encodes the seed-count floor so this cannot silently regress.

---

## E-015 — E-014's explanation for the DP collapse is WRONG — 2026-08-18

**Model:** Claude Opus 5 · **Phase:** 1 (reopened) · **Decision:** D-022
`scripts/dp_collapse.py`. **This entry refutes a hypothesis this log proposed one day earlier.**

**What E-014 claimed (as an explicitly untested hypothesis).** DP's policy is optimal for a model estimated over 133 of 576 states, so on shifts that stray outside that estimated core it has no useful guidance and falls back on the D-011 absorbing-self-loop convention. Predicted: per-seed DP reward should fall as the share of off-core steps rises.

**The test.** Run the DP policy on all 30 eval seeds. For each, measure the share of steps spent in states never seen during estimation (*off-core state*), the share where the action DP chose was never observed from that state (*off-core pair* — the D-011 convention actually firing), and the resulting reward. Severity-sort is run on the identical seeds as a control: if DP's bad seeds are just *hard* seeds, severity-sort should suffer on them too.

**Result — the hypothesis is refuted, decisively.**

| measure | value |
|---|---|
| off-core **state** share | **0.0% on every one of the 30 seeds** |
| off-core **pair** share | **0.0% on every one of the 30 seeds** |
| corr(off-core share, DP reward) | undefined — zero variance |
| corr(severity reward, DP reward) | **+0.085** |

**DP never leaves its estimated core.** Not once, on any eval seed. Every state it visits was seen during estimation, and every action it takes was observed from that state, so **the D-011 convention never fires at evaluation time at all.** Both D-011 and the coverage figure are exonerated as explanations. The correlation the hypothesis predicted cannot even be computed, because the predictor is constant.

**And it is not seed difficulty either.** corr(severity, DP) = +0.085 — essentially none. DP fails on seeds where severity-sort does fine:

| seed | DP reward | severity reward |
|---|---|---|
| 101 | **+470.8** | +287.1 |
| 102 | +136.6 | −277.2 |
| 127 | **−985.8** | −106.5 |
| 128 | **−755.0** | **+233.2** |

DP's range across seeds is roughly −986 to +471. On seed 128 it loses 755 while severity-sort gains 233 on the same shift.

### The new hypothesis — also untested, and flagged as such

If the states are in-core, then the model is being consulted about situations it *has* seen, and still giving bad advice. The likeliest remaining explanation is **distribution shift in the estimate, not gaps in it**: `P̂`/`R̂` were counted under a **uniform-random** policy (E-004), but DP then behaves nothing like random — it bulk-closes ~97% of the time. The transitions that actually follow *DP's own* actions are therefore not the transitions the estimate was built from, even though the states are familiar. That is the textbook model-based RL failure: a model accurate for `π_random` used to plan for `π_greedy`.

This is a **hypothesis with no test behind it yet**, stated so the next session does not have to rediscover the reasoning. The obvious test: re-estimate `P̂`/`R̂` from rollouts of the DP policy itself and check whether value iteration on that model produces a policy whose predicted value matches its measured reward. If it does, distribution shift is confirmed and D-004's caveat needs strengthening from "optimal for the estimated model" to "optimal for a model of a policy it does not follow".

### What this changes

D-004's caveat has been correct all along, but for a subtler reason than anyone was stating. Everyone — including E-014 — read "optimal for the estimated model, not the true environment" as being about *coverage*. It is not. Coverage is fine on the eval distribution. The gap is between the policy the model describes and the policy being planned.

### Why this is worth the entry

E-014 offered its explanation with an explicit "this is a hypothesis, not a measurement" and a named test. Running that test took one script and refuted it in one number — 0.0%, thirty times over. **A labelled hypothesis is cheap to kill; an unlabelled one becomes folklore.** Had E-014 asserted the coverage story as fact, it would now be in the report, sounding entirely plausible, and wrong.

---

## E-016 — 20 DQN runs collapsed to BULK_CLOSE. The Huber delta was 1.0. — 2026-08-19

**Model:** Claude Opus 5 · **Phase:** 3 · **Decisions:** D-029, D-030
`results/dqn_runs/dqn_delta1_E016/` (20 runs, kept). **The entire first Phase 3 sweep is discarded as a result and kept as evidence.**

### What was run

30 control runs x 20000 episodes were launched overnight on the config at commit `c4e613e`. Twenty finished before the sweep was stopped. Every one of them failed in the same way.

| measure | DQN (20 runs) | fifo | random | severity_sort | tabular Q-learning |
|---|---|---|---|---|---|
| recall@deadline | **0.0086** | 0.141 | 0.545 | 0.826 | 0.73 |
| total reward | **-480 to -520** | -702.2 | -270.9 | +50.6 | +270.9 |

Recall is an order of magnitude below the worst baseline in the project. The greedy diagnostic was **pinned at -515.4 at 36 of 40 checkpoints**, from episode 500 — before the agent had learned anything — to episode 20000. Training reward moved the wrong way as exploration decayed: -266.8 over episodes 0-2000 (epsilon ~1.0) down to -403.4 over 18000-20000 (epsilon 0.05). **The learned greedy policy was worse than random**, and more exploration made it better.

### It was not a broken network

The obvious guess — dead units, collapsed weights, a wiring bug — is wrong, and checking cost one script:

| measure | value | reading |
|---|---|---|
| std of Q ACROSS STATES, per action | 15.1, 15.4, 15.7, 14.8, 14.8 | Q varies with the state |
| std of Q ACROSS ACTIONS, per state | mean 3.98 | actions are distinguished |
| best-minus-second-best gap | mean 4.79, min 0.10 | the argmax is not a coin flip |
| greedy action distribution | **BULK_CLOSE 99.37%** | the policy is degenerate, the network is not |
| final training loss | ~0.04-0.3 | it fits its targets *precisely* |

A network converging to a low loss while producing a useless policy means it learned the wrong objective, not that it failed to learn.

### Root cause

`F.huber_loss(predicted, target)` was called without a `delta`, so it used torch's default of **1.0**. Measured over 2381 real transitions:

```
per-step reward : mean -2.167  std 46.448  min -1499.5  max +1.5
TD error        : median |0.139|   p99 |0.967|   max |1454.6|
fraction of |TD| > 1.0 (the delta) : 0.9%
```

`env_default.yaml` prices burying a real incident at **-150** and an end-of-shift miss at **-200 x asset multiplier**. Those 0.9% of transitions are the environment's entire lesson, and every one of them sat deep in Huber's *linear* regime, where the gradient is flat. The bug in one number, from two otherwise identical agents given one backup each:

```
routine error (TD = -1)    grad norm  2.543705
buried incident (TD = -150) grad norm 2.579709
ratio  1.014          <- a 150x larger error, a 1.4% larger gradient
```

So the agent learned the small frequent rewards to high precision and was told, in effect, that burying a real incident is a rounding error. BULK_CLOSE pays +0.5 per junk alert closed and appeared to cost nothing. It took it 99.4% of the time.

The comment in `training_default.yaml` said Huber was chosen because it is "less sensitive to the large negative outlier rewards". That was exactly backwards: **those penalties are the signal, not outliers to be suppressed.** Tabular Q-learning uses the raw TD error and reaches recall 0.73 on the identical environment — the comparison that should have raised the question earlier.

### Choosing the replacement — and a criterion that was not good enough

Delta was swept over 10/25/50/100/200 x 3 seeds x 3000 episodes. The selection rule was written down first: any run whose final greedy sits in the always-bulk-close band (below -450) counts as collapsed; a delta with any collapsed run is disqualified; survivors ranked on mean greedy over the last third; ties broken on volatility.

| delta | collapsed@final | checkpoints in collapse band | last-third mean | SEM | volatility |
|---|---|---|---|---|---|
| 10 | **3/3** | **36/36** | -515.4 | 0.0 | 0.0 |
| 25 | **1/3** | 14/36 | -284.9 | 106.1 | 157.5 |
| 50 | 0/3 | 0/36 | 4.8 | 41.2 | 128.0 |
| 100 | 0/3 | 1/36 | 27.6 | 39.8 | 120.7 |
| 200 | 0/3 | 0/36 | 24.6 | 38.1 | 93.0 |

**The rule as written picks delta = 100, and it should not have been followed.**

```
d50  vs d100 : |diff| 22.8  SEM 57.3  ratio 0.40   NOT RESOLVABLE
d50  vs d200 : |diff| 19.8  SEM 56.2  ratio 0.35   NOT RESOLVABLE
d100 vs d200 : |diff|  3.0  SEM 55.1  ratio 0.05   NOT RESOLVABLE
```

A 3.0-point margin against a standard error of 55 is noise, and ranking on it is the identical error E-008 made and E-014 retracted. The rule was under-specified: it had no resolvability gate. What the experiment actually establishes is a threshold, not a ranking — **delta must be at least 50**; above that this design cannot choose, and with a spread of ~69 across seeds no feasible number of seeds would.

The value was therefore taken from the config rather than the data. **200 is the largest named single-event penalty in `env_default.yaml`**, so every individual penalty the agent must learn stays quadratic and only the compound multi-miss tail (observed to -1499.5) is linearised. See D-029.

### Verification

Through the real trainer and shipped config, 3 runs x 3000 episodes — 15% of the training budget:

| | before (delta 1.0) | after (delta 200) |
|---|---|---|
| recall@deadline | 0.0086 | **0.48 +- 0.21** |
| total reward | -480 to -520 | -49.4 +- 136.6 |
| MTTD (min) | — | 17.1 (severity_sort 28.3, oracle 39.8) |
| greedy curve | pinned at -515.4 | 113.9 / 61.7 / 9.7 / -118 / 153.1 / 130.1 |

**Stated plainly: the collapse is fixed, the agent is not yet good.** At 3000 episodes it is still below severity_sort on recall (0.48 vs 0.84) and reward (-49 vs +40), still volatile, and 16 of 90 eval episodes caught nothing at all. Whether the full 20000-episode budget closes that gap is the open question the corrected sweep exists to answer.

`config.py` now refuses `huber_delta < 50` with this measurement in the error message, and `tests/test_dqn.py` asserts that a -150 penalty moves the network more than 10x as much as a routine -1 error — the assertion that fails on the old code.

### Why this is worth the entry

The failure was silent in every direction that matters. The loss curve looked *excellent* — converging to 0.04. The network was healthy by every structural check. Nothing errored, nothing warned, and 20 runs completed successfully. The only visible symptom was a number in a results table being bad, and the project's own habit of treating a suspicious result as a bug report rather than a finding is the only reason it was caught before being written up as "DQN underperforms tabular Q-learning".

The near-miss is worth recording too. The plan had been to let all 60 runs finish overnight and analyse in the morning. Had that happened, the ablations would have been ablations of a broken agent, and the phase's conclusion would have been drawn from 60 runs of a policy that closes every alert unread.

---

## E-017 — Phase 3 gate NOT met. Replay is essential; the target network is harmful. — 2026-08-19

**Model:** Claude Opus 5 · **Phase:** 3 · **Decisions:** D-029, D-030 · **Supersedes nothing** — E-016 remains the record of the discarded first sweep.
`results/dqn_runs/{dqn, dqn_no_replay, dqn_no_target_network}/`, config hash `679eaa992c7f`.

### Setup

46 runs x 20000 episodes, all at `huber_delta: 200.0` (D-029). Control **n=30**, each ablation **n=8**.

**The ablations ran at 8, not the 15 D-027 specified.** The machine could not sustain the sweep — see D-030 — and with a hard deadline the choice was 8 per ablation or an unequal, arbitrary number of both. Precision was kept where the gate turns (the control) and cut where D-027 already argued it matters less. 8 is above the CONSTRAINTS #3 floor of 5. Stated here rather than reported as whatever count happened to finish.

### Result 1 — the exit criterion is NOT met

Exit criterion: *"DQN matches or beats tabular Q-learning on the same evaluation seeds, and the two ablations visibly destabilise training in the plots."* **Both halves fail.**

| agent | recall@deadline | total reward | MTTD (min) |
|---|---|---|---|
| tabular q_learning | **0.73** | — | **46.5** |
| severity_sort | 0.826 | +50.6 | 26.5 |
| random | 0.545 | -270.9 | 98.4 |
| **DQN control (n=30)** | **0.48 +- 0.19** | **-46.87 +- 145.22** | 45.78 +- 51.40 |
| fifo | 0.141 | -702.2 | 198.6 |

The learning curve has **plateaued**: last quarter 18.5 against previous quarter -14.6, a difference of +33.1 inside a run-to-run spread of +-150.0. More episodes would buy nothing. This is a converged agent that is worse than the lookup table it was meant to improve on, not an undertrained one.

**On reward the comparison is not resolvable, and that is stated before the sign.** Rolling both agents over the identical 30 eval seeds (`compare_dqn_tabular.py`, the D-028 paired protocol):

```
PAIRED per eval seed - total reward, DQN minus tabular:
  mean difference : -75.71      std of the diff : 292.50
  standard error  :  53.40      DQN wins on     : 13/30 seeds
  |mean| / SEM    :   1.42      -> NOT RESOLVABLE at 30 seeds
```

So the gate fails on **recall and MTTD**, where the gaps are 6-8 SEM, not on reward. An earlier reading in this session claimed the reward gap was ~12 SEM by comparing the 30-run DQN mean against E-008's +270.9 — a different protocol, not a same-seed comparator. That comparison was wrong and is retracted here. The `47.6` printed by `aggregate_dqn.py` is likewise a hardcoded string, not a computed value; `compare_dqn_tabular.py` is the only authoritative comparator.

### Result 2 — the premise of the phase was confirmed, and it still did not help

```
states the DQN visited at eval       : 42/576
agreement per VISIT                  : 1687/5626 = 30.0%
buckets where the DQN chose >1 action: 21/42
```

In **half the buckets it visited, the DQN chose different actions for situations the 576-bucket discretisation merges.** The continuous state genuinely distinguishes cases the buckets cannot — which is exactly what Phase 3 set out to test. The extra resolution is real. It simply did not convert into better triage.

That is the most interesting sentence in this entry: the hypothesis behind the phase was *correct* and the phase still failed its gate.

### Result 3 — replay is essential, and our own instability measures could not see it

```
no replay, all 8 runs:  recall 0.0000   reward -520.5   final greedy -515.4
```

Identical to four decimal places across eight different seeds. That is the E-016 always-BULK_CLOSE collapse: without replay the DQN does not learn at all. **Replay is not a refinement in this environment; it is load-bearing.**

**`scripts/dqn_ablations.py` reported the opposite:**

```
condition            runs   final   volatility   end std   drawdown
control                30    18.5        167.8      44.2      635.2
no replay               8  -515.4          0.0       0.0        0.0
  no replay: NO clear destabilisation (all three within 1.5x of control) - a NEGATIVE RESULT
```

All three measures — volatility, end-std, drawdown — are **0.00**, because a collapsed constant policy is perfectly *stable*. The ratios read `x0.00`, which the script's rule interprets as "less unstable than control". The measures were defined before looking at the data, which was the right instinct and is why they were trusted; but they detect **wobble**, and are blind to a **flatline**.

The script's printed verdict is a defect, not a finding, and the conclusion is the reverse of what it says. Recorded as a follow-up owing a `BUG_003`: any instability measure used as a phase gate must first check that the agent learned *anything*, because "did not move" and "did not destabilise" are indistinguishable to a variance statistic.

### Result 4 — the target network makes this agent WORSE

|  | n | recall | total reward | volatility | drawdown |
|---|---|---|---|---|---|
| control (replay + target net) | 30 | 0.481 +- 0.190 | -46.9 +- 147.7 | 167.8 | 635.2 |
| **no target network** | 8 | **0.588 +- 0.093** | **+43.5 +- 67.6** | **59.0** | **308.9** |

```
reward diff (no_target - control) = +90.3   SEM 36.0   ratio 2.51  -> RESOLVABLE
recall diff                       = +0.107  SEM 0.048  ratio 2.25  -> RESOLVABLE
```

Removing one of DQN's two headline stabilisers improved recall, reward, volatility and drawdown simultaneously. Both differences clear the ratio-2 bar this project uses (R6, E-014). Every one of the 8 runs finished above the control mean on the greedy diagnostic (98.6 to 141.3, against control's 18.5).

**Honest limits on this one.** n=8 against n=30, and ratios of 2.5 and 2.25 clear the bar without being overwhelming — this is a result to state, not to build on. The mechanism is unexplained. A plausible story is that with `target_update_every: 1000` gradient steps the frozen target is stale enough to hold the estimate back on a 19,461-parameter network that is not at risk of the divergence the target network exists to prevent — but that is a hypothesis with no test attached, and per E-015's lesson it is labelled as such rather than written as explanation.

### What this phase actually established

1. The DQN does not beat tabular Q-learning here. **Gate not met, and not restated** — the same treatment D-012 and D-020 gave Phases 1 and 2. A human decides whether to amend it.
2. **Experience replay is essential.** Without it, learning fails completely.
3. **The target network is counterproductive** at this scale, by a resolvable margin.
4. The continuous state does distinguish situations the buckets merge (21 of 42 visited buckets), so the phase's premise held even though its gate did not.
5. Two of our own analysis tools were wrong in ways that would have been reported as fact: a hardcoded comparator in `aggregate_dqn.py`, and instability measures in `dqn_ablations.py` that score total collapse as maximal stability.

### Why this is worth the entry

Three of the four results here contradict something we expected. The one that should travel furthest is Result 3: **we wrote the instability measures down in advance precisely so we could not fit them to the data afterwards, and they still gave the wrong answer** — because choosing a metric early protects against one failure mode (post-hoc rationalisation) and not against another (measuring the wrong quantity). Pre-registration is necessary and not sufficient. The only reason the error was caught is that the per-run numbers were read directly instead of trusting the summary line.

---

## E-018 — REINFORCE runs, and at 300 episodes it has become severity-sort exactly — 2026-08-23

**Model:** Claude Opus 5 · **Phase:** 4 · **Decisions:** none yet — this is a diagnostic, not a result.
Smoke runs only. Artefacts deleted after measurement (D-018); regenerate with the command below.

### Status of these numbers

**NOT a result and must not be quoted as one.** One repeat, 300 episodes, no seed variation:
CONSTRAINTS #3 requires >=5 seeds for any headline figure. This entry exists because two of
the observations change what the next session should do, and because the project logs what it
measured rather than only what it wanted.

```powershell
.\.venv\Scripts\python.exe scripts/train_reinforce.py --episodes 300 --repeats 1 --eval-every 300 --no-plot
```

### 1. The agent trains, and the cost is ~24 min per full run

300 episodes took ~22 s of training (36.9 s wall including imports and one 10-episode greedy
diagnostic). That is **~0.073 s/episode**, so a 20,000-episode run projects to **~24 minutes** —
the same order as the DQN's measured 27 min.

**This projection is stated with D-024's and D-030's warning attached**: it was taken on a short
run on a quiet machine, which is exactly the condition under which this project's compute
estimates have been wrong twice. Treat 24 min as a lower bound, not a plan.

### 2. At 300 episodes the greedy policy IS severity-sort, bit for bit

| | recall@deadline | total reward | std |
|---|---|---|---|
| REINFORCE greedy, 300 episodes | **0.8443** | **40.44** | +-220.12 |
| severity_sort (E-014, 30 eval seeds) | **0.84** | **40.4** | +-220.1 |

Identical to every digit reported. Per CONSTRAINTS #5 this was treated as a bug report and
checked before being written down.

**It is not a bug.** `baselines.py:55-56` defines `severity_sort` as a constant policy —
`act()` returns `PULL_HIGHEST_SEVERITY` unconditionally. Counting the greedy REINFORCE policy's
actions across all 30 evaluation episodes gives **`PULL_HIGHEST_SEVERITY` 1131 times and nothing
else**. Two policies that emit the same action in every state produce identical trajectories on
identical seeds, so identical metrics are the correct outcome, not a coincidence.

**What it means, stated carefully.** After 300 episodes REINFORCE has *rediscovered the industry
baseline* — which is a real thing to have learned from scratch, and also means it has **matched
severity-sort, not beaten it**, against a Phase 4 exit criterion that asks for beating it.

**What is genuinely open:** whether this is a way-point or a terminus. The greedy reading is
already degenerate at 300 episodes — one action, every state — which is what premature
convergence looks like. `reinforce:` has no entropy bonus (the `actor_critic:` section does have
`entropy_coef`), so nothing in the configuration resists a policy sharpening early. Whether
20,000 episodes moves off this point or is pinned to it is the first question the next run
answers, and **either answer is publishable** — "policy gradient converges to the human heuristic
and stops" would be a genuinely interesting finding about this environment's reward.

### 3. The gradient is being clipped by two orders of magnitude

Logged pre-clip policy-gradient norms: **1584.68** at episode 10, **2228.17** at episode 300,
against `grad_clip_norm: 10.0`. The clip is therefore active on essentially every step, at a
ratio of ~150-220x.

That makes the update, in practice, a **fixed-size step in the gradient's direction** — the
magnitude information the `(G_t - b(s_t))` factor carries is discarded before the optimiser sees
it. It is doing what the config asks and nothing is failing, which is precisely the shape of
problem E-016 was: a defensible-looking setting quietly changing what algorithm is running.

**Not changed here.** `grad_clip_norm` is a tuning parameter and the eval seeds have already been
touched by these smoke runs' evaluation step, so tuning against anything measured today would be
the mistake CONSTRAINTS #2 exists to prevent. The clean move is a train-seed-only comparison at
two or three clip values, run as a named experiment. Flagged for a human in `HANDOVER.md`.

### What this changes for the next session

1. The full run is affordable (~24 min x repeats) — but it needs approval before launching
   (CLAUDE.md: >10 min).
2. Do not report any Phase 4 number against "beats severity-sort" until the humans have taken
   the one decision covering Phases 1-3's unmet gates. Matching severity-sort exactly is precisely
   the awkward case that decision has to cover.
3. The clipping ratio deserves a named experiment before the full sweep, not after.


---

## E-019 - the REINFORCE gradient clip does not distinguish itself on reward, and two other things fell out - 2026-08-25

**Model:** Claude Opus 5 · **Phase:** 4 · **Machine:** Diya's PC · **Decisions:** none taken; `grad_clip_norm` left at 10.0.

E-018 flagged that pre-clip policy-gradient norms were 1584-2228 against a configured
`grad_clip_norm` of 10.0, so the clip fires on essentially every step and the update becomes a
**fixed-size step along the gradient direction** - the `(G_t - b(s_t))` magnitude divided out
before the optimiser sees it. E-018 declined to change it, because tuning against numbers
measured on already-touched eval seeds is what CONSTRAINTS #2 forbids. This is the clean
train-seed-only replacement.

```powershell
.\.venv\Scripts\python.exe scripts/reinforce_clip_experiment.py
```

3 values x 3 repeats x 1500 episodes, trained on `reinforce.clip_experiment_seed_start`
(1800000 - its own block, D-016), measured on the train-diagnostic seeds (1-10).
**The eval block was never touched**, and `_assert_no_eval_seeds` makes that a runtime failure
rather than a comment. Total wall time **3.7 min**.

### Status of these numbers

**Reduced budget, and not quotable beside a headline REINFORCE result.** 1500 x 3 against a
reported run's 20000 x 5. These rank clip values; they do not measure REINFORCE.

### 1. The headline: the clip value does not clear the noise floor

| clip | greedy reward (mean +- std) | mean pre-clip norm | clip fired |
|---|---|---|---|
| 10.0 (shipped) | -369.8 +- 205.9 | 3144.0 | 100.0% |
| 100.0 | -202.1 +- 223.2 | 3020.4 | 99.7% |
| 2000.0 | -224.2 +- 205.9 | 1828.4 | 30.4% |

Spread **BETWEEN** values: **74.4**. Spread **WITHIN** a value: **211.6**.

The between-value spread is a third of the within-value spread, so any ranking read off this
table is a random draw. **`grad_clip_norm` stays at 10.0** - not because 10.0 was vindicated,
but because nothing here justifies moving it, and moving it on a difference the noise swallows
is precisely the mistake E-012 documented.

Note what the third column does confirm: at 10.0 and 100.0 the clip fires on ~100% of updates,
and only at 2000.0 does it fall to 30.4%. So E-018's *mechanical* claim is correct - the clip
really is active on essentially every step at the shipped value. It simply does not follow that
this costs anything measurable in reward.

### 2. The greedy policy is degenerate at EVERY clip value

The per-run greedy diagnostics are not spread across a range. They land repeatedly on the same
few numbers:

```
clip 10.0     -515.4, -515.4,  -78.7
clip 100.0     -78.7, -515.4,  -12.2
clip 2000.0    -78.7, -515.4,  -78.7
```

Seven of nine runs produced exactly -515.4 or -78.7. Those are constant-action policies, and
**-515.4 is the same value Phase 3's collapsed DQN produced (E-016)** - the BULK_CLOSE signature.

So the premature convergence E-018 observed at 300 episodes **is not caused by the gradient
clip**. It happens at a clip firing 100% of the time and at one firing 30% of the time, equally.
That rules out the most obvious suspect, and it is why `actor_critic:` was given an entropy bonus
rather than inheriting REINFORCE's exploration story unchanged (see E-020).

### 3. The sampled policy BEATS its own argmax, and the eval protocol reads the argmax

The finding worth carrying into the report, because it questions how every Phase 4 number gets
measured. Sampled reward over the last 100 training episodes, beside the greedy read of the
*same* policy at the *same* moment:

| clip | sampled (last 100) | greedy read |
|---|---|---|
| 10.0 | +111.7, +25.7, +59.0 | -515.4, -515.4, -78.7 |
| 100.0 | +50.1, +22.4, +6.0 | -78.7, -515.4, -12.2 |
| 2000.0 | +6.5, +22.1, +33.3 | -78.7, -515.4, -78.7 |

**Every one of the nine runs collects positive reward while sampling and scores negative when
read greedily.** These are not different policies - the greedy read is `argmax_a pi(a|s)` of the
policy that produced the sampled column.

That is not a paradox, and the explanation is the point: the argmax of a spread-out policy throws
away the mixing that was doing the work. If pi puts 0.4 on PULL_HIGHEST_SEVERITY and 0.35 on
BULK_CLOSE, its argmax is a *pure* severity policy, and the agent that earned +111 was playing
neither. **A stochastic policy's argmax is not that policy**, and this environment appears to
reward the mixture over either pure strategy.

**What this puts in question.** `train_reinforce.py` and `train_actor_critic.py` both evaluate
through a `_GreedyView`, and E-018's headline finding - "REINFORCE has become severity-sort
exactly" - is a statement about the argmax, not about the agent. Whether Phase 4 reports the
sampled policy, the greedy one, or both is **a decision for the humans**, now listed in
`HANDOVER.md`. It is not a decision to take while looking at which one scores better.

### What this changes for the next session

1. `grad_clip_norm` is settled at 10.0 and needs no further work.
2. The degenerate-greedy question moves from "maybe the clip" to "the policy genuinely sharpens
   early", which is what the actor-critic's entropy bonus exists to resist.
3. **A new decision is owed** - greedy vs sampled evaluation - and it affects every Phase 4
   number, including E-018's.

---

## E-020 - `entropy_coef: 0.01` is ~1000x too small, and it is the third instance of one pattern - 2026-08-25

**Model:** Claude Opus 5 · **Phase:** 4 · **Machine:** Diya's PC · **Decisions:** none taken; `entropy_coef` left at 0.01.

### Status of these numbers

**The sweep RAN on 2026-08-25** (4 values x 3 repeats x 80 episodes, 6.7 min) and its result is
in "The sweep result" below. `entropy_coef` was set to **1.0** on the strength of it.

**Budget caveat, stated up front.** 80 episodes x 3 repeats against a headline run's 20000 x 5.
That is enough to answer the structural question this sweep was built for - does the policy stay
non-degenerate - and **not** enough to rank values on reward. The reward column is reported below
and explicitly not used.

### How it surfaced

The first smoke run of `scripts/train_actor_critic.py` (60 episodes, 1 repeat) collapsed:

```
repeat 0  ep 30/60  sampled -508.1  td_err_std 32.79  entropy 0.000  greedy(train-diag) -515.4
repeat 0  ep 60/60  sampled -658.3  td_err_std 32.90  entropy 0.000  greedy(train-diag) -515.4
```

Eval-seed recall **0.0000**, reward **-520.55**. The greedy diagnostic sat on **-515.4** - again
the Phase 3 BULK_CLOSE collapse value. Per CONSTRAINTS #5 this was treated as a bug report before
being written down.

### The diagnostic: not a bug, a scale mismatch

Three entropy coefficients, 20 episodes, seed 0, on the actor-critic train block. Policy entropy:

| coef | ep0 | ep2 | ep4 | ep9 | ep19 |
|---|---|---|---|---|---|
| 0.01 (shipped) | 0.9112 | 0.0061 | 0.0003 | 0.0003 | **0.00001** |
| 1.0 | 1.0165 | 1.1164 | 1.2280 | 0.9021 | 1.3291 |
| 50.0 | 1.6013 | 1.6074 | 1.6071 | 1.6074 | 1.6077 |

Uniform over 5 actions is `ln 5 = 1.6094`; zero means one action in every state.

At 0.01 the actor's **gradient norm** follows the entropy down: 34.40, 0.86, 0.32, 0.03, 0.00.
The policy saturates, `grad ln pi` vanishes, and learning stops. At 50.0 entropy pins to uniform
and the policy never commits to anything. At 1.0 it holds near 1.0, and that run produced rewards
of **+225.6** (ep 4) and **+19.3** (ep 19) where the 0.01 run reached **-1364.5**.

**The arithmetic.** The actor's loss is `I * delta * ln pi`. Measured `|delta|` reaches **1410**
in this environment. The bonus is `entropy_coef * H(pi)` with `H <= ln 5 = 1.609`, so at
coefficient 0.01 it contributes at most **0.016** against a term three to four orders of
magnitude larger. It is not a weak preference for exploration - it is arithmetically incapable
of affecting the update.

### The pattern, which is worth more than the fix

**This is the third time this project has hit the same defect shape.**

| | the default | what it faced | what it silently did |
|---|---|---|---|
| E-016 | `huber_delta` 1.0 (torch's) | penalties of -150 to -200 | flattened catastrophes to routine errors; 20 runs collapsed |
| E-019 | `grad_clip_norm` 10.0 | gradient norms 1584-2228 | replaced the update's magnitude with a constant |
| E-020 | `entropy_coef` 0.01 | TD errors reaching 1410 | bonus arithmetically inert; policy saturates in 5 episodes |

In all three: **nothing errors, the loss curve looks healthy, and a defensible-looking default
quietly replaces the algorithm with a different one.** The generalisation for the report: *a
hyperparameter whose scale was never checked against the environment's reward scale is not a
tuning choice, it is an untested assumption.* Two of the three were caught only because a number
in a results table looked wrong, which is a bad detection mechanism.

### What was built and deliberately not run

`scripts/actor_critic_entropy_experiment.py`, with `actor_critic.entropy_experiment_seed_start`
(2200000) as its own block. It sweeps 0.01 / 0.1 / 1.0 / 10.0 - three orders of magnitude,
because the question is one of scale rather than which nearby value is best - at 80 episodes x 3
repeats, a budget set by the ~10-minute no-approval limit at a measured **~0.6 s/episode**.

It reports two verdicts separately, and that separation is deliberate: **reward** against the
noise floor (expected to be inconclusive at this budget), and **COLLAPSE** - which values left
the policy degenerate. The second is structural rather than a noisy mean, and it is the one the
sweep is for.

3-episode smoke, confirming the mechanism behaves as predicted (final entropy):

| coef | 0.01 | 0.1 | 1.0 | 10.0 |
|---|---|---|---|---|
| final entropy | 0.0263 | 0.2419 | 1.1138 | 1.5660 |

Monotone in the coefficient, approaching `ln 5` from below.

### The sweep result

```powershell
.\.venv\Scripts\python.exe scripts/actor_critic_entropy_experiment.py
```

4 values x 3 repeats x 80 episodes on `actor_critic.entropy_experiment_seed_start` (2200000),
measured on the train-diagnostic seeds. 6.7 min.

| entropy_coef | greedy reward (mean +- std) | mean entropy | final entropy | mean abs TD |
|---|---|---|---|---|
| 0.01 (was shipped) | -515.4 +- 0.0 | 0.0087 | **0.0000** | 2.89 |
| 0.1 | -515.4 +- 0.0 | 0.0158 | **0.0000** | 2.81 |
| **1.0 (now shipped)** | -515.4 +- 0.0 | 0.9890 | **0.8279** | 7.14 |
| 10.0 | -515.4 +- 0.0 | 1.5639 | **1.5707** | 20.42 |

**COLLAPSE verdict: degenerate at 0.01 and 0.1.** The diagnostic is confirmed at three seeds
each: the shipped value could not hold the policy open, and neither could a 10x increase.

### The reward column is VACUOUS, and that is a finding about the harness

Every one of the twelve runs scored **exactly -515.4, std 0.0**. The script duly reported
"the between-value spread does NOT clear the noise floor", which is true and also describes the
wrong failure. The greedy diagnostic here is not *noisy* - it is **constant**. Any policy whose
argmax is BULK_CLOSE in every state earns exactly that on the fixed train-diagnostic seeds,
deterministically, so the metric has no ability to discriminate at all.

**Two consequences, and the second is the more important.**

1. `scripts/actor_critic_entropy_experiment.py`'s reward verdict cannot distinguish "the values
   are indistinguishable" from "the metric is saturated". A future version should say so. Logged
   rather than fixed, because the sweep it was built for has already answered its question.
2. **The entropy bonus fixes the sampled policy and does NOT fix the argmax.** At coefficient 1.0
   the policy is genuinely stochastic (entropy 0.83) and its greedy read is *still* constant
   BULK_CLOSE. This is E-019 section 3's finding arriving from the opposite direction, and it
   sharpens the decision owed there: if the argmax is degenerate at every setting that keeps the
   agent learning, then evaluating Phase 4 through a `_GreedyView` may be measuring an artefact
   of the argmax rather than the agent. **80 episodes is far too short to call this permanent** -
   it is a flag for the full run, not a conclusion.

### Why 1.0, stated so the choice can be defended

Bounded from both sides by the structural criterion, not by reward:

- **0.01, 0.1 - ruled out.** Final entropy 0.0000 on every repeat. The policy collapses to one
  action and `grad ln pi` vanishes, so learning stops.
- **10.0 - ruled out.** Final entropy 1.5707 against a uniform maximum of `ln 5 = 1.6094` -
  **97.6% of maximum**. The bonus dominates the TD signal; the policy is barely expressing a
  preference about anything. Note `|TD|` climbing to 20.42 here against 7.14 at 1.0: the critic
  is chasing a policy that will not settle.
- **1.0 - chosen.** Final entropy 0.8279: comfortably away from collapse, and comfortably below
  uniform, so the policy is both exploring and committing.

**Sampled reward is reported and deliberately not used as the reason.** Means: -674.5 (0.01),
-526.0 (0.1), -157.2 (1.0), -213.3 (10.0). The 1.0-vs-10.0 gap of ~56 sits against within-value
spreads of ~72 and ~47 and does not clear. **1.0 is not claimed to be the best-scoring value** -
it is the only value that is neither collapsed nor uniform.

### Compute facts for the next machine

Measured on Diya's PC this session. Both correct E-018's projection:

- **REINFORCE: ~0.016 s/episode** (13,500 episodes in 3.7 min) - **4.5x faster** than E-018's
  0.073 estimate. A 20000 x 5 run projects to **~27 min**, not ~2 h.
- **Actor-critic: ~0.6 s/episode** - **~37x costlier per episode than REINFORCE**, because it
  updates every STEP rather than once per episode. A 20000-episode run projects to **~3.3 h per
  repeat**, making it by far the most expensive item in Phase 4 and the one that decides the
  phase's schedule.

The asymmetry is not a defect and belongs in the report: it is the price of online learning, and
it is the other half of the trade that bought the actor-critic a 6x smaller tiny-MDP anchor
(40 episodes x 200 steps against REINFORCE's 60 x 800).

### What the next session must do first

1. ~~Run the E-020 sweep and set `entropy_coef` from it.~~ **DONE 2026-08-25.** `entropy_coef`
   is 1.0, chosen on the collapse boundary. 191 tests pass with it, tiny-MDP anchor included.
2. Take the **greedy-vs-sampled evaluation decision** from E-019 section 3. This sweep made it
   more urgent, not less: the argmax was constant BULK_CLOSE at **every** coefficient tested,
   including the two where the sampled policy stayed healthy.
3. Only then consider full runs, budgeting from the actor-critic figure above, not REINFORCE's.
   The full run is also the first honest test of whether the argmax stays degenerate past 80
   episodes.
