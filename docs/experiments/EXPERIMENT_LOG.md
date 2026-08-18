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
