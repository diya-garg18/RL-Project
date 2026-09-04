# ROADMAP.md — Build order

Work top to bottom. Tick boxes as things are genuinely done (i.e. tested, not just written). Each phase ends with an **exit criterion** — a concrete, checkable statement. Do not start the next phase until the current one's exit criterion is met and `TEST_CHECKLIST.md` passes.

**Current status (2026-09-04):**
- **THE GATE DECISION IS TAKEN (D-033, Diya, 2026-08-25).** The exit criteria for Phases 1, 2, 3 and 4 stay **exactly as written**, and a phase that fails its criterion closes *built-but-not-passed*. Four honest failures become the report's spine. This resolves the "still owed" item that had been blocking the write-up since Phase 1, and it covers Phase 4 in advance so the criterion cannot be adjusted after the numbers arrive.
- **Phase 4** — **CLOSED as built-but-not-passed** (E-022, E-023, 2026-09-04). All boxes done except the optional PPO. Both learners measured at full budget: REINFORCE reproduces severity_sort exactly in 3 runs of 5; the actor-critic scores recall 0.6316 sampled against 0.0022 greedy (D-036). Sample-efficiency ordering is clear — 9,273 / 294,545 / 1,567,392 steps to reach 40.4. Gate not met, not restated.
- **Phase 0** — closed, gate **passes** on the 30-seed block (oracle strictly best on total reward, 168.0 vs 40.4). One piece of its amendment *rationale* is weakened by E-014, but the criterion itself holds.
- **Phase 1** — **CLOSED as built-but-not-passed** (D-022). Criterion 2 falsified by E-014 (DP −201.2 on 30 seeds, not +305.9 on 5). Gate deliberately not amended a second time. E-015 then refuted the stated *cause* as well: DP never leaves its estimated core.
- **Phase 2** — **CLOSED as built-but-not-passed.** All 8 boxes complete; exit criterion not met and deliberately not restated (D-020).
- **Phase 3** — **CLOSED as built-but-not-passed** (E-017). All six boxes done; gate not met and not restated. Replay proved essential, the target network counterproductive.

> **The eval seed block was widened 5 → 30 on 2026-08-17 (D-019) and every agent re-measured (E-014).** Any number in this file not marked "30-seed" predates that and may be a 5-seed figure. The lesson is worth more than any single result: every number was computed correctly, reported with its standard deviation, and reproduced deterministically — and one of them had **the wrong sign**. Reporting a standard deviation is not the same as reading it.

---

## Phase 0 — Foundation (Week 1, first half)

*Goal: a working simulator you can run a random agent against and get sensible numbers out of.*

- [x] `git init`, verify `.gitignore`
- [x] Create and activate a venv; install `requirements.txt` *(pinned after verified install; torch 2.13.0 works on Python 3.13)*
- [x] `config.py` — load and validate `config/env_default.yaml` into typed objects; fail loudly on missing keys
- [x] `alerts.py` — the `Alert` dataclass (see `ARCHITECTURE.md` §4)
- [x] `generator.py` — Poisson arrivals; sample severity, asset criticality, verify cost, type; assign `is_true_incident` and `deadline_min`
- [x] **Calibration check:** PASSED 2026-08-13, human-verified (E-001): 168.7 alerts/shift, 3.34% incidence, r=0.323; robust on two untuned seed blocks. Recorded in `EXPLAIN.md` Part 8.
- [x] `state.py` — `discretise(env_state) -> int` (0..575) and `featurise(env_state) -> np.ndarray`
- [x] `env.py` — `SOCTriageEnv` with `reset(seed)` and `step(action)`; the 5 actions; the reward function from brief §3.5; 480-minute termination
- [x] `agents/base.py` — the `Agent` interface: `act(state)`, `update(...)`, `save()`, `load()`
- [x] `agents/baselines.py` — random, FIFO, severity-sort, cheapest-first, oracle-greedy
- [x] `runner.py` — run N episodes, fixed seed, emit `EpisodeRecord` JSON
- [x] `evaluation/metrics.py` — MTTD, recall@deadline, wasted minutes, critical misses, composite cost
- [x] Tests: env determinism under a fixed seed; reward accounting sums correctly; **`test_no_ground_truth_leakage`** (no observation ever encodes `is_true_incident`); action-4 never closes more than 10 *(7 tests passing)*
- [x] Baseline comparison table over 5 seeds, written to `results/` *(E-002)*

**Exit criterion (amended twice — 2026-08-14, both approved by Diya):** `python scripts/run_baselines.py` prints a table of the baselines × all metrics with mean ± std; **the oracle is strictly best on mean total reward** (the MDP's objective); random and FIFO sit clearly at the bottom on recall. *(First amendment: "random is worst" → FIFO is far worse, textbook overloaded-queue behaviour — E-002 obs. 1. Second amendment: "oracle strictly best on recall" → E-003 showed that in this deliberately coarse action space no honest greedy oracle can reliably out-recall severity-camping (64% of incidents are sev-3 by construction, D-007); the oracle's information advantage is decisive on total reward instead (145 vs 51 over 30 seeds). Both are findings about the design, documented, not bugs.)*

> If the oracle isn't clearly best on total reward or the naive strategies aren't clearly at the bottom, the environment is broken. Fix it before proceeding — everything downstream depends on this being right.

**✅ PHASE 0 COMPLETE — 2026-08-14.** Gate evidence: E-001 (calibration), E-003 (baselines + 30-seed diagnostic, superseding E-002), 7 passing tests, human reproduction of the table by Diya, gate wording approved by Diya.

---

## Phase 1 — MDP formalisation & Dynamic Programming (Week 1, second half) — CO1, CO2

- [x] `agents/dp.py` — estimate `P̂(s'|s,a)` and `R̂(s,a)` by counting transitions over 50k random-policy episodes *(E-004)*
- [x] Report state-coverage: 133/576 states, 589/2880 (s,a) pairs visited. Unvisited pairs handled explicitly as absorbing self-loops, reward 0 — **D-011**.
- [x] Implement **value iteration** (and **policy iteration**) from scratch on the estimate *(VI 1075 sweeps; VI/PI agree 100%)*
- [x] Plot the convergence curve (max value change per sweep) *(results/dp_convergence.png)*
- [x] Evaluate the DP policy in the real environment; add it to the baseline table *(E-004)*
- [x] **For the report:** hand-work a 5-state Markov Reward Process on paper, show the Bellman equations explicitly, verify against code *(FEATURE_001 — derivation in `docs/features/FEATURE_001_mrp_worked_example.md`, code in `src/soc_triage/mrp_example.py`, four-route agreement to 7.11e-15 in `tests/test_mrp_bellman.py`. **The humans still owe the unaided reproduction** — tracked under TEST_CHECKLIST "The human check", not here.)*

**Exit criterion (ORIGINAL — superseded 2026-08-16, kept for the record):** ~~value iteration converges (Δ < 1e-4), the DP policy beats severity-sort on recall@deadline, and `EXPLAIN.md` states plainly that this policy is optimal *for the estimated model*, not for the true environment.~~

**Exit criterion (AMENDED 2026-08-16 — D-012, approved by Pranav; Diya countersign pending):**
1. Value iteration converges (Δ < 1e-4) and policy iteration independently agrees with it on ≥95% of states.
2. The DP policy achieves the **highest mean total reward** of any agent on the evaluation seeds — total reward being the MDP's actual objective.
3. The Bellman machinery is verified against an answer derived **outside** the code — a hand-worked MRP (FEATURE_001).
4. `EXPLAIN.md` states plainly that the policy is optimal *for the estimated model*, not for the true environment.
5. The gap between the reward DP maximises and the triage quality it delivers is **recorded as a headline finding**, not treated as a defect.

*Why amended (same shape as the two Phase 0 amendments): the original asked DP to win on **recall**, but DP optimises the **reward**, and those are not the same objective. Requiring a reward-maximiser to top a metric it is not maximising is a category error in the gate, not a failure in the agent. E-004 showed the DP policy scores recall 0.43 vs severity-sort's 0.87 while earning the highest reward measured (306 > oracle 214 > severity 154), by using BULK_CLOSE as paid waiting ~97% of the time and abandoning 57% of real incidents. The hand-written reward genuinely rates this optimal — verified against per-step breakdowns and reproduced in the true environment, so it is not an artefact of the estimated model. `PROJECT_BRIEF.md` §3.5 says this trap is deliberate; patching the reward would delete the Phase 5 RLHF motivation, so the reward stands and the gate moves. Alternative rejected: treat the reward as a bug to fix — see D-012.*

> ⛔ **PHASE 1 — BUILT, CRITERION 2 FALSIFIED. Closed unpassed 2026-08-18 (E-014, E-015, D-019, D-020, D-022).**
>
> The amended criterion required the DP policy to achieve the **highest mean total reward of any agent** on the evaluation seeds. That was measured on 5 seeds. On the widened 30-seed block (D-019) the DP policy scores **−201.2 ± 438.5 — the worst of any planned or learned agent** — against the +305.9 recorded below, with recall falling 0.43 → 0.23 and the largest variance of any agent.
>
> E-004 is **not** retracted and not altered; it stands as recorded on the seeds it used (CONSTRAINTS #4). What is retracted is the claim that criterion 2 is satisfied.
>
> Likely cause (**hypothesis, untested**): D-004 compounded by D-011 — DP's policy is optimal for a model estimated over 133 of 576 states, so on shifts that stray outside that core it has no useful guidance. Testable by correlating per-seed DP reward against distance from the visited core. **Not done.**
>
> **✅ BOTH DECISIONS TAKEN 2026-08-18 (D-022, Pranav).**
>
> **Status: BUILT — CRITERION FALSIFIED ON BETTER MEASUREMENT.** The gate is **not** amended a second time. D-012's amendment was legitimate (the original criterion contained a category error); this would not be — nothing is wrong with criterion 2 as written, it is simply false at −201.2. Rewriting a criterion because the result came out wrong, having already rewritten it once, would taint the first amendment retrospectively.
>
> **And the cause stated above was ALSO wrong** — tested and refuted in **E-015**. Off-core share is **0.0% on all 30 eval seeds**, for states *and* state-action pairs. DP never leaves its estimated core, and D-011's convention never fires at evaluation time. Coverage and D-011 are both exonerated. The control rules out seed difficulty too: corr(severity, DP) = **+0.085**, and on seed 128 DP loses 755 where severity-sort gains 233.
>
> **Remaining explanation — untested, and labelled as such:** `P̂`/`R̂` were counted under a *uniform-random* policy, but DP bulk-closes ~97% of the time, so the transitions following its own actions are not the ones the model was built from — even though the states are familiar. **Distribution shift in the estimate, not gaps in it.** Test named in E-015: re-estimate from DP-policy rollouts and check whether the plan's predicted value matches its measured reward.
>
> **This sharpens D-004 for the report.** "Optimal for the estimated model, not the true environment" has been read throughout — including by E-014 — as being about *coverage*. It is not. The gap is between the policy the model describes and the policy being planned.

**⛔ SUPERSEDED — the assessment below was made on 5 eval seeds and criterion 2 no longer holds. Kept verbatim for the record.** ~~**✅ PHASE 1 COMPLETE — 2026-08-16.**~~ Gate evidence as assessed at the time, all five criteria then believed met: VI converged Δ 9.95e-05 in 1075 sweeps with VI/PI agreement 100% (E-004) · DP total reward 305.9 ± 127.6 vs oracle 214.1 and severity-sort 153.7 on eval seeds 101–105 (E-004) · four-route Bellman verification agreeing to 7.11e-15, including the shipped `value_iteration` reproducing a hand-derived value function (FEATURE_001, E-005) · D-004/D-011 caveats written in `EXPLAIN.md` · the reward-hacking finding logged as E-004 and carried into Phase 5 as its primary motivation. 14 tests passing.

> **The finding this phase is actually remembered for:** exact planning found the reward exploit two phases before anyone was looking for it. Every later agent optimises the same reward, so expect the bulk-close hack to reappear in Phase 2 and Phase 3 — and *that continuity* (DP hacks → Q-learning hacks → RLHF fixes) is the report's spine.

---

## Phase 2 — Tabular model-free RL (Week 2) — CO2

- [x] `agents/monte_carlo.py` — first-visit MC control with ε-greedy *(FEATURE_006, E-010, D-017 — recall 0.71±0.02, reward 177.3±91.7, best MTTD 18.6±3.0. Verified on the tiny MDP against the ε-soft target, not q\*.)*
- [x] `agents/sarsa.py` — on-policy TD control *(FEATURE_006, E-010, D-017 — recall 0.74±0.01, reward 324.1±81.6, the highest of any agent in the project.)*
- [x] `agents/q_learning.py` — off-policy TD control, ε-greedy with decay *(FEATURE_003, E-007, D-015 — built test-first; reproduces the hand-derived `q_*` to 9.24e-14 on the tiny MDP, correct policy after 10 episodes. **Not yet run on the 576-state environment** — the update rule is verified, performance is not.)*
- [x] Learning curves: reward per episode, smoothed over 100 episodes, 5 seeds each *(FEATURE_004, E-008 — `scripts/train.py`, `results/q_learning_curve.png`. **Caveat: the curve does not visibly converge** after epsilon floors; learner instability and environment variance have not been separated, so no convergence claim is made.)*
- [x] Convergence comparison against the Phase 1 DP solution (max-norm distance between Q-tables, and policy agreement %) *(FEATURE_006, E-011 — `scripts/compare_agents.py`. Policy agreement over commonly-visited states is **22–44%**, max-norm |ΔQ| 116–320. The naive 'all 576' figure of 83–86% is manufactured by states neither agent visited, where both fall back to a convention.)*
- [x] **Print the learned policy as a readable table** — for each `time_left` bucket, which action wins in which queue state. This is a headline figure for the report and the viva. *(FEATURE_005, E-009 — `scripts/policy_table.py` → `results/policy_table.md`.)*
  - For **Q-learning**, bulk-closing rises 25.3% → 36.0% → 46.2% into the crunch while severity-first falls 34.9% → 28.0% → 15.4%.
  - ⚠️ **PARTIALLY RETRACTED 2026-08-17 (E-013).** Running the same figure for the other two learners shows the trend **does not replicate**: Monte Carlo rises like Q-learning (23.1% → 28.6% → 42.9%), but **SARSA falls — 47.4% → 51.9% → 25.0%, the opposite direction**, on the same environment and reward. With only 12–14 visited states per crunch bucket, a direction disagreement across algorithms is fully consistent with noise. **The per-algorithm figures stand; the claim that there is a consistent interpretable strategy shift does not.** Both readings E-009 offered are now under-supported.
  - **Caveat:** coverage is 121/576 states, and the crunch column rests on **13 states**. 455 unvisited states would have printed as a confident `PULL_HIGHEST_SEVERITY` via the argmax tie-break — the agent now records visit counts purely so they print as `·` instead.
- [x] Ablations: learning rate, γ, ε-decay schedule *(FEATURE_006, E-012 — `scripts/ablations.py`, measured on train-diagnostic seeds, never eval.* **None of the three clears the noise floor.** *Between-config spread is smaller than or comparable to within-config spread in every sweep; the default config alone produced 75, −34 and 47. Reported as a negative result rather than filled in.)*
- [x] Tests: Q-learning converges on a tiny hand-checkable 2-state MDP with a known answer *(ticked 2026-09-01. All three sub-boxes below were already complete; only this parent was left unticked, and `HANDOVER.md` had been reading it as outstanding work behind Phase 4. It was not.)*
  - [x] **The 2-state MDP itself, hand-solved and verified** *(FEATURE_002, E-006, D-014 — `src/soc_triage/tiny_mdp.py`, 13 tests. `q_* = [[10.0, 6.7], [10.7, 13.0]]` derived on paper, Bellman-optimality residual 1.78e-15, and `agents/dp.value_iteration` reproduces it.)* **Built first, ahead of the learners** — deliberately out of box order, because an anchor built afterwards cannot say whether a disagreement is the learner's fault or its own (D-014).
  - [x] **Q-learning measured against it** *(FEATURE_003, E-007 — `tests/test_tabular.py`, 20 tests. Includes the single-backup assertion that distinguishes Q-learning from SARSA, which no convergence test can.)*
  - [x] SARSA and Monte Carlo measured against it *(`tests/test_on_policy.py` — split out of `test_tabular.py`, which was near the 500-line limit. Both are graded against `tiny_mdp.epsilon_soft_q(epsilon)`, NOT `HAND_COMPUTED_Q`: they are on-policy, so their fixed point is q_pi for the epsilon-greedy policy they follow, and grading them against q\* would mark a correct implementation badly broken. The soft target is itself anchored — at epsilon=0 it reproduces the pen-and-paper q\* exactly.)*

**Exit criterion:** Q-learning beats severity-sort on recall@deadline and MTTD across 5 seeds (report mean ± std), and the printed policy table shows a *behaviourally interpretable* strategy shift as time runs out.

**⛔ PHASE 2 CLOSED 2026-08-17 AS *BUILT BUT NOT PASSED* (D-020). All 8 boxes complete; the exit criterion is NOT met and is deliberately NOT restated.**

Final assessment on the **30-seed** eval block (D-019, E-014):

| agent | recall | reward | reward std |
|---|---|---|---|
| oracle_greedy | **0.87** | **168.0** | ±232.9 |
| q_learning | 0.72 | 47.6 | **±52.0** |
| sarsa | 0.66 | 40.5 | **±49.4** |
| severity_sort | 0.84 | 40.4 | ±220.1 |
| monte_carlo | 0.70 | −16.4 | ±77.0 |
| dp | 0.23 | −201.2 | ±438.5 |

> **Recall half — FAILS.** All three learners lose to severity-sort: 0.66–0.72 against 0.84.
>
> **Policy-table half — FAILS.** Reported satisfied in E-009; **withdrawn** by E-013 because the strategy shift reverses direction depending on which learner produced it, so it is not a property of the task.
>
> **And the reward consolation prize is gone too.** On 5 seeds the learners appeared to beat severity-sort by 100+. On 30 seeds it is 47.6 and 40.5 against 40.4, inside a ±220 spread — indistinguishable. They pay the recall and get nothing reliable for it.
>
> **The gate is NOT restated, and that is the decision** (D-020). Phase 1's gate was legitimately amended because it contained a category error — it asked a reward-maximiser to top a metric it does not optimise. No such error exists here: the learners simply did not do the thing. Amending now would be tuning the criterion to the result, which is the exact failure this project exists to avoid. Restating it on reward *consistency*, where the learners genuinely do win (±50 vs ±220), was the most tempting option and was rejected for the same reason — nobody set out to optimise variance.
>
> **What Phase 2 achieved anyway:** three algorithms hand-written and each verified against a pen-and-paper answer before touching the real environment; an ablation study honest enough to report that none of its effects clear the noise (E-012); a policy renderer that marks absence of data rather than inventing a preference (E-009); a retraction when a finding failed to replicate (E-013); and the discovery that the project's own evaluation protocol was too weak to support its conclusions (E-014). That last one is worth more than a passed gate. Q-learning recall **0.73 ± 0.03** against severity-sort's **0.87** — fails. MTTD 22.0 vs 23.0 — marginally better, well inside the spread. Reward 270.9 vs 153.7 — wins clearly, on the metric the gate does not use.
>
> The flag below was right. BULK_CLOSE_LOW_RISK accounts for **62.3%** of the learned policy's actions (DP: ~97%). Same exploit, less extreme, found by a completely different algorithm — which is the evidence that the pathology is in the reward, not in DP.
>
> **The gate is left unmet and unamended, deliberately**, exactly as D-012 requires: a criterion contradicted by measurement gets decided by a human with real numbers in hand, not patched by whoever ran the experiment. **Two decisions are owed** — the gate itself, and the more serious eval-seed representativeness problem E-008 uncovered (every agent, oracle included, scores 120–230 higher on eval seeds than on train seeds, with per-seed spread several times the effect being measured). The second affects every experiment in the project, not just this phase.

> ⚠ **Flagged, not yet amended (E-003 implication 2, E-004, D-012).** This gate has the same shape as the one Phase 1 had to restate: Q-learning maximises the *same exploitable reward* DP did, so it may well repeat the bulk-close hack and land below severity-sort on recall while winning on reward. **Do not pre-emptively weaken this criterion** — run Q-learning first, then decide with real numbers in hand, exactly as Phases 0 and 1 did. If the hack reappears, that is a *result* (it demonstrates the pathology is in the reward, not in the algorithm), and the gate gets restated on the objective with the recall gap recorded. If Q-learning beats severity-sort on recall anyway, that is the more interesting outcome and needs explaining, not just celebrating.

> This is the moment the project becomes real. Take the win seriously — and then check it isn't a bug.

---

## Phase 3 — Deep Q-Networks (Week 3, first half) — CO3

- [x] `agents/dqn.py` — MLP Q-network over the 17-dim continuous state *(FEATURE_007, E-017 — `QNetwork` + `DQNAgent`, [128,128] relu, 19,461 parameters. Input scaling by fixed domain divisors, D-023.)*
- [x] Experience replay buffer (write it by hand) *(FEATURE_007 — `agents/replay.py`, five parallel numpy arrays, 8 tests. **E-017 found it is load-bearing, not a refinement**: all 8 no-replay runs scored recall 0.0000.)*
- [x] Target network with periodic hard update *(FEATURE_007 — refreshed every 1000 **gradient** steps, not env steps. **E-017 found it makes this agent worse** — removing it improved recall 0.481→0.588 and reward −46.9→+43.5, ratios 2.25 and 2.51.)*
- [x] ε-greedy with decay; gradient clipping; Huber loss *(FEATURE_007, D-029. **The Huber delta was the phase's central bug** — left at torch's default of 1.0 against penalties of −150 to −1499, it flattened every catastrophe to the size of a routine error and collapsed 20 runs to BULK_CLOSE. See E-016 and BUG_002. Now 200.0, with a loader guard refusing anything below 50.)*
- [x] Compare DQN-on-continuous-state vs tabular-on-discretised-state *(E-017 — `scripts/compare_dqn_tabular.py`, paired per eval seed (D-028). **DQN loses**: recall 0.48 vs 0.73. Reward is **not resolvable** — paired |mean|/SEM = 1.42. In 21 of 42 visited buckets the DQN chose different actions for situations the discretisation merges, so the phase's premise held even though its gate did not.)*
- [x] **Ablation (required):** train DQN with replay off, and with the target network off *(E-017 — n=8 each, not the 15 D-027 specified; the machine could not sustain it against the deadline, see D-030. **Neither ablation shows "instability" in the sense the box expects**, and for opposite reasons: no-replay flatlines completely (volatility 0.00) and no-target-network is *more* stable than the control. `dqn_ablations.py` mis-reported the first as "no destabilisation" — BUG_003.)*

**Exit criterion:** DQN matches or beats tabular Q-learning on the same evaluation seeds, and the two ablations visibly destabilise training in the plots.

> ⚠️ **EXIT CRITERION NOT MET (E-017), and deliberately not restated.** Both halves
> fail. The DQN reaches recall 0.48 against tabular Q-learning's 0.73 on the same
> eval seeds, and its learning curve has **plateaued** — a converged agent that is
> worse than the lookup table, not an undertrained one. Neither ablation
> "visibly destabilises training": one destroys learning outright without any
> instability signature, the other improves on the control.
>
> Following the precedent of D-012 (Phase 1) and D-020 (Phase 2), the gate is
> left **unmet and unamended** until a human decides. Three phases have now
> failed their originally-written exit criteria, and that pattern is itself a
> finding about how the criteria were written.
>
> **Phase 3 is therefore CLOSED as built-but-not-passed.** All six work items are
> genuinely complete; the criterion they were meant to satisfy is not.

---

## Phase 4 — Policy gradient & actor–critic (Week 3, second half) — CO3, CO4

- [x] `agents/reinforce.py` — Monte Carlo policy gradient, with a baseline for variance reduction *(FEATURE_008, E-018. Built and tested; **no full training run yet**.)*
- [x] `agents/actor_critic.py` — separate actor and critic heads *(FEATURE_009, D-034, 18 tests. Bootstrapping is the criterion, not the head count. `entropy_coef` set to 1.0 from E-020.)*
- [x] Sample-efficiency comparison: DQN vs REINFORCE vs actor–critic (reward vs environment steps) *(`scripts/compare_sample_efficiency.py` **built and smoke-verified**; the three trainers now record `episode_steps` so the x-axis is real rather than estimated — actor-critic takes ~88 steps/shift against REINFORCE's ~47, so per-episode plotting would have been wrong. **NOT YET RUN**: needs the full training runs, which are Pranav's machine.)* **RUN 2026-09-04 (E-023): REINFORCE reaches severity_sort's 40.4 in 9,273 environment steps, the DQN in 294,545, the actor-critic in 1,567,392.**
- [x] Show REINFORCE's variance explicitly *(`scripts/variance_demo.py`, **E-021**. Measured: the coefficient std is 146.94 unbaselined, 147.68 baselined, 30.89 for the actor-critic. **The textbook baseline reduction did NOT replicate (1.00x)** — the value head is not accurate enough at this budget for it to pay off — while bootstrapping's 4.78x is structural. A negative result on the half everyone quotes, and the better interview answer for it.)*
- [ ] *Optional, cut first if time is short:* PPO with clipped objective

**Exit criterion:** all three learners train to a policy beating severity-sort, and the sample-efficiency plot shows a clear ordering you can explain.

**PHASE 4 CLOSES BUILT-BUT-NOT-PASSED - 2026-09-04 (E-022, E-023).** The criterion is
**not restated**; D-033 governs, as it did for Phases 1, 2 and 3. Fourth consecutive phase
to close this way, and that pattern is itself the reportable finding.

*Clause 2 (MET):* the sample-efficiency ordering is clear and explainable - REINFORCE
reaches severity_sort's 40.4 in **9,273** environment steps, the DQN in **294,545**, the
actor-critic in **1,567,392**.

*Clause 1 (NOT MET, all three):* REINFORCE 70.52 +- 37.62 - above severity_sort's 40.44 but
by less than its own spread; actor-critic **-74.02 +- 38.47**; DQN 30.8. None reaches
severity_sort's recall of 0.8443 either (0.7763, 0.6316, 0.48).

*What it established instead:* policy gradient **rediscovers severity_sort exactly** and
stops (E-022 - three runs of five match it to four decimals, two with a vanished
gradient); the textbook baseline variance reduction **did not replicate** while
bootstrapping's did (E-021); and **how you read a stochastic policy can invert its result**
- the actor-critic scores recall 0.6316 sampled against 0.0022 greedy, and whether that gap
is transient or permanent depends on the exploration mechanism, not the algorithm family
(E-023, D-036).

---

## Phase 5 — RLHF (Weeks 4–5) — CO4. **This phase is the project. Do not cut it.**

### 5a — Collect preferences

> **Data layer built and tested 2026-09-04, session 12** (FEATURE_011, D-037 to D-039).
> 71 tests in 0.30 s, plus 15 config tests. Full suite 286 passed. Nothing is measured
> yet because nothing has been labelled yet — that is human time, not compute.

- [x] `rlhf/pairs.py` — sample pairs of `EpisodeRecord`s **run on the same alert stream** under different policies *(FEATURE_011, D-037/D-038 — 25 tests. Deterministic: two builds are byte-identical, because labels reference `pair_id`. Balanced over all 36 policy pairings, seeded side-swap against position bias, round-robin double-labelling. Exercised end to end on the 270 real eval-seed records that share a config: 300 pairs, 28 pairings split 10/11, 50 double labels across all 28.)*
- [x] Episode summary renderer: action timeline + outcome cards (caught & when, missed, minutes wasted). Ground truth *is* shown to the labeller — they are judging outcomes, not guessing. *(`rlhf/summary.py`, FEATURE_011, D-039 — 15 tests. Every reward field is stripped and two tests enforce it. Ran over all 300 real EpisodeRecords with zero failures. **Known limitation:** missed incidents are counts, not per-incident cards — the record does not carry the end-of-shift queue.)*
- [x] `rlhf/store.py` — SQLite: pair id, labeller id, choice (left/right/tie), timestamp, time-taken *(FEATURE_011 — 18 tests. CHECK and UNIQUE live in the schema, not only in Python, because Diya's UI writes to the same file. CONSTRAINTS #23 is enforced by there being no column to put a name in.)*
- [ ] **`scripts/generate_pairs.py`** — run the 9 policies on the pair seed block and write the records stage 1b consumes. *(NOT BUILT. The only piece a missing artefact can block; all nine artefacts do exist on Pranav's machine. Needs a decision on which training repeat to show — REINFORCE's five differ by 0.7763 ± 0.0833 and three reproduce severity_sort exactly, so the choice is not cosmetic.)*
- [ ] Labelling UI (simplest thing that works — a local FastAPI page, or even a CLI with rendered text) *(**Diya's box** — `PROJECT_BRIEF.md` §9. It reads `results/rlhf/pairs.json` and writes through `rlhf/store.py`. It must never read `pairs_key.json`.)*
- [ ] **Collect 300 labelled pairs.** Include 50 pairs labelled by *both* Pranav and Diya.
- [x] Compute and record **Cohen's κ** on those 50 overlapping pairs *(`rlhf/agreement.py` + `scripts/report_kappa.py`, FEATURE_011 — 13 tests, written by hand against a paper-worked 10-pair example giving exactly 6/11. **The code is built and verified; the κ itself is unmeasured** because no pairs have been labelled. Undefined cases return None with a reason rather than a flattering 1.0.)*

### 5b — Train the reward model
- [ ] `rlhf/reward_model.py` — small MLP `r̂(state, action)`; Bradley–Terry loss over trajectory returns:
      `P(τ_A ≻ τ_B) = σ(Σ r̂_A − Σ r̂_B)`, trained with binary cross-entropy
- [ ] 80/20 train/held-out split on pairs; report both accuracies
- [ ] Sanity check: does `r̂` broadly agree with the hand-written reward? Where it *disagrees* is the interesting part — investigate and write it up.

### 5c — Re-train the policy
- [ ] Re-train Q-learning and DQN using `r̂` in place of the hand reward
- [ ] Collect **fresh held-out preference labels** comparing the RLHF policy against the hand-reward policy — this is the headline result
- [ ] Compare both policies on ground-truth metrics too

**Exit criterion:** one graph showing held-out human agreement for hand-reward vs learned-reward policies, plus an honest statement of what the ground-truth metrics did. *If the RLHF policy is preferred by humans but worse on ground truth, report that — it's a more interesting finding than a clean win.*

---

## Phase 6 — Audit, evaluation, delivery (Week 6) — CO5

- [ ] `evaluation/audit.py` — the four reward-hacking experiments from brief §7:
  - [ ] Bulk-close exploit frequency
  - [ ] Reward-model train vs held-out gap
  - [ ] State-visitation overlap (OOD drift)
  - [ ] Cohen's κ (from 5a)
- [ ] Final comparison table: all agents × all metrics × 5 seeds, mean ± std
- [ ] All report figures generated by a single reproducible script
- [ ] Dashboard (`web/`) — live queue, chosen action, reward breakdown, plain-English justification via Groq/Llama *(cut this before cutting anything above)*
- [ ] Final report with the limitations section from brief §12 included verbatim
- [ ] Presentation deck
- [ ] **Viva prep:** both team members work through `INTERVIEW_PREP.md` and can write the four core functions from memory

**Exit criterion:** `python scripts/reproduce_all.py` regenerates every number and figure in the report from scratch, and both students can independently explain any file in `src/`.

---

## Cut order if time runs out

1. PPO (Phase 4, optional)
2. The Groq/Llama justification layer
3. The React dashboard → fall back to matplotlib plots + a CLI replay
4. Actor–critic (keep REINFORCE)

**Never cut:** Phase 5 (RLHF) or the Phase 6 audit. Those two are the entire differentiator. Every other team in the class will have a Q-learning agent; almost none will have collected real human preference data and audited their own reward model.
