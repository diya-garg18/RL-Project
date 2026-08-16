# ROADMAP.md — Build order

Work top to bottom. Tick boxes as things are genuinely done (i.e. tested, not just written). Each phase ends with an **exit criterion** — a concrete, checkable statement. Do not start the next phase until the current one's exit criterion is met and `TEST_CHECKLIST.md` passes.

**Current phase:** Phase 2 — tabular model-free RL. Phases 0 and 1 are closed (2026-08-14, 2026-08-16).

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

**✅ PHASE 1 COMPLETE — 2026-08-16.** Gate evidence, all five criteria met: VI converged Δ 9.95e-05 in 1075 sweeps with VI/PI agreement 100% (E-004) · DP total reward 305.9 ± 127.6 vs oracle 214.1 and severity-sort 153.7 on eval seeds 101–105 (E-004) · four-route Bellman verification agreeing to 7.11e-15, including the shipped `value_iteration` reproducing a hand-derived value function (FEATURE_001, E-005) · D-004/D-011 caveats written in `EXPLAIN.md` · the reward-hacking finding logged as E-004 and carried into Phase 5 as its primary motivation. 14 tests passing.

> **The finding this phase is actually remembered for:** exact planning found the reward exploit two phases before anyone was looking for it. Every later agent optimises the same reward, so expect the bulk-close hack to reappear in Phase 2 and Phase 3 — and *that continuity* (DP hacks → Q-learning hacks → RLHF fixes) is the report's spine.

---

## Phase 2 — Tabular model-free RL (Week 2) — CO2

- [ ] `agents/monte_carlo.py` — first-visit MC control with ε-greedy
- [ ] `agents/sarsa.py` — on-policy TD control
- [x] `agents/q_learning.py` — off-policy TD control, ε-greedy with decay *(FEATURE_003, E-007, D-015 — built test-first; reproduces the hand-derived `q_*` to 9.24e-14 on the tiny MDP, correct policy after 10 episodes. **Not yet run on the 576-state environment** — the update rule is verified, performance is not.)*
- [x] Learning curves: reward per episode, smoothed over 100 episodes, 5 seeds each *(FEATURE_004, E-008 — `scripts/train.py`, `results/q_learning_curve.png`. **Caveat: the curve does not visibly converge** after epsilon floors; learner instability and environment variance have not been separated, so no convergence claim is made.)*
- [ ] Convergence comparison against the Phase 1 DP solution (max-norm distance between Q-tables, and policy agreement %)
- [x] **Print the learned policy as a readable table** — for each `time_left` bucket, which action wins in which queue state. This is a headline figure for the report and the viva. *(FEATURE_005, E-009 — `scripts/policy_table.py` → `results/policy_table.md`.)*
  - **The strategy shift exists and is monotonic across all three buckets:** working alerts by severity falls 34.9% → 28.0% → 15.4% as time runs out; bulk-closing rises 25.3% → 36.0% → 46.2%.
  - **Two readings, not separated by the data.** Either an analyst-like escalation under deadline pressure, or the E-008 reward hack intensifying where end-of-shift miss charges make it most profitable. Both fit; a per-action reward decomposition inside the crunch bucket would settle it.
  - **Caveat:** coverage is 121/576 states, and the crunch column rests on **13 states**. 455 unvisited states would have printed as a confident `PULL_HIGHEST_SEVERITY` via the argmax tie-break — the agent now records visit counts purely so they print as `·` instead.
- [ ] Ablations: learning rate, γ, ε-decay schedule
- [ ] Tests: Q-learning converges on a tiny hand-checkable 2-state MDP with a known answer
  - [x] **The 2-state MDP itself, hand-solved and verified** *(FEATURE_002, E-006, D-014 — `src/soc_triage/tiny_mdp.py`, 13 tests. `q_* = [[10.0, 6.7], [10.7, 13.0]]` derived on paper, Bellman-optimality residual 1.78e-15, and `agents/dp.value_iteration` reproduces it.)* **Built first, ahead of the learners** — deliberately out of box order, because an anchor built afterwards cannot say whether a disagreement is the learner's fault or its own (D-014).
  - [x] **Q-learning measured against it** *(FEATURE_003, E-007 — `tests/test_tabular.py`, 20 tests. Includes the single-backup assertion that distinguishes Q-learning from SARSA, which no convergence test can.)*
  - [ ] SARSA and Monte Carlo measured against it, in the same file

**Exit criterion:** Q-learning beats severity-sort on recall@deadline and MTTD across 5 seeds (report mean ± std), and the printed policy table shows a *behaviourally interpretable* strategy shift as time runs out.

> **Second half of the gate — the policy table — IS satisfied** (E-009): the shift is present, monotonic, and interpretable. The recall half is not.
>
> ❌ **MEASURED 2026-08-16 (E-008) — NOT MET.** Q-learning recall **0.73 ± 0.03** against severity-sort's **0.87** — fails. MTTD 22.0 vs 23.0 — marginally better, well inside the spread. Reward 270.9 vs 153.7 — wins clearly, on the metric the gate does not use.
>
> The flag below was right. BULK_CLOSE_LOW_RISK accounts for **62.3%** of the learned policy's actions (DP: ~97%). Same exploit, less extreme, found by a completely different algorithm — which is the evidence that the pathology is in the reward, not in DP.
>
> **The gate is left unmet and unamended, deliberately**, exactly as D-012 requires: a criterion contradicted by measurement gets decided by a human with real numbers in hand, not patched by whoever ran the experiment. **Two decisions are owed** — the gate itself, and the more serious eval-seed representativeness problem E-008 uncovered (every agent, oracle included, scores 120–230 higher on eval seeds than on train seeds, with per-seed spread several times the effect being measured). The second affects every experiment in the project, not just this phase.

> ⚠ **Flagged, not yet amended (E-003 implication 2, E-004, D-012).** This gate has the same shape as the one Phase 1 had to restate: Q-learning maximises the *same exploitable reward* DP did, so it may well repeat the bulk-close hack and land below severity-sort on recall while winning on reward. **Do not pre-emptively weaken this criterion** — run Q-learning first, then decide with real numbers in hand, exactly as Phases 0 and 1 did. If the hack reappears, that is a *result* (it demonstrates the pathology is in the reward, not in the algorithm), and the gate gets restated on the objective with the recall gap recorded. If Q-learning beats severity-sort on recall anyway, that is the more interesting outcome and needs explaining, not just celebrating.

> This is the moment the project becomes real. Take the win seriously — and then check it isn't a bug.

---

## Phase 3 — Deep Q-Networks (Week 3, first half) — CO3

- [ ] `agents/dqn.py` — MLP Q-network over the ~20-dim continuous state
- [ ] Experience replay buffer (write it by hand — it's ~30 lines and a classic interview question)
- [ ] Target network with periodic hard update
- [ ] ε-greedy with decay; gradient clipping; Huber loss
- [ ] Compare DQN-on-continuous-state vs tabular-on-discretised-state
- [ ] **Ablation (required):** train DQN with replay off, and with the target network off. Show the instability. This is the cleanest possible demonstration of *why* those two tricks exist and it will be asked about in the viva.

**Exit criterion:** DQN matches or beats tabular Q-learning on the same evaluation seeds, and the two ablations visibly destabilise training in the plots.

---

## Phase 4 — Policy gradient & actor–critic (Week 3, second half) — CO3, CO4

- [ ] `agents/reinforce.py` — Monte Carlo policy gradient, with a baseline for variance reduction
- [ ] `agents/actor_critic.py` — separate actor and critic heads
- [ ] Sample-efficiency comparison: DQN vs REINFORCE vs actor–critic (reward vs environment steps)
- [ ] Show REINFORCE's variance explicitly — it is high, and being able to say *why* (full-return estimates, no bootstrapping) is a strong interview answer
- [ ] *Optional, cut first if time is short:* PPO with clipped objective

**Exit criterion:** all three learners train to a policy beating severity-sort, and the sample-efficiency plot shows a clear ordering you can explain.

---

## Phase 5 — RLHF (Weeks 4–5) — CO4. **This phase is the project. Do not cut it.**

### 5a — Collect preferences
- [ ] `rlhf/pairs.py` — sample pairs of `EpisodeRecord`s **run on the same alert stream** under different policies
- [ ] Episode summary renderer: action timeline + outcome cards (caught & when, missed, minutes wasted). Ground truth *is* shown to the labeller — they are judging outcomes, not guessing.
- [ ] `rlhf/store.py` — SQLite: pair id, labeller id, choice (left/right/tie), timestamp, time-taken
- [ ] Labelling UI (simplest thing that works — a local FastAPI page, or even a CLI with rendered text)
- [ ] **Collect 300 labelled pairs.** Include 50 pairs labelled by *both* Pranav and Diya.
- [ ] Compute and record **Cohen's κ** on those 50 overlapping pairs

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
