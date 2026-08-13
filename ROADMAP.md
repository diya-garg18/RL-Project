# ROADMAP.md — Build order

Work top to bottom. Tick boxes as things are genuinely done (i.e. tested, not just written). Each phase ends with an **exit criterion** — a concrete, checkable statement. Do not start the next phase until the current one's exit criterion is met and `TEST_CHECKLIST.md` passes.

**Current phase:** Phase 0 — nothing built yet.

---

## Phase 0 — Foundation (Week 1, first half)

*Goal: a working simulator you can run a random agent against and get sensible numbers out of.*

- [ ] `git init`, verify `.gitignore`
- [ ] Create and activate a venv; install `requirements.txt`
- [ ] `config.py` — load and validate `config/env_default.yaml` into typed objects; fail loudly on missing keys
- [ ] `alerts.py` — the `Alert` dataclass (see `ARCHITECTURE.md` §4)
- [ ] `generator.py` — Poisson arrivals; sample severity, asset criticality, verify cost, type; assign `is_true_incident` and `deadline_min`
- [ ] **Calibration check:** write a script that generates 100 shifts and prints: alerts/shift, % true incidents, and the **Pearson correlation between severity and `is_true_incident`**. Tune the generator until incidence ≈ 3% and that correlation lands in **0.30–0.40**. Record the final numbers in `EXPLAIN.md`. *(This is the assumption the whole project rests on — §4.2 of the brief. Get it right and write it down.)*
- [ ] `state.py` — `discretise(env_state) -> int` (0..575) and `featurise(env_state) -> np.ndarray`
- [ ] `env.py` — `SOCTriageEnv` with `reset(seed)` and `step(action)`; the 5 actions; the reward function from brief §3.5; 480-minute termination
- [ ] `agents/base.py` — the `Agent` interface: `act(state)`, `update(...)`, `save()`, `load()`
- [ ] `agents/baselines.py` — random, FIFO, severity-sort, cheapest-first, oracle-greedy
- [ ] `runner.py` — run N episodes, fixed seed, emit `EpisodeRecord` JSON
- [ ] `evaluation/metrics.py` — MTTD, recall@deadline, wasted minutes, critical misses, composite cost
- [ ] Tests: env determinism under a fixed seed; reward accounting sums correctly; **`test_no_ground_truth_leakage`** (no observation ever encodes `is_true_incident`); action-4 never closes more than 10
- [ ] Baseline comparison table over 5 seeds, written to `results/`

**Exit criterion:** `python scripts/run_baselines.py` prints a table of all 6 baselines × 5 metrics with mean ± std, and the oracle is strictly best on recall while random is worst.

> If the oracle isn't clearly best or random isn't clearly worst, the environment is broken. Fix it before proceeding — everything downstream depends on this being right.

---

## Phase 1 — MDP formalisation & Dynamic Programming (Week 1, second half) — CO1, CO2

- [ ] `agents/dp.py` — estimate `P̂(s'|s,a)` and `R̂(s,a)` by counting transitions over 50k random-policy episodes
- [ ] Report state-coverage: how many of the 576 states were visited, and how often. Unvisited states must be handled explicitly (document the choice in `DECISIONS.md`).
- [ ] Implement **value iteration** (and **policy iteration**) from scratch on the estimate
- [ ] Plot the convergence curve (max value change per sweep)
- [ ] Evaluate the DP policy in the real environment; add it to the baseline table
- [ ] **For the report:** hand-work a 5-state Markov Reward Process on paper, show the Bellman equations explicitly, verify against code

**Exit criterion:** value iteration converges (Δ < 1e-4), the DP policy beats severity-sort on recall@deadline, and `EXPLAIN.md` states plainly that this policy is optimal *for the estimated model*, not for the true environment.

---

## Phase 2 — Tabular model-free RL (Week 2) — CO2

- [ ] `agents/monte_carlo.py` — first-visit MC control with ε-greedy
- [ ] `agents/sarsa.py` — on-policy TD control
- [ ] `agents/q_learning.py` — off-policy TD control, ε-greedy with decay
- [ ] Learning curves: reward per episode, smoothed over 100 episodes, 5 seeds each
- [ ] Convergence comparison against the Phase 1 DP solution (max-norm distance between Q-tables, and policy agreement %)
- [ ] **Print the learned policy as a readable table** — for each `time_left` bucket, which action wins in which queue state. This is a headline figure for the report and the viva.
- [ ] Ablations: learning rate, γ, ε-decay schedule
- [ ] Tests: Q-learning converges on a tiny hand-checkable 2-state MDP with a known answer

**Exit criterion:** Q-learning beats severity-sort on recall@deadline and MTTD across 5 seeds (report mean ± std), and the printed policy table shows a *behaviourally interpretable* strategy shift as time runs out.

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
