# DECISIONS.md — Why, not just what

> Field Guide habit #2 and #14. Code shows *what* changed; this file shows *why*. **Append-only** — never edit or delete a past entry. If a decision is reversed, write a new entry that supersedes it and link back.
>
> Every entry records the model version that made the call, because model behaviour shifts between versions and "the AI decided this" is not a full answer six months later.

**Entry format:**

```
## D-nnn — <short title>
**Date:** · **Model:** · **Phase:** · **Status:** active | superseded by D-nnn
**Decision:** what was decided
**Why:** the reasoning
**Alternatives rejected:** what else was considered, and why not
**Consequences:** what this makes easier, and what it makes harder
```

---

## D-001 — Project topic: SOC alert triage

**Date:** 2026-08-13 · **Model:** Claude Opus 5 · **Phase:** pre-0 · **Status:** active

**Decision:** Build an RL agent that prioritises security alerts for a human analyst, with an RLHF layer.

**Why:** Three requirements had to be met at once — (a) genuinely sequential, so it's real RL and not classification in costume; (b) a reward that genuinely cannot be hand-written, so RLHF is necessary rather than decorative (7 lectures of the syllabus depend on it); (c) matching both team members' actual backgrounds. Alert triage satisfies all three. Diya works in Digital Trust / Cyber Assurance at KPMG; Pranav ran a cybersecurity club for three years and has a track record in rigorous ML evaluation.

**Alternatives rejected:**
- *RLHF negotiation agent* — architecturally impressive but too much of it would be un-explainable in an interview, which was the binding constraint.
- *Adaptive study planner* — safe and explainable, but generic (it's Anki with extra steps), and it overlaps Pranav's existing AcePlan project.
- *Cost-aware LLM cascade router* — genuinely strong and very current, but single-turn routing is closer to a contextual bandit than an MDP, weakening the Dynamic Programming phase.
- *Vulnerability patch scheduling* — same skeleton, better public data (EPSS/KEV), but much less demoable. Kept as the fallback if the professor insists on real data.

**Consequences:** Requires building a simulator, since no public dataset of analyst triage decisions exists. That's the project's main external-validity weakness and is documented in `PROJECT_BRIEF.md` §12 and `EXPLAIN.md` Part 9.

---

## D-002 — Actions are triage *rules*, not individual alerts

**Date:** 2026-08-13 · **Model:** Claude Opus 5 · **Phase:** pre-0 · **Status:** active

**Decision:** The action space is 5 fixed triage strategies (pull-highest-severity, pull-oldest, pull-most-critical-asset, pull-cheapest, bulk-close-low-risk) rather than "select one of the N queued alerts".

**Why:** "Pick one of N" gives a variable-sized action space where N changes every step and can reach the hundreds. That breaks tabular Q-learning outright and makes DQN awkward. Fixing the action space at 5 keeps the Q-table at 576×5, keeps every syllabus algorithm applicable without modification, and makes the learned policy human-readable — a security manager can read the policy table and agree or disagree with it.

**Alternatives rejected:**
- *Pointer/attention network over the queue* — handles variable actions properly, but neither student could reproduce it in an interview. Violates the teaching constraint.
- *Top-k fixed slots (e.g. always choose among the top 10 by severity)* — still couples the action space to a hand-written heuristic, and less interpretable than a rule-choice framing.

**Consequences:** Gives up the ability to express fine-grained per-alert preferences; the agent can only choose among five orderings. Accepted deliberately: interpretability and explainability are worth more here than expressiveness. **This is the most likely interview question about the project** — see `INTERVIEW_PREP.md` Q3.

---

## D-003 — Severity is deliberately made a weak predictor of ground truth

**Date:** 2026-08-13 · **Model:** Claude Opus 5 · **Phase:** pre-0 · **Status:** active

**Decision:** In the generator, `severity` correlates with `is_true_incident` at a target Pearson r of only 0.30–0.40, while the *combination* of alert type, asset criticality and verify cost carries substantially more signal.

**Why:** This reflects the widely-reported industry reality that vendor severity labels are poor predictors of real incidents — it's the reason alert fatigue exists. It's also the mechanism by which a learned policy can beat the severity-sort baseline.

**Alternatives rejected:** Making severity highly predictive — which would make severity-sort near-optimal and leave nothing to learn.

**Consequences:** This is an assumption we are *building into* the world, and it partly determines our headline result. It must be stated plainly in the report, in `EXPLAIN.md` Part 9, and in the viva. Not disclosing it would make the result misleading. The calibration is a Phase 0 exit gate precisely so it can't be quietly fudged later.

---

## D-004 — Dynamic Programming runs on an estimated model, not the true one

**Date:** 2026-08-13 · **Model:** Claude Opus 5 · **Phase:** pre-0 · **Status:** active

**Decision:** For the DP phase, estimate `P̂(s'|s,a)` and `R̂(s,a)` by counting transitions over ~50k random-policy episodes, then run value/policy iteration on that estimate.

**Why:** The true transition dynamics of a queue with Poisson arrivals and heterogeneous alerts aren't analytically tractable over the 576-state discretisation. Model estimation from rollouts is a legitimate, teachable technique and still demonstrates value iteration, policy iteration and the Bellman equations properly.

**Alternatives rejected:** Building a separate toy MDP with hand-written transition probabilities — cleaner mathematically but disconnected from the actual project. *Partially adopted anyway:* a 5-state MRP is hand-worked on paper for the report, purely to show the Bellman equations explicitly.

**Consequences:** The resulting policy is optimal **for the estimated model**, not for the true environment, and every mention of it must say so. States with sparse or zero visit counts need an explicit handling rule — to be decided and logged in Phase 1.

---

## D-005 — All syllabus algorithms are hand-written; no RL libraries

**Date:** 2026-08-13 · **Model:** Claude Opus 5 · **Phase:** pre-0 · **Status:** active

**Decision:** No Stable-Baselines3 or equivalent for Monte Carlo, SARSA, Q-learning, DQN, REINFORCE, actor–critic, or the reward model.

**Why:** The binding constraint on this project is that both students must be able to write the core functions from memory in an interview. A library call teaches nothing and cannot be defended under questioning.

**Alternatives rejected:** Using SB3 to move faster. Speed is not the bottleneck here; explainability is.

**Consequences:** Slower, and more chances for subtle bugs — which is why Phase 2 includes a tiny hand-checkable 2-state MDP with a known analytical answer as a correctness test. SB3 remains permitted as an optional *cross-check* for PPO only, and would require its own entry here.

---

## D-006 — `EXPLAIN.md` added beyond the Field Guide's nine documents

**Date:** 2026-08-13 · **Model:** Claude Opus 5 · **Phase:** pre-0 · **Status:** active

**Decision:** Add a tenth document, `EXPLAIN.md`, updated every session, holding a plain-English explanation of everything the project does and why.

**Why:** All nine Field Guide documents are written for someone already fluent in the codebase. None serves the "explain it to me like I know nothing" purpose, which is exactly what's needed for viva preparation, for onboarding a teammate, and for satisfying Field Guide habit #15 ("if you can't explain it in your own words, you're not ready to accept it").

**Alternatives rejected:** Folding plain-English explanations into `ARCHITECTURE.md` — would blur that file's purpose and make it too long to skim at session start.

**Consequences:** One more file to keep current. `CLAUDE.md` makes updating it a mandatory part of the session-end protocol.

---

## D-007 — Truth model: multiplicative lifts, with severity_lift as an explicit config knob

**Date:** 2026-08-13 · **Model:** Claude Fable 5 · **Phase:** 0 · **Status:** active

**Decision:** `P(true incident) = base_rate × type_lift × severity_lift[severity] × asset_lift[criticality]`, capped at 0.95. All lift arrays live in `config/env_default.yaml` as TUNE-marked values. Calibrated final values: `base_rate` 0.0135, `severity_lift` [0.1, 0.35, 2.4, 15.0], `asset_lift` [0.8, 1.2, 1.4].

**Why:** The brief (§4.2) demands severity be *weakly* correlated with truth (Pearson r 0.30–0.40), but the scaffold config had no mechanism linking them at all — type lifts alone give r ≈ 0. A per-severity multiplier is the simplest mechanism that creates the link, is tunable without touching code, and is explainable in one sentence. Multiplicative composition (rather than additive or logistic) keeps every factor independently interpretable: "this type is 3× riskier", "severity 3 is 15× the base".

**Alternatives rejected:**
- *Logistic model over features* — statistically tidier, but harder to explain and tune knob-by-knob; violates the teaching constraint for no measurable benefit.
- *Severity sampled conditionally on truth* (draw truth first, then severity) — equivalent maths, but inverts the causal story we tell (attackers don't consult the vendor's severity label) and makes the per-type lift awkward.

**Consequences:** Achieving r ≥ 0.30 with a 3% positive rate mathematically forces incident concentration at the top severity (P(true|sev 3) ≈ 30%, ~64% of incidents arrive at severity 3). Severity-sort therefore becomes a respectably strong baseline — accepted, because beating a strong industry baseline honestly is worth more than beating a strawman. The base rate had to drop from 0.03 to 0.0135 because the population-averaged lift product is ≈ 2.2.

---

## D-008 — Time-of-day modulation of incident rate: deferred

**Date:** 2026-08-13 · **Model:** Claude Fable 5 · **Phase:** 0 · **Status:** active

**Decision:** The generator does not modulate `P(true incident)` by time of day, although brief §4.2 mentions it.

**Why:** It affects neither Phase 0 calibration target (overall incidence, severity correlation) and no planned metric depends on it. Adding it now would be building ahead of need (CONSTRAINTS #18 in spirit) and would add a knob nobody would tune or defend.

**Alternatives rejected:** A sinusoidal or two-regime (day/night) modulation — trivially addable later inside `generate_shift` if a phase ever needs it; the config schema can grow a `time_of_day_lift` list then.

**Consequences:** The simulated world is slightly flatter than the brief's full description. If it is ever added, the calibration must be re-run and re-recorded, and this entry superseded.

---

## D-009 — Reward-timing semantics the brief leaves open

**Date:** 2026-08-14 · **Model:** Claude Fable 5 · **Phase:** 0 · **Status:** active

**Decision:** Three interpretations fixed in `env.py`: (a) detection delay is measured at the moment investigation *starts*; (b) the end-of-shift −200×mult penalty applies only to never-investigated true incidents whose deadline expired *within* the shift — an incident whose dwell budget outlives the shift is the next shift's problem; (c) a bulk-closed true incident is charged −150×mult once at closure and not charged again at shift end.

**Why:** (a) the decision moment is what the agent controls; (b) charging for deadlines that haven't expired yet would punish physics, not policy; (c) double-charging would make bulk-close's expected value incoherent and distort the reward-hacking analysis that action exists to enable.

**Alternatives rejected:** Delay measured at investigation *completion* — defensible, but couples the delay penalty to verify cost, which the false-positive penalty already prices.

**Consequences:** Every reward number downstream depends on these three lines. They must be stated in the report's MDP section, and both students should be able to defend each in one sentence.

---

## D-010 — Oracle is a path-clearing greedy, an upper bound in expectation only

**Date:** 2026-08-14 · **Model:** Claude Fable 5 · **Phase:** 0 · **Status:** active

**Decision:** `oracle_greedy` (i) catches any reachable true incident, most-urgent-deadline first; (ii) when an incident is unreachable, clears the pull-rule path with the fewest blockers, using bulk-close only when the sweep would remove blockers on that chosen path; (iii) otherwise waits as cheaply as possible.

**Why:** The first oracle (catch-if-reachable, else wait cheap) lost to severity-sort on recall (0.72 vs 0.85) — mid-tier incidents can stay off every rule's argmax for a whole shift, and a hygiene-loop bug made it bulk-close junk forever while a blocked incident sat. Path-clearing fixes both; final recall 0.86 vs severity-sort's 0.85.

**Alternatives rejected:** Full lookahead/planning over arrivals — no longer explainable in five minutes, and the ceiling it adds is not needed for the exit gate.

**Consequences:** The oracle is a bound *in expectation over seeds*, not per-stream: on eval seed 101 it loses to severity-sort by one incident through end-game timing (an incident arriving at minute 464 while the oracle was mid-investigation). This caveat is documented in EXPLAIN.md and must accompany any use of the word "upper bound" in the report. Also note: severity-sort's 0.85 recall confirms the D-007 prediction that it is a strong opponent.
