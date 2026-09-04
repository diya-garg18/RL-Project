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

---

## D-011 — Unvisited (state, action) pairs in the DP estimate → absorbing self-loop, reward 0

**Date:** 2026-08-14 · **Model:** Claude Fable 5 · **Phase:** 1 · **Status:** active

**Decision:** For any (s, a) never observed in the 50k random-policy rollouts, set `P̂(s | s, a) = 1` (deterministic self-loop) and `R̂(s, a) = 0`, rather than a uniform prior or a pessimistic/optimistic constant.

**Why:** ROADMAP requires unvisited states be handled *explicitly*, not left as silent zeros. With random-policy exploration only 589/2880 (s,a) pairs are ever seen — the unvisited ones are essentially unreachable under any sensible policy. A self-loop with reward 0 keeps their value pinned near 0, so value iteration never has a reason to *prefer* an unknown action; inventing transition mass toward real states would inject fiction into the Bellman backups.

**Alternatives rejected:**
- *Optimistic init (large +R̂)* — would lure the greedy policy into unexplored actions the model knows nothing about; the opposite of what we want from a reference policy.
- *Uniform P̂ over all 576 states* — mathematically smooth but meaningless: it asserts knowledge (equiprobable transitions) we do not have.

**Consequences:** The DP policy is only trustworthy on the visited core of the state space; on unvisited states it defaults toward low-value actions. This is acceptable for a reference ceiling but is another reason the DP "optimum" is optimum *for the estimated model* only (compounds D-004). If a later phase visits states DP never saw, its policy there is uninformed — note it if it ever matters.

---

## D-012 — Phase 1 exit gate restated on total reward; the reward hack is kept and featured

**Date:** 2026-08-16 · **Model:** Claude Opus 5 · **Phase:** 1 · **Status:** active
**Approved by:** Pranav (2026-08-16). **Diya countersign: pending** — the two Phase 0 amendments were both Diya-approved and this one should match that bar before the report cites it.

**Decision:** The Phase 1 exit criterion is restated from *"the DP policy beats severity-sort on recall@deadline"* to *"the DP policy achieves the highest mean total reward of any agent on the evaluation seeds"*, plus an added requirement that the Bellman machinery be verified against an answer derived outside the code. The hand-written reward function is **left exactly as it is**, and the reward-hacking behaviour it permits is recorded as a headline finding rather than patched.

**Why:** The original gate asked a reward-maximiser to top a metric it does not maximise. DP computes an optimal policy for `R̂`; recall@deadline is a *diagnostic* we compute afterwards, and nothing in value iteration is pointed at it. E-004 measured the consequence: DP earns 305.9 ± 127.6 total reward against oracle 214.1 and severity-sort 153.7, while scoring recall 0.43 against severity-sort's 0.87 — it uses BULK_CLOSE as paid waiting ~97% of the time and abandons 57% of real incidents. That was checked before being accepted (CONSTRAINTS #5): the arithmetic reconciles against per-step reward breakdowns, and the policy reproduces the same behaviour in the *true* environment, so it is not an artefact of the estimated model (D-004) or of unvisited-state handling (D-011).

`PROJECT_BRIEF.md` §3.5 states the exploit is deliberate — the reward is *designed* to be gameable, because a reward nobody can write correctly by hand is the entire argument for learning one from human preferences in Phase 5. The hack is therefore the project working as intended, discovered two phases earlier than expected because exact planning finds exploits that gradient-based learning only stumbles into.

This is the third gate amendment (after the two in Phase 0) and follows the same rule: when measurement contradicts a criterion, establish which of the two is wrong before changing either. Here the criterion was wrong.

**Alternatives rejected:**
- *Patch the reward so bulk-close is no longer profitable.* Rejected on three counts. It changes the MDP, which CONSTRAINTS #15 forbids without explicit human sign-off. It invalidates every number in E-002, E-003 and E-004. And it deletes the Phase 5 motivation — an un-gameable hand-written reward would make RLHF a bolted-on exercise rather than a necessity, which is precisely the failure mode `PROJECT_BRIEF.md` §2 argues this project avoids.
- *Declare Phase 1 failed and re-run DP with a recall-shaped objective.* Rejected: optimising a metric directly to make a gate pass is tuning-to-the-test, and would produce a "DP" that is not the textbook DP the syllabus requires.
- *Keep the gate and mark it permanently unmet.* Rejected: it blocks the roadmap on a criterion now known to be unsatisfiable in principle, and buries the most interesting result of the phase under an apparent failure.

**Consequences:** Phase 1 is closed. The DP row must never be reported as "best triage policy" without the recall figure beside it — the reward number alone is misleading, and the report's honesty depends on the pair travelling together. Phase 2's exit criterion has the same structural weakness and is now flagged in `ROADMAP.md`, but **deliberately not pre-emptively amended**: Q-learning gets run first and the gate gets decided on real numbers, the same way this one was. If the bulk-close hack recurs under Q-learning and DQN, that recurrence is itself evidence that the pathology lives in the reward rather than in any one algorithm — which is the strongest possible setup for Phase 5.

---

## D-013 — The worked MRP's constants live in code, not in config

**Date:** 2026-08-16 · **Model:** Claude Opus 5 · **Phase:** 1 · **Status:** active

**Decision:** The five-state MRP in `src/soc_triage/mrp_example.py` — its transition matrix, transition rewards, discount factor and hand-derived value function — is defined as module-level constants, not loaded from `config/*.yaml`. A deliberate, narrow exception to CONSTRAINTS #9.

**Why:** CONSTRAINTS #9 targets *tunables* — "a number someone might want to change." These are the opposite. They are the definition of a worked example whose entire value is that a human derived `V = [52/11, −4, +20, 0, 0]` from them on paper. Exposing them in YAML would invite an edit that leaves the derivation in `FEATURE_001` and the frozen `HAND_COMPUTED_V` silently disagreeing with the config — turning an external correctness anchor into a test that checks nothing. Constants in the module keep the numbers, the derivation and the assertion within one reading distance of each other.

The discount factor deserves its own note: the example uses γ = 0.9, *not* the project's 0.99 from `config/training_default.yaml`. That is chosen so the only non-integer value stays an exact fraction (52/11) rather than an unmemorable decimal. Anyone reading the example must not mistake its γ for the project's.

**Alternatives rejected:**
- *A `config/mrp_example.yaml`.* Literal compliance with #9, but it separates the numbers from the derivation that depends on them and creates exactly the drift risk described above.
- *Inlining the constants into `tests/test_mrp_bellman.py`.* Would satisfy "not in src", but then `scripts/run_mrp_example.py` could not import them for the report output, and the MRP would be untestable from anywhere else.

**Consequences:** One module in `src/` contains hard-coded numbers, which will look like a #9 violation to anyone reading the constraint without this entry — hence the pointer to D-013 in the module docstring. If the example is ever changed, `FEATURE_001`'s derivation, `HAND_COMPUTED_V`, and the arithmetic asserted in `test_expected_rewards_match_hand_arithmetic` must all be redone together, or the tests will catch the mismatch immediately (which is the intended safety net).

---

## D-014 — Phase 2 gets its own hand-solved anchor, built before the learners

**Date:** 2026-08-17 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active
**Approved by:** Pranav (2026-08-17), before implementation.

**Decision:** A second worked example, `src/soc_triage/tiny_mdp.py` — a two-state, two-action MDP whose optimal action-value function is derived on paper (FEATURE_002) — is built **first** in Phase 2, ahead of Monte Carlo, SARSA and Q-learning. Its constants live in code rather than config, extending D-013's narrow exception to CONSTRAINTS #9 to this file for identical reasons.

**Why:** FEATURE_001's MRP checks the Bellman backup for `V`. It cannot check `Q` at all — an MRP has no actions — so Phase 2's three learners would otherwise have no external anchor whatsoever. Their only available check would be agreement with each other, and agreement is not correctness: three implementations of the same misunderstanding agree perfectly and are all wrong. That is precisely the failure mode this project's documentation exists to prevent.

The ordering is the other half of the decision, and it is a deliberate departure from the written box order in `ROADMAP.md` (where the test is the last box, not the first). An anchor built *after* the learners is ambiguous — when Q-learning disagrees with it, either could be at fault. Built first and cross-checked against `agents.dp.value_iteration`, which Phase 1 already validated against an independently hand-derived answer (E-005), the anchor arrives pre-trusted and every later disagreement is unambiguously the learner's fault. The trust chain is: human arithmetic → `value_iteration` (E-005) → `tiny_mdp` (E-006) → the Phase 2 learners.

The fixture's design was constrained by what each property rules out, not by what looks tidy: continuing rather than episodic (so a TD learner cannot pass without bootstrapping), differing optimal actions per state (so a constant-action stub fails), deterministic (so the assertions stay exact), and — the one that took three attempts — a **wide decision margin**. See the alternatives below.

**Alternatives rejected:**
- *Reuse the 5-state MRP.* Wrong shape: no actions, so no `Q` to check.
- *Skip the anchor and cross-check the three learners against each other, and against the Phase 1 DP Q-table on the real environment.* This is the roadmap's other Phase 2 box and it is worth doing, but it is not a substitute. The DP Q-table is itself only optimal for an *estimated* model (D-004), so agreement with it confirms consistency, not correctness.
- *A tidier reward design with a narrow margin.* The first two candidate MDPs gave the optimal action a winning margin of 0.1 and 0.3 on values around 10 — a 1–3% gap. A tabular learner sitting within 3% of `q_*` is entirely ordinary, and would flip the greedy policy at random, producing a flaky test that gets muted rather than a strict one that gets trusted. Rejected in favour of margins of 3.3 and 2.3, recorded as `MIN_ACTION_GAP` so a later edit cannot erode them silently.
- *Padding the 2-action MDP to 5 actions with zero-reward self-loops* so `dp.value_iteration` (which loops over `N_ACTIONS = 5`) accepts it. Rejected: that adds a genuinely new action worth 0 — harmless in this MDP where all values are positive, silently wrong in any MDP with negative values. Duplicating the real actions cyclically can only tie the maximum, never beat it, so `q_*` is provably preserved.

**Consequences:** Two modules in `src/` now contain hard-coded numbers and will look like CONSTRAINTS #9 violations to anyone reading the constraint without D-013 and this entry — both module docstrings carry the pointer. If the tiny MDP is ever changed, FEATURE_002's derivation, `HAND_COMPUTED_Q`, `HAND_COMPUTED_V`, `HAND_COMPUTED_POLICY`, `MIN_ACTION_GAP` and the per-entry arithmetic in `test_hand_computed_q_matches_the_pen_and_paper_arithmetic` must all be redone together — the tests catch the mismatch immediately, which is the intended safety net.

One design consequence is worth flagging forward: under the tiny MDP's optimal policy the agent never leaves `QUIET`, so `BUSY` is reachable only by exploration. This makes the fixture a live demonstration of why ε > 0 is necessary — good teaching material, but also a trap. A learner tested with ε pinned to 0 will fail on `Q(BUSY, ·)` for a reason that has nothing to do with its update rule. Noted in `TEST_CHECKLIST.md` so the next session does not debug the wrong thing.

---

## D-015 — Epsilon decays in an explicit `end_episode()` hook, not inferred from `done`

**Date:** 2026-08-17 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active

**Decision:** `QLearningAgent` exposes `end_episode()`, which applies one geometric decay step to epsilon and floors it. The caller — `scripts/train.py`, once it exists — must invoke it once per episode. Decay is deliberately **not** performed inside `update()`, and **not** triggered by `done=True`.

**Why:** `config/training_default.yaml` specifies `decay: 0.9995  # per episode`. A 480-step shift contains hundreds of `update()` calls, so applying the same factor per step would drive epsilon from 1.0 to its 0.05 floor inside a *single episode* instead of over ~6000. The agent would stop exploring almost immediately, and the symptom — a learning curve that rises briefly then flattens — reads as "the algorithm has converged", not as "the exploration schedule is broken by three orders of magnitude". Silent, plausible-looking failures are the expensive kind.

Inferring the boundary from `done=True` was the tempting alternative because it needs no caller cooperation. It fails on continuing tasks: `tiny_mdp` (D-014) has no terminal state and never sets `done`, so epsilon would never decay at all in exactly the fixture used to verify the agent. Worse, it fuses two genuinely separate concepts — *termination*, which changes the Bellman target, and *episode boundary*, which changes the exploration schedule. On a truncated continuing task those two differ, and code that treats them as one is the source of the classic "agent learns the world ends every 200 steps" bug. Keeping them apart in the API keeps them apart in the students' heads, which is the point.

**Alternatives rejected:**
- *Decay inside `update()` per step.* Wrong by ~500x per episode, and invisible.
- *Decay when `done=True`.* Breaks on continuing tasks; conflates termination with truncation.
- *An episode counter inside the agent that the agent increments itself.* The agent cannot know where an episode ends without being told; this just hides the same coupling behind a less obvious interface.

**Consequences:** Any training loop that forgets `end_episode()` runs at fixed `epsilon_start` forever — a real failure mode, mitigated by `test_epsilon_does_not_decay_on_update` documenting the intent and by this entry. `agents/base.py` was **not** modified to add `end_episode()` to the base `Agent` interface: baselines have no exploration schedule, and widening the interface to suit one subclass would push a no-op onto five agents that do not need it (CONSTRAINTS #18 — don't refactor what you weren't asked to touch). If SARSA and Monte Carlo both end up needing it, promoting it to the base class at that point is the right moment, not now.

A second, smaller decision recorded here rather than separately: `load_training_config` now parses the `epsilon` and `q_learning` sections, which already existed in the YAML but went unread while Phase 1 needed only `common` and `dp`. Three range checks were added — `alpha` in (0, 1], `0 <= min <= start <= 1`, `decay` in (0, 1]. These fail at load time because an out-of-range alpha yields a diverging Q-table and an incoherent epsilon schedule yields an agent that never explores; both present as algorithm bugs and are expensive to trace back to a typo in a config file.

---

## D-016 — Q-learning trains on its own seed block, one fresh shift per episode

**Date:** 2026-08-17 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active
**Approved by:** Pranav (2026-08-17), before the first training run.

**Decision:** Training episodes draw seeds from a dedicated block, `q_learning.train_seed_start: 200000`, giving one distinct shift per episode (20,000 per run, offset per repeat). The existing `seeds.train` list (1–10) is **retained but repurposed** as a *training diagnostic* set — the seeds the learning curve is measured on, never trained on. Evaluation seeds (101–105) remain untouched and are read exactly once, at the end of a run.

**Why:** `config/training_default.yaml` specifies `n_episodes: 20000` against a `seeds.train` list of ten. Taken literally, a run replays the same ten alert streams 2,000 times each. A tabular agent with 2,880 state-action cells and 20,000 episodes of practice on ten fixed shifts is being invited to memorise those shifts rather than learn a triage policy, and the resulting eval-seed gap would be blamed on the algorithm rather than on the training distribution.

The fix follows an existing precedent rather than inventing a convention: `dp.estimation_seed_start: 10000` exists for exactly this reason — Phase 1 needed 50,000 episodes and ten seeds could not supply them (D-004). Phase 2 needs 100,000 across five repeats. Blocks are disjoint by construction — train-diag 1–10, eval 101–105, calibration 1000–3099, DP estimation 10000–59999, Q-learning training 200000+ — and `load_training_config` now rejects a start below 100000 rather than trusting the comment, per CONSTRAINTS #2's requirement that seed separation be enforced in code.

Keeping seeds 1–10 as the diagnostic set is the second half of the decision and matters just as much. The learning curve has to be plotted against *something*, and plotting it against the evaluation seeds would be tuning against them by eye — a violation of CONSTRAINTS #2 that leaves no trace in the code.

**Alternatives rejected:**
- *Cycle the ten train seeds 2,000 times.* The literal reading of the config, and the cheapest. Rejected: it makes memorisation the most likely explanation for any train/eval gap, which would contaminate the interpretation of every Phase 2 result.
- *Cycle the ten, but measure and report the memorisation gap.* Genuinely tempting — it turns a weakness into evidence. Rejected because it spends the phase's headline result on a self-inflicted artefact rather than on the algorithm, and because the gap it would measure is confounded by the seed-difficulty effect discovered in E-008.
- *Draw training seeds at random each episode.* Loses reproducibility: two runs would face different shifts and could not be compared.

**Consequences:** Q-learning's training distribution is now far wider than DP's estimation distribution, so the two are not learning from equivalent experience — worth stating whenever E-004 and E-008 are compared. Any future learner (SARSA, Monte Carlo, DQN, REINFORCE) should use this same block, offset per algorithm, or its results will not be comparable to Q-learning's.

D-016 note added 2026-08-17: SARSA (400000+) and Monte Carlo (600000+) received their own blocks under the same rule, and `load_training_config` now rejects duplicate or too-close blocks rather than trusting the YAML comments.

E-008 then found something this decision did not anticipate and does not fix: **the eval block itself is unrepresentative.** Every agent, oracle included, scores 120–230 reward higher on seeds 101–105 than on seeds 1–10, with per-seed standard deviations several times larger than the differences being reported. That is a separate and more serious problem, it affects every experiment in the project rather than just Phase 2, and it is recorded in E-008 as an open decision for the humans rather than acted on here.

---

## D-017 — On-policy learners are graded against an ε-soft target, not against q\*

**Date:** 2026-08-17 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active

**Decision:** SARSA and first-visit Monte Carlo are tested against `tiny_mdp.epsilon_soft_q(epsilon)` — the action-value function of the ε-greedy policy they actually follow — rather than against `HAND_COMPUTED_Q`. Q-learning continues to be graded against `HAND_COMPUTED_Q`.

**Why:** Q-learning is off-policy and converges to q\* (S&B §6.5). SARSA (§6.4) and MC control (§5.4) are on-policy: their fixed point is q_π for the ε-greedy policy being followed, which is **not** q\* whenever ε > 0. On the tiny MDP at ε = 0.1 the two differ by more than 1.5 — `q*(QUIET,WAIT) = 10.0` against an ε-soft value of 8.54. A correct SARSA graded against q\* would have looked badly broken, and the obvious "fix" would have been to break SARSA until it matched.

The new target is not allowed to float free. `epsilon_soft_q(0.0)` must reproduce `HAND_COMPUTED_Q` to machine precision, because at ε = 0 the ε-greedy policy *is* the greedy policy and the backup reduces to the Bellman optimality equation. `test_epsilon_soft_q_collapses_to_q_star_as_epsilon_goes_to_zero` asserts exactly that, which keeps the on-policy target anchored to the same pen-and-paper answer everything else in Phase 2 is anchored to.

It is also a genuinely independent computation: `epsilon_soft_q` takes the exact expectation over next actions, whereas SARSA samples one. Agreement between them is evidence, not tautology.

**Alternatives rejected:**
- *Grade SARSA and MC against `HAND_COMPUTED_Q` with a loose tolerance.* The gap is not error, it is a real property of the algorithms; hiding it behind a tolerance would teach the students the opposite of the on-policy/off-policy distinction the report is meant to explain.
- *Drive ε to ~0 during the test so all three converge to q\*.* Textbook GLIE, and it would work — but on this fixture BUSY is only reachable by exploring, so ε → 0 starves half the MDP and the test would fail for an unrelated reason.
- *Compare only policies, not values.* Robust but weak: it would pass an agent whose values were wildly wrong as long as their ranking survived.

**Consequences:** Tolerances differ by algorithm and were set from measurement, not assumption — Q-learning 1e-9 (measured 9.24e-14), SARSA 0.15 (worst 0.100 over 8 seeds), MC 0.40 (worst 0.272). The looser two are not slack: SARSA's residual is constant-α noise, which shrinks with α (0.113 → 0.080 → 0.041 for α 0.05 → 0.01 → 0.002) but *not* with more episodes, and MC is the higher-variance estimator by construction.

**A related correction, recorded here because it invalidated a committed comment.** `tiny_mdp.HORIZON = 200` was documented as harmless for truncation on the grounds that γ²⁰⁰ ≈ 7e-10. That reasoning is right for the return measured from t=0 and **wrong for Monte Carlo**, which computes a return from *every* timestep — at t=199 the missing tail is the entire value. Measured bias against the ε-soft target: 2.75 at HORIZON=50, 0.47 at 200, 0.09 at 800. `MC_HORIZON = 800` was added, set where the bias falls below the constant-α noise the TD learners already carry. The original comment was not a typo; it was a plausible argument applied to the wrong quantity, which is why it survived review.

---

## D-018 — A reduced training run may not overwrite a full run's artefacts

**Date:** 2026-08-17 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active

**Decision:** `scripts/train.py` writes `results/<agent>_Q.npy` and `_visits.npy` only when the run matches the configured budget (full `n_episodes`, full `eval_every`, ≥ 5 repeats). Any reduced run writes to `results/smoke/` instead and says so.

**Why:** This is a bug fix disguised as a policy, and the bug is worth recording. A `--episodes 200 --repeats 1` smoke test silently replaced the artefacts of a completed 20,000-episode run. Nothing errored. The stale file was a valid `.npy` of the right shape, and the corruption surfaced only later and indirectly, as an unexplained drop in state coverage — 121 states in E-009 against 81 in the first run of `compare_agents.py` — which initially looked like a bug in the comparison script.

CONSTRAINTS #4 forbids deleting or overwriting an experiment result. That rule was written with "don't delete a run that looked worse" in mind, but the more likely failure is this one: an accidental overwrite by a routine command, with no warning and no error. Making the config-faithful path the only one that can write to `results/` moves the constraint from a rule people must remember into the code.

**Alternatives rejected:**
- *Remember not to smoke-test with the same output paths.* This is precisely the class of discipline that fails under time pressure, and it had already failed once within an hour of the script being written.
- *Timestamp every artefact.* Solves overwriting but breaks every downstream script's fixed path, and clutters `results/` with runs nobody will read.
- *Refuse to write anything on a reduced run.* Tempting, but smoke tests genuinely benefit from inspectable output; routing it to `results/smoke/` keeps that without the risk.

**Consequences (D-018):** `results/smoke/` may accumulate stale files. It is gitignored along with the rest of `results/`, and nothing reads from it. Scripts that consume artefacts (`policy_table.py`, `compare_agents.py`) read only from `results/` and will fail loudly with the exact command needed if a full run has not been done — which is the correct behaviour, since a reduced run's Q-table is not a result.

---

## D-019 — The evaluation seed block is widened from 5 seeds to 30

**Date:** 2026-08-17 · **Model:** Claude Opus 5 · **Phase:** 2 (consequences for 0 and 1) · **Status:** active
**Approved by:** Pranav (2026-08-17), on the recommendation recorded in E-008. **Diya countersign: pending** — this is a larger change than D-012 and should carry her sign-off before the report cites any number affected by it.

**Decision:** `seeds.eval` becomes `[101..130]`. The original five (101–105) are **kept inside** the block. Every agent in the project has been re-measured on the widened block (E-014). No prior experiment entry has been altered or deleted.

**Why:** CONSTRAINTS #3 requires at least 5 seeds and that floor was honoured throughout — but nobody had checked whether 5 samples could resolve the effects being claimed, and they could not. Severity-sort's per-seed reward standard deviation is ±220 against inter-agent differences of ~100, so the standard error of a 5-seed mean was roughly the size of every finding built on it. Two independent lines of evidence forced the change: E-008 found that all agents including the oracle scored 120–230 higher on seeds 101–105 than on seeds 1–10, and E-012 found that deliberate order-of-magnitude hyperparameter changes could not be separated from run-to-run noise.

Thirty was chosen from that arithmetic rather than by preference: 220/√30 ≈ 40, comfortably below the effects of interest, and it matches the 30-seed diagnostic already used in E-003 so the two are directly comparable.

Keeping 101–105 inside the block is the second half of the decision and matters as much as the widening. It makes every pre-widening number a *sub-sample* of the new measurement, so old and new results can be discussed in the same breath and the change can be honestly described as adding seeds rather than replacing them. `tests/test_eval_protocol.py` asserts all five are still present.

**Alternatives rejected:**
- *Leave the block at 5 and add a caveat.* This was the status quo for one session and it is not viable: E-014 shows a 5-seed measurement did not merely widen the error bars on Phase 1's headline result, it got the **sign wrong**. A caveat on a number that is wrong by 500 reward is not honesty, it is decoration.
- *Widen to a fresh, disjoint block (e.g. 200–229).* Cleaner in one sense — no overlap with anything previously reported — but it orphans every prior result, since old and new would share no seeds and could not be compared at all.
- *Report on both 5 and 30 seeds side by side indefinitely.* Rejected as a permanent arrangement: it invites quoting whichever column is more flattering. The 5-seed column appears in E-014 exactly once, for the specific purpose of showing how badly it misled.

**Consequences, and they are large:**

1. **Phase 1's amended exit criterion is falsified.** D-012 criterion 2 required the DP policy to achieve the highest mean total reward of any agent. On 30 seeds DP scores **−201.2 ± 438.5, the worst of any planned or learned agent**, against +305.9 on 5 seeds. Phase 1's status is therefore **reopened** — see D-020.
2. **Phase 2's exit criterion fails on both halves**, and more comprehensively than before: no learner reliably beats severity-sort on reward either. See D-020.
3. **Phase 0's exit criterion still passes** (the oracle remains strictly best on total reward, 168.0 vs 40.4), so Phase 0 is not reopened. But the *rationale* attached to its second amendment — that no honest greedy oracle can reliably out-recall severity-camping — is weaker: on 30 seeds the oracle out-recalls it, 0.87 to 0.84. That sentence should not be repeated as established.
4. **The reward-hacking narrative must be restated, not abandoned.** Still true: the reward is exploitable and every agent trades recall away chasing it. No longer true: that the trade pays. The exploit was profitable on five particular shifts and does not generalise — which is a *stronger* argument for Phase 5, since the objective turns out to be not merely misaligned but unstable.
5. **One genuinely new positive finding**, invisible at 5 seeds: the learned policies are roughly four times more consistent shift-to-shift than the heuristics (±50 against ±220). Nothing in the reward function values that, which is itself evidence for learning a reward from humans.
6. Every future experiment costs slightly more to evaluate. Negligible — evaluation was never the bottleneck; training is.

**The lesson worth carrying, stated plainly:** every number in this project was computed correctly, reported with its standard deviation, and reproduced deterministically — and one of them had the wrong sign. E-002 printed ±218.7 beside a mean of 153.7 and nobody drew the inference. **Reporting a standard deviation is not the same as reading it.** Compare the spread to the size of the claimed effect *before* believing the effect.

---

## D-020 — Phase 2 closes with its exit criterion NOT met; Phase 1 is reopened

**Date:** 2026-08-17 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active
**Approved by:** Pranav (2026-08-17). **Diya countersign: pending**, together with D-019.

**Decision:** Phase 2 is closed as **built but not passed**. All eight roadmap boxes are complete; the exit criterion is not met and **is not being restated to make it pass**. Phase 1 is **reopened**, its amended criterion 2 having been falsified by E-014.

**Why the gate is not restated.** This is the decision that matters, and the temptation was real. Phase 1's gate was legitimately amended once (D-012) because the criterion asked a reward-maximiser to top a metric it does not optimise — a category error in the criterion. The same move is *not* available here. Phase 2's criterion asked the learners to beat severity-sort on recall and MTTD, and on an honest 30-seed measurement they lose on recall (0.66–0.72 vs 0.84) and are indistinguishable on reward. There is no category error to correct; the agents simply did not do the thing. Amending the gate now would be tuning the criterion to the result, which is the precise failure `CONSTRAINTS.md` and this project's whole thesis exist to prevent.

The second half of the criterion — "the printed policy table shows a behaviourally interpretable strategy shift as time runs out" — was reported satisfied in E-009 and that assessment is **withdrawn** (E-013): the shift reverses direction depending on which learner produced it, so it is not a property of the task.

**What Phase 2 did achieve**, and it is not nothing: three textbook algorithms written by hand and each verified against a pen-and-paper answer before touching the real environment; an ablation study honest enough to report that none of its effects clear the noise; a policy-rendering tool that marks absence of data rather than inventing a preference; and the discovery that the project's evaluation protocol was too weak to support its own conclusions. The last of those is worth more than a passed gate.

**Why Phase 1 is reopened rather than quietly annotated.** CONSTRAINTS #4 forbids erasing a result, and E-004 stands as recorded. But leaving Phase 1 marked ✅ COMPLETE when its load-bearing criterion is known false would be a worse offence than the original error: it would mean the roadmap asserts something the log contradicts. Reopened means the phase's status is honest; it does not mean the work is discarded.

**What the humans must now decide** (this is where it stops being mine to call):
1. Whether Phase 1's gate is re-amended a second time, or Phase 1 is accepted as "built, criterion falsified on better measurement" in the same shape as Phase 2. Re-amending twice on the same phase deserves scrutiny — the more honest option is probably the latter.
2. Whether DP's collapse is investigated before Phase 3. The hypothesis in E-014 (D-004 + D-011: DP is lost outside its 133-state estimated core) is testable by correlating per-seed DP reward against how far each shift strays from that core. It has **not** been tested, and DP is the report's Phase 1 centrepiece.
3. Whether Phase 3 (DQN) proceeds now. It optimises the same unstable reward on the same environment, so it will likely reproduce the same pattern — which is informative, but it is a fifth data point on a question already answered rather than a new one.

**Consequences:** the report's spine changes shape. It was "DP games the reward → the learners game it too → RLHF fixes it." It becomes "four methods all sacrifice recall to a hand-written reward, none of them reliably profits by it, and the measurement that made it look profitable was too small to trust — so the objective itself must be learned." That is a harder story to tell and a considerably more honest one.

**Alternatives rejected:**
- *Restate the Phase 2 gate on total reward, as D-012 did for Phase 1.* Fails on the facts: on 30 seeds the learners do not beat severity-sort on reward either (47.6 and 40.5 against 40.4, inside a ±220 spread). There is no metric on which they clearly win, so any restatement would be reverse-engineered from the result.
- *Restate it on reward **consistency**, where the learners genuinely do win (±50 vs ±220).* This is the most tempting option and it is still goalpost-moving: nobody set out to optimise variance, no criterion mentioned it, and elevating a finding discovered after the fact into the gate it passes is exactly the pattern to avoid. It is recorded as a finding in E-014 and left there.
- *Delay closing Phase 2 until the gate passes.* Open-ended, and there is no reason to expect a tabular method to close a 0.12–0.18 recall gap on this state encoding. Phase 3's function approximation is the designed answer to that.

---

## D-021 — Phase work is split into meaningful commits, alternating between teammates, with the balance tracked in code

**Date:** 2026-08-17 · **Model:** Claude Opus 5 · **Phase:** cross-cutting · **Status:** active
**Requested by:** Pranav (2026-08-17). Added as CONSTRAINTS #24–26 with his explicit authorisation to edit that file.

**Decision:** Every phase is divided into meaningful commits. The two students alternate machines, and **each must have real commits in the history**. Claude reports the balance via `scripts/commit_balance.py` at the start and end of every session and says plainly when the work should move to the other person. Handover threshold: **3 commits**. A `.mailmap` collapses split author identities so the measurement is accurate.

**Why:** The git history is part of what gets evaluated, and it will not reflect an even split by accident. Whoever happens to be at the keyboard during a long session accumulates commits fast — measured at the time of this decision, the split was **Diya 17 / Pranav 7**, with Phase 0 entirely Diya's (12 commits) and Phase 2 entirely Pranav's (4). Neither student chose that; it is just what happened. Nobody notices the drift while it is happening, and by the time it is visible it is expensive to correct, which is why the check is automated and Claude raises it unprompted rather than waiting to be asked.

The `.mailmap` matters more than it looks: Diya had committed under two identities (her personal address and the GitHub web-editor noreply address), which split her contributions across two names in `git shortlog` and GitHub's contributor graph — making the imbalance look *worse* than it was. The mailmap rewrites nothing; it only tells git how to collapse identities when summarising.

**The boundary, stated explicitly because it is the whole risk in this decision.** The requirement is that the *work* be evenly split, not that the *record* be made to look even. A commit under a name is a claim that the person did that work and can explain it; an examiner may ask either student to walk through any commit bearing their name. So the history is balanced by **handing over at the right time** — never by committing on someone's behalf, re-attributing authorship, or padding with cosmetic commits. `CONSTRAINTS.md` #24 says this in the constraint itself, and `scripts/commit_balance.py` repeats it in its own docstring, because that is where someone tempted to shortcut it will be looking.

**Alternatives rejected:**
- *Rewrite the existing history to rebalance authorship.* This would have equalised the count in one command. Rejected on two grounds: the commits are already pushed, so it needs a force-push over shared history; and it would attribute work to whoever the rebalance favoured regardless of who did it, which is the exact misrepresentation #24 forbids.
- *Track the balance by eye.* Tried implicitly for five sessions and it produced a 17/7 split. The point of a script is that it runs whether or not anyone remembers to care.
- *A tighter threshold than 3 commits.* Rejected because it would force handovers mid-feature, and CONSTRAINTS #25 exists precisely because a handover with failing tests costs more than the balance gains.
- *Counting lines changed rather than commits.* More faithful to effort in principle, but trivially gamed by documentation volume — this project writes a great deal of prose — and GitHub's contributor view counts commits, which is what will actually be looked at.

**Consequences:** Pranav is 10 commits behind at the time of writing, so **Phase 3 should run on his machine** until the gap closes to within 3. Phase 3 is DQN and naturally divides into several commits (network, replay buffer, target network, training loop, the two required ablations), so it can absorb that without padding. Future phases should be planned with the split in mind rather than corrected afterwards.

One honest limitation of D-021: commit *count* is a weak proxy for contribution. Two students could satisfy this constraint perfectly while one did the thinking and the other typed. The constraint is worth having anyway — it makes the obvious failure visible — but it is not a substitute for both students being able to explain the code, which is what `INTERVIEW_PREP.md` and the teaching constraint in `CLAUDE.md` exist for.

---

## D-022 — Phase 1 closes as "built, criterion falsified on better measurement"; its stated cause was wrong too

**Date:** 2026-08-18 · **Model:** Claude Opus 5 · **Phase:** 1 · **Status:** active
**Decided by:** Pranav (2026-08-18), choosing between the two options D-020 put to the humans. **Diya countersign: pending**, with D-012, D-019 and D-020.

**Decision:** Phase 1 is **not** re-amended a second time. It closes in the same shape as Phase 2 — **built, criterion falsified on better measurement**. Its work stands; its gate does not.

**Why not amend again.** D-012 already amended Phase 1 once, legitimately: the original criterion asked a reward-maximiser to top a metric it does not optimise, which is a category error in the criterion. A second amendment would be different in kind. Nothing is wrong with criterion 2 as written — "the DP policy achieves the highest mean total reward of any agent" is a perfectly sound thing to require. It is simply **false** on an honest measurement: −201.2 against severity-sort's +40.4 and the oracle's +168.0. Rewriting a criterion because the result came out the wrong way, having already rewritten it once, is precisely the pattern this project exists to avoid. Two amendments to one phase would make the gate look like something fitted to the outcome, which would taint the first amendment retrospectively as well.

**And the explanation offered for the collapse was also wrong.** E-014 proposed — explicitly as an untested hypothesis — that DP fails on shifts straying outside its 133-state estimated core, falling back on D-011's convention. E-015 tested it and refuted it: **off-core share is 0.0% on all 30 eval seeds**, for states *and* for state-action pairs. DP never leaves its core, and D-011's convention never fires at evaluation time. The control rules out seed difficulty too — corr(severity reward, DP reward) = +0.085, and on seed 128 DP loses 755 where severity-sort gains 233.

The remaining explanation, **stated as an untested hypothesis and labelled as one**: `P̂`/`R̂` were counted under a uniform-random policy, but DP bulk-closes ~97% of the time, so the transitions following its own actions are not the transitions the model was built from — even though the states are familiar. Distribution shift in the estimate, not gaps in it. The test is named in E-015: re-estimate the model from DP-policy rollouts and check whether the resulting plan's predicted value matches its measured reward.

**Consequences.**
- `ROADMAP.md` Phase 1 reads **built / criterion falsified**, with E-004 intact and its gate assessment withdrawn (CONSTRAINTS #4).
- **D-004's caveat needs restating in the report.** "Optimal for the estimated model, not the true environment" has been read throughout — including by E-014 — as being about *coverage*. E-015 shows coverage is fine on the eval distribution. The real gap is between the policy the model describes and the policy being planned. That is a sharper and more interesting statement of the same caveat, and it is the version the report should carry.
- Phases 1 and 2 now both close unpassed. That is an uncomfortable pair to present and it is the honest one: four methods, one exploitable reward, no method reliably beating a severity sort, and a measurement protocol that had to be fixed mid-project before any of it could be seen.

**Alternatives rejected:**
- *Amend criterion 2 to "highest reward among model-based methods" or similar.* Reverse-engineered from the result; see above.
- *Leave Phase 1 open indefinitely pending the distribution-shift test.* The test is worth running, but the gate's status does not depend on its outcome — DP scores −201.2 either way. Closing on the measurement and leaving the explanation open is the accurate split.

## D-023 — DQN input scaling uses fixed domain divisors, held in `training_default.yaml`

**Date:** 2026-08-18 · **Model:** Claude Opus 5 · **Phase:** 3 · **Status:** active
**Approved by:** Pranav (2026-08-18), before implementation.

**Decision:** Each `featurise()` column is divided by a fixed constant taken from the domain — queue length by 150, ages by the 480-minute shift, severity by 3, and so on. The divisors live in `config/training_default.yaml` under `dqn.feature_scales`, **not** in `config/env_default.yaml`, and `state.feature_scale_vector()` orders them to match `FEATURE_NAMES`, raising if any column is missing or unknown.

**Why:** Measured across 888 observations, the columns span a 470× range: `max_age_min` reaches 473.57 while every `frac_type_*` stays inside [0, 1]. Fed raw to an MLP, the two age columns dominate every gradient the network ever takes. Scaling here is a correctness requirement, not an optimisation — and it was only visible because the spread was measured before the design was written rather than assumed.

The location matters more than it looks. `runner.config_hash()` content-hashes `env_default.yaml` into every `EpisodeRecord` written since Phase 0. Adding a key there would change the hash and orphan every prior result from the config that produced it — a silent break in the traceability the whole project rests on. `training_default.yaml` is not hashed, so it is safe.

**Alternatives rejected:**
- *Running mean/std normalisation.* Standard, and wrong here: the statistics would drift during training, so a state's encoding in episode 1 differs from episode 20000, and training and evaluation would not agree on what a state even is. Fixed domain constants keep the encoding stable and explainable in an interview.
- *Put the divisors in `env_default.yaml` where the other domain constants live.* Correct on tidiness, disqualifying on traceability. See above.
- *Skip scaling and let the network learn it.* It can, eventually, via the first layer's weights — but it wastes capacity and the failure mode is silent: no error, just a worse number.

**Consequences:** The divisors are domain facts and must never be tuned. Tuning them would make them hyperparameters fitted to the evaluation, which CONSTRAINTS #2 forbids. `test_scale_vector_rejects_a_missing_column` and `..._an_unknown_column` exist because a partial mapping would leave one column unscaled and the bug would surface only as a slightly worse result.

---

## D-024 — `train_freq: 4` — the decision stands, its original justification did not

**Date:** 2026-08-18 · **Model:** Claude Opus 5 · **Phase:** 3 · **Status:** active, **stated reason corrected same day**

**Decision:** The DQN takes one gradient step every 4 environment steps, so a run is 20,000 episodes — identical to the tabular learners' budget.

**Why (as originally argued):** A pre-design probe measured 1.107 ms per batch-64 gradient step. At one step per transition a 20,000-episode run projected to 18.3 minutes; at one in four, 4.6 minutes. `train_freq: 4` was therefore presented as what made the full budget affordable, so the DQN-vs-tabular comparison would carry no budget confound.

**The correction.** Both numbers in that argument were wrong. Re-measured against a real training loop: **9.87 ms** per gradient step and **0.709 ms** per `act()` call, roughly 10× the pre-design figures, putting a training episode at ~204 ms and a 20,000-episode run at **~68 minutes**, not 4.6. The original probe measured neither the optimiser step, the gradient clipping, nor the per-step forward pass that action selection performs — it timed a fragment and the fragment was reported as the whole.

`torch.set_num_threads(1)` was ruled out as the cause: it is the *fastest* of 1, 4 and 8 threads on a 19,461-parameter network (159 / 172 / 375 ms per episode). A component breakdown found no single hotspot — forward+backward 3.1 ms, Adam 2.9 ms, clipping 0.77 ms, target forward 0.75 ms — just per-op framework overhead on a small net.

**Why the decision survives its reasoning:** the *comparison* argument was always the real one. Matching the tabular learners' 20,000 episodes exactly is what removes the budget confound, and that is true whether a run takes 4.6 minutes or 68. Only the affordability claim was false, and the answer to that was parallelism (D-027), not a smaller budget.

**Consequences:** recorded here rather than quietly amended, because this is the same failure as E-015 — a decision that was right for a reason that turned out to be wrong. A number carried from a probe into a design without being re-measured against the real thing is not evidence. Anyone quoting "4.6 minutes per run" from the design spec is quoting a retracted figure.

---

## D-025 — Phase 3 gets its own trainer rather than extending `scripts/train.py`

**Date:** 2026-08-18 · **Model:** Claude Opus 5 · **Phase:** 3 · **Status:** active

**Decision:** `scripts/train_dqn.py` is a separate file. `scripts/train.py` is not modified.

**Why:** The two trainers differ in what they save (a Q-table and visit counts versus network weights), what they plot (the DQN adds a loss curve the tabular learners do not have), and what they switch (two ablation flags). Merging them produces a script whose every second line is an `isinstance` check. The rule of three applies: the shared harness is extracted when a third trainer appears, not in anticipation of one — the same reasoning that produced `agents/tabular.py` only once SARSA and Monte Carlo existed.

There is a second reason, specific to this project: CONSTRAINTS #11 requires that nothing from a later phase be needed to run an earlier one. Leaving `train.py` untouched makes that trivially true rather than something to verify.

**Alternatives rejected:** *Add a `--agent dqn` branch to `train.py`.* Cheaper today, and it puts Phase 3 code on the path of every Phase 2 reproduction — so a Phase 3 bug could break a Phase 2 rerun.

**Consequences:** the honesty machinery — greedy diagnostic on train seeds, eval seeds read once at the end, the `results/smoke/` guard — is duplicated rather than shared. That duplication is deliberate and is the cost being paid for phase isolation. It is also the thing to collapse first when Phase 4 adds a third trainer.

---

## D-026 — "Replay off" means a batch of one on the latest transition

**Date:** 2026-08-18 · **Model:** Claude Opus 5 · **Phase:** 3 · **Status:** active

**Decision:** Under `no_replay`, `update()` learns from exactly the transition just observed, as a batch of one, instead of sampling the buffer. Everything downstream — target computation, loss, clipping, optimiser step — is unchanged.

**Why:** This is what "DQN without experience replay" means: online Q-learning with a function approximator, which is the thing replay was introduced to fix. Keeping it a batch of one rather than a separate code path means the only difference between the conditions is the *data*, not the arithmetic.

**The confound, stated rather than hidden:** the ablated condition also has a batch size of 1 against the control's 64, so it takes noisier gradients for two reasons at once — correlated samples *and* a smaller batch. These cannot be separated without a third condition (batch-64 sampled from only the most recent 64 transitions), which is not run. Any conclusion from this ablation is therefore about "replay as a package", not about decorrelation in isolation.

**Alternatives rejected:** *Keep batch size 64 by repeating the latest transition 64 times.* Equal batch size, but the gradient is identical to a batch of one scaled by 64 — it changes the effective learning rate and nothing else, which would be a worse confound wearing a disguise.

**Consequences:** the confound must appear in any write-up of the ablation. `test_no_replay_ablation_trains_on_a_single_transition` pins the behaviour at a single backup, because an ablation that did nothing and an ablation that was never wired up look identical in a training plot.

---

## D-027 — Training repeats run as parallel single-repeat processes; 30 control runs, 15 per ablation

**Date:** 2026-08-18 · **Model:** Claude Opus 5 · **Phase:** 3 · **Status:** active
**Approved by:** Pranav (2026-08-18), after the compute budget was re-measured.

**Decision:** `train_dqn.py --only-repeat K` runs one repeat and writes its result as JSON; `scripts/run_dqn_sweep.py` keeps a fixed number of those in flight; `scripts/aggregate_dqn.py` combines them. The Phase 3 sweep is **30 control runs and 15 per ablation**, all at the full 20,000 episodes.

**Why parallel:** each training process is single-threaded and uses ~301 MB (both measured, not estimated), so N repeats run as N processes on N cores. Sequentially, 60 runs × 20,000 episodes is ~68 hours; ten at a time it is ~8. That is the only reason the full episode budget survived contact with the real per-episode cost (D-024). `seed_base` depends only on `repeat_index` and `n_episodes`, never on how many repeats are running, so a parallel repeat faces exactly the alert stream it would have faced sequentially.

**Why 30 and not 5:** CONSTRAINTS #3's minimum of five has already proved insufficient in this project. E-014 found every headline comparison in Phases 0–2 rested on a sample far too small for this environment's variance — the constraint was honoured and the measurement was still misleading. The control is compared against tabular Q-learning at 47.6 ± 52.0, a spread wider than any plausible effect, so the standard error is the binding quantity. Thirty runs shrinks it by √6 against five. Doubling the *episodes* instead would not shrink it at all.

**Why the ablations get fewer:** an ablation is compared against the control and should fail obviously. If its effect only emerges at 30 runs, that is a negative result worth reporting, not something to buy precision for. Precision is spent where the phase gate actually turns.

**Consequences:** repeats are addressable by index, so a sweep can be *extended* later — repeats 30–39 are simply more indices — without recomputing anything. The scheduler holds each launch under a ceiling on total memory used (checked before every launch, not once at startup) so an unattended overnight run cannot push the machine into swap; it waits and prints why, then resumes by itself.

---

## D-028 — The DQN-vs-tabular comparison is reported paired per evaluation seed

**Date:** 2026-08-18 · **Model:** Claude Opus 5 · **Phase:** 3 · **Status:** active

**Decision:** `scripts/compare_dqn_tabular.py` reports total reward both unpaired (mean ± std, as Phase 2 reported it) and **paired**: the per-seed difference DQN − tabular across the 30 evaluation shifts, with its standard error and the ratio |mean| / SEM.

**Why:** both agents run on the identical 30 shifts, so the per-seed difference cancels the shift-to-shift variance that dominates the unpaired spreads — severity-sort's ±220.1 against a mean of 40.4. E-013 asked for a lower-variance evaluation protocol and E-014 showed why: the spread of this environment is several times larger than the effects being claimed. Pairing is that protocol, it uses data the runs already produce, and it costs no extra compute.

This is not a new claim about the agents; it is a better estimator of the same quantity. The unpaired numbers are still printed so nothing is hidden by the change.

**Alternatives rejected:** *Report only the unpaired means, as Phases 0–2 did.* Consistent with prior tables, and it is precisely the presentation that produced the E-008 error later retracted in E-014.

**Consequences:** the script prints |mean| / SEM and states plainly that below about 2 the difference is not resolvable at 30 seeds, whichever way it points. The Phase 3 exit criterion — "DQN matches or beats tabular Q-learning" — may well be decided by a spread rather than a mean, and this was recorded before any result existed, in the same spirit as D-012 and D-020.

---

## D-029 — The Huber delta is 200, taken from the reward table rather than from the sweep

**Date:** 2026-08-19 · **Model:** Claude Opus 5 · **Phase:** 3 · **Status:** active
**Approved by:** Pranav (2026-08-19), after the delta sweep was run at his request.

**Decision:** `dqn.huber_delta: 200.0` in `config/training_default.yaml`, passed explicitly to `F.huber_loss`. The loader refuses any value below 50.

**Why a delta exists at all:** it did not, until now. `F.huber_loss(predicted, target)` was called with torch's default of 1.0, and that single omission destroyed the entire first Phase 3 sweep (E-016). Below the delta Huber is quadratic and the gradient scales with the error; above it the gradient is flat. With the delta at 1.0, this environment's -150 and -200 penalties produced the same gradient as a routine +-1 mis-estimate — measured ratio **1.014 for a 150x larger error** — so the agent never learned to avoid them and collapsed to BULK_CLOSE 99.4% of the time.

**Why 200, and why not from the data.** A 5x3 sweep (delta 10/25/50/100/200, 3 seeds, 3000 episodes) established a *threshold*: delta 10 collapsed 3/3 seeds and delta 25 collapsed 1/3, while 50, 100 and 200 collapsed 0/3. It established nothing beyond that. Pairwise |difference|/SEM on the score was 0.05 to 0.40 — the pre-registered rule nominally selected delta=100 on a 3.0-point margin against a standard error of 55, which is precisely the mistake E-008 made and E-014 retracted. The rule lacked a resolvability gate and was not followed; that is recorded in E-016 rather than quietly corrected.

200 comes from `env_default.yaml` instead: it is the **largest named single-event penalty** (`end_of_shift_missed: -200.0`, with `bulk_close_true_incident: -150.0` beneath it). At that delta every individual penalty the agent must learn stays in the quadratic regime where the gradient still carries magnitude, and only the compound multi-miss tail — observed down to -1499.5 when several incidents expire together — is linearised. That is Huber doing the job it is for, and it is a sentence either student can defend without reciting a sweep table.

**Alternatives rejected:**
- *delta = 50 (~1 std of per-step reward, 46.4).* Defensible, and tested no worse. Rejected because it pushes the -150 buried-incident event into the linear regime — the exact signal E-016 proved is load-bearing.
- *Reward clipping to [-1, 1], as in Mnih et al.* This is what makes delta=1 correct in the original paper. Disqualifying here: the *magnitude ordering* of the penalties (-150 for burying one, -500 for missing a critical one) is the triage signal. Clipping deletes what the agent must learn.
- *Plain MSE.* delta=1000 tested indistinguishably from 200, and over the observed range is MSE in all but name. It discards the outlier protection for nothing.

**Consequences:** the divisor is a *domain* fact like the D-023 feature scales and must not be tuned against results — tuning it would make it a hyperparameter fitted to the evaluation, which CONSTRAINTS #2 forbids. The loader's `>= 50` guard carries the measurement in its error text so an unattended overnight sweep cannot silently repeat E-016, and `test_a_buried_incident_moves_the_network_more_than_a_routine_error` fails on the old code.

---

## D-030 — Sweep parallelism and memory limits are set from measurement, and the first numbers were wrong

**Date:** 2026-08-19 · **Model:** Claude Opus 5 · **Phase:** 3 · **Status:** active, **supersedes the parallelism figure in D-027**

**Decision:** `--max-parallel` defaults to 8 and `--process-gb` to 0.95. For a long unattended sweep on this machine, launch with `--max-parallel 5`.

**Why 8 and not 10:** D-027 chose 10 on the assumption that more parallelism means more throughput. Measured, aggregate throughput peaks at 8 and then *falls*:

| parallel | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| runs/hour | 28.8 | 39.0 | **48.6** | 47.3 | 47.6 |

The first sweep therefore ran at a setting slower than the alternative, not faster — consistent with 6 P-cores plus E-cores on the i7-13650HX, where the extra processes land on cores costing more than they add.

**Why `--process-gb` moved from 0.31 to 0.95:** 0.31 GB was the *working set*. Private commit is ~940 MB, which is why the machine sat at 81.5% used with 10 running rather than the ~20% the old figure predicted. The memory guard predicts with this number before every launch, so understating it made the guard launch too eagerly — the same class of error as the free-memory floor that guard replaced.

**The correction that matters most, and it is not a tuning parameter.** Even at 8 the first sweep degraded from 27 min/run to **195 min/run** over five hours, with CPU at 19% and ~96000 page faults/sec against almost no disk reads. The cause was almost certainly **Mem Reduct**, a third-party memory tool set to auto-clean whenever memory exceeded 90%, with "Working set" among the regions it clears. Clearing a working set calls `EmptyWorkingSet` on every process; the trainers' resident pages are evicted and must immediately be faulted back in, which is exactly the soft-fault storm observed. Early runs were fast because memory sat near 69%, below the trigger; runs slowed 7x once the baseline crossed 90%. Recorded as the *likely* cause — the tool's own log was not inspected — because the evidence is circumstantial but fits every measurement.

**Consequences:** benchmark numbers taken on a freshly rebooted, otherwise-idle machine are not predictions about an 8-hour unattended run. The 800-episode benchmark above projected 9.9 min/run; reality was 27 at best and 195 at worst. The benchmark measured the right quantity under the wrong conditions, which is the same failure as D-024's compute-budget probe — a fragment measured accurately and generalised wrongly. Any future capacity claim in this project should say what else was running on the machine.

---

## D-031 — `config.py` is split into a package, and the split is behaviour-neutral by construction

**Date:** 2026-08-23 · **Model:** Claude Opus 5 · **Phase:** 4 · **Status:** active

**Decision:** `src/soc_triage/config.py` (657 lines) becomes the package `src/soc_triage/config/`, in four files: `validation.py` (ConfigError plus the three shared checks), `environment.py` (the env YAML loader), `training.py` (the training YAML loader), and `__init__.py` (re-exports only).

**Why now, and why it is Phase 4 work rather than tidying.** It was the only file in the repo over the 500-line limit (CONSTRAINTS #12) and the violation predated Phase 3. Phase 4 adds two more config sections — `reinforce` and `actor_critic` — plus their seed-block checks, so the choice was to split it or to break the limit further with every Phase 4 commit. Splitting first is cheaper than splitting later with two more phases' worth of loaders inside it.

**How the split was chosen.** Along the seam that already existed: two YAML files, two loaders, sharing nothing but three validation helpers. No function was rewritten, reordered or renamed — the bodies were moved verbatim, which is what makes the next paragraph true.

**Why `__init__.py` re-exports everything.** Twenty-seven call sites across `src/`, `scripts/` and `tests/` import from `soc_triage.config`. Re-exporting means none of them changed, so **the existing 130 tests are a real check on the move**: they exercise the same public names through the same import path, and a passing suite means the split changed nothing observable. Had the imports been rewritten instead, the tests would have been checking the new arrangement rather than checking the move. Verified: 130 passed before, 130 passed after.

**Known cost.** The word "config" now names two things one directory apart — `config/` at the repo root holds YAML, `soc_triage.config` is the code that reads it. That collision is called out at the top of `__init__.py` rather than left for a reader to trip over. Renaming the package would have touched all 27 call sites and destroyed the property in the paragraph above, which was judged the worse trade.

---

## D-032 — The 17 input divisors move from `dqn:` to a shared `features:` block

**Date:** 2026-08-23 · **Model:** Claude Opus 5 · **Phase:** 4 · **Status:** active, **extends D-023** (which chose the divisors and where the file they live in)

**Decision:** `feature_scales` leaves the `dqn:` section of `training_default.yaml` and becomes `features.scales`, loaded into a new `FeaturesConfig` on `TrainingConfig`. `DQNConfig` no longer carries it. The values are unchanged, all 17 of them.

**Why.** Phase 4's REINFORCE and actor-critic read the same 17 columns from `state.featurise` as the DQN does, so they need the same divisors. Three ways to give them those: read `dqn.feature_scales` from a Phase 4 agent (Phase 4 depending on Phase 3's config section, and a lie about what the section means), duplicate the 17 numbers under `reinforce:` (two copies of a domain constant, which is one copy that can drift), or promote. The divisors were never DQN hyperparameters — the shift is 480 minutes long and severity runs 0-3 whatever is consuming them. D-023 put them under `dqn:` because the DQN was the only thing that could use them, not because they belong to it.

**What the duplicate would actually have cost, which is the real argument.** The Phase 4 deliverable is a sample-efficiency comparison of DQN vs REINFORCE vs actor-critic. If those agents can be scaled differently and nothing fails, the comparison silently stops being a comparison of algorithms and becomes partly a comparison of preprocessing — and it would look completely normal in the results table. That is the same shape of defect as E-016 and BUG_003: no error, no crash, just a number that means something other than what its label says. `tests/test_feature_scales.py::test_the_dqn_section_does_not_keep_a_private_copy_of_the_scales` fails if the copy is ever re-added.

**Backwards compatibility deliberately not provided.** The loader raises on a config with no `features:` block rather than falling back to `dqn.feature_scales`. A silent fallback would let a stale config from before this change train an agent on unscaled columns — the exact failure D-023 exists to prevent. Covered by `test_a_missing_features_block_is_refused`.

**Verified:** 132 passed (130 before, minus 5 scale tests moved out of `test_dqn_config.py`, plus 7 in the new `test_feature_scales.py`).


---

## D-033 - The four exit criteria stay exactly as written; phases close built-but-not-passed

**Date:** 2026-08-25 · **Model:** Claude Opus 5 · **Phase:** 4 · **Decided by:** Diya · **Status:** active, **covers D-012, D-020, E-017 and Phase 4**

**Decision.** The exit criteria for Phases 1, 2, 3 and 4 are **not amended**. Where a phase fails
its criterion it closes as *built-but-not-passed*, and the honest failures become the project's
spine in the report. This is option (b) of the three `HANDOVER.md` listed.

**What was owed and why it was blocking.** Three phases had already closed against unmet criteria
with the gate deliberately left for a human (D-022, D-020, E-017). `HANDOVER.md` recorded that as
"THE BIG ONE" - no longer three footnotes but a pattern about how the criteria were written,
needing one decision covering all of them. Phase 4 then made it urgent rather than merely tidy:
E-018 found REINFORCE at 300 episodes reproducing `severity_sort`'s metrics **to every digit
reported**, against a Phase 4 criterion demanding the learners *beat* severity-sort. Matching it
exactly is the awkward boundary case, and taking the decision after seeing a full run's number is
the exact failure this project exists to avoid.

**Why (b) rather than restating the gates on total reward.** Option (a) had a real argument -
total reward is the MDP's actual objective, and D-012 legitimately amended Phase 1's criterion on
precisely that ground. What makes it wrong *now* is timing. D-012 was defensible because Phase 1's
original criterion contained a **category error**: it asked a reward-maximiser to top a metric
(recall) it does not optimise. No such error exists in Phases 2, 3 or 4 - those learners simply
did not do the thing their criteria describe. Rewriting four criteria after four results are known
would also taint D-012 retrospectively, converting a principled amendment into the first move of a
pattern of moving goalposts. D-020 already rejected this reasoning once for Phase 2, including its
most tempting version (restating Phase 2 on reward *consistency*, where the learners genuinely
win); applying the same logic across all four is consistency, not stubbornness.

**What this decision explicitly does NOT do.** It does not declare the phases failed as
engineering. Every work item in Phases 1 and 3 is genuinely complete and independently verified,
and the pattern of four unmet criteria is itself a reportable finding about criterion-writing -
worth more than four passes would have been.

**Consequence for Phase 4.** Its criterion stands: *all three learners train to a policy beating
severity-sort, and the sample-efficiency plot shows a clear ordering you can explain.* When the
full runs happen, whatever they produce is reported against that sentence unchanged. If REINFORCE
matches severity-sort rather than beating it, Phase 4 closes built-but-not-passed like the others,
and "policy gradient rediscovers the human heuristic and stops there" is written up as the finding
it is.

---

## D-034 - The actor-critic uses two separate networks and an entropy bonus that is not in the textbook

**Date:** 2026-08-25 · **Model:** Claude Opus 5 · **Phase:** 4 · **Status:** active

**Decision.** `agents/actor_critic.py` implements S&B 2nd ed. 13.5 (one-step episodic
actor-critic) with three choices worth recording, two of them departures.

**1. Two separate networks, not a shared trunk with two heads.** The shared trunk is the more
common engineering choice and it is the wrong one here. It couples the actor's and the critic's
gradients through weights they both own, and the resulting behaviour cannot be explained in five
minutes (CONSTRAINTS #13). Separate networks also make the comparison against `reinforce.py`'s
policy-plus-baseline pair like-for-like, which matters because those two agents differ in exactly
one respect and the code should not add a second difference.

**2. An entropy bonus, which S&B 13.5 does not have.** `entropy_coef: 0.0` recovers the textbook
update exactly, and `test_zero_entropy_coefficient_is_the_textbook_update` pins that. It is here
because E-018 found REINFORCE's greedy policy degenerate - one action in every state - by 300
episodes, and E-019 then ruled out the gradient clip as the cause across nine runs at three clip
values. A policy-gradient method has no epsilon to hold exploration open; the only thing that can
keep pi spread out is a term that pays for spread. **The departure is declared in the module
docstring under its own heading**, because an addition to a textbook algorithm that goes
unmentioned is how a student ends up defending code they cannot name.

**3. The bootstrap is dropped at the shift boundary.** `done=True` means the target is `r` alone,
not `r + gamma*v(s')`. The shift ends at 480 minutes with the end-of-shift penalty already
charged, so there is no future left to value. Worth knowing for the viva: bootstrapping through a
**time limit** rather than a true terminal is a real bias in many published implementations. Here
the shift end is a genuine episode end rather than a truncation, so treating it as terminal is
correct rather than merely conventional.

**What makes it an actor-critic, stated because it is the likeliest thing to be caught out on.**
Not the network count - `reinforce.py` also has two networks and is not an actor-critic. The
difference is one term: REINFORCE's coefficient is `G_t - b(s_t)`, the **observed** return; this
agent's is `r + gamma*v(s') - v(s)`, the successor's **estimate**. `tests/test_reinforce.py` pins
that REINFORCE does not bootstrap and `tests/test_actor_critic.py` pins that this does; the two
files are deliberate mirror images.

**Verified:** 18 new tests, including a tiny-MDP anchor recovering `HAND_COMPUTED_POLICY` on 3
seeds and an arithmetic anchor - a self-looping state paying 1.0 has `v = 1 + gamma*v = 10` at
gamma 0.9, which a non-bootstrapping critic could never reach from single-step rewards of 1.0.
188 passed overall.

**One thing found by the anchor rather than by reasoning, kept because it is instructive.** The
tiny-MDP fixture does *not* copy REINFORCE's 10x learning-rate scaling. At `critic_lr` 0.05 the
critic collapses to a constant, returning 10.000 for both states to seven digits, and the policy
anchor fails on seed 1. The critic takes one step per **step** where REINFORCE's baseline takes
one per **episode** - 200x more updates on this fixture - so scaling its rate up as well
over-corrects. The actor keeps the 10x; the critic runs at the shipped 0.005.

---

## D-035 - Two hyperparameters found suspect this session were NOT changed, and each got a named experiment instead

**Date:** 2026-08-25 · **Model:** Claude Opus 5 · **Phase:** 4 · **Status:** active. **Both experiments have now run; `entropy_coef` was set to 1.0 from E-020 the same day** - see the outcome note at the end.

**Decision.** `reinforce.grad_clip_norm` stays at 10.0 and `actor_critic.entropy_coef` stays at
0.01, despite both being demonstrated this session to be doing something other than what they
appear to do. Each gets a dedicated train-seed-only experiment with its own seed block rather
than an edit.

**Why not just fix them.** Both are tuning parameters, and CONSTRAINTS #2 puts tuning behind two
gates: never against evaluation seeds, and never on evidence that cannot carry the claim. The
evidence available at the moment of temptation was, in both cases, too thin - a smoke run for the
clip (E-018) and a 20-episode single-seed probe for the entropy coefficient. Editing a config on
that basis produces a number nobody can defend, and it is how a project ends up unable to say why
any of its settings are what they are.

**What the two experiments cost, and why the shape is repeatable.** E-019 (clip) ran in 3.7 min
and returned a clean negative: between-value spread 74.4 against a within-value spread of 211.6,
so the clip value does not clear the noise floor and 10.0 stays by default rather than by
vindication. E-020 (entropy) is built but **not run** - session compute was scoped to build-only -
and it is the one that matters, because the diagnostic behind it is not ambiguous: at 0.01 the
bonus contributes at most 0.016 against TD errors reaching 1410, and the policy saturates within
five episodes with the actor's gradient norm falling to 0.00.

**The seed-block convention this establishes.** A tuning experiment gets its **own** block, never
the block whose numbers get reported: `reinforce.clip_experiment_seed_start` (1800000) and
`actor_critic.entropy_experiment_seed_start` (2200000), both enforced by the loader against every
other block. Otherwise a value chosen on the control's own alert streams launders tuning into the
result it is later quoted beside. That is D-016's principle extended from learners to experiments.

**The generalisation, which is the reportable part.** E-016 (`huber_delta` 1.0 against penalties
of -150 to -200), E-019 (`grad_clip_norm` 10.0 against norms of 1584-2228) and E-020
(`entropy_coef` 0.01 against TD errors of 1410) are three instances of one defect: **a
hyperparameter whose scale was never checked against the environment's reward scale is not a
tuning choice, it is an untested assumption.** Nothing errors in any of the three, the loss curve
looks healthy in all of them, and two were caught only because a number in a results table looked
wrong. Phase 6's audit should include a scale check rather than relying on that.

**OUTCOME (same day).** E-020 ran: 4 values x 3 repeats x 80 episodes, 6.7 min. It bounded the
coefficient from both sides - collapse (final entropy 0.0000) at 0.01 **and** at 0.1, and 97.6%
of uniform entropy at 10.0 - and `entropy_coef` was set to **1.0** on that structural criterion.
191 tests pass with the new value, tiny-MDP anchor included.

**This vindicates the decision to wait, and it is worth saying why.** The 20-episode eyeball that
prompted the experiment showed 1.0 looking healthy, and would have produced the same number. What
it could not have produced is the *reason*: that 0.1 collapses too, so the answer was never a
small bump; that 10.0 is too large in a specific measurable way; and that the greedy metric is
saturated rather than noisy. A value set from the eyeball would have been right by luck and
indefensible in a viva. The experiment cost 6.7 minutes.

**One thing the sweep found that nobody was looking for.** The greedy reward column is identical
across all twelve runs - exactly -515.4, std 0.0 - because any policy whose argmax is BULK_CLOSE
everywhere scores exactly that on fixed seeds. So the entropy bonus fixes the *sampled* policy and
leaves the *argmax* degenerate at every coefficient tested. Together with E-019 section 3 that
makes the greedy-vs-sampled evaluation decision the most consequential item still owed.

---

## D-036 - Phase 4 reports each algorithm as the policy it optimised: sampled for policy gradient, greedy for value-based

**Date:** 2026-09-01 · **Model:** Claude Opus 5 · **Phase:** 4 · **Decided by:** Pranav · **Status:** active. **Taken before any full Phase 4 run existed**, which is the point.

**Decision.** Phase 4's headline number for `reinforce` and `actor_critic` is the reward of the
**sampled** policy - the stochastic policy the agent actually plays. The greedy read
(`argmax_a pi(a|s)`, the existing `_GreedyView`) is **kept and reported alongside as a named
diagnostic**, never as the headline. Phases 0-3 are untouched: value-based agents continue to be
reported greedy.

The rule, stated so it is not a per-phase preference: **each algorithm is reported as the policy
its own objective optimised.**

**Why the two cases are not analogous, which is the whole argument.** Q-learning's target is
`q_*`, so the greedy policy read off `Q` is exactly the object the algorithm was learning;
epsilon is a training scaffold bolted on for exploration and removing it at evaluation recovers
the intended policy. `train.py:224` and `train_dqn.py:357` both do `agent.epsilon = 0.0` before
touching the eval seeds, and every number in E-014 and E-017 is therefore an argmax number.
Policy gradient maximises `J(theta) = E[return | pi_theta]` - the expected return **of the
stochastic policy**. Its argmax appears nowhere in that objective. Evaluating a policy-gradient
agent through its argmax measures a policy that was never optimised, and the current protocol -
`train_reinforce.py:271` and `train_actor_critic.py:272`, both routing final eval through
`_GreedyView` - treats the two cases as if they were the same.

**The evidence that this is not a theoretical distinction here.** E-019 section 3: in **nine runs
out of nine**, the sampled policy collected positive reward (+6.0 to +111.7) while the greedy read
of the *same policy at the same moment* scored strongly negative (-12.2 to -515.4). Seven of the
nine greedy reads were exactly **-515.4** or **-78.7** - constant-action policies, and -515.4 is
the same value Phase 3's collapsed DQN produced (E-016, the BULK_CLOSE signature). E-020 then
found the same degeneracy from the other direction: the greedy column was identical across all
twelve runs, **-515.4 with std 0.0**, at every entropy coefficient tested - including coefficient
1.0, where final entropy was 0.8279 and the sampled policy was healthy. The entropy bonus fixes
the policy and does not fix its argmax.

The mechanism is not a paradox and belongs in the viva answer: if `pi` puts 0.4 on
PULL_HIGHEST_SEVERITY and 0.35 on BULK_CLOSE, its argmax is a *pure* severity policy, and the
agent that earned +111.7 was playing neither pure strategy. **A stochastic policy's argmax is not
that policy**, and this environment appears to reward the mixture over either pure strategy.

**The decisive reason, and it is a measurement argument rather than a preference.** A greedy
column that reads exactly -515.4 with std 0.0 across twelve runs spanning a 1000x range of
entropy coefficients - including settings where the agent demonstrably learned and settings where
it demonstrably collapsed - **has no ability to discriminate**. E-020 already recorded that its
reward verdict "cannot distinguish 'the values are indistinguishable' from 'the metric is
saturated'". Grading Phase 4 on that metric would produce a fourth built-but-not-passed phase as
an artefact of the instrument rather than as a result about the agent, and CONSTRAINTS #5's logic
runs in this direction too: a catastrophic number from a metric already known to be measuring the
wrong object is a bug report, not a finding.

**What was given up, stated plainly because it is a real cost.** Phase 4's headline is **not
directly comparable to Phase 3's DQN number**, because the two phases report different objects.
The exit criterion names `severity_sort`'s 40.4 (30-seed mean, E-014), which is a deterministic
policy's score. Any write-up comparing REINFORCE's sampled reward against 40.4 or against the
DQN's 0.48 recall must say which convention each side uses. The alternative - keeping everything
greedy for comparability - buys that comparability by reporting a number the evidence above says
is uninformative, which is the worse trade.

**One asymmetry this decision does NOT resolve, flagged for box 3.** "Sampled" does not mean the
same thing for the DQN. Its training-time reward is epsilon-greedy behaviour with epsilon floored
at **0.05** (`config/training_default.yaml`: start 1.0, min 0.05, decay 0.9995) - 5% deliberate
random actions, a training artefact nobody would deploy. For REINFORCE and the actor-critic the
sampling **is** the policy. `scripts/compare_sample_efficiency.py` plots a sampled and a greedy
curve for all three agents; it must **label the convention per agent** rather than implying one
curve type means one thing. That is a presentation requirement on box 3, not a further decision.

**What this obliges before any run starts.** There is currently no sampled-eval path: both
trainers evaluate the final policy through `_GreedyView`. Adding one is a code change to
`train_reinforce.py` and `train_actor_critic.py`, and it must land **before** the full runs, not
after. Both agents already hold `self._rng = np.random.default_rng(seed)` and sample through it,
so a sampled number is reproducible - but the eval RNG will be **explicitly reseeded** rather than
inherited from the end of training, so the reported number does not depend on how many episodes
preceded it.

**Precedent followed.** D-033 settled that criteria are decided with the criterion in hand and
never after the result is known. E-019's own next-session list put this decision at item 2, ahead
of "only then consider full runs". Both are honoured: this was taken while no full Phase 4 run
existed, so it cannot have been chosen to favour a number.

---

## D-037 - The RLHF data layer reads EpisodeRecord files, never live agents

**Date:** 2026-09-04 (session 12) · **Taken by:** Pranav · **Model:** Claude Opus 5
**Applies to:** `src/soc_triage/rlhf/`, FEATURE_011

`rlhf/pairs.py` imports no agent, no environment and no torch. It reads
EpisodeRecord JSON off disk and pairs the records. The alternative - having the
pair builder instantiate policies and run them - was rejected for three reasons,
in order of weight.

**1. It keeps the dependency arrow pointing the right way.** CONSTRAINTS #11 says
nothing from a later phase may be required to run an earlier one. The inverse
discipline is worth just as much: the RLHF layer must not drag Phase 3/4
machinery behind it. As built, the whole of `rlhf/` runs on a clone with no
`torch` and no `results/`, which is also why its 71 tests execute in under a
second while the Phase 4 suites take minutes.

**2. Seven of the nine policies need trained parameters, and those are
gitignored.** `dp`, `q_learning`, `sarsa` and `monte_carlo` need `*_Q.npy`;
`dqn`, `reinforce` and `actor_critic` need `*.pt`. Coupling pair construction to
agent loading would make the pair set unbuildable on a fresh clone. Coupling it
to records instead means the thing to regenerate is a set of records, and
`runner.save_records` already writes them.

**3. The contract was designed in Phase 0 and this is the first thing to use it.**
`runner.py`'s docstring already promised "the interchange format that evaluation,
**the RLHF pair builder**, and the dashboard all consume". Honouring that rather
than inventing a second path is the cheaper decision and the more honest one.

**The cost, stated.** There is now a seam: `scripts/generate_pairs.py` must run
the policies and write records before `pairs.py` can do anything. That script is
the only piece of Phase 5a that a missing artefact can block, and it is
deliberately the last piece in the build order.

---

## D-038 - Labelled episodes get their own seed block, and the pair set is blinded in two files

**Date:** 2026-09-04 (session 12) · **Taken by:** Pranav · **Model:** Claude Opus 5
**Applies to:** `config/training_default.yaml` (`rlhf:`), `rlhf/pairs.py`

Three sub-decisions, taken together because they all concern what a labeller is
allowed to see.

**Seeds: `rlhf.pair_seed_start: 3000000`, a block of its own (D-016 convention).**
The labelled episodes may not come from the eval block `[101..130]`, and the
reason is CONSTRAINTS #2 applied one level up. The reward model is *fitted* to
human judgements of these episodes; Phase 5c re-trains policies on that reward
model; Phases 5 and 6 then evaluate those policies on the eval seeds. Labelling
eval-seed episodes would put human judgement of eval-seed outcomes inside the
reward the policy maximises - evaluation information entering training, laundered
through a person, with no test anywhere that could catch it. The train block
`[1..10]` is permitted but rejected on a weaker ground: ten alert streams across
300 pairs means a labeller sees the same shift about thirty times, and boredom
and memory become confounds. Twelve dedicated seeds fix that.

**Pool: nine policies, `oracle_greedy` excluded.** The oracle reads
`is_true_incident` by design - the sanctioned CONSTRAINTS #1 exception - so it
would win nearly every pair it appeared in. A pair whose answer is a foregone
conclusion costs a labeller twenty seconds and teaches the Bradley-Terry model
almost nothing, because the logistic loss has vanishing gradient exactly where
the prediction is already confident and correct. Wide rather than narrow is
deliberate: brief section 7 asks for state-visitation overlap between the RLHF
policy and the policies that generated the labels, and that question is only
answerable if the pair policies covered a wide behavioural range.

**Blinding: two files, not one.** `pairs.json` is what the labelling UI reads and
contains no policy name anywhere - which required stripping `run_id`, since it
reads `sarsa-seed3000004` and names the policy in passing. `pairs_key.json` holds
the names and the side assignment and is read only by analysis code. Two distinct
biases are being defended against: **name bias**, where a labeller who can see
"random" against "dqn" stops judging outcomes, and **position bias**, where people
pick the left option more often than chance. The second is handled by a seeded
side-swap recorded as `swapped`, so the analysis can undo it exactly or measure
the bias rather than assuming it cancelled. The test checks the *written file* by
substring search rather than inspecting the object, because that is the only form
that catches a name arriving through a field nobody thought about.

**What was given up.** The blinding costs a second artefact that must travel with
the first, and a rule that the UI reads only one of them. That rule is not
enforced by anything except the fact that `pairs.json` genuinely does not contain
the names - which is the strongest form available without a service boundary, and
stronger than a promise in a README.

---

## D-039 - The labeller never sees the hand-written reward

**Date:** 2026-09-04 (session 12) · **Taken by:** Pranav · **Model:** Claude Opus 5
**Applies to:** `rlhf/summary.py`

`summarise_episode` drops every reward field: the per-step `reward`, the
`reward_breakdown`, and `outcome["total_reward"]`. Two tests fail if any of them
reappears - one walking the nested structure by key name and by value, one over
the text rendering, because those are two separate surfaces a number could
escape through.

The reasoning is the premise of the whole phase. PROJECT_BRIEF section 3.5 says
the reward numbers are invented, and section 6.1 says that is precisely why we ask
humans for comparisons instead. A labeller who can see the hand reward is partly
reading it back, and a Bradley-Terry model fitted to preferences contaminated
that way would be an expensive re-derivation of `config/env_default.yaml` -
arriving at a learned reward that agrees with the hand reward, and looking like a
successful result while establishing nothing.

**Ground truth is shown, and that is not the same thing.** PROJECT_BRIEF section
6.2: "Ground truth *is* shown to the labeller - they are judging outcomes, not
guessing." So the summary shows which investigations turned out to be real
incidents, how long each took to catch, how many were missed and how many were on
crown-jewel assets. CONSTRAINTS #1 forbids ground truth in an **agent
observation**; a summary is environment-side output for a person, the same
category as `env.episode_outcome()` itself.

**Known limitation, recorded rather than hidden.** Missed incidents are reported
as counts, not per-incident cards, because the EpisodeRecord does not carry the
alerts left unhandled in the queue at the end of a shift - only investigated and
bulk-closed alerts appear in `steps[]`. Fixing that means changing `runner.py`, a
Phase 0 module that wrote all 300 existing record files. The trade is not worth
it: the outcome block already gives the count, the criticality breakdown and the
buried count. If per-incident cards later prove necessary the upgrade is an
additive key and old files stay readable.

---

## D-040 - The single-label pairs are split 125/125 by a fixed rule, not by whoever sits down first

**Date:** 2026-09-05 (session 13) · **Taken by:** Diya · **Model:** Claude Opus 5
**Applies to:** `labelling/queue.py`, `config/training_default.yaml` (`rlhf.labellers`)

300 pairs, of which 50 are double-labelled so that Cohen's kappa exists at all.
That leaves 250 needing one opinion each, and somebody has to decide whose.

**Chosen: a deterministic assignment computed from `pairs.json` before anyone
starts.** The double-labelled pairs go to every labeller; the singles are dealt
round-robin in `pair_id` order. Two labellers therefore get 125 singles apiece and
175 judgements each, 350 in total.

**Rejected: a shared queue serving whichever pair nobody has answered yet.** It is
simpler, it needs no assignment rule at all, and it self-balances if both people
work at the same pace. It loses on three counts. The split becomes an artefact of
who had a free evening rather than a decision anybody made; it cannot be
reconstructed afterwards except by reading timestamps out of the database; and two
people labelling simultaneously race each other for the same pair. A project whose
central claim is that its numbers are reproducible should not have to explain in a
viva that the division of its only human-generated data was decided by scheduling.

**Why sorting by `pair_id` costs no randomness.** The worry with dealing alternate
pairs is that one labeller could systematically receive one kind of comparison.
That cannot happen here, because `pairs.build_pairs` already shuffles the draft
before numbering it — deliberately, so that a labeller working through the file
does not meet every alpha-vs-beta comparison in one block. File order is therefore
already random with respect to which policies are being compared, and the queue
gets a balanced spread without ever inspecting what a pair contains. Which it must
not: the pairs are blinded (D-038).

**What was given up.** If one labeller stops at 90, the other cannot silently
cover the gap — the assignment is a recorded fact, not a suggestion. That is the
intended trade, but it means an unfinished half needs a deliberate decision rather
than momentum. Validation refuses a labeller list that does not divide the singles
evenly, so the fairness assumption fails loudly at load.

---

## D-041 - The labeller id is bound at launch and the request cannot override it

**Date:** 2026-09-05 (session 13) · **Taken by:** Diya · **Model:** Claude Opus 5
**Applies to:** `scripts/label_ui.py`, `labelling/app.py`

`store.add_label` needs a `labeller_id`, and CONSTRAINTS #23 requires it to be
opaque — the schema has nowhere to record a name, an employer or a client.

**Chosen:** `python scripts/label_ui.py --labeller L1`. The page never asks. The
id is a closure variable inside the app, the `Answer` model has no field for it,
and unknown keys in a request body are dropped, so a POST cannot claim to be
somebody else. The launcher validates the id against `rlhf.labellers` before
opening anything.

**Rejected: a text box on the page.** Two failure modes, both landing on the one
artefact in the project that cannot be regenerated. The visible one is that a text
box invites a real name or an email address, and once that is in the SQLite the
only remedies are editing the irreplaceable file or discarding labels. The
dangerous one is silent: a stale value left over from the other person's turn
records this person's judgements under that person's id. That does not merely
mislabel data — it corrupts **kappa specifically**, because kappa's premise is that
the two sets of judgements came from two independent people. A run where forty of
L1's answers are filed as L2's would still produce a coefficient, and it would
look like a result.

**Alternative also rejected: infer the id from the OS user.** It would remove the
flag, but the two students alternate machines continuously (CONSTRAINTS #24-26),
so the OS user is a fact about the laptop and not about the labeller.

---

## D-042 - The answer timer is measured in the browser, capped, and stored as unknown past the cap

**Date:** 2026-09-05 (session 13) · **Taken by:** Diya · **Model:** Claude Opus 5
**Applies to:** `labelling/render.py`, `labelling/app.py`, `config/training_default.yaml`

`store.add_label` takes `seconds_taken: float | None`, and Pranav's docstring
already gives half the reasoning: a defaulted `0.0` "would read as an instant
decision and would silently skew any later analysis of how long people spent
thinking."

**Chosen:** the page records when the pair was displayed and when the answer was
clicked, using `performance.now()` so a clock correction mid-sitting cannot move
it, and sends the elapsed seconds with the answer. Above
`rlhf.max_seconds_per_pair` (300) the stored value is `None`. The server applies
the same cap again, and also stores `None` for a negative elapsed time.

**Why the cap.** Open a pair, walk away, come back twenty minutes later and click.
Without a cap the row asserts 1200 seconds of deliberation. That is not a large
number, it is a **false** one, and it would drag any mean computed from the column
while looking entirely plausible. `None` asserts "we do not know how long this
took", which is true. This is Pranav's `None`-over-`0.0` rule applied at the other
end of the range, and the fact that it is the same rule twice is the argument for
it rather than a coincidence.

**Why the server re-checks what the browser already checked.** Not defence in
depth for its own sake: a page can be edited or its POST replayed by hand, and
this is the only column in the project whose corruption would be invisible.

**Rejected: server-side timing** (serve to POST). It measures the same quantity
slightly worse, adds network latency to the figure, and has the identical
coffee-break flaw — so it buys nothing for the JavaScript it saves.

**What is still not measured.** Time spent re-reading a pair after scrolling away
and back is counted; time spent thinking about a pair before it is displayed
cannot be. The column is evidence about attention, not a stopwatch on cognition,
and the Phase 6 audit should read it that way.

---

## D-043 - `httpx2` added as the first test-only dependency

**Date:** 2026-09-05 (session 13) · **Taken by:** Diya · **Model:** Claude Opus 5
**Applies to:** `requirements.txt`, `tests/test_labelling_app.py`

FastAPI's `TestClient` refuses to import without `httpx2`, so without it the
labelling routes could not be tested at all — and those routes are what write the
irreplaceable labels. Approved under CONSTRAINTS #8 and pinned at `2.12.0`. It is
the first entry in `requirements.txt` that exists purely for tests, and installing
it pulls `httpcore2 2.12.0` and `truststore 0.10.4` with it, which is worth
knowing before the other machine runs `pip install`.

**Rejected: no new dependency — start `uvicorn` in a subprocess and drive it with
`urllib.request` from the standard library.** Genuinely attractive, because it
would test the actual server the humans run rather than an in-process substitute,
catching launcher bugs `TestClient` cannot see. It lost on cost and reliability: a
second or two of startup per test file on a suite that already takes eight minutes
on Diya's machine, plus port-binding flakiness that produces failures unrelated to
the code under test. A test that fails for reasons of its own teaches nothing and
eventually gets deleted for crying wolf.

**What this leaves uncovered, stated rather than papered over.** Nothing in the
suite exercises `scripts/label_ui.py` end to end. It was verified by running it
instead — output in FEATURE_012 §13 — and it is deliberately thin so that
everything worth testing sits behind `create_app`. The rejected approach is still
the right one if the launcher ever grows logic of its own.

---

## D-044 - `_clean_seconds` rejects NaN explicitly rather than relying on SQLite to absorb it

**Date:** 2026-09-05 (session 13) · **Taken by:** Diya · **Model:** Claude Sonnet 5
**Applies to:** `labelling/app.py`

Found by re-deriving the D-042 cap's failure modes rather than trusting an
automated review notification whose body arrived corrupted — only its one-line
summary named `training_validation.py` and "parser-differential" was legible, so
the claim was verified from scratch instead of acted on blind (R5).

**The bug.** `nan < 0` and `nan > max_seconds` are both `False` in IEEE 754, so the
original range check let a `NaN` seconds value straight through untouched.
Python's `json` module accepts the bare `NaN` token by default, which strict JSON
(RFC 8259) forbids — a real parser-differential: what the wire format permits and
what the validation logic assumes are two different things. Confirmed reachable
end to end through the real FastAPI route with a hand-built request, not just at
the language level.

**Why it looked fine and was not.** `_clean_seconds(nan, 300)` returned `nan`
itself, and that `nan` happened to come back as `None` after a round trip through
SQLite — its C driver silently converts NaN to NULL on write to a REAL column.
Undocumented, engine-specific behaviour, not a guarantee `sqlite3` or any
successor makes. The function's own contract — reject the impossible, return
`None` — was being met by luck rather than by the code, and a test that only
checked the round trip through the database would never have caught the
difference. The fix adds `if value != value: return None` (NaN is the one float
unequal to itself) ahead of the range check, and a test calls `_clean_seconds`
directly so the assertion is about the function's contract, not about what a
particular database happens to do with a bad input.

**Rejected: catch this in the `Answer` pydantic model instead** (`allow_inf_nan`).
Would stop NaN earlier, but the model also has to admit ordinary out-of-range and
negative values so `_clean_seconds` can turn them into the same `None` — moving
one case to the boundary and leaving the rest where they are creates two places
that decide what counts as "unknown" instead of one.
