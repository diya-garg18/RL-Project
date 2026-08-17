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

**Date:** 2026-08-16 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active
**Approved by:** Pranav (2026-08-16), before implementation.

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

**Date:** 2026-08-16 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active

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

**Date:** 2026-08-16 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active
**Approved by:** Pranav (2026-08-16), before the first training run.

**Decision:** Training episodes draw seeds from a dedicated block, `q_learning.train_seed_start: 200000`, giving one distinct shift per episode (20,000 per run, offset per repeat). The existing `seeds.train` list (1–10) is **retained but repurposed** as a *training diagnostic* set — the seeds the learning curve is measured on, never trained on. Evaluation seeds (101–105) remain untouched and are read exactly once, at the end of a run.

**Why:** `config/training_default.yaml` specifies `n_episodes: 20000` against a `seeds.train` list of ten. Taken literally, a run replays the same ten alert streams 2,000 times each. A tabular agent with 2,880 state-action cells and 20,000 episodes of practice on ten fixed shifts is being invited to memorise those shifts rather than learn a triage policy, and the resulting eval-seed gap would be blamed on the algorithm rather than on the training distribution.

The fix follows an existing precedent rather than inventing a convention: `dp.estimation_seed_start: 10000` exists for exactly this reason — Phase 1 needed 50,000 episodes and ten seeds could not supply them (D-004). Phase 2 needs 100,000 across five repeats. Blocks are disjoint by construction — train-diag 1–10, eval 101–105, calibration 1000–3099, DP estimation 10000–59999, Q-learning training 200000+ — and `load_training_config` now rejects a start below 100000 rather than trusting the comment, per CONSTRAINTS #2's requirement that seed separation be enforced in code.

Keeping seeds 1–10 as the diagnostic set is the second half of the decision and matters just as much. The learning curve has to be plotted against *something*, and plotting it against the evaluation seeds would be tuning against them by eye — a violation of CONSTRAINTS #2 that leaves no trace in the code.

**Alternatives rejected:**
- *Cycle the ten train seeds 2,000 times.* The literal reading of the config, and the cheapest. Rejected: it makes memorisation the most likely explanation for any train/eval gap, which would contaminate the interpretation of every Phase 2 result.
- *Cycle the ten, but measure and report the memorisation gap.* Genuinely tempting — it turns a weakness into evidence. Rejected because it spends the phase's headline result on a self-inflicted artefact rather than on the algorithm, and because the gap it would measure is confounded by the seed-difficulty effect discovered in E-008.
- *Draw training seeds at random each episode.* Loses reproducibility: two runs would face different shifts and could not be compared.

**Consequences:** Q-learning's training distribution is now far wider than DP's estimation distribution, so the two are not learning from equivalent experience — worth stating whenever E-004 and E-008 are compared. Any future learner (SARSA, Monte Carlo, DQN, REINFORCE) should use this same block, offset per algorithm, or its results will not be comparable to Q-learning's.

D-016 note added 2026-08-16: SARSA (400000+) and Monte Carlo (600000+) received their own blocks under the same rule, and `load_training_config` now rejects duplicate or too-close blocks rather than trusting the YAML comments.

E-008 then found something this decision did not anticipate and does not fix: **the eval block itself is unrepresentative.** Every agent, oracle included, scores 120–230 reward higher on seeds 101–105 than on seeds 1–10, with per-seed standard deviations several times larger than the differences being reported. That is a separate and more serious problem, it affects every experiment in the project rather than just Phase 2, and it is recorded in E-008 as an open decision for the humans rather than acted on here.

---

## D-017 — On-policy learners are graded against an ε-soft target, not against q\*

**Date:** 2026-08-16 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active

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

**Date:** 2026-08-16 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active

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

**Date:** 2026-08-16 · **Model:** Claude Opus 5 · **Phase:** 2 (consequences for 0 and 1) · **Status:** active
**Approved by:** Pranav (2026-08-16), on the recommendation recorded in E-008. **Diya countersign: pending** — this is a larger change than D-012 and should carry her sign-off before the report cites any number affected by it.

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

**Date:** 2026-08-16 · **Model:** Claude Opus 5 · **Phase:** 2 · **Status:** active
**Approved by:** Pranav (2026-08-16). **Diya countersign: pending**, together with D-019.

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
