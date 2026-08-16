# TEST_CHECKLIST.md — What "done" means

> Field Guide habit #8. Not a vibe check — actual commands, actual expected outputs. AI claiming success and code actually working are two different facts.

Run the relevant block **and paste the real output** before marking any roadmap task complete.

---

## Every session, before commit

```bash
pytest -q                          # all tests pass
python -c "import soc_triage"      # package imports cleanly
git status                         # nothing from results/ or *.pt staged
```

Expected: all tests pass, clean import, no forbidden files staged.

---

## Phase 0 — Foundation

```bash
python scripts/calibrate_generator.py     # (100 shifts is built in; edit N_SHIFTS in the script to vary)
```
**Must show:** ~170 alerts/shift · true-incident rate 2.5–3.5% · **severity↔truth Pearson r between 0.30 and 0.40**.
If r is outside that band, stop and retune. This gates the whole project (`DECISIONS.md` D-003).
*Verified PASS 2026-08-13 (E-001): 168.7 / 3.34% / r=0.323.*

```bash
python scripts/run_baselines.py           # eval seeds come from config seeds.eval
```
**Must show:** the 5 Phase-0 baselines × all metrics with mean ± std (the DP row joins in Phase 1), where **`oracle` is strictly best on mean total reward**, and random and FIFO sit clearly at the bottom on recall. The script asserts both gates and prints PASS/FAIL for each. *Verified PASS 2026-08-14 (E-003); re-verified on the new device 2026-08-16 — oracle 214.1, severity-sort 153.7, cheapest-first −467.6, random −229.9, FIFO −658.7.*

**Both amendments to this criterion are approved and folded in above** (Diya, 2026-08-14). For the record, since the original wording is still quoted in older documents: (1) "random is worst" was wrong — FIFO is far worse (0.141 recall at 30 seeds), a textbook overloaded-queue result, E-002 obs. 1; (2) "oracle strictly best on **recall**" was unachievable — no honest greedy oracle can out-recall severity-camping when 64% of incidents carry severity 3 by construction (D-007), so the gate moved to total reward, where the oracle's information advantage is decisive (E-003). Both are findings about the design, documented, not bugs.

```bash
pytest tests/ -v
```
Required tests, by actual name (the 7 Phase-0 tests, all passing; suite total is now 14 — the other 7 are Phase 1's `test_mrp_bellman.py`):
- `test_env_deterministic_under_fixed_seed` — same seed ⟹ identical trajectory and outcome
- `test_different_seeds_differ`
- `test_reward_breakdown_sums_to_step_reward` — per-step breakdown sums exactly to the step reward
- `test_no_ground_truth_leakage` — observations unchanged when hidden truth is flipped **(never weaken or skip this one)**
- `test_snapshot_carries_no_precomputed_truth_fields` — snapshot field whitelist
- `test_bulk_close_never_exceeds_cap`
- `test_clock_terminates_at_shift_end`

---

## Phase 1 — Dynamic Programming

```bash
python scripts/run_dp.py
```
**Must show:** value iteration converged with Δ < 1e-4; the sweep count; state coverage (how many of 576 states were visited, and the minimum visit count); and **value iteration and policy iteration agreeing on ≥95% of states**. Disagreement means one is wrong — investigate, don't average.
*Verified PASS 2026-08-14 (E-004); re-verified 2026-08-16: Δ 9.95e-05 in 1075 sweeps, VI/PI agreement 100%, coverage 133/576 states and 589/2880 pairs.*

**Also must show — and must be read, not skipped:** the DP row's **total reward beside its recall**. Currently reward 305.9 ± 127.6 (highest of any agent) with recall 0.43 ± 0.17 (below random's 0.52). That pair is the E-004 reward-hacking finding. Quoting the reward alone is misleading and D-012 forbids it.

```bash
pytest tests/test_mrp_bellman.py -v
python scripts/run_mrp_example.py
```
**Must show:** all four routes to the five-state MRP's value function agreeing — by hand, closed form, iterative evaluation, and `agents/dp.value_iteration` — with the largest disagreement below 1e-9. *Verified PASS 2026-08-16 (E-005): 7/7 tests, largest disagreement 7.11e-15.*

This is the only check in the project that validates the Bellman backup against an answer derived **outside** the code. VI/PI agreement above cannot do it — they share the same equation. **If this fails, fix `agents/dp.py`; never adjust the expected values.** They were derived by a human on paper (`docs/features/FEATURE_001_mrp_worked_example.md`) and changing them to make the test pass would destroy the only external anchor Phase 1 has.

---

## Phase 2 — Tabular

```bash
pytest tests/test_tiny_mdp.py -v
```
**Must show:** 13 passing tests establishing the hand-derived `q_*` for the 2-state MDP, including `agents/dp.value_iteration` reproducing it. *Verified PASS 2026-08-16 (E-006): 13/13, Bellman optimality residual 1.78e-15 against a 1e-12 tolerance.*

This is the Phase 2 counterpart of `test_mrp_bellman.py` and carries the same rule: **if it fails, fix the learner — never adjust the expected values.** They were derived by a human on paper (`docs/features/FEATURE_002_tiny_mdp_qstar.md`). The anchor is verified load-bearing: injecting a 0.1 error into any entry of `HAND_COMPUTED_Q` moves the residual to ~0.10, thirteen orders of magnitude above tolerance.

```bash
pytest tests/test_tabular.py -v
```
**Q-learning: verified PASS 2026-08-16 (E-007) — 20 tests.** `max |Q − q*| = 9.237e-14`, correct policy after 10 episodes, identical across 5 seeds. Groups:

- `test_q_learning_converges_to_the_hand_derived_q_star` — reproduces `tiny_mdp.HAND_COMPUTED_Q` to < 1e-9
- `test_q_learning_recovers_the_optimal_policy` — the behaviour, checked separately from the values
- `test_update_bootstraps_off_the_max_not_the_behaviour_action` — **the one that matters most.** Q-learning and SARSA converge to the same answer on this fixture, so `sarsa.py` pasted into `q_learning.py` would pass every convergence test. This pins the difference at a single backup: target 10.0, not 0.95.
- `test_terminal_update_does_not_bootstrap` · `test_single_update_matches_hand_arithmetic` · `test_alpha_zero_learns_nothing`
- `test_epsilon_decays_geometrically_and_stops_at_the_floor` · `test_epsilon_does_not_decay_on_update` (D-015)
- `test_ties_break_toward_the_lower_action_index` — not an edge case; a zero-init table ties on step one of every run
- `test_different_seeds_explore_differently_but_reach_the_same_fixed_point` — characterisation, guards a real investigation (E-007)
- `test_q_table_shape` — (576, 5) · `test_q_table_starts_at_zero` — zero init, not optimistic
- Config: alpha and epsilon range checks reject bad YAML at load time

Still to write, same fixture: `test_sarsa_solves_toy_mdp` (on-policy) and `test_monte_carlo_solves_toy_mdp` (truncated at `tiny_mdp.HORIZON`).

> Watch for one specific false pass here: under the tiny MDP's optimal policy the agent never leaves `QUIET`, so `BUSY` is reached only by exploring. A learner run with ε = 0 will never see half the MDP. If a learner passes on `Q(QUIET, ·)` and produces garbage on `Q(BUSY, ·)`, the exploration schedule is the suspect, not the update rule.

```bash
python scripts/train.py --agent q_learning --seeds 5 --eval
```
**Must show:** learning curve rising then flattening; final policy beating severity-sort on recall@deadline and MTTD, reported as mean ± std over 5 **evaluation** seeds (disjoint from training seeds).

---

## Phase 3 — DQN

```bash
python scripts/train.py --agent dqn --seeds 5 --eval
python scripts/ablate_dqn.py          # replay off, target net off
```
**Must show:** DQN ≥ tabular on the same eval seeds; both ablations visibly less stable in the plots. If turning off replay *doesn't* hurt, replay probably isn't wired in — check before reporting.

---

## Phase 4 — Policy gradient

```bash
python scripts/train.py --agent reinforce --seeds 5 --eval
python scripts/train.py --agent actor_critic --seeds 5 --eval
```
**Must show:** both beat severity-sort; the sample-efficiency plot has a clear, explainable ordering; REINFORCE's higher variance is visible.

---

## Phase 5 — RLHF

```bash
python scripts/build_pairs.py
```
**Must show:** every pair matches two episodes on the **same seed** (identical alert stream). A pair on mismatched streams is invalid data — assert it.

```bash
python scripts/train_reward_model.py
```
**Must show:** train accuracy **and** held-out accuracy, separately. Held-out ≥ 65% is the bar for the model being informative at all. A large train/held-out gap is a reward-hacking audit finding — record it, don't hide it.

```bash
python scripts/kappa.py
```
**Must show:** Cohen's κ over the 50 double-labelled pairs, with n. Report whatever it is. Low agreement is a legitimate finding about how ill-defined "good triage" is.

---

## Phase 6 — Final

```bash
python scripts/reproduce_all.py
```
**Must:** regenerate every number and figure in the report from scratch, with no manual steps. If it needs a human to intervene, it isn't reproducible yet.

```bash
python scripts/audit.py
```
**Must show:** all four reward-hacking experiments with their results — including any that found nothing. "We looked for X using method Y and did not find it" is a valid result and belongs in the report.

---

## The human check (Field Guide habit #15)

Not automatable, and the most important one:

- [ ] Pranav can write, from memory: the Q-learning update, ε-greedy, the replay buffer, and the Bradley–Terry loss
- [ ] Diya can explain, from memory: the state/action/reward design, why actions are rules not alerts, and how preference labels become a reward model
- [ ] Both can explain any file in `src/` without reading it first
- [ ] Both can state the project's limitations without looking them up

If any of these fail, the project is not done — regardless of what the metrics say.
