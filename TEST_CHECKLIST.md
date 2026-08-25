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
Required tests, by actual name (the 7 Phase-0 tests, all passing; **suite total is now 76** — see the Phase 1, Phase 2 and evaluation-protocol blocks below):
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

**Also must show — and must be read, not skipped:** the DP row's **total reward beside its recall**.

> ⚠️ **The 5-seed figures once quoted here are superseded.** On the 30-seed block (D-019, E-014) DP scores **reward −201.2 ± 438.5 with recall 0.23 ± 0.24 — the worst of any planned or learned agent**, not the +305.9 / 0.43 measured on 5 seeds. Phase 1 is reopened on that basis. Quoting either number alone is misleading and D-012 forbids it; quoting the 5-seed pair at all now requires saying it is superseded.

```bash
pytest tests/test_mrp_bellman.py -v
python scripts/run_mrp_example.py
```
**Must show:** all four routes to the five-state MRP's value function agreeing — by hand, closed form, iterative evaluation, and `agents/dp.value_iteration` — with the largest disagreement below 1e-9. *Verified PASS 2026-08-16 (E-005): 7/7 tests, largest disagreement 7.11e-15.*

This is the only check in the project that validates the Bellman backup against an answer derived **outside** the code. VI/PI agreement above cannot do it — they share the same equation. **If this fails, fix `agents/dp.py`; never adjust the expected values.** They were derived by a human on paper (`docs/features/FEATURE_001_mrp_worked_example.md`) and changing them to make the test pass would destroy the only external anchor Phase 1 has.

---

## The evaluation protocol itself

```bash
pytest tests/test_eval_protocol.py -v
```
**Verified PASS 2026-08-17 (E-014, D-019) — 5 tests.** These guard the measuring instrument, not any agent:

- `test_eval_block_is_large_enough_to_resolve_the_effects_being_reported` — **at least 30 eval seeds.** Five satisfied CONSTRAINTS #3 and was still too few: severity-sort's per-seed reward std is ±220 against effects of ~100, so a 5-seed mean had a standard error the size of every finding built on it. One conclusion (DP's reward) came out with the **wrong sign**.
- `test_the_original_five_eval_seeds_are_still_in_the_block` — 101–105 must stay inside 101–130, so every pre-widening result remains a sub-sample rather than being orphaned.
- `test_eval_block_avoids_every_other_reserved_seed_range` — an eval seed colliding with a training block would mean evaluating on a shift the agent trained on. No test result would reveal it.
- plus uniqueness and train/eval disjointness.

> **Before believing any difference between agents, compare it to the spread.** Reporting a standard deviation is not the same as reading it — this project shipped ±218.7 beside a mean of 153.7 for three phases without drawing the inference.

## Phase 2 — Tabular

```bash
pytest tests/test_tiny_mdp.py -v
```
**Must show:** 13 passing tests establishing the hand-derived `q_*` for the 2-state MDP, including `agents/dp.value_iteration` reproducing it. *Verified PASS 2026-08-17 (E-006): 13/13, Bellman optimality residual 1.78e-15 against a 1e-12 tolerance.*

This is the Phase 2 counterpart of `test_mrp_bellman.py` and carries the same rule: **if it fails, fix the learner — never adjust the expected values.** They were derived by a human on paper (`docs/features/FEATURE_002_tiny_mdp_qstar.md`). The anchor is verified load-bearing: injecting a 0.1 error into any entry of `HAND_COMPUTED_Q` moves the residual to ~0.10, thirteen orders of magnitude above tolerance.

```bash
pytest tests/test_tabular.py -v
```
**Q-learning: verified PASS 2026-08-17 (E-007) — 20 tests.** `max |Q − q*| = 9.237e-14`, correct policy after 10 episodes, identical across 5 seeds. Groups:

- `test_q_learning_converges_to_the_hand_derived_q_star` — reproduces `tiny_mdp.HAND_COMPUTED_Q` to < 1e-9
- `test_q_learning_recovers_the_optimal_policy` — the behaviour, checked separately from the values
- `test_update_bootstraps_off_the_max_not_the_behaviour_action` — **the one that matters most.** Q-learning and SARSA converge to the same answer on this fixture, so `sarsa.py` pasted into `q_learning.py` would pass every convergence test. This pins the difference at a single backup: target 10.0, not 0.95.
- `test_terminal_update_does_not_bootstrap` · `test_single_update_matches_hand_arithmetic` · `test_alpha_zero_learns_nothing`
- `test_epsilon_decays_geometrically_and_stops_at_the_floor` · `test_epsilon_does_not_decay_on_update` (D-015)
- `test_ties_break_toward_the_lower_action_index` — not an edge case; a zero-init table ties on step one of every run
- `test_different_seeds_explore_differently_but_reach_the_same_fixed_point` — characterisation, guards a real investigation (E-007)
- `test_q_table_shape` — (576, 5) · `test_q_table_starts_at_zero` — zero init, not optimistic
- Config: alpha and epsilon range checks reject bad YAML at load time

- `test_visits_are_counted_per_state_action` · `test_unvisited_states_are_reported_as_unvisited_not_as_action_zero` — **display-honesty tests** (FEATURE_005). Unvisited states have all-zero Q rows, so `argmax` returns action 0; without the visit count, 455 of 576 cells in the headline policy table would print as a confident `PULL_HIGHEST_SEVERITY` the agent never chose.

```bash
pytest tests/test_on_policy.py -v
```
**SARSA and Monte Carlo: verified PASS 2026-08-17 (E-010) — 18 tests.** Split from `test_tabular.py` at the 500-line limit, and because these two share something Q-learning does not.

> ⚠️ **They are NOT graded against `HAND_COMPUTED_Q`** (D-017). Both are on-policy and converge to `tiny_mdp.epsilon_soft_q(ε)`, which at ε = 0.1 differs from q\* by more than 1.5. If a future session "fixes" them to match q\*, it will have broken two correct algorithms. The soft target is anchored by `test_epsilon_soft_q_collapses_to_q_star_as_epsilon_goes_to_zero`.

- `test_sarsa_bootstraps_off_a_worse_action_when_exploration_picks_one` — **the SARSA/Q-learning separator.** Target 0.95, not 5.00. Without it, `sarsa.py` could be a copy of `q_learning.py` and every convergence test would still pass.
- `test_sarsa_actually_takes_the_action_it_bootstrapped_off` — the on-policy invariant. A leaked commitment yields an off-policy hybrid that still converges.
- `test_monte_carlo_is_first_visit_not_every_visit` — the MC separator. 2.615, not 2.8075, from identical episode data.
- `test_monte_carlo_learns_nothing_until_the_episode_ends` · `test_monte_carlo_clears_its_buffer_between_episodes`
- `test_every_learner_has_its_own_training_seed_block` — D-016, enforced in code

**Tolerances differ by algorithm and were all set from measurement:** Q-learning 1e-9 (measured 9.24e-14), SARSA 0.15 (worst 0.100 over 8 seeds), MC 0.40 (worst 0.272). The looser two are not slack — SARSA's residual is constant-α noise that shrinks with α but not with more episodes, and MC is the higher-variance estimator.

**MC trains at `MC_HORIZON` (800), not `HORIZON` (200).** MC computes a return from every timestep, so a 200-step truncation biases it downward by up to 0.47 (2.75 at HORIZON=50). Do not "simplify" the two horizons into one.

### Phase 2 analysis scripts

```bash
python scripts/train.py --agent {q_learning,sarsa,monte_carlo}   # ~3 min each
python scripts/policy_table.py --agent <name>                    # box 6
python scripts/compare_agents.py                                 # box 5, needs run_dp.py first
python scripts/ablations.py                                      # box 7, ~4 min
```

**A reduced run (`--episodes`, `--repeats < 5`) writes to `results/smoke/` and cannot overwrite a full run's artefacts** (D-018). If `compare_agents.py` reports coverage that disagrees with E-009/E-011, suspect a stale artefact first.

**Replication rule, learned the hard way (E-013):** any *behavioural* claim about a learned policy must be checked on all three learners before it is written down. The Q-learning strategy-shift finding was internally consistent, monotonic and plausible — and SARSA runs the opposite direction.

### Regenerating the Phase 2 artefacts

```bash
python scripts/train.py           # 5 runs x 20000 episodes, ~2.8 min
python scripts/policy_table.py    # reads results/*.npy, writes results/policy_table.md
```
`results/` is gitignored. The pipeline is deterministic under its seeds: the second training run of 2026-08-17 reproduced E-008 to the digit (recall 0.73 ± 0.03, reward 270.9 ± 105.5, MTTD 22.0 ± 15.6). **If a re-run does not reproduce those numbers, something has changed that shouldn't have — investigate before using any new figure.**

`scripts/policy_table.py` self-checks its own `encode`/`decode` over all 576 state ids on every run. A transposed mixed-radix unpack yields a table that still looks plausible, which is exactly the error that survives review.

> Watch for one specific false pass here: under the tiny MDP's optimal policy the agent never leaves `QUIET`, so `BUSY` is reached only by exploring. A learner run with ε = 0 will never see half the MDP. If a learner passes on `Q(QUIET, ·)` and produces garbage on `Q(BUSY, ·)`, the exploration schedule is the suspect, not the update rule.

```bash
python scripts/train.py --agent q_learning --seeds 5 --eval
```
**Must show:** learning curve rising then flattening; final policy beating severity-sort on recall@deadline and MTTD, reported as mean ± std over 5 **evaluation** seeds (disjoint from training seeds).

---

## Phase 3 — DQN

Run all of these from the repo root with `.\.venv\Scripts\python.exe`.

| # | Check | Command | Passing means |
|---|---|---|---|
| 1 | Whole suite | `-m pytest tests/ -q` | **122 passed** |
| 2 | Config fails loudly | `-m pytest tests/test_dqn_config.py -q` | 11 passed — a zero divisor, a too-small `learning_starts`, or a duplicated seed block is rejected at load time, not mid-training |
| 3 | Replay buffer | `-m pytest tests/test_replay.py -q` | 8 passed — capacity, overwrite order, dtypes, and that all five arrays stay aligned |
| 4 | Agent, stabilisers, ablations | `-m pytest tests/test_dqn.py -q` | 13 passed |
| 5 | Tiny-MDP anchor | `-m pytest tests/test_dqn.py -k tiny_mdp -q` | 2 passed — max \|Q − q*\| < 0.05, measured 0.013–0.017 over 3 seeds at 120 episodes |
| 6 | Analysis guards | `-m pytest tests/test_dqn_analysis.py -q` | 14 passed — incomparable runs are refused, undefined metrics dropped not zeroed |
| 7 | Trainer runs | `scripts/train_dqn.py --episodes 40 --repeats 1 --eval-every 20 --no-plot` | completes; output says `results/smoke/` **not** `results/` |
| 8 | Parallel repeat matches sequential | `scripts/train_dqn.py --only-repeat 3 --episodes 40 --eval-every 40 --no-plot` | JSON shows `seed_base 1000120` = 1000000 + 3×40 |
| 9 | Sweep scheduler | `scripts/run_dqn_sweep.py --control-runs 2 --ablation-runs 1 --episodes 40 --max-parallel 2` | 4 runs; control on the 1000000 block, both ablations on 1200000, flags set |

**Before believing any ablation result**, run check 4 and confirm
`test_no_replay_ablation_trains_on_a_single_transition` and
`test_no_target_network_ablation_bootstraps_off_the_online_net` pass. An
ablation that did nothing and an ablation that was never wired up look
identical in a plot; only these two tests tell them apart.

**Housekeeping that has bitten this project twice.** Before committing, sweep
for stray zero-byte files (`git status --porcelain` — BUG_001; several appeared
during Phase 3) and delete any smoke-test output under `results/dqn_runs/`, or a
40-episode run can be averaged into a 20,000-episode sweep. `aggregate_dqn.py`
refuses mismatched runs, but the cheapest fix is not to create them.

**Not yet checkable — needs a training run:** the exit criterion itself, "DQN
matches or beats tabular Q-learning on the same evaluation seeds, and the two
ablations visibly destabilise training in the plots."

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


---

## Phase 4 - actor-critic and the two tuning experiments (added 2026-08-25)

### The suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q          # expect 191 passed
```

**Wall time is not stable.** 56 s and 8 min 49 s were both observed on the same commit in
one session, on machine load alone. Do not treat a slow run as a hang.

### Actor-critic alone

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_actor_critic.py tests/test_actor_critic_config.py -q
```

Expect **32 passed** (18 agent + 14 config), ~100 s.

The two that matter most, and why:

| test | what breaks if it goes |
|---|---|
| `test_the_critic_bootstraps` | the agent silently becomes REINFORCE-with-a-baseline |
| `test_the_critic_converges_to_the_hand_computable_value` | `v = 1 + gamma*v = 10` at gamma 0.9. A non-bootstrapping critic converges to 1.0, so this is arithmetic proof the target contains `v(s')` |

### Smoke runs before any long run

```powershell
.\.venv\Scripts\python.exe scripts/train_actor_critic.py --episodes 60 --repeats 1 --eval-every 30
.\.venv\Scripts\python.exe scripts/reinforce_clip_experiment.py --episodes 20 --repeats 1
.\.venv\Scripts\python.exe scripts/actor_critic_entropy_experiment.py --episodes 3 --repeats 1
```

**What to look at in the actor-critic smoke, and it is not the reward.** Watch the
`entropy` column. At the shipped `entropy_coef: 1.0` a healthy run holds entropy around
**0.8-1.0** - clearly off the floor, clearly below the uniform maximum of `ln 5 = 1.609`.

- Entropy falling to `0.000` means the policy has collapsed to one action and learning has
  stopped. That was the pre-E-020 state at `entropy_coef: 0.01`; if it reappears at 1.0,
  something has regressed.
- Entropy pinned near `1.609` means the bonus is dominating the TD signal and the policy is
  not committing to anything. That was E-020's finding at 10.0.

**Do NOT read the greedy diagnostic as a health signal in a short run.** It sat on exactly
-515.4 in all twelve E-020 runs regardless of coefficient, because any policy whose argmax is
BULK_CLOSE everywhere scores exactly that on fixed seeds. It is saturated, not informative.

### The experiments

```powershell
.\.venv\Scripts\python.exe scripts/reinforce_clip_experiment.py            # E-019, ~3.7 min, DONE
.\.venv\Scripts\python.exe scripts/actor_critic_entropy_experiment.py      # E-020, 6.7 min, DONE -- set entropy_coef to 1.0
```

Both print the **between-value spread beside the within-value spread** and say plainly when
the first does not exceed the second. Read that line before reading the ranking - E-008 and
E-012 both turned on that comparison being made rather than assumed.

Both refuse to run if a training seed collides with the evaluation block
(`_assert_no_eval_seeds`). That is CONSTRAINTS #2 enforced in code rather than by comment.
