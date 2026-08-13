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
python scripts/calibrate_generator.py --shifts 100
```
**Must show:** ~170 alerts/shift · true-incident rate 2.5–3.5% · **severity↔truth Pearson r between 0.30 and 0.40**.
If r is outside that band, stop and retune. This gates the whole project (`DECISIONS.md` D-003).

```bash
python scripts/run_baselines.py --seeds 5
```
**Must show:** a table of 6 baselines × 5 metrics with mean ± std, where `oracle` is strictly best on recall@deadline and `random` is worst.

```bash
pytest tests/test_env.py -v
```
Required tests, by name:
- `test_determinism_under_seed` — same seed ⟹ identical trajectory
- `test_reward_accounting` — sum of per-step rewards == episode total
- `test_no_ground_truth_leakage` — no observation array contains or is derivable from `is_true_incident` **(never weaken or skip this one)**
- `test_bulk_close_caps_at_ten`
- `test_clock_never_exceeds_shift_length`
- `test_empty_queue_advances_clock`

---

## Phase 1 — Dynamic Programming

```bash
python scripts/run_dp.py
```
**Must show:** value iteration converged with Δ < 1e-4; the sweep count; state coverage (how many of 576 states were visited, and the minimum visit count); and **value iteration and policy iteration agreeing on ≥95% of states**. Disagreement means one is wrong — investigate, don't average.

---

## Phase 2 — Tabular

```bash
pytest tests/test_tabular.py -v
```
- `test_q_learning_solves_toy_mdp` — converges to the known analytical answer on a hand-checkable 2-state MDP
- `test_epsilon_decays_to_floor`
- `test_q_table_shape` — (576, 5)

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
