# FEATURE_004 — `scripts/train.py`, the training entry point

**Status:** done *(the script; the Phase 2 exit criterion it produced numbers for is **not met** — see E-008)*
**Phase:** 2 · **Owner:** Pranav · **Started:** 2026-08-17 · **Finished:** 2026-08-17
**Model(s) used:** Claude Opus 5. Training-seed decision (D-016) approved by Pranav before the run.

---

## What and why

`agents/q_learning.py` (FEATURE_003) was verified on a 2-state fixture and had never touched the real environment. This is the harness that connects it to the 576-state SOC MDP, runs it repeatedly, and reports honestly.

It completes ROADMAP Phase 2 boxes 3 (partly) and 4, and produces the first learning result in the project.

## Approach

Four stages, and the ordering is the design:

1. Train on a dedicated seed block — one fresh shift per episode (D-016).
2. Every `eval_every` episodes, freeze exploration to 0 and measure the greedy policy on the **train-diagnostic** seeds (1–10).
3. Repeat the whole run 5 times with different agent seeds *and* different training-seed slices.
4. **Only at the very end**, look at the evaluation seeds.

Stage 4 happens once, after every training decision has already been made, so nothing in the script can tune against the evaluation seeds. Stage 2 exists because a learning curve has to be plotted against something, and plotting it against eval seeds would be tuning against them by eye — a CONSTRAINTS #2 violation that leaves no trace in code.

Stage 3 varies both the agent seed and the training shifts. Varying only the agent seed would leave all five runs facing an identical alert stream and would understate the real variability.

## Design decisions made here

**`agent.end_episode()` is called explicitly in the training loop** (D-015). This is the single line the entire epsilon schedule depends on, and forgetting it fails silently — epsilon stays at 1.0, the agent explores at random forever, and there is no error message. Commented as such at the call site.

**Diagnostic evaluation saves and restores epsilon, and passes `learn=False`.** Measuring must not change what is being measured, and the curve should show what the learned policy is worth rather than what a partly-random agent happens to score.

**Reward-hack-relevant metrics travel together.** The output table prints recall, reward and MTTD on one line per agent, because D-012 established that the reward number alone is misleading for this environment.

**The script prints a warning that its own two spreads are not comparable.** The `q_learning` row's ± is across training runs; the baseline rows' ± is across eval seeds. Presenting them in one table without saying so would be quietly misleading, so the script says so itself rather than relying on the reader to remember.

**Episode records are discarded during training.** `run_episode` builds a full `EpisodeRecord`; 100,000 of them would not fit in memory. Only `outcome.total_reward` is kept per episode. Reusing the tested runner and throwing the record away was preferred over writing a second, leaner env-agent loop that could drift from the first.

## Files touched

| File | New/Modified | What changed |
|---|---|---|
| `scripts/train.py` | **New** | The training entry point |
| `config/training_default.yaml` | Modified | `q_learning.train_seed_start: 200000` (D-016) |
| `src/soc_triage/config.py` | Modified | `QLearningConfig.train_seed_start` + a seed-block collision check |
| `tests/test_tabular.py` | Modified | 2 tests: the new field loads; a colliding seed block is rejected |

The config change was made test-first — the seed-collision test was watched to fail (`DID NOT RAISE ConfigError`) before the validation existed.

## How it was verified

Smoke test first, then timing, then the full run — deliberately in that order, so a bug would surface in seconds rather than after a full training run:

```
--episodes 100 --repeats 1   ->  ran end to end
--episodes 500 --repeats 1   ->  1.3 s  =>  0.003 s/episode
projected full run           ->  ~4 min (measured: 2.8 min)
```

The projection was checked against the 10-minute threshold in `CLAUDE.md` that requires human sign-off before a long run. It came in under, so the run proceeded.

**Cross-check that the harness is wired correctly:** the baseline rows the script prints — severity_sort 153.7 ± 218.7, oracle_greedy 214.1 ± 207.6 — reproduce E-002 and E-004 exactly. The evaluation path is the same one Phase 1 used, and it still gives the same answers.

Full result in **E-008**. In brief: recall 0.73 ± 0.03, reward 270.9 ± 105.5, MTTD 22.0 ± 15.6, against severity_sort's 0.87 / 153.7 / 23.0. **The Phase 2 exit criterion is not met** — it requires beating severity-sort on recall, and 0.73 < 0.87.

## What was tried that didn't work

**Nothing in the script itself failed** — but two results forced investigation before anything was written down, and one of them turned out to matter more than the training run.

**The train-diagnostic curve sat at 40–60 reward while the eval number came out at 270.9.** The obvious reading is overfitting to the training block. It is not: the same *baselines* show the same gap. Severity-sort scores −78.7 on seeds 1–10 and +153.7 on seeds 101–105; even the oracle gains 120 points moving between the two sets. **The five evaluation seeds are systematically easier than seeds 1–10**, and the per-seed spread (±325 for severity_sort) is several times larger than the ~117-point effect E-008 reports. This affects E-002, E-003 and E-004 as much as E-008. Escalated to the humans as an open decision rather than fixed, because widening the eval block invalidates the comparability of every prior experiment.

**The learning curve does not visibly converge.** After epsilon floors at ~episode 6000 the diagnostic keeps swinging between −262.9 and +121.3 with no trend. Given the variance finding above, most of that is probably shift-to-shift noise rather than an unstable learner — but the two have not been separated, so "Q-learning converged" is not claimed.

## Follow-ups left open

- **The eval-seed representativeness decision** (E-008). The most important open item in the project right now; it is not a Phase 2 issue.
- **The Phase 2 gate.** Unmet and deliberately unamended, following D-012's precedent that a criterion contradicted by measurement gets decided by a human, with real numbers in hand.
- Separate learner instability from environment variance before making any convergence claim.
- Fix the table so the two ± columns are the same quantity.
- ROADMAP boxes 1, 2, 5, 6, 7: SARSA, Monte Carlo, the DP convergence comparison, the readable policy table, and the ablations.

## Plain-English summary

This is the script that finally puts the learning agent to work on the real simulated SOC. It plays 20,000 shifts, five separate times over, and only looks at the "exam" shifts once at the very end — so nothing it does during practice can be secretly tuned to the exam.

The agent learned something real. It earns far more reward than the rule-of-thumb baseline (271 against 154). But it catches **fewer** genuine incidents (73% against 87%), and looking at what it actually does explains why: it spends 62% of its time hitting the bulk-close button. That is the same shortcut the Phase 1 planner found, arrived at by a completely different method. Two unrelated algorithms independently discovering the same exploit is strong evidence that the problem is the **reward we wrote**, not the algorithms — which is precisely the argument for learning a reward from human judgement later in the project.

So the Phase 2 target is not met, and we are not moving the target to make it look met. That decision belongs to the humans, the same way it did in Phase 1.

There is also a bigger problem, found by accident. We noticed the agent scoring much better on the exam shifts than on the practice ones, and checked whether it was cheating. It wasn't — **the five exam shifts are simply easier than the practice shifts**, for every agent we tested, including the theoretical best-possible one. Worse, shifts vary so wildly that the noise is several times bigger than the differences we have been reporting. That casts doubt on comparisons made throughout the project so far, not just this one. We have changed nothing in response, because fixing it means re-running everything, and that is a call for the humans to make.
