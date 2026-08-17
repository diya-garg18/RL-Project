# FEATURE_006 — Completing Phase 2: SARSA, Monte Carlo, comparison, ablations

**Status:** done — **all 8 ROADMAP Phase 2 boxes built.** The phase's *exit criterion* is separately not met, and two decisions remain owed to the humans (E-008).
**Phase:** 2 · **Owner:** Pranav · **Started:** 2026-08-16 · **Finished:** 2026-08-16
**Model(s) used:** Claude Opus 5.

---

## What and why

Phase 2 stood at 5 of 8 boxes: the tiny-MDP anchor, Q-learning, `train.py`, learning curves and the policy table. This closes the remaining three — SARSA, Monte Carlo, the DP convergence comparison and the ablations — and in doing so **retracts part of an earlier interpretation** (E-013).

## What was built

| Box | Deliverable | Result |
|---|---|---|
| 1 | `agents/monte_carlo.py` | first-visit MC control, S&B §5.4 |
| 2 | `agents/sarsa.py` | on-policy TD control, S&B §6.4 |
| 5 | `scripts/compare_agents.py` | E-011 |
| 7 | `scripts/ablations.py` | E-012 |
| — | `agents/tabular.py` | shared base extracted at the third implementation |

Plus `scripts/policy_table.py --agent`, which is what surfaced E-013.

## The design decisions that mattered

**On-policy learners are graded against an ε-soft target, not q\*** (D-017). SARSA and MC converge to `q_π` for the ε-greedy policy they follow, which at ε = 0.1 differs from q\* by more than 1.5 on this fixture. Grading them against `HAND_COMPUTED_Q` would have marked correct implementations as broken. `tiny_mdp.epsilon_soft_q` computes that target and is anchored by the requirement that it reproduce q\* exactly at ε = 0.

**A shared `TabularAgent` base, extracted at the third implementation and not before.** The point was not saving keystrokes: each agent file now contains essentially nothing but its own update rule, so `sarsa.py` beside `q_learning.py` shows the difference in a few lines instead of burying it in forty lines of identical boilerplate. That is the part the students must write from memory.

**SARSA's `a'` problem.** `Agent.update` provides no next action, and widening that interface would force four other agents to supply something meaningless. SARSA therefore selects `a'` during `update` and commits to it, and `act()` returns exactly that. The on-policy property depends entirely on the commitment being honoured — a leak would produce a hybrid that still converges and still passes every value test — so `test_sarsa_actually_takes_the_action_it_bootstrapped_off` asserts it directly.

**Ablations are measured on the train-diagnostic seeds, never on eval** (E-012). An ablation is tuning, and CONSTRAINTS #2 forbids tuning against evaluation seeds whether the reading is done by a program or by a person.

## Results, in one place

All three learners, eval seeds, mean ± std across 5 training runs:

| agent | recall | reward | MTTD |
|---|---|---|---|
| sarsa | 0.74 ± 0.01 | **324.1 ± 81.6** | 23.3 ± 11.7 |
| q_learning | 0.73 ± 0.03 | 270.9 ± 105.5 | 22.0 ± 15.6 |
| monte_carlo | 0.71 ± 0.02 | 177.3 ± 91.7 | **18.6 ± 3.0** |
| dp | 0.43 ± 0.17 | 305.9 ± 127.6 | 6.3 |
| severity_sort | **0.87 ± 0.16** | 153.7 ± 218.7 | 23.0 ± 12.4 |

**The agreement is the finding, not the ranking.** All three learners land at recall 0.71–0.74 and all three fall short of severity-sort's 0.87. With DP, that is four methods sharing no update rule, trained on disjoint seed blocks, converging on the same trade: more reward, less recall. **The reward function selects this behaviour, not the algorithm** — D-012's prediction, now the best-supported claim in the project.

Do **not** report "SARSA is the best tabular method". The 177–324 spread between learners is the same order as the eval-seed variance E-008 measured, and E-012 shows deliberate hyperparameter changes fail to clear that same noise.

## What was tried that didn't work

**A guessed tolerance, for the third time.** SARSA's convergence test shipped at 0.25 and failed at 0.294. Investigating rather than loosening showed the residual is constant-α noise: it shrinks with α (0.113 → 0.080 → 0.041) but **not** with more episodes (0.276 → 0.294 → 0.113 → 0.139 — a random walk). Tolerance reset from measurement across 8 seeds. This project has now made the same mistake three times (E-006 action gap, E-007 convergence, here); the rule is measure the fixture, then set the bar.

**A committed comment that was wrong.** `HORIZON = 200` was justified by γ²⁰⁰ ≈ 7e-10. True for the return from t=0; wrong for Monte Carlo, which computes a return from every timestep, so at t=199 the missing tail is the whole value. Measured bias: 2.75 at HORIZON=50, 0.47 at 200, 0.09 at 800. `MC_HORIZON = 800` added. The comment was not a typo but a plausible argument applied to the wrong quantity, which is why it survived review (D-017).

**A smoke test silently destroyed a real result.** `--episodes 200` overwrote a completed 20,000-episode Q-table. Nothing errored; the file was valid and the wrong size only showed up later as unexplained coverage loss (121 → 81 states) that first looked like a bug in `compare_agents.py`. Fixed structurally — reduced runs now write to `results/smoke/` (D-018). CONSTRAINTS #4 was written against deliberate deletion; the likelier failure is an accidental overwrite by a routine command.

**A PowerShell `Set-Content` round-trip corrupted `train.py`** — every em dash became `â€”` and a BOM was prepended. Repaired by targeted replacement. **Never rewrite a source file through PowerShell's file cmdlets; use the editor.** Logged in `FLOW.md` gotchas alongside the related `python -c` quoting failure.

**And one thing that went wrong in the write-up itself.** The first draft of E-012 contained an `alpha=0.02` row that had scrolled off the captured output and was filled in from nothing — `63.9 ± 50.1` against a true `76.0 ± 50.0`. Caught on re-read, corrected by re-running (the sweep is deterministic under its seeds, and every other row reproduced exactly). Recorded because a fabricated number in an experiment log is the most damaging single failure available to this project, and it happened while writing up a section about not over-reading data.

## The retraction

**E-013 partially retracts E-009.** E-009 reported a monotonic, interpretable strategy shift in Q-learning's policy — bulk-closing rising 25.3% → 36.0% → 46.2% into the crunch — and offered two readings. Running the same figure for the other learners was the obvious check and had not been done:

| agent | >240m | 60–240m | <60m | direction |
|---|---|---|---|---|
| q_learning | 25.3% | 36.0% | 46.2% | rises |
| monte_carlo | 23.1% | 28.6% | 42.9% | rises |
| **sarsa** | 47.4% | 51.9% | **25.0%** | **falls** |

SARSA learned the reverse. Two of three agree; one contradicts. With 12–14 visited states per crunch bucket, that is fully consistent with the trends being noise.

**The per-algorithm figures stand; the interpretation does not.** "There is a behaviourally interpretable strategy shift" is not established. Nothing failed to catch this — the Q-learning figure was internally consistent, monotonic across three buckets, and had a plausible story attached. It took running the identical analysis on two more agents. **Replication across algorithms should be the default for any behavioural claim here, not an afterthought.**

## Files touched

| File | New/Modified |
|---|---|
| `src/soc_triage/agents/tabular.py` · `sarsa.py` · `monte_carlo.py` | **New** |
| `scripts/compare_agents.py` · `scripts/ablations.py` | **New** |
| `tests/test_on_policy.py` | **New** — 18 tests |
| `src/soc_triage/agents/q_learning.py` | reduced to its update rule |
| `src/soc_triage/tiny_mdp.py` | `epsilon_soft_q`, `MC_HORIZON`, corrected HORIZON comment |
| `src/soc_triage/config.py` | sarsa/monte_carlo sections + seed-block collision checks |
| `scripts/train.py` | `--agent`, per-agent artefacts, reduced-run guard |
| `scripts/policy_table.py` | `--agent` |
| `scripts/run_dp.py` | saves `dp_Q.npy` / `dp_visits.npy` for box 5 |

## How it was verified

```
$ .\.venv\Scripts\python.exe -m pytest tests/ -q
....................................................................... [100%]
71 passed in 6.86s
```

Every new test was watched to fail first. RED evidence: `ModuleNotFoundError: No module named 'soc_triage.agents.monte_carlo'`, `ImportError: cannot import name 'epsilon_soft_q'`, `AttributeError` on `sarsa.train_seed_start`.

DP re-run reproduced E-004 exactly (133/576 coverage, 1075 sweeps, Δ 9.95e-05, VI/PI 100%, recall 0.43, reward 305.9 ± 127.6). The ablation sweep reproduced row-for-row on re-run.

## Follow-ups left open

- **The two decisions from E-008** remain the top items: the eval-seed block, and the Phase 2 gate. E-012 sharpens the first — hyperparameter effects are invisible under current measurement noise.
- Why Monte Carlo has the best MTTD (18.6 ± 3.0) and much the tightest spread while being weakest on reward. Unexplained.
- Why SARSA beats DP on reward (324.1 vs 305.9). Most likely D-004 — DP is optimal for an estimated model built from 133 visited states — but not investigated.
- Whether the learners' 22–44% mutual policy agreement with near-identical performance means a broad plateau of equivalent strategies, or metrics too coarse to separate them (E-011).
- The crunch-bucket reward decomposition, **downgraded** by E-013: there may be no stable effect to explain.

## Plain-English summary

Phase 2 asked for three learning algorithms. The other two are now written: SARSA, which is like Q-learning but honest about its own mistakes — it accounts for the fact that it sometimes acts randomly — and Monte Carlo, which doesn't guess at all and just waits to see how each shift actually ends.

All three end up in the same place. Each earns far more points than the sensible rule-of-thumb baseline while catching **fewer** real incidents. Together with the planner from Phase 1, that's four completely different methods finding the same loophole. When every method cheats the same way, the problem is the rules, not the players — which is exactly the case for teaching the system what humans actually want, later in the project.

Two honest results worth more than a clean win. First, we tested whether the usual tuning knobs matter, and **none of them clears the noise** — the same setting run three times varies more than any change we made to it. Second, and more uncomfortable: last session we reported that the agent visibly changes strategy as the shift runs out, with a nice story attached. Running that same check on the two new algorithms shows one of them does the **opposite**. So we've withdrawn the claim. The measurements stand; the story doesn't.

Nothing broke to reveal that. Every test passed the whole time. It took deliberately repeating an analysis we'd already done once and been pleased with.
