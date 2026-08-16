# FEATURE_001 — Five-state MRP, hand-worked and machine-verified

**Status:** done
**Phase:** 1 · **Owner:** Pranav · **Started:** 2026-08-16 · **Finished:** 2026-08-16
**Model(s) used:** Claude Opus 5 (implementation, derivation, docs). Gate decision approved by Pranav; Diya countersign pending.

---

## What and why

The last unticked Phase 1 box: *"hand-work a 5-state Markov Reward Process on paper, show the Bellman equations explicitly, verify against code."*

The reason it matters is narrower than it looks. Phase 1 shipped a 576-state DP solver whose only correctness evidence was **internal**: value iteration converged, and policy iteration agreed with it 100%. But VI and PI share `greedy_policy` and the same backup expression — if the Bellman equation itself were written wrong, both would converge, agree with each other perfectly, and be wrong together. Nothing in Phase 1 compared the solver to an answer derived **outside** the code.

This feature supplies that external anchor: an MRP small enough to solve with a pen, and a test that demands `agents/dp.py` reproduce the pen's answer.

## Roadmap link

`ROADMAP.md` Phase 1, final checkbox. Closes Phase 1's task list.

## Approach

Agreed before implementation:

1. Define a five-state MRP thematically matched to the project (an alert's journey through a shift), with numbers chosen so the arithmetic stays exact.
2. Derive V by hand, showing the Bellman equation per state.
3. Verify by **four** independent routes, not one — hand, closed form, iterative evaluation, and the project's own solver.
4. Make route 4 possible by expanding the MRP into a degenerate MDP whose five actions are identical, so `max_a` collapses and value iteration must return the MRP's value function.

Route 4 is the load-bearing one. Routes 2 and 3 only check the new file against itself.

## The MRP

An alert's journey through a shift, coarsened to five states. `CONFIRMED` and `MISSED` are absorbing.

| | QUIET | BACKLOG | INVESTIGATING | CONFIRMED | MISSED |
|---|---|---|---|---|---|
| **QUIET** | 0.50 | 0.25 | 0.25 | — | — |
| **BACKLOG** | — | — | — | 0.40 | 0.60 |
| **INVESTIGATING** | — | — | — | 0.80 | 0.20 |
| **CONFIRMED** | — | — | — | 1.00 | — |
| **MISSED** | — | — | — | — | 1.00 |

Rewards are collected **on the transition**, `r(s, s')`: a minute of clock costs −1 from QUIET; reaching CONFIRMED pays +20 from BACKLOG (caught late) or +30 from INVESTIGATING (caught fast); reaching MISSED costs −20. Absorbing states pay nothing further.

γ = **0.9** for this example only — *not* the project's 0.99. Chosen so the one non-integer value stays an exact small fraction a student can write down.

The QUIET self-loop is deliberate: without it the chain is acyclic and solves by back-substitution, which would not exercise the fixed-point machinery the real solver depends on.

## The derivation, by hand

**Step 1 — expected one-step reward** (Sutton & Barto 2nd ed. eq. 3.5): `R(s) = Σ_s' P(s'|s) · r(s,s')`

```
R(QUIET)         = 0.50(−1) + 0.25(−1) + 0.25(−1)  = −1
R(BACKLOG)       = 0.40(+20) + 0.60(−20)           = 8 − 12   = −4
R(INVESTIGATING) = 0.80(+30) + 0.20(−20)           = 24 − 4   = +20
R(CONFIRMED)     = R(MISSED)                       = 0
```

**Step 2 — the Bellman equation** (S&B eq. 3.14): `V(s) = R(s) + γ · Σ_s' P(s'|s) · V(s')`

Absorbing states first. `V(CONFIRMED) = 0 + 0.9·V(CONFIRMED)` ⟹ `0.1·V = 0` ⟹ **V(CONFIRMED) = 0**. Same for MISSED. (This is a real check, not a triviality: it confirms the payoff was booked once on entry rather than paid out on every absorbing step, which would diverge.)

Both successors of BACKLOG and INVESTIGATING are now known to be 0:

```
V(BACKLOG)       = −4  + 0.9[0.4(0) + 0.6(0)]  = −4
V(INVESTIGATING) = +20 + 0.9[0.8(0) + 0.2(0)]  = +20
```

QUIET depends on itself, so it needs a solve rather than a substitution:

```
V(QUIET) = −1 + 0.9[0.5·V(QUIET) + 0.25·V(BACKLOG) + 0.25·V(INVESTIGATING)]
         = −1 + 0.45·V(QUIET) + 0.9[0.25(−4) + 0.25(20)]
         = −1 + 0.45·V(QUIET) + 0.9(−1 + 5)
         = −1 + 0.45·V(QUIET) + 3.6
         = 2.6 + 0.45·V(QUIET)

0.55·V(QUIET) = 2.6   ⟹   V(QUIET) = 2.6/0.55 = 52/11 = 4.7272…
```

**Answer:** `V = [52/11, −4, +20, 0, 0]`

**Sanity reading.** QUIET is worth *more* than BACKLOG (+4.7 vs −4) even though QUIET pays −1 every step and BACKLOG pays nothing directly. That is the discounted future talking: from QUIET there is a 25% chance of routing into INVESTIGATING, which is worth +20, whereas BACKLOG is a coin-flip that lands on −20 more often than +20. An agent should prefer an idle queue to a stale one. That is the whole intuition behind a value function, in one comparison.

## Alternatives considered

- **γ = 0.99 to match the project.** Rejected: `V(QUIET) = 2.6/(1 − 0.495)` is an unmemorable decimal, and the point of the exercise is arithmetic a human can carry.
- **An acyclic five-state chain.** Rejected: solves by pure back-substitution, so it would never detect a broken fixed-point iteration.
- **Putting the MRP constants in `config/*.yaml`** per CONSTRAINTS #9. Rejected — see **D-013**. They are the definition of a worked example, not tunables; a config edit would silently invalidate the derivation this test checks against.
- **Testing only the closed form.** Rejected: it would check the new file against itself and prove nothing about the shipped solver.

## Files touched

| File | New/Modified | What changed |
|---|---|---|
| `src/soc_triage/mrp_example.py` | New | MRP definition, `expected_rewards`, `solve_linear`, `evaluate_iteratively`, `as_degenerate_mdp` |
| `tests/test_mrp_bellman.py` | New | 7 tests; the load-bearing one runs `agents/dp.value_iteration` against the hand answer |
| `scripts/run_mrp_example.py` | New | Prints the derivation and the four-route comparison, for the report and viva |
| `ROADMAP.md` | Modified | Final Phase 1 box ticked; exit criterion restated (D-012); stale "Current phase" line fixed |
| `TEST_CHECKLIST.md` | Modified | Phase 1 block gains the two new commands; stale Phase 0 wording synced to the approved gate |
| `FLOW.md` | Modified | Flow C gains the verification path |
| `ARCHITECTURE.md` | Modified | New module registered |
| `EXPLAIN.md` | Modified | Part 9 (reward hacking) and Part 10 (the MRP) |
| `DECISIONS.md` | Modified | D-012, D-013 appended |
| `docs/experiments/EXPERIMENT_LOG.md` | Modified | E-005 appended |

## What was tried that didn't work

**Chasing round numbers.** The first several designs tried to force `V(QUIET)` to a whole number by tuning γ and the self-loop probability together. γ=0.5 with a 0.4 self-loop gives exactly 1.75 — but γ=0.5 is so far from any discount factor used in practice that the example stops resembling the thing it is teaching, and a 0.5 discount makes the future nearly irrelevant, which is the opposite of the lesson. Abandoned in favour of γ=0.9 and an exact fraction. `52/11` turns out to be *better* teaching material than a round number: it forces the reader to actually solve the equation instead of pattern-matching an answer.

**Booking the incident payoff as `R(CONFIRMED) = +10`.** The obvious first design. It diverges — an absorbing state with non-zero reward collects it forever, so `V(CONFIRMED) = 10/(1−γ) = 100`, and the terminal swamps every other value in the chain. Fixed by moving to per-transition rewards `r(s,s')` and folding them into `R(s)` via eq. 3.5, which is both the textbook definition and the same quantity `estimate_model` produces as `R_hat`. `test_absorbing_states_have_zero_value` exists specifically to catch anyone re-introducing this.

## How it was verified

```
$ .\.venv\Scripts\python.exe -m pytest tests/test_mrp_bellman.py -v
============================= test session starts =============================
platform win32 -- Python 3.13.1, pytest-9.1.1, pluggy-1.6.0
collected 7 items

tests/test_mrp_bellman.py::test_transition_matrix_is_stochastic PASSED   [ 14%]
tests/test_mrp_bellman.py::test_expected_rewards_match_hand_arithmetic PASSED [ 28%]
tests/test_mrp_bellman.py::test_closed_form_matches_hand_answer PASSED   [ 42%]
tests/test_mrp_bellman.py::test_quiet_state_value_is_the_exact_fraction PASSED [ 57%]
tests/test_mrp_bellman.py::test_iterative_evaluation_converges_to_the_same_answer PASSED [ 71%]
tests/test_mrp_bellman.py::test_absorbing_states_have_zero_value PASSED  [ 85%]
tests/test_mrp_bellman.py::test_project_value_iteration_reproduces_hand_answer PASSED [100%]

============================== 7 passed in 0.14s ==============================
```

```
$ .\.venv\Scripts\python.exe scripts\run_mrp_example.py
Four routes to V, which must agree:
  state                by hand   closed form   iterative  agents/dp.py
  QUIET               4.727273      4.727273    4.727273      4.727273
  BACKLOG            -4.000000     -4.000000   -4.000000     -4.000000
  INVESTIGATING      20.000000     20.000000   20.000000     20.000000
  CONFIRMED           0.000000      0.000000    0.000000      0.000000
  MISSED              0.000000      0.000000    0.000000      0.000000

  iterative evaluation: 44 sweeps
  agents/dp.py value iteration: 44 sweeps, final delta 8.88e-15
  largest disagreement with the hand-derived answer: 7.11e-15
```

Largest disagreement 7.11e-15 — floating-point noise, ~15 significant figures of agreement.

## Follow-ups left open

- The **pen-and-paper** half remains a genuine human task. This file supplies the reference solution; it does not substitute for Pranav and Diya each reproducing the derivation unaided before the viva. `TEST_CHECKLIST.md` "The human check" is where that gets ticked.
- The MRP is a *reward* process, not a decision process — it has no actions and therefore no policy improvement. If a future session wants a hand-checkable **MDP** (for Phase 2's Q-learning correctness test), that is a separate 2-state object; `ROADMAP.md` Phase 2 already has a box for it.

## Plain-English summary

Our dynamic-programming code solved a 576-state problem, and we had no way to check its answer by hand — we only knew it agreed with itself. So we built a tiny five-state version of the same problem, one small enough to solve with a pen and paper in about five minutes, and worked out the answer by hand: an idle queue is worth about +4.73, a stale backlog is worth −4, and an alert actively being investigated is worth +20. Then we fed that same tiny problem to the real code and required it to produce the same three numbers. It did, matching to fifteen decimal places. That means the equation inside our big solver is genuinely the textbook one, not merely something that happens to settle on a stable answer.
