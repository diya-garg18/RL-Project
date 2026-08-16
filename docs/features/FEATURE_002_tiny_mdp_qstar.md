# FEATURE_002 — Two-state MDP, hand-solved, as the Phase 2 correctness anchor

**Status:** done
**Phase:** 2 · **Owner:** Pranav · **Started:** 2026-08-16 · **Finished:** 2026-08-16
**Model(s) used:** Claude Opus 5 (design, derivation, implementation, docs). Design approved by Pranav before implementation.

---

## What and why

Phase 2 writes three tabular learners — Monte Carlo, SARSA and Q-learning — by hand. All three produce a `(576, 5)` Q-table on an environment far too large to check with a pen. Left there, the only available check is agreement *between* the learners, and agreement is not correctness: if all three implement the same misunderstanding of the Bellman backup they will agree beautifully and all be wrong.

This feature builds the smallest MDP whose optimal action-value function can be derived on paper, so each learner can be pointed at a **known answer** before it is pointed at the real environment.

It is the action-value counterpart of FEATURE_001:

| | Checks the backup for | Guards |
|---|---|---|
| FEATURE_001 — 5-state MRP | `V` | `agents/dp.py` (Phase 1) |
| FEATURE_002 — 2-state MDP | `Q` | the tabular learners (Phase 2) |

## Roadmap link

`ROADMAP.md` Phase 2, final box: *"Tests: Q-learning converges on a tiny hand-checkable 2-state MDP with a known answer."*

Built **first** rather than last, which is a deliberate departure from the written box order. The reasoning: the anchor is only useful if it is trusted, and it can only be trusted before a learner exists to be blamed for a disagreement. Built afterwards, a mismatch between Q-learning and the fixture is ambiguous. Built first and cross-checked against the already-verified Phase 1 solver, any later mismatch is unambiguously the learner's fault.

## The MDP

Two states, two actions, deterministic, continuing (no terminal state), γ = 0.9.

| State | Meaning |
|---|---|
| `QUIET` (0) | queue under control, nothing urgent waiting |
| `BUSY` (1) | a backlog has built up |

| Action | Meaning |
|---|---|
| `WAIT` (0) | do not pull an alert this step |
| `WORK` (1) | pull an alert and investigate it |

| From | Action | To | Reward | Reading |
|---|---|---|---|---|
| QUIET | WAIT | QUIET | **+1** | a calm queue is a good place to be |
| QUIET | WORK | BUSY | **−5** | digging into a quiet queue wastes time and a backlog forms behind you |
| BUSY | WAIT | BUSY | **−1** | ignoring a backlog does not clear it |
| BUSY | WORK | QUIET | **+4** | working the backlog clears it |

## The derivation, by hand

**Step 1 — guess the policy.** Claim: WAIT in QUIET, WORK in BUSY. (Verified in step 3; a guess is legitimate here because step 3 checks it against the Bellman optimality equation, which admits no other solution.)

**Step 2 — solve for `V` under that policy** (S&B eq. 3.14, one action per state so no max):

```
V(QUIET) = +1 + 0.9 · V(QUIET)          ← the self-loop: needs a genuine solve
  0.1 · V(QUIET) = 1
       V(QUIET) = 10

V(BUSY)  = +4 + 0.9 · V(QUIET)
         = 4 + 9
         = 13
```

`V(QUIET) = 1/(1 − γ)` is the geometric series a student should recognise on sight: +1 forever, discounted.

**Step 3 — expand to `q_*`** (S&B eq. 3.20: `q_*(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) · max_a' q_*(s',a')`, which for deterministic transitions is just `R(s,a) + γ · V(s')`):

```
Q(QUIET, WAIT) = +1 + 0.9 · V(QUIET) = +1 + 9.0  = 10.0   ← best in QUIET
Q(QUIET, WORK) = −5 + 0.9 · V(BUSY)  = −5 + 11.7 =  6.7
Q(BUSY,  WAIT) = −1 + 0.9 · V(BUSY)  = −1 + 11.7 = 10.7
Q(BUSY,  WORK) = +4 + 0.9 · V(QUIET) = +4 + 9.0  = 13.0   ← best in BUSY
```

**Answer:** `Q* = [[10.0, 6.7], [10.7, 13.0]]`, `V* = [10, 13]`, `π* = [WAIT, WORK]`.

The row maxima are 10.0 and 13.0, which reproduce the `V` from step 2 — so the guessed policy is confirmed greedy with respect to its own value function, which is exactly the optimality condition. The guess was right.

**Sanity reading.** BUSY is worth *more* than QUIET (13 vs 10) even though BUSY is the bad state. That looks wrong for about five seconds and is worth sitting with, because it is the single most common misreading of a value function. `V(BUSY) = 13` is not "being in a backlog is pleasant" — it is "from BUSY, one WORK step pays +4 **and** returns you to QUIET, so you collect the +4 *on top of* everything QUIET was already worth." The one-off bonus for cleaning up beats the steady +1 stream, but you cannot farm it: getting back to BUSY to collect it again costs −5, which is strictly worse than the 3.3 it would gain. The optimal policy is therefore "stay in QUIET", and `Q(QUIET, WORK) = 6.7` is precisely the arithmetic that forbids the exploit. A value function that let you loop for profit here would be the tiny-MDP version of the Phase 1 bulk-close hack — and it is worth noting that this fixture was checked for exactly that before being trusted.

## Design constraints, and what each one rules out

Four properties were required. Each exists to fail a specific broken learner that would otherwise pass:

1. **Exact arithmetic** — the expected values are integers and one-decimal fractions, so the target is a number a human wrote, not a number a program produced.
2. **Continuing, not episodic** — no terminal state, so a learner cannot succeed by averaging complete returns without ever bootstrapping. TD methods must genuinely use `V(s')`.
3. **The optimal action differs between the two states** — WAIT in QUIET, WORK in BUSY. An agent hard-wired to `return 0` fails. Had both states preferred the same action, a constant-action stub would have scored a perfect policy match.
4. **Deterministic transitions** — nothing here tests whether a learner can average out sampling noise; that is the real environment's job. Determinism keeps the assertions exact and the runtime at ~0.4 s.

A fifth property fell out of the design rather than being asked for, and is the most useful one for the viva: **under the optimal policy the agent never leaves QUIET**, so BUSY is reachable only by exploring. A learner with ε pinned to 0 never sees half the MDP and fails on `Q(BUSY, ·)`. The fixture demonstrates why exploration is not optional.

## Alternatives considered

- **Reuse the 5-state MRP from FEATURE_001.** Rejected: an MRP has no actions, so it cannot check a `Q`-table at all — it is exactly the wrong shape for Phase 2.
- **A 2-state MDP with a terminal state.** Would make Monte Carlo simpler (real episodes, no truncation). Rejected because it also lets a TD learner pass without ever bootstrapping through a non-terminal step, which is the specific bug most worth catching.
- **Padding the 2-action MDP up to 5 actions with zero-reward self-loops**, so `agents/dp.value_iteration` (which loops over its module constant `N_ACTIONS = 5`) would accept it. Rejected: that adds a *genuinely new* action worth 0. Harmless in this MDP where all values are positive, silently wrong in any MDP with negative values. Padding by **duplicating** the real actions cyclically (0,1,0,1,0) can only tie the maximum, never beat it, so `q_*` is provably untouched — and `test_padding_preserves_the_answer` asserts it.

## Files touched

| File | New/Modified | What changed |
|---|---|---|
| `src/soc_triage/tiny_mdp.py` | **New** | The MDP, the frozen hand-derived `Q*`/`V*`/`π*`, `step()`, `bellman_optimality_residual()`, `greedy_from_q()`, `pad_actions()` |
| `tests/test_tiny_mdp.py` | **New** | 13 tests verifying the anchor itself |
| `docs/features/FEATURE_002_tiny_mdp_qstar.md` | **New** | This file |
| `tests/test_mrp_bellman.py` | Modified | Docstring pointed at a filename that does not exist (`phase1-mrp-worked-example.md`); corrected to `FEATURE_001_mrp_worked_example.md` |

No learner was written. `tests/test_tabular.py` and `agents/q_learning.py` are the next request.

## What was tried that didn't work

**Two earlier reward designs, both rejected for a margin too narrow to test against.** The first attempt was `(QUIET,WAIT)→+1`, `(QUIET,WORK)→0`, `(BUSY,WAIT)→0`, `(BUSY,WORK)→+2`, giving `Q* = [[10, 9.9], [9.9, 11]]`. Arithmetically clean, and useless: the optimal action in QUIET wins by **0.1** on a value of 10 — a 1% margin. A learner within 1% of `q_*`, which is a perfectly ordinary state for a tabular method after a finite number of episodes, would flip the policy at random and the test would be flaky rather than strict. The second attempt (`−2` and `+4`) widened the QUIET margin only to 0.3. The final design pushes the needless-investigation penalty to `−5`, which buys margins of **3.3** and **2.3**. `MIN_ACTION_GAP` is recorded in the module so the margin cannot be eroded by a later edit without a test failing.

The general lesson, which applies to every fixture and not just this one: a test fixture needs its correct answer to be *far* from its wrong answers, not merely different from them. Designing for a clean-looking number first and a decision margin second gets this exactly backwards.

**Passing the mutation check to PowerShell via `python -c` with a here-string.** PowerShell's native-command argument parsing stripped the inner double quotes, and Python received `print(residual,` — an unclosed paren, five lines in. Not worth debugging: the fix is to write the throwaway script to the scratchpad directory and run it as a file. Logged in `FLOW.md` gotchas as well, since it is the second quoting-related tooling failure this project has hit.

## How it was verified

```
$ .\.venv\Scripts\python.exe -m pytest tests/test_tiny_mdp.py -v
============================= test session starts =============================
platform win32 -- Python 3.13.1, pytest-9.1.1, pluggy-1.6.0
collected 13 items

tests/test_tiny_mdp.py::test_transition_model_is_a_probability_model PASSED [  7%]
tests/test_tiny_mdp.py::test_transitions_are_deterministic PASSED        [ 15%]
tests/test_tiny_mdp.py::test_step_agrees_with_the_transition_matrix PASSED [ 23%]
tests/test_tiny_mdp.py::test_the_worked_transitions_are_the_ones_documented PASSED [ 30%]
tests/test_tiny_mdp.py::test_hand_computed_q_satisfies_bellman_optimality PASSED [ 38%]
tests/test_tiny_mdp.py::test_hand_computed_q_matches_the_pen_and_paper_arithmetic PASSED [ 46%]
tests/test_tiny_mdp.py::test_v_is_the_row_max_of_q PASSED                [ 53%]
tests/test_tiny_mdp.py::test_greedy_policy_matches_the_hand_derivation PASSED [ 61%]
tests/test_tiny_mdp.py::test_optimal_action_differs_between_the_two_states PASSED [ 69%]
tests/test_tiny_mdp.py::test_action_gap_is_wide_enough_to_test_against PASSED [ 76%]
tests/test_tiny_mdp.py::test_padding_preserves_the_answer PASSED         [ 84%]
tests/test_tiny_mdp.py::test_project_value_iteration_reproduces_hand_answer PASSED [ 92%]
tests/test_tiny_mdp.py::test_value_iteration_reconstructs_the_full_q_table PASSED [100%]

============================= 13 passed in 0.36s ==============================
```

Full suite, and a clean package import:

```
$ .\.venv\Scripts\python.exe -m pytest tests/ -q
...........................                                              [100%]
27 passed in 0.38s

$ .\.venv\Scripts\python.exe -c "import soc_triage"
soc_triage imports cleanly
```

**Mutation check — the tests are load-bearing.** A passing test proves nothing unless it can fail. The Bellman residual was re-run with deliberately wrong entries:

```
residual, correct Q        : 1.776e-15
residual, Q(BUSY,WORK)  13.0 -> 13.1: 0.1000
residual, Q(QUIET,WAIT) 10.0 ->  9.9: 0.0900
residual, Q(QUIET,WORK)  6.7 ->  6.8: 0.1000
policy if Q(QUIET,WORK)=99 : [1 1] vs hand [0 1]
```

The correct table sits at `1.8e-15`, floating-point noise on `0.9 × 13`. A 0.1 error produces a residual of ~0.1 — **thirteen orders of magnitude** above the `1e-12` tolerance. There is no plausible wrong answer that slips through.

## Follow-ups left open

- `tests/test_tabular.py` and the three learners. This fixture is imported, not modified, by that work.
- **The humans still owe the unaided derivation.** The same gap FEATURE_001 left open: the box is ticked because the derivation and its verification exist, not because Pranav and Diya can each reproduce it cold. Tracked in `TEST_CHECKLIST.md` → "The human check". This one is a genuinely fair viva question — two states, four numbers, γ = 0.9 — and should be practised.

## Plain-English summary

Before we let three learning algorithms loose on a 576-state problem nobody can check by hand, we built a two-state version small enough to solve on paper. It has two situations — a calm queue and a backlogged one — and two things you can do in each: wait, or work an alert. Doing the arithmetic by hand gives four numbers that say exactly how good each choice is in each situation, and the best move genuinely differs between the two: wait when things are calm, work when there's a backlog.

Those four numbers are now frozen in the code as the right answer. Every learning algorithm we write next has to reproduce them before we believe anything it says about the real problem. We also checked the check — we deliberately typed in wrong numbers to confirm the test actually notices, and it does, by a factor of about ten trillion.
