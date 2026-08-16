# FEATURE_003 — Tabular Q-learning agent

**Status:** done *(the agent and its tests; not yet run on the real 576-state environment)*
**Phase:** 2 · **Owner:** Pranav · **Started:** 2026-08-16 · **Finished:** 2026-08-16
**Model(s) used:** Claude Opus 5 (implementation, tests, docs). Built test-first.

---

## What and why

The first learning algorithm in the project. Everything before it either planned with a known model (`agents/dp.py`) or did not learn at all (`agents/baselines.py`). This is the first agent that improves from experience.

`agents/q_learning.py` implements off-policy TD control — Sutton & Barto 2nd ed. §6.5:

```
Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```

Written by hand, no RL library (CONSTRAINTS #7), with explicit loops in place of `np.argmax` and `Q[s].max()` (CONSTRAINTS #14) because both students must be able to reproduce it from memory.

## Roadmap link

`ROADMAP.md` Phase 2, box 3 (`agents/q_learning.py`) and the second half of box 8 (`tests/test_tabular.py`). Boxes 1, 2 and 4–7 remain open.

## Approach — test-first

The tests were written and **watched to fail** before the agent existed:

```
tests\test_tabular.py:28: in <module>
    from soc_triage.agents.q_learning import QLearningAgent
E   ModuleNotFoundError: No module named 'soc_triage.agents.q_learning'
```

That matters here more than usual. This project's thesis is that a claim of correctness needs evidence, and a test written after the code passes on its first run — which proves the test was written to match what the code does, not what it should do.

The tests are in three groups, deliberately ordered:

1. **Mechanics** — Q-table shape and zero init, epsilon schedule, action selection, tie-breaking.
2. **Update rule** — single hand-computed backups.
3. **Convergence** — train on `tiny_mdp` (FEATURE_002), reproduce `q_*`.

Group 2 exists because group 3 is not sufficient. A subtly wrong backup still converges to *something close* on a two-state problem, and — more sharply — **SARSA and Q-learning converge to the same answer here**. Copy-pasting `sarsa.py` into `q_learning.py` would pass every convergence test in the file. The single-step tests pin the arithmetic so the two algorithms cannot be confused:

```
Q = 0, alpha = 0.5, gamma = 0.9, r = 1, Q(BUSY) = [1.0, 10.0]

Q-learning target: 1 + 0.9 * max(1, 10) = 10.0   ->  Q = 0 + 0.5(10.0)  = 5.0
SARSA-ish target:  1 + 0.9 * Q(BUSY,WAIT) = 1.9  ->  Q = 0 + 0.5(1.9)   = 0.95
```

`test_update_bootstraps_off_the_max_not_the_behaviour_action` asserts 5.0 and explicitly asserts *not* 0.95.

## Design decisions made here

**Epsilon decays in `end_episode()`, not in `update()`.** `config/training_default.yaml` specifies `decay: 0.9995  # per episode`. Decaying per *step* instead would drive epsilon to its 0.05 floor inside a single 480-minute shift, and the symptom — a learning curve that plateaus early — looks nothing like the cause. It cannot be inferred from `done` either, because the tiny MDP is continuing and never sets it. So the episode boundary is stated explicitly by the caller. `scripts/train.py` must call `end_episode()` once per episode; `test_epsilon_does_not_decay_on_update` guards the wrong version.

**Zero initialisation, not optimistic.** Optimistic initial values are a legitimate exploration technique and a *different algorithm*; adopting them would change every number the report quotes. Pinned by `test_q_table_starts_at_zero` so the choice cannot drift in silently without a DECISIONS entry.

**Ties break toward the lower action index**, matching `agents.dp.greedy_policy` (strict `>`, first index wins). Not an edge case: a zero-initialised table ties on every action at step one of every run, so an arbitrary rule would make runs unreproducible across seeds — and would make the Q-learning and DP policies incomparable rather than merely different.

**The agent owns its own RNG, seeded separately from the environment.** Changing the exploration schedule must not shift the alert stream, or two runs stop being comparable and every ablation in box 7 becomes meaningless.

**No default hyperparameters in the constructor.** Every value is injected. A default in a constructor is a magic number wearing a disguise (CONSTRAINTS #9).

## Files touched

| File | New/Modified | What changed |
|---|---|---|
| `src/soc_triage/agents/q_learning.py` | **New** | The agent: `act`, `update`, `end_episode`, `greedy_policy`, `_argmax` |
| `tests/test_tabular.py` | **New** | 20 tests across mechanics / update rule / convergence / config |
| `src/soc_triage/config.py` | Modified | Added `EpsilonConfig` and `QLearningConfig` to `TrainingConfig`, plus range validation |
| `docs/bugs/BUG_001_stray_zero_byte_files.md` | Modified | New trigger found — Python return annotations (`-> list[int]`) |

The YAML itself needed **no change**: `epsilon` and `q_learning` already existed in `config/training_default.yaml`. The loader simply had not read them, by design — it parsed `common` and `dp` only, because Phase 1 needed only those. Phase 2 opening them is the loader catching up to the phase, not new configuration.

Three range checks were added at load time: `alpha` in (0, 1], `0 <= epsilon.min <= epsilon.start <= 1`, and `epsilon.decay` in (0, 1]. An out-of-range alpha produces a diverging Q-table and an incoherent epsilon schedule produces an agent that never explores — both present as algorithm bugs and cost an afternoon to trace back to a typo in YAML.

## What was tried that didn't work

**The convergence tolerance was set two orders of magnitude too loose on the first pass.** The test shipped with `max_error < 1e-2`, chosen by guessing what a tabular learner "usually" achieves. Measuring instead of guessing showed the real figure is **9.24e-14** — machine precision — because the tiny MDP is deterministic, so the TD target carries no sampling noise and a constant alpha converges exactly rather than hovering. A 1e-2 tolerance would have accepted a genuinely wrong backup. Tightened to 1e-9. The lesson is the same one FEATURE_002 recorded about the action gap: **measure the fixture, then set the tolerance; do not assume a plausible-looking number.**

**A suspicious result that turned out to be correct.** All five seeds converged to *identical* Q-tables — max error 9.24e-14, standard deviation exactly 0. Under CONSTRAINTS #5 that is a bug report until proven otherwise, and the obvious explanation was that the seed was being ignored. It was not: early trajectories genuinely diverge (`10011101…` vs `11000011…`, with different intermediate Q-tables after 3 episodes), they simply converge to the same place. A deterministic MDP has a unique fixed point and no target noise, so exploration order changes how fast it is reached, not where. `test_different_seeds_explore_differently_but_reach_the_same_fixed_point` now locks both halves of that down, so the next person to notice the zero variance does not repeat the investigation — and a genuinely broken seed fails loudly.

That test is honestly labelled as **characterisation**, written after the investigation rather than test-first. Everything else in the file was written before the agent existed.

## How it was verified

RED — the tests failing before the agent existed:

```
E   ModuleNotFoundError: No module named 'soc_triage.agents.q_learning'
1 error in 0.27s
```

GREEN — after implementing:

```
$ .\.venv\Scripts\python.exe -m pytest tests/ -q
...............................................                          [100%]
47 passed in 2.56s
```

Convergence on the hand-solved MDP (FEATURE_002's `q_*`):

```
learned Q:            hand-derived q_*:
 [[10.   6.7]          [[10.   6.7]
  [10.7 13. ]]          [10.7 13. ]]

max |Q - q*| = 9.237e-14      policy: [0 1]   hand: [0 1]
```

Speed of convergence, and stability across seeds:

```
   10 episodes: max err 6.248e-02   policy correct: True
   50 episodes: max err 9.237e-14   policy correct: True
  500 episodes: max err 9.237e-14   policy correct: True
 2000 episodes: max err 9.237e-14   policy correct: True

5 seeds @ 500 episodes: mean 9.237e-14   std 0.000e+00   policy [0 1] every time
```

The policy is correct after **10 episodes**; the values take ~50 to reach machine precision. That ordering — behaviour converging well before values do — is worth remembering for the viva.

## Follow-ups left open

- **The agent has never been run on the real 576-state environment.** Everything above is on a 2-state fixture. Passing here proves the update rule is right; it proves nothing about performance on the SOC MDP, and no Phase 2 exit criterion has been tested.
- `scripts/train.py` does not exist. It must call `end_episode()` once per episode — the runner currently has no such hook.
- `agents/sarsa.py` and `agents/monte_carlo.py`, on the same fixture.
- Learning curves, the DP convergence comparison, the readable policy table, and the ablations — ROADMAP boxes 4–7, all still open.

## Plain-English summary

This is the project's first algorithm that actually learns. It keeps a big table — one row per situation, one column per possible action — holding its current guess at how good each action is. Every time it acts and sees what happened, it nudges one number in that table toward the reward it just got plus its own best guess about what follows. That single line of arithmetic is the entire algorithm.

The "off-policy" part is the interesting bit. While learning, the agent deliberately does random things now and then so it discovers options it would otherwise never try. But when it updates its table, it assumes it will behave *perfectly* from the next step onward — not randomly. So it explores like an amateur and learns like an expert. That's the one difference from SARSA, the next algorithm we'll write, and we wrote a test specifically to make sure the two can never be confused with each other.

We wrote the tests first and watched them fail, then built the agent until they passed. On the hand-solved two-state problem it reproduces the pen-and-paper answer to fourteen decimal places, and it works out the right *strategy* within about ten practice runs — long before the numbers themselves settle down.

Two things this does **not** mean. It has never been run on the real problem, only on the tiny practice one. And one result looked too good — every random seed gave byte-identical answers — so we stopped and checked whether the randomness was broken. It wasn't: the practice problem has exactly one right answer and no noise, so every route arrives at the same destination.
