# Phase 3 design — Deep Q-Networks

**Date:** 2026-08-18
**Author:** Pranav Upadhyay (design produced with Claude Opus 5, approved by Pranav before implementation)
**Roadmap phase:** 3 — Deep Q-Networks (Week 3, first half), CO3
**Status:** approved, not yet implemented

---

## 1. Why this phase exists

Phase 2's tabular learners see the world through `discretise()` — 576 buckets built
from five coarse features. That encoding throws away everything about *queue
composition*: whether the fifty waiting alerts are all one type or spread across
six, whether the verify costs are cheap or expensive, how many incidents have
already been confirmed this shift. `featurise()` keeps all of it in 17 floats.

DQN is the honest test of whether that discarded information is worth anything.
It is the same objective, the same environment, the same exploration schedule —
only the state representation and the function approximator change.

**Prior expectation, recorded before measurement:** Phase 2 found that all three
tabular learners land inside the noise of a hand-written heuristic on total
reward (q_learning 47.6 ± 52.0 vs severity_sort 40.4 ± 220.1, E-014). There is no
reason to expect DQN to escape a pathology that lives in the reward function
rather than in the algorithm. The bulk-close exploit found by DP (~97% of actions)
and inherited by Q-learning (62.3%) should be expected to reappear here.
If it does, that is a **result**, not a failure — it is the third independent
algorithm to find the same exploit, which is the strongest available evidence
that the objective is at fault. See §10.

---

## 2. Measurements taken before designing

All figures below were measured on this machine on 2026-08-18, not estimated.

### 2.1 Episode length and compute cost

| quantity | value | method |
|---|---|---|
| env steps per shift, random policy | **49.9** (min 40, max 59) | 10 episodes, train seeds 1–10 |
| env steps per shift, severity_sort | **38.9** (min 32, max 43) | 10 episodes, train seeds 1–10 |
| env step + `featurise` | ~0.02 ms | 10 episodes timed |
| `featurise()` alone | 0.8 µs/call | 2000 calls |
| batch-64 gradient step, 17→128→128→5, Adam + Huber | **1.107 ms** | 500 steps |
| single forward pass (one `act`) | **0.056 ms** | 2000 calls |

Consequence: a gradient step on *every* env step at the tabular budget of 20 000
episodes is ~1M updates ≈ **18.3 min per run**, ≈ 4.6 hours across the main run
plus two ablations at 5 repeats each. That is why `train_freq` exists (§5.3).

### 2.2 Feature magnitudes — the finding that shaped the design

888 observations collected under random and severity_sort policies over train
seeds 1–10:

| col | feature | min | mean | p99 | max |
|---|---|---|---|---|---|
| 0 | queue_len | 0.00 | 45.19 | 128.00 | 148.00 |
| 1 | mean_severity | 0.00 | 0.74 | 1.50 | 2.00 |
| 2 | max_severity | 0.00 | 2.33 | 3.00 | 3.00 |
| 3 | mean_age_min | 0.00 | 82.27 | 219.55 | 237.79 |
| 4 | max_age_min | 0.00 | 183.62 | 454.56 | **473.57** |
| 5 | mean_asset_criticality | 0.00 | 0.58 | 1.16 | 2.00 |
| 6 | max_asset_criticality | 0.00 | 1.77 | 2.00 | 2.00 |
| 7 | mean_verify_cost_min | 0.00 | 12.33 | 18.61 | 22.86 |
| 8–13 | frac_type_0 … frac_type_5 | 0.00 | 0.10–0.23 | 0.25–0.50 | 0.50–1.00 |
| 14 | time_left_norm | 0.01 | 0.54 | 1.00 | 1.00 |
| 15 | alerts_handled | 0.00 | 18.45 | 40.00 | 45.00 |
| 16 | incidents_confirmed | 0.00 | 1.29 | 5.00 | 6.00 |

**A 470× spread between the largest and smallest column.** Fed raw into a 128-unit
MLP, the two age columns dominate every gradient and the six type-fraction columns
contribute essentially nothing. Input scaling is therefore not an optimisation —
it is a correctness requirement, and it is §3.

---

## 3. Input scaling

Each column is divided by a fixed constant before entering the network.

**The divisors are domain constants, not tuned values.** Every one is a number the
students can justify in a viva from the MDP definition alone:

| column(s) | divisor | justification |
|---|---|---|
| `queue_len` | 150 | observed ceiling; queue length is unbounded in principle |
| `mean_severity`, `max_severity` | 3 | `severity.levels` runs 0–3 |
| `mean_age_min`, `max_age_min` | 480 | the shift length — an alert cannot be older than the shift |
| `mean_asset_criticality`, `max_asset_criticality` | 2 | `asset_criticality.levels` runs 0–2 |
| `mean_verify_cost_min` | 30 | ceiling of `verify_cost_min.options` |
| `frac_type_0` … `frac_type_5` | 1 | already fractions summing to 1 |
| `time_left_norm` | 1 | already normalised by construction |
| `alerts_handled` | 50 | observed ceiling (max 45) |
| `incidents_confirmed` | 10 | observed ceiling (max 6) |

### 3.1 Where the divisors live, and why not the obvious place

They go in **`config/training_default.yaml`** under `dqn.feature_scales`, as a
mapping keyed by feature name, validated against `state.FEATURE_NAMES`.

They do **not** go in `config/env_default.yaml`, even though that is where
environment constants normally live. `env_default.yaml` is content-hashed by
`runner.config_hash()` and the resulting `config_hash` is written into **every**
`EpisodeRecord` produced since Phase 0. Adding a key to it would change that hash
and orphan every Phase 0, 1 and 2 result from the config that produced it.
`training_default.yaml` is not hashed (`scripts/train.py:177` hashes only the env
config), so it is the safe home.

Scaling is applied **inside `DQNAgent`**, not inside `state.featurise()`.
`featurise()` is the observation contract shared with Phase 4; how a particular
approximator preprocesses it is the approximator's business.

### 3.2 Rejected alternative

A running mean/std normaliser (Welford) adapts automatically and needs no
constants. Rejected because its state changes between training and evaluation,
which makes a saved policy's behaviour depend on the order it saw its training
data — an extra thing to explain, an extra thing to get wrong at eval time, and a
silent source of train/eval mismatch. Fixed divisors are deterministic and
explainable in one sentence each.

---

## 4. Modules

| File | Responsibility | New/changed |
|---|---|---|
| `src/soc_triage/agents/replay.py` | `ReplayBuffer` — uniform-sampling ring buffer | new |
| `src/soc_triage/agents/dqn.py` | `QNetwork` + `DQNAgent` | new |
| `src/soc_triage/config.py` | `DQNConfig` dataclass, loader, validation | changed (additive) |
| `config/training_default.yaml` | `dqn:` block extended | changed (additive) |
| `scripts/train_dqn.py` | training entry point | new |
| `scripts/dqn_ablations.py` | the two required ablations | new |
| `scripts/compare_dqn_tabular.py` | roadmap box 5 | new |
| `tests/test_replay.py` | buffer behaviour | new |
| `tests/test_dqn.py` | agent behaviour + tiny-MDP anchor | new |

**Nothing in Phase 0–2 is modified.** `scripts/train.py`, `scripts/ablations.py`
and `scripts/compare_agents.py` are tested, their outputs are logged experiments,
and CONSTRAINTS #11 requires Phase 2 to keep reproducing if Phase 3 breaks.
The only edits to existing files are additive: a new dataclass and loader section
in `config.py`, and new keys in the `dqn:` block of `training_default.yaml`.

### 4.1 Why a separate trainer rather than extending `scripts/train.py`

`train.py` is typed to `TabularAgent` throughout — it passes `n_states`, saves
`Q.npy` and `visits.npy`, and its `AGENTS` dict is `dict[str, type[TabularAgent]]`.
DQN has no Q-table and no visit counts. Branching it would push a 293-line file
toward the 500-line limit (CONSTRAINTS #12) and entangle Phase 2's tested path.

This follows the repo's own rule-of-three, stated in `agents/tabular.py`: shared
machinery was extracted at the **third** implementation, not the second. A second
trainer is a duplicate; a third (Phase 4's REINFORCE) is the signal to extract a
shared harness into `src/soc_triage/training.py`. Noted here so Phase 4 does it
deliberately rather than accreting a fourth copy.

---

## 5. `DQNAgent` — the algorithm

Deep Q-Network (Mnih et al. 2015), the form taught in Sutton & Barto 2nd ed.
§16.5 as function-approximation Q-learning (§6.5) plus two stabilisers.

### 5.1 The update

For a minibatch of transitions `(s, a, r, s', done)` sampled uniformly from the
replay buffer:

```
target  = r                                    if done
        = r + gamma * max_a' Q_target(s', a')  otherwise

loss    = Huber( Q_online(s, a),  target )
```

`target` is computed under `torch.no_grad()`. Gradients are clipped to
`grad_clip_norm` before `optimizer.step()`. `gamma` is `common.gamma` (0.99) —
the same discount the tabular learners use, for the same reason recorded in the
config: end-of-shift misses are charged at the very end, so the agent must value
distant consequences.

The `done` semantics are inherited verbatim from `q_learning.py`: `done` means
**terminated**, never merely truncated. This environment terminates at 480 shift
minutes, which is a genuine terminal state — the end-of-shift missed-incident
penalty is charged there — so bootstrapping past it would invent value that does
not exist.

**Style (CONSTRAINTS #14):** the batched `max` and `gather` are tensor operations,
because that is where the batching lives and an explicit Python loop over 64
samples would be slower without being clearer. The *structure* of the update —
target, loss, clip, step — stays a readable sequence of named steps, and the
docstring states the rule in the same form as `q_learning.py` does. The teaching
constraint is that a student can write this function from memory; five tensor
lines with the rule written above them satisfies that better than a loop does.

### 5.2 Target network

A frozen copy of the online network, hard-copied every `target_update_every`
**gradient steps** (not env steps, not episodes). Between copies its parameters do
not move at all — asserted in tests, because a target network that silently
tracks the online network is the classic implementation bug and produces exactly
the instability the ablation is meant to demonstrate.

### 5.3 Timing parameters

| parameter | value | meaning |
|---|---|---|
| `learning_starts` | 1000 | env steps to fill the buffer before the first gradient step |
| `train_freq` | **4** (new key) | one gradient step every 4 env steps |
| `target_update_every` | 1000 | gradient steps between hard copies |
| `batch_size` | 64 | |
| `replay_capacity` | 100 000 | ~2000 episodes of history at 50 steps/episode |

`train_freq: 4` is the decision that makes this phase affordable: it keeps the
20 000-episode budget identical to the tabular learners — so the DQN-vs-tabular
comparison has no episode-budget confound — while cutting wall-clock from 18.3
min/run to ~4.6 min/run. Four is the standard DQN value, not a number chosen to
hit a time target.

### 5.4 Exploration

DQN reads the **same `epsilon:` config block** as the tabular learners
(start 1.0, min 0.05, decay 0.9995 per episode) and decays in `end_episode()`,
matching `TabularAgent.end_episode` exactly.

Deliberate: if DQN and tabular Q-learning differ in results, the difference should
be attributable to representation and approximation, not to one of them having
explored more. Decaying per episode rather than per step is the same D-015 trap
documented for Phase 2 — per-step decay reaches the floor inside a single shift.

`agent.epsilon` remains a plain settable attribute so the greedy-diagnostic
pattern from `train.py` (`epsilon = 0.0`, `learn=False`, restore in `finally`)
transfers unchanged.

### 5.5 Ablation flags

Two booleans on the agent, sourced from `dqn.ablations` in config:

- **`use_replay = False`** — no buffer. Each gradient step trains on a batch
  containing only the single transition just observed. Maximally correlated,
  no reuse; the textbook reason replay exists.
  *The batch size differs from the control (1 vs 64) and the writeup must say so
  explicitly* — the ablation confounds decorrelation with batch size, and that
  confound is acceptable only if it is stated rather than hidden.
- **`use_target_network = False`** — the bootstrap target is computed from the
  online network, so the regression target moves every step.

Implemented as branches inside one class rather than as subclasses. Three
near-identical classes would obscure the two lines that actually differ, which
are the two lines the viva will ask about.

---

## 6. Replay buffer

Hand-written (CONSTRAINTS #7). A fixed-capacity ring buffer over five parallel
numpy arrays (`obs`, `action`, `reward`, `next_obs`, `done`), with a write cursor
and a size counter.

```
push(obs, action, reward, next_obs, done) -> None    # overwrites oldest at capacity
sample(batch_size) -> tuple of 5 arrays              # uniform WITH replacement
__len__() -> int                                     # current fill, not capacity
```

Sampling uses the **agent's own seeded RNG**, so a run reproduces exactly.

Parallel arrays rather than a `deque` of tuples: sampling returns contiguous
arrays ready for `torch.from_numpy` with no per-sample Python work, and the
capacity semantics are visible in two integers instead of hidden in a container.

---

## 7. Verification

### 7.1 The tiny-MDP anchor — not required by the roadmap, required by this project

Phase 2's discipline (D-014) was: build a two-state MDP with a pen-and-paper
`q_*` **before** the learner, so that a disagreement is unambiguously the
learner's fault. `tiny_mdp.py` and its hand-derived
`q_* = [[10.0, 6.7], [10.7, 13.0]]` already exist and are verified to a Bellman
residual of 1.78e-15.

DQN gets the same anchor: the two states are presented as one-hot 2-vectors,
and the network's learned Q-values are asserted to approach `HAND_COMPUTED_Q`.

**The anchor runs at `tiny_mdp`'s own gamma of 0.9, not the project's 0.99.**
`HAND_COMPUTED_Q` was derived on paper at 0.9 (D-014); testing against it at 0.99
would compare the network to the wrong answer and the failure would look like a
convergence problem. `tiny_mdp.py`'s deliberate divergence from the project
discount is already flagged in HANDOVER as a thing that looks like a CONSTRAINTS
#9 violation and is not.
This is the only test in the phase that can distinguish "the update rule is
correct" from "the update rule is plausible and the environment is too noisy to
tell".

**The tolerance will be set from a measured run, not chosen for plausibility.**
Function approximation will not reach the 9.24e-14 that tabular Q-learning
achieved; the honest procedure is to run it, observe the converged error across
several seeds, and set the threshold above the observed spread — then record the
observed value in the test's docstring so a future regression is visible.
Guessing a tolerance is how this project previously got one wrong by twelve
orders of magnitude.

### 7.2 `tests/test_replay.py`

- capacity is respected; the oldest entry is the one overwritten
- `len()` grows to capacity then stops
- `sample` returns the declared shapes and dtypes
- two buffers with the same seed produce identical samples
- sampling before `learning_starts` worth of data is never requested (the agent's
  guard, asserted from the agent side)

### 7.3 `tests/test_dqn.py`

- `obs_kind == "cont"` and the runner feeds it `featurise` output
- feature scaling is applied, and the scale vector matches `FEATURE_NAMES` order
- **the target network does not move between hard updates** (parameters compared
  before and after a gradient step)
- a hard update makes target and online parameters equal
- gradient clipping fires on an artificially large loss
- epsilon decays once per `end_episode()` call and never below `min`
- `use_replay=False` trains on exactly one transition (batch dimension is 1)
- `use_target_network=False` computes its target from the online network —
  asserted at a single backup, in the spirit of
  `test_update_bootstraps_off_the_max_not_the_behaviour_action`, because a
  converged agent cannot distinguish the two
- the tiny-MDP anchor of §7.1
- two runs with the same seed produce identical parameters

---

## 8. Training entry point — `scripts/train_dqn.py`

Structurally parallel to `scripts/train.py`, and for the same reasons:

1. train on the dedicated `dqn.train_seed_start` block, one fresh shift per episode
2. every `eval_every` episodes, freeze exploration and measure the greedy policy
   on the **train-diagnostic** seeds (1–10) — never on eval seeds
3. repeat over several agent seeds (CONSTRAINTS #3)
4. **only at the very end**, evaluate on the 30 eval seeds

Carried over deliberately from `train.py`:

- the `results/smoke/` guard — a reduced run (`--episodes`, `--repeats < 5`) writes
  to `results/smoke/` so it can never overwrite a full run's artefact. This bit the
  project once (D-018): a 200-episode smoke test silently replaced a
  20 000-episode Q-table and surfaced later as an unexplained coverage drop.
- the note that baseline std is across eval seeds while learner std is across
  training runs, and the two are not comparable.

Artefacts: `results/dqn.pt` (state dict, gitignored — CONSTRAINTS #19),
`results/dqn_curve.png`.

### 8.1 Seed block

`dqn.train_seed_start: 1000000`. Disjoint from train-diagnostic (1–10), eval
(101–130), calibration (1000–3099), DP estimation (10000–59999), q_learning
(200000+), sarsa (400000+), monte_carlo (600000+) and ablations (800000+).
Enforced by the existing distinctness check in `config.py`, extended to include
DQN — not trusted to the YAML comment (CONSTRAINTS #2).

`torch.manual_seed` is set per run alongside the numpy RNG. CPU-only, single
thread, no cuDNN — so bitwise reproducibility is achievable and is asserted.

---

## 9. Comparison and ablations

### 9.1 `scripts/compare_dqn_tabular.py` — roadmap box 5

During evaluation rollouts, log `(discretise(snap), dqn_action)` at every step.
Compare against tabular Q-learning's greedy action on **the same states**.
Report:

- policy agreement over states **actually visited during evaluation**
- the visited-state count, alongside the total of 576
- the standard metric table: recall@deadline, total reward, MTTD on the 30 eval
  seeds

The restriction to visited states is E-011's correction carried forward: the naive
"agreement over all 576 states" figure of 83–86% was manufactured almost entirely
by states neither agent had ever seen, where both fall back to the same
tie-breaking convention. Agreement over commonly-visited states was 22–44%.

**Rejected:** projecting the DQN onto all 576 buckets by evaluating a
representative feature vector per bucket. There is no true inverse of
`featurise`, so the "representative vector" would be an invention, and this
project already has one retraction (E-013) caused by reading structure into a
figure built on 13 states.

### 9.2 `scripts/dqn_ablations.py` — roadmap box 6

Three configurations — control, `no_replay`, `no_target_network` — each at the
full 20 000-episode budget, 3 repeats each, plotted on one axis. Same structure
as the existing `scripts/ablations.py`, self-contained in the same way.

Ablation training seeds come from their own block, distinct from
`dqn.train_seed_start`, placed in config rather than hardcoded in the script.

**What counts as a result here:** the roadmap asks the ablations to "visibly
destabilise training". Phase 2's ablation study (E-012) found that none of α, γ
or ε-decay cleared the noise floor and reported that as a negative result rather
than filling it in. The same standard applies: if replay-off and target-off do
**not** visibly destabilise, that gets reported as a negative result, and the
first thing to check is whether the implementation is actually disabling what it
claims to disable (which is what the single-backup tests in §7.3 exist to rule
out).

---

## 10. Exit criterion, and the risk to it

**Roadmap exit criterion, unchanged:** DQN matches or beats tabular Q-learning on
the same evaluation seeds, and the two ablations visibly destabilise training in
the plots.

**Recorded risk, before any measurement.** On the 30-seed block, tabular
Q-learning scores 47.6 ± 52.0 and severity_sort 40.4 ± 220.1. The bar DQN must
match sits well inside its own spread. A DQN result of, say, 60 ± 55 would satisfy
"matches or beats" on the mean while being statistically indistinguishable, and
writing that down as a pass would be exactly the error E-014 caught the project
making at 5 seeds.

Therefore, when the numbers arrive: **compare the difference to the spread before
declaring the gate met** (the standing lesson of E-014), and state plainly whether
"matches" means "is genuinely comparable" or merely "has a larger mean".

Consistent with D-012 and D-020, this criterion is **not** pre-emptively weakened.
It is run first and decided by a human with real numbers in hand.

---

## 11. Commit plan

Seven commits, each one logical change, each test-first (CONSTRAINTS #17, #24):

1. `phase3: DQN config section — loader, validation, feature scales`
2. `phase3: hand-written uniform replay buffer`
3. `phase3: Q-network and DQN agent — target net, Huber loss, gradient clipping`
4. `phase3: DQN training entry point`
5. `phase3: DQN vs tabular Q-learning policy comparison`
6. `phase3: replay-off and target-network-off ablations`
7. `phase3: documentation — FEATURE_007, EXPLAIN, FLOW, ARCHITECTURE, HANDOVER`

At session start the balance was Diya 17 / Pranav 9 (gap 8, IMBALANCED,
CONSTRAINTS #26). Seven genuine commits on Pranav's machine brings the gap to
roughly 1. No commit here is padding — the decomposition is the one the work has
anyway.

---

## 12. Compute budget

| run | runs | cost |
|---|---|---|
| main training, 20 000 episodes × 5 repeats | 5 | ~23 min |
| ablations, 3 configs (control + 2) × 3 repeats | 9 | ~41 min |
| **total** | **14** | **~64 min** |

The ablation script re-runs the control configuration rather than reusing the
main run's curves, because a three-way plot is only readable if all three lines
have the same repeat count and the same seed block. That duplication costs ~14
minutes and is the price of a comparison that is not quietly unmatched.

Derived from the measured 1.107 ms/gradient step and ~50 env steps/episode at
`train_freq: 4`. Each run over ~10 minutes is confirmed with a human before
launch (CLAUDE.md "Ask the humans before") and logged in
`docs/experiments/EXPERIMENT_LOG.md`.

---

## 13. Decisions requiring a `DECISIONS.md` entry

- **D-023** — input scaling by fixed domain divisors, held in
  `training_default.yaml` rather than `env_default.yaml`, to avoid changing
  `config_hash` and orphaning every prior result (§3.1). Alternative rejected:
  running mean/std normaliser (§3.2).
- **D-024** — `train_freq: 4` so the episode budget can match the tabular
  learners exactly rather than being cut to fit the clock (§5.3).
- **D-025** — a separate `scripts/train_dqn.py` rather than extending
  `scripts/train.py`; shared harness deferred to Phase 4 under rule-of-three
  (§4.1).
- **D-026** — "replay off" means batch-of-one on the latest transition, with the
  batch-size confound stated rather than hidden (§5.5).
