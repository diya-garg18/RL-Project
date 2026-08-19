# FEATURE_007 — Deep Q-Network over the continuous state

**Phase:** 3
**Status:** built and unit-tested. **First sweep FAILED (E-016) and is kept as a negative result; corrected sweep running.** The exit criterion is still unmeasured.
**Date started:** 2026-08-18
**Model:** Claude Opus 5

---

## What and why

Phases 1 and 2 gave every learner the same 576-bucket view of the world from `state.discretise()`. That view throws away most of what an analyst would actually look at: the queue's *composition*, the spread of ages, how much verification cost is sitting in front of them. Phase 3 replaces the table with a neural network reading `state.featurise()` — 17 continuous numbers — to find out whether the discarded detail is worth anything.

That is the honest framing. The DQN is not here because it is more advanced; it is here to answer a question the discretisation raises.

## Roadmap link

Phase 3, all six boxes: `agents/dqn.py`, a hand-written replay buffer, a target network with periodic hard update, ε-greedy decay + gradient clipping + Huber loss, the DQN-vs-tabular comparison, and the two required ablations.

**Exit criterion:** "DQN matches or beats tabular Q-learning on the same evaluation seeds, and the two ablations visibly destabilise training in the plots."

## Approach

The algorithm is the §6.5 Q-learning backup with the table replaced by a function approximator, plus the two devices that make that replacement stable:

```
tabular:  Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
DQN:      minimise ( r + gamma * max_a' Q_target(s',a') - Q_online(s,a) )^2
```

The bracketed TD error is the same in both. What changed is that the step no longer moves one table cell — it moves every weight, and so moves the estimate for every state at once. That generalisation is the entire reason to do this and the entire reason it is unstable without help.

- **Experience replay** (`agents/replay.py`) — a hand-written ring buffer of five parallel numpy arrays. Breaks the correlation between consecutive transitions within a shift, and lets each transition be learned from more than once.
- **Target network** — a frozen copy refreshed every `target_update_every` *gradient* steps (not environment steps; with `train_freq: 4` these differ by 4×). Without it the regression target moves every time the weights move, and the network chases a value it is itself changing.
- **Input scaling** — fixed domain divisors (D-023). Measured spread across the 17 columns is 470×, so this is a correctness requirement, not a tweak.
- **Huber loss and gradient clipping** — this environment's missed-incident penalty produces occasional large negative rewards, which MSE would square into a single step that wrecks the network.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Running mean/std input normalisation | Statistics drift during training, so a state's encoding differs between episode 1 and episode 20000, and between training and evaluation. Fixed domain constants are stable and explainable. |
| Scale divisors in `env_default.yaml` | That file is content-hashed into every `EpisodeRecord` since Phase 0. Adding a key orphans every prior result from the config that produced it. (D-023) |
| Extend `scripts/train.py` with a `--agent dqn` branch | Puts Phase 3 code on the path of every Phase 2 reproduction, so a Phase 3 bug could break a Phase 2 rerun. Rule of three: extract the shared harness when a third trainer appears. (D-025) |
| A deque of transition tuples for the buffer | Parallel arrays return contiguous batches ready for `torch.from_numpy` with no per-sample Python work, and the capacity semantics are two printable integers rather than container behaviour. |
| Repeat the latest transition 64× under `no_replay` | Equalises batch size but produces a gradient identical to a batch of one scaled by 64 — it changes the effective learning rate and nothing else. (D-026) |

## Files touched

| File | Change |
|---|---|
| `src/soc_triage/agents/dqn.py` | new — `QNetwork`, `DQNAgent` |
| `src/soc_triage/agents/replay.py` | new — `ReplayBuffer` |
| `src/soc_triage/state.py` | added `feature_scale_vector()` |
| `src/soc_triage/config.py` | added `DQNConfig`, its loader section and validation |
| `config/training_default.yaml` | filled in the `dqn:` block; added `train_freq`, seed blocks, `feature_scales` |
| `scripts/train_dqn.py` | new — trainer, plus `--only-repeat` for parallel execution |
| `scripts/run_dqn_sweep.py` | new — parallel scheduler with a memory ceiling |
| `scripts/aggregate_dqn.py` | new — combines parallel repeats, with comparability guards |
| `scripts/compare_dqn_tabular.py` | new — policy agreement + paired metric comparison |
| `scripts/dqn_ablations.py` | new — the three instability measures |
| `tests/test_dqn_config.py`, `test_replay.py`, `test_dqn.py`, `test_dqn_analysis.py` | new — 46 tests |

Nothing in Phases 0–2 was modified. Every edit to an existing file is additive.

## What was tried that didn't work

**The compute budget was wrong by 15×, and the error shaped the design.** A pre-design probe measured 1.107 ms per gradient step and 0.056 ms per forward pass. Against a real training loop those are **9.87 ms** and **0.709 ms** — the probe had timed a fragment (no optimiser step, no clipping, no per-step action selection) and the fragment was reported as the whole. A 20,000-episode run is ~68 minutes, not the 4.6 the spec claimed. `torch.set_num_threads(1)` was ruled out as the cause — it is the fastest of 1/4/8 threads here — and a breakdown found no hotspot, just per-op overhead on a 19,461-parameter net. `train_freq: 4` survives on its *comparison* argument; only its affordability argument was false. (D-024)

**The tiny-MDP anchor looked like a failure and was not.** At 60 episodes, max |Q − q*| was 0.42–0.43 across three seeds. Every entry was low by a near-identical 0.43 — a *uniform* offset, which is what an unconverged bootstrap looks like and not what a wrong backup looks like. Measuring the error against episode count settled it: 0.429 → 0.017 → 0.006 at 60 / 120 / 250 episodes, a 25× collapse rather than a plateau at a floor. Had a tolerance of 0.5 been guessed up front, the test would have passed at 60 episodes and the distinction would never have been drawn.

**Two guards had to be written after the failure they prevent nearly happened.** A 40-episode smoke test left result files that an existence-check would have skipped, silently mixing them into a 20,000-episode sweep. And the first version of the sweep's memory guard was expressed as a floor on free memory ("keep 0.9 GB free"), which would have let it fill the machine to 94% — the exact limit it existed to enforce.

**`summarise()` returns `None` for MTTD when nothing was caught.** The first trainer assumed a float and crashed. That path is load-bearing rather than defensive: an agent that catches nothing is precisely what a destabilised ablation looks like.

## How it was verified

| Check | Command | Result |
|---|---|---|
| Full suite | `.\.venv\Scripts\python.exe -m pytest tests/ -q` | **122 passed** |
| Config validation (11) | `pytest tests/test_dqn_config.py -q` | 11 passed |
| Replay buffer (8) | `pytest tests/test_replay.py -q` | 8 passed |
| Agent + ablations + anchor (13) | `pytest tests/test_dqn.py -q` | 13 passed |
| Analysis guards (14) | `pytest tests/test_dqn_analysis.py -q` | 14 passed |
| Tiny-MDP anchor | inside `test_dqn.py` | max \|Q − q*\| < 0.05, measured 0.013–0.017 at 120 episodes |
| Trainer end-to-end | `train_dqn.py --episodes 40 --repeats 1 --no-plot` | ran, wrote to `results/smoke/` |
| Parallel mode | `train_dqn.py --only-repeat 3 --episodes 40` | `seed_base 1000120`, matching the sequential path exactly |
| Sweep scheduler | `run_dqn_sweep.py --control-runs 2 --ablation-runs 1 --episodes 40` | 4 runs, correct seed blocks and ablation flags |

Both stabilisers and both ablations are pinned at a **single backup**, not by a convergence curve — a target network that silently tracks the online network still trains and still converges on an easy problem, and the only symptom is the instability the ablation is meant to demonstrate.

## What the first sweep found — read before quoting anything above

The 20 completed runs of the first sweep **all collapsed to BULK_CLOSE** (99.4% of actions, recall 0.0086). Cause: `F.huber_loss` was called without a `delta`, so torch's default of 1.0 applied against penalties of -150 to -1499, and a 150x larger error produced a 1.014x larger gradient. See **E-016**, **BUG_002**, **D-029**.

That invalidates one line of the "Approach" section above, which is left in place rather than edited so the reasoning error stays visible: *"Huber loss and gradient clipping — this environment's missed-incident penalty produces occasional large negative rewards, which MSE would square into a single step that wrecks the network."* The premise was right and the conclusion backwards. Those penalties are not outliers to suppress; they are the entire triage signal. The correct statement is that Huber is right **and its delta must be matched to the reward scale** — 200, the largest named single-event penalty in `env_default.yaml`.

## Follow-ups left open

- **The exit criterion is still unmeasured.** The corrected sweep was still running when this was written. Verified only at 3000 episodes x 3 runs: recall 0.48 +- 0.21, still below severity_sort's 0.84.
- **`run_dqn_sweep.py` launches `train_dqn.py` without `-u`**, so a running repeat's `.log` stays empty until it finishes. Progress is only visible in the scheduler log. Worth fixing between sweeps.
- The `no_replay` ablation carries a batch-size confound (D-026) that must appear in any write-up.
- Both ablations draw from the same seed block (1200000). Each is compared against the control, which is on a different block, so the comparison that matters is unconfounded; ablation-vs-ablation is paired as a side effect.
- The `seed_starts` error message in `config.py` still says "eval (101-105)"; the block has been 101–130 since D-019. Left alone deliberately — correcting it inside a Phase 3 commit would mix concerns.
- `config.py` is 640 lines, over the 500-line limit. It was already 539 at Phase 3's start, so the violation predates this work, but Phase 3 made it worse and it is now the only file in the repo over the limit.

## Plain-English summary

Up to now, every agent in this project saw the world through 576 pigeonholes. Two shifts that were meaningfully different — one with a backlog of low-severity noise, one with three critical alerts about to breach — could land in the same pigeonhole and be treated identically.

This feature replaces the pigeonholes with a small neural network that reads seventeen numbers describing the situation directly. The hope is that the extra detail lets it triage better. The honest position is that we do not yet know, because no full training run has happened.

Two tricks make it work at all, and both are switchable off so we can show what they are worth. **Replay** stores past situations and learns from a random mixture of them, instead of only from what just happened — otherwise the network only ever sees one shift's worth of very similar moments in a row. **A target network** is a frozen copy used to say what the future is worth, refreshed occasionally; without it the network is chasing a target it moves every time it learns, like trying to hit a dot that jumps whenever you aim.
