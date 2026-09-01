# BUG_004 — both Phase 4 trainers reported a single training run as the result

**Status:** fixed
**Severity:** wrong-results
**Phase:** 4 · **Found:** 2026-09-01 · **Fixed:** 2026-09-01
**Model(s) used:** Claude Opus 5

---

## Symptom

`scripts/train_reinforce.py` and `scripts/train_actor_critic.py` trained `--repeats N`
agents, then evaluated **one** of them — whichever happened to be last:

```python
for repeat_index in repeats:
    agent, ... = train_one_run(...)
    final_agent = agent            # <- overwritten every iteration
...
eval_records = run_episodes(env, _GreedyView(final_agent), tuple(cfg.seeds.eval), ...)
payload = {..., "eval_summary": summary}
```

The other N−1 trained agents were saved to disk and never evaluated. The number written
to `eval_summary` was therefore one training run, and its `std` described **variation
across the 30 eval seeds**, not variation across training runs.

Nothing errored. The output looked exactly like every other phase's output.

## Why it matters

CONSTRAINTS #3: *"Never report a single run. Every headline number is mean ± std over at
least 5 seeds."* Every other phase already honoured this — `train.py` builds
`per_run_summaries` and aggregates with `across_runs`, and `aggregate_dqn.py` does the
same for the DQN sweep. Phase 4 was the only phase that did not, and there is **no Phase 4
aggregation script** that would have repaired it downstream: `compare_sample_efficiency.py`
reads `episode_rewards` / `episode_steps` / `curves`, never `eval_summary`. A search of
`scripts/`, `src/` and `tests/` found `eval_summary` written in exactly two places and read
in none, so the defect would have surfaced only when a human quoted the number.

D-036 made it urgent: it designates the Phase 4 eval number the phase's **headline**, so
shipping the sampled-evaluation path onto this code would have produced an inadmissible
headline with a spread that meant the wrong thing.

## How large is the error

Measured on a deliberately tiny smoke run (3 repeats × 20 episodes) — the budget is far too
small to learn anything, which is the point: it shows the spread between runs, not a result.

```
  repeat 0: recall 0.1568  reward   -665.3
  repeat 1: recall 0.0000  reward   -520.5
  repeat 2: recall 0.8443  reward     47.5     <- the only one the old code evaluated

  ACROSS 3 RUNS (this is the reportable number):
    recall_at_deadline: 0.3337 +- 0.3667  (over 3 run(s))
    total_reward:    -379.4480 +- 307.6071  (over 3 run(s))
```

The old code would have reported **recall 0.8443, reward +47.5** — the best of the three,
by luck of iteration order — where the honest aggregate is **0.3337 ± 0.3667** and
**−379.4 ± 307.6**. The reported value was not merely noisy; it sat outside one standard
deviation of the mean, and it was the single most favourable run available.

This is the project's own recurring lesson in a new place: E-014 found a headline number
with the wrong sign because a ±218 spread went unread. Here the spread was not merely
unread — it was never computed.

## Fix

Both trainers now keep every trained agent, evaluate all of them on the eval seeds, and
report `eval_across_runs` (mean, std and `n_runs` per metric) alongside `eval_per_run`.
The single-run `eval_summary` key is gone rather than kept as an alias, because a key that
name would still read as "the" summary.

Three details worth keeping:

1. **`None` is preserved, not zeroed.** `summarise` reports `mttd_min` as `None` when a run
   caught no incidents. Such runs are dropped from that metric's average and the metric
   reads `undefined` if none survive. This path is load-bearing, not padding: E-020's
   collapsed actor-critic runs scored recall 0.0000, and the smoke run above exercised it
   (`mttd_min` aggregated over 2 of 3 runs).
2. **A run count below `MIN_RUNS_TO_REPORT` prints a refusal**, not a silent number. It is
   not an error — `--only-repeat` is the parallel-sweep pattern (D-027) and those per-run
   files get aggregated later — but a one-run `std` of 0.00 must never be quoted as a
   measured spread.
3. **`MIN_RUNS_TO_REPORT = 5` lives in `evaluation/metrics.py`**, not in `config/`. It is a
   protocol floor from CONSTRAINTS #3 rather than a tunable, the same reasoning that puts
   `MIN_EVAL_SEEDS = 30` in `tests/test_eval_protocol.py`.

## Verification

- `.\.venv\Scripts\python.exe -m pytest tests/ -q` → **191 passed** (114.34 s)
- REINFORCE smoke `--episodes 20 --repeats 3 --no-plot` → three per-repeat lines, an
  across-runs block, and the under-count warning; output quoted above
- actor-critic smoke `--episodes 10 --repeats 3 --no-plot` → same shape, and exercised the
  `mttd_min: undefined on all 3 runs` branch

## Still open, found while fixing this

**Neither Phase 4 trainer has a smoke-run guard.** `train_reinforce.py` writes to
`results/reinforce_runs/<tag>/` and `train_actor_critic.py` to
`results/actor_critic_runs/actor_critic/` regardless of `--episodes`, so a 20-episode
smoke run lands exactly where a 20,000-episode run's file goes. That is the D-018 trap the
tabular trainers avoid via `results/smoke/`, and the same one HANDOVER warns about for the
DQN ("delete `results/dqn_runs/` after any smoke test"). The smoke artefacts from this fix
were deleted by hand. **Guard this before the full Phase 4 runs**, because a stale smoke
file averaged into a real sweep is silent and the file looks entirely valid.
