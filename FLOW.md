# FLOW.md — How execution travels

> Field Guide habit #4. Bugs live in the gaps between files. This traces what calls what, in what order.
>
> **Status:** planned flows — no code exists yet. Each flow is marked ⬜ (not built) or ✅ (built and verified). Update the marker and correct the detail the moment a flow becomes real.

**Last updated:** 2026-08-13

---

## Flow A — One training run ⬜

The main loop. Everything else is a variation on this.

```
scripts/train.py
  └─ config.load("config/env_default.yaml", "config/training_default.yaml")
  └─ env = SOCTriageEnv(env_config)
  └─ agent = build_agent(name, training_config)
  └─ runner.train(env, agent, n_episodes, seeds)
       │
       └─ for each episode:
            ├─ env.reset(seed)
            │    └─ generator.generate_shift(seed) ──▶ list[Alert]   (pre-generated, released by arrival_time)
            │    └─ queue ← alerts with arrival_time <= 0
            │    └─ returns initial observation
            │
            ├─ loop until done:
            │    ├─ obs = state.discretise(env_state)   OR  state.featurise(env_state)
            │    ├─ action = agent.act(obs)
            │    ├─ next_obs, reward, done, info = env.step(action)
            │    │     ├─ _select_alert(action)          ← the 5 triage rules live here
            │    │     ├─ _advance_clock(verify_cost)
            │    │     ├─ _admit_new_arrivals()          ← alerts whose arrival_time <= now
            │    │     ├─ _compute_reward(...)           ← reward table, brief §3.5
            │    │     └─ _check_termination()           ← clock >= 480, then end-of-shift misses charged
            │    └─ agent.update(obs, action, reward, next_obs, done)
            │
            └─ runner writes EpisodeRecord ──▶ results/runs/<run_id>/
```

**Key ordering detail to get right:** new arrivals are admitted **after** the clock advances but **before** the next observation is built. Get this backwards and the agent sees a stale queue — a bug that would show up as mysteriously poor performance rather than a crash.

**Second detail:** the full shift's alerts are generated up front from the seed and then *released* by arrival time. That makes episodes reproducible and lets two different policies face an identical alert stream — which is what paired comparison (and the RLHF pair builder) depends on.

---

## Flow B — Baseline evaluation ⬜

```
scripts/run_baselines.py
  └─ for agent_name in [random, fifo, severity, cheapest, oracle]:
       └─ for seed in EVAL_SEEDS:            ← disjoint from TRAIN_SEEDS, enforced in config
            └─ runner.evaluate(env, agent, seed)  ──▶ EpisodeRecord
  └─ evaluation.metrics.summarise(records)   ──▶ per-agent metric dict
  └─ evaluation.compare.table(...)           ──▶ results/baselines.md  +  stdout
```

Baselines never call `agent.update()` — they have nothing to learn. The `Agent` base class provides a no-op `update()` so the runner needs no special-casing.

---

## Flow C — Dynamic Programming (Phase 1) ⬜

```
scripts/run_dp.py
  └─ agents/dp.estimate_model(env, n_episodes=50_000, policy=random)
  │      └─ counts[s][a][s'] ++ ,  reward_sum[s][a] += r
  │      └─ normalise ──▶ P_hat (576×5×576), R_hat (576×5)
  │      └─ report coverage: visited states, min/median visit count
  ├─ agents/dp.value_iteration(P_hat, R_hat, gamma, theta)  ──▶ V, policy
  ├─ agents/dp.policy_iteration(P_hat, R_hat, gamma)        ──▶ V, policy  (cross-check: policies should agree)
  └─ runner.evaluate(env, DPAgent(policy), EVAL_SEEDS)
```

`value_iteration` and `policy_iteration` must converge to the same policy. If they don't, one of them is wrong — that disagreement is the test.

---

## Flow D — RLHF (Phase 5) ⬜

Three separate stages. Don't fuse them; each produces an artefact the next consumes.

```
STAGE 1 — build pairs
scripts/build_pairs.py
  └─ load EpisodeRecords from results/runs/
  └─ rlhf.pairs.build(records)
       └─ match episodes sharing the same seed (identical alert stream), different agents
       └─ render each side: action timeline + outcome cards
  └─ rlhf.store.insert_pairs(...)        ──▶ rlhf.db (SQLite)

STAGE 2 — collect labels (human in the loop)
web/labeller  (or scripts/label_cli.py)
  └─ GET  /pair/next        ← unlabelled pair for this labeller
  └─ POST /pair/{id}/label  { choice: left|right|tie, labeller_id, seconds_taken }
       └─ rlhf.store.record_label(...)
  └─ 50 pairs are deliberately served to BOTH labellers ──▶ Cohen's κ

STAGE 3 — train the reward model
scripts/train_reward_model.py
  └─ rlhf.store.load_labels()            ──▶ 80/20 train/held-out split
  └─ rlhf.reward_model.train(...)
       └─ for each pair (τ_A, τ_B, label):
            R_A = Σ r_hat(s,a) over τ_A
            R_B = Σ r_hat(s,a) over τ_B
            loss = BCE( sigmoid(R_A − R_B), label )     ← Bradley–Terry
  └─ save ──▶ results/reward_model.pt
  └─ report train accuracy AND held-out accuracy       ← the gap is an audit signal
```

Then Flow A runs again with `reward_fn = reward_model` instead of the hand-written reward. That is the **only** change — the env exposes a reward-function hook so no learning code needs to know which reward it's optimising.

---

## Flow E — Live dashboard (Phase 6, optional) ⬜

```
Browser (React)
  └─ POST /api/episode/start  { agent, seed }
  │     └─ FastAPI ──▶ runner.step_generator(env, agent)   (a Python generator, one step per call)
  └─ GET  /api/episode/step   (polled, or streamed via SSE)
        └─ returns { queue_snapshot, action_taken, reward_breakdown, justification }
              └─ justification ← Groq/Llama call, templated from `info`, cached per (state, action)
```

The LLM only ever **describes** a decision the RL agent already made. It never influences the decision. Keep that boundary absolute — it's the difference between "RL project with an explanation layer" and "an LLM wrapper", and an examiner will ask.

---

## Gotchas discovered while building

*Append here whenever a flow surprises you. This section is worth more than the diagrams above once the project is real.*

- *(none yet)*
