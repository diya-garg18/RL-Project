# FLOW.md — How execution travels

> Field Guide habit #4. Bugs live in the gaps between files. This traces what calls what, in what order.
>
> **Status:** Flows B, C and C2 are built and verified. Flow A is partly built (env, runner, encoders — `scripts/train.py` still to come). Flows D and E are still planned. Each flow is marked ⬜ (not built) or ✅ (built and verified). Update the marker and correct the detail the moment a flow becomes real.

**Last updated:** 2026-08-17

---

## Flow A — One training run ✅ *(built & verified 2026-08-17 — FEATURE_004, E-008; entry point `scripts/train.py`)*

**Three seed blocks, three different jobs — the part of this flow most worth getting right:**

```
scripts/train.py
  └─ for repeat in 0..4:                       ← 5 runs; one run is not a result
       ├─ agent = QLearningAgent(seed=repeat)
       ├─ seed_base = q_learning.train_seed_start + repeat * n_episodes   (200000+, D-016)
       ├─ for episode in 0..n_episodes-1:
       │    ├─ runner.run_episode(env, agent, seed_base + episode, learn=True)
       │    │      └─ ONE FRESH SHIFT PER EPISODE — never a replayed seed
       │    ├─ keep only outcome.total_reward   ← 100k full EpisodeRecords will not fit in memory
       │    └─ agent.end_episode()              ← THE epsilon decay (D-015). Omit it and
       │                                          epsilon sticks at 1.0, silently, forever.
       └─ every eval_every episodes:
            └─ greedy_diagnostic(seeds 1-10)    ← epsilon pinned to 0, learn=False,
                                                   TRAIN-diagnostic seeds only
  └─ ONCE, at the end: evaluate on seeds 101-105 and print the comparison table
```

`greedy_diagnostic` saves and restores epsilon around the measurement — measuring must not change what is being measured.

**Why the eval seeds are read exactly once, at the end.** Every training decision is already made by then, so nothing in the script can tune against them. Plotting the learning curve against eval seeds would be tuning against them *by eye*, which CONSTRAINTS #2 forbids just as firmly and which leaves no trace in code. Hence the separate train-diagnostic block.

Measured: 2.8 min for 5 × 20,000 episodes. Result in E-008 — **the Phase 2 exit criterion is not met**, and E-008's second finding is that the eval seed block is not representative.

### Flow A (original plan, for reference)

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

## Flow B — Baseline evaluation ✅ *(built & verified 2026-08-14 — E-002; entry point is scripts/run_baselines.py, table lands in results/baselines.md)*

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

## Flow C — Dynamic Programming (Phase 1) ✅ *(built & verified 2026-08-14 — E-004; entry point scripts/run_dp.py; VI/PI agreed 100%)*

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

### Flow C2 — the external correctness check (Phase 1, E-005) ✅ *(built & verified 2026-08-16 — FEATURE_001)*

Flow C's VI/PI cross-check proves the two solvers are *consistent*, not *correct* — they share `greedy_policy` and the same Bellman expression, so a wrong equation would make both agree and both be wrong. This path is the only one that compares the solver to an answer derived outside the code.

```
scripts/run_mrp_example.py                          tests/test_mrp_bellman.py
  └─ mrp_example.expected_rewards(P, r)  ──▶ R(s)          (S&B eq. 3.5)
  ├─ mrp_example.solve_linear(P, R, γ)        ──▶ V   closed form (I − γP)⁻¹R
  ├─ mrp_example.evaluate_iteratively(...)    ──▶ V   iterative backups (§4.1)
  ├─ mrp_example.as_degenerate_mdp(P, R, 5)   ──▶ (P_mdp, R_mdp) shaped (5,5,5)/(5,5)
  │     └─ agents/dp.value_iteration(...)     ──▶ V   THE SHIPPED SOLVER
  └─ compare all of the above against mrp_example.HAND_COMPUTED_V
```

Note the direction of the arrow: `mrp_example.py` depends on `agents/dp.py`, never the reverse. The DP module has no idea this check exists, which is what makes it a check.

All four routes must agree to ~1e-9. Measured disagreement: **7.11e-15**. If this test ever fails, the Bellman backup in `agents/dp.py` has been broken — fix `dp.py`, never the expected values (they came from a human with a pen; see `docs/features/FEATURE_001_mrp_worked_example.md`).

### Flow C3 — the Phase 2 correctness anchor (E-006) ✅ *(built & verified 2026-08-17 — FEATURE_002)*

Flow C2 checks the Bellman backup for **V**. The Phase 2 learners produce **Q**, which C2 cannot check at all — an MRP has no actions. This path establishes a `q_*` derived on paper, and ties it to the solver C2 already validated.

```
tests/test_tiny_mdp.py
  └─ tiny_mdp.bellman_optimality_residual(HAND_COMPUTED_Q, P, R, γ)   (S&B eq. 3.20)
  │     └─ must be 0 everywhere — checks the frozen constants with NO solver involved
  ├─ tiny_mdp.greedy_from_q(HAND_COMPUTED_Q)  ──▶ π*  must equal HAND_COMPUTED_POLICY
  └─ tiny_mdp.pad_actions(P, R, 5)            ──▶ (P_wide, R_wide) shaped (2,5,2)/(2,5)
        └─ agents/dp.value_iteration(...)     ──▶ V   THE SHIPPED SOLVER, already
                                                     trusted via Flow C2
        └─ rebuild Q(s,a) = R(s,a) + γ·P[s,a]@V  and compare to HAND_COMPUTED_Q
```

Same dependency direction as C2: `tiny_mdp.py` depends on `agents/dp.py`, never the reverse.

**Why `pad_actions` duplicates rather than invents.** `dp.value_iteration` loops over its module constant `N_ACTIONS = 5`, so a 2-action MDP must be widened first. The padding repeats the real actions cyclically (0,1,0,1,0), so the `max_a` ranges over copies of genuine actions only and `q_*` is provably unchanged. Padding with zero-reward self-loops would insert a *new* action worth 0 — harmless here where all values are positive, silently wrong in any MDP with negative values.

Measured: Bellman residual on the hand-derived table **1.78e-15**; a deliberately injected 0.1 error moves it to **0.10**, thirteen orders of magnitude above the 1e-12 tolerance. The anchor detects wrong answers.

Every learner enters this flow at the same point — but **not against the same target** (D-017):

```
                              Q-learning (off-policy)  ──▶  HAND_COMPUTED_Q   (q*)
tiny_mdp.step()  ──▶  train  ─┤
                              SARSA / Monte Carlo      ──▶  epsilon_soft_q(ε)  (q_π)
                              (on-policy)                    │
                                                             └─ anchored: at ε=0 it
                                                                reproduces q* exactly
```

On-policy learners evaluate the ε-greedy policy they follow, so their fixed point is `q_π`, not `q*` — at ε = 0.1 the two differ by more than 1.5 on this fixture. Grading SARSA against `HAND_COMPUTED_Q` would mark a correct implementation as broken.

```
tests/test_tabular.py
  └─ for each episode:
       ├─ state = random start        ← so neither state is starved
       ├─ for HORIZON steps:
       │    ├─ action = agent.act(state)                 ← epsilon-greedy
       │    ├─ next_state, reward = tiny_mdp.step(...)
       │    └─ agent.update(state, action, reward, next_state, done=False)
       │                                                   ▲
       │                        TRUNCATION, NOT TERMINATION ┘ — the task is
       │                        continuing; done=True here would teach the
       │                        agent the world ends every 200 steps
       └─ agent.end_episode()          ← the ONLY place epsilon decays (D-015)
  └─ compare agent.Q against HAND_COMPUTED_Q, agent.greedy_policy() against HAND_COMPUTED_POLICY
```

Measured: `max |Q − q*| = 9.24e-14` after 500 episodes; correct policy after 10.

---

## Flow F — DQN training, run in parallel (Phase 3) 🟡 *(built 2026-08-18 — FEATURE_007; **no training result yet**)*

The Phase 2 path ran repeats one after another inside a single process. The DQN
cannot: a 20,000-episode run is ~68 minutes (measured), so 60 runs sequentially
is ~68 hours. Each training process is single-threaded and uses ~301 MB, so the
repeats are run as separate processes instead and combined afterwards.

```
scripts/run_dqn_sweep.py                     the scheduler
   |
   |  for each (condition, repeat_index):
   |     wait until launching one more stays under the memory ceiling
   |     subprocess ->
   |
   +--> scripts/train_dqn.py --only-repeat K [--no-replay | --no-target-network]
   |       |
   |       +-- build_agent()          config -> DQNAgent (weights seeded by K)
   |       +-- train_one_run()        20000 x runner.run_episode(learn=True)
   |       |      |
   |       |      +-- runner._encode()  obs_kind "cont" -> state.featurise()
   |       |      +-- DQNAgent.act()    scale -> online net -> argmax (ties low)
   |       |      +-- DQNAgent.update() push to ReplayBuffer; every train_freq
   |       |      |                     steps: sample, target from the FROZEN
   |       |      |                     net, Huber loss, clip, Adam step; every
   |       |      |                     target_update_every GRADIENT steps:
   |       |      |                     hard-copy online -> target
   |       |      +-- agent.end_episode()   epsilon decay, once per episode
   |       |      +-- greedy_diagnostic()   every eval_every, on TRAIN seeds 1-10
   |       |
   |       +-- run_episodes(cfg.seeds.eval)   ONCE, at the very end
   |       +-- writes results/dqn_runs/<tag>/repeat<K>.json  (+ .pt, .log)
   |
   v
scripts/aggregate_dqn.py     refuses to average runs that disagree on episode
   |                         count, config hash or ablation flags; reports the
   |                         curve, the metric table, the SEM, and whether the
   |                         learning curve has plateaued
   +--> results/dqn_curve.png

scripts/compare_dqn_tabular.py   policy agreement over VISITED states only,
   |                             plus total reward paired per eval seed
   +--> results/dqn_vs_tabular.md

scripts/dqn_ablations.py         volatility / divergence / drawdown, all three
   |                             defined before looking at the data
   +--> results/dqn_ablations.{png,md}
```

**The three things that make this path different from Flow A, and why:**

1. **`--only-repeat K` writes JSON instead of printing a table.** `seed_base`
   depends only on `repeat_index` and `n_episodes` — never on how many repeats
   are running — so a parallel repeat faces exactly the alert stream it would
   have faced sequentially. That is what makes the two paths comparable rather
   than merely similar, and it also means a sweep can be *extended* later by
   running higher indices.
2. **The scheduler checks memory before every launch, not once at startup.**
   An eight-hour unattended run shares the machine with whatever else is open;
   available memory here fell from 8.0 GB to 3.8 GB in half an hour with no
   training running at all.
3. **Aggregation is a separate step that can refuse.** A directory of JSON files
   looks valid whatever is in it, so the comparability checks live there rather
   than being assumed.

**Ablations follow the identical path** with one flag added, so the only
difference between conditions is the algorithm — never the harness.

---

## Flow G — REINFORCE training (Phase 4) 🟡 *(built 2026-08-23 — FEATURE_008; smoke-tested only, see E-018)*

```
scripts/train_reinforce.py
  └─ load_env_config + load_training_config          config/*.yaml
  └─ build_agent()                                   ReinforceAgent, shared feature scales (D-032)
  └─ for each repeat:
       └─ for each episode:
            └─ runner.run_episode(learn=True)
                 └─ agent.act(obs)                   SAMPLES from softmax pi(.|s) — no argmax
                 └─ agent.update(...)                buffers (s, a, r) only; learns nothing yet
            └─ agent.end_episode()                   <-- THE ENTIRE UPDATE HAPPENS HERE
                 1. G_t by one backwards pass
                 2. subtract baseline v(s_t)         detached; never enters the target
                 3. weight by gamma^t
                 4. policy step on -sum coeff*ln pi
                 5. fit the baseline to those returns
            └─ every eval_every: greedy_diagnostic() TRAIN-diagnostic seeds only
       └─ agent.save()                               results/reinforce_runs/<tag>/
  └─ ONCE, at the very end: run_episodes on eval seeds via _GreedyView
```

**The two things this flow does differently from Flow F (DQN):**

1. **`end_episode()` is not a housekeeping call, it is the algorithm.** In Flow A and Flow F,
   forgetting it leaves epsilon stuck (D-015) — bad, but the agent still learns. Here, forgetting
   it means **nothing is ever learned at all**, because `update()` only buffers.
2. **"Greedy" needs a wrapper.** The DQN went greedy by pinning epsilon to 0. REINFORCE has no
   epsilon — its randomness *is* the policy — so `_GreedyView` reads the argmax of the action
   probabilities without touching the agent's RNG or its buffer. The sampled reward and the greedy
   diagnostic answer two genuinely different questions, and both are logged.

---

## Flow H — Actor-critic training (Phase 4) 🟡 *(built 2026-08-25 — FEATURE_009; smoke-tested only, and the shipped entropy_coef breaks it — see E-020)*

Entry point: `scripts/train_actor_critic.py`. Same skeleton as Flow G, with one structural
difference that is the whole point of the algorithm.

```
main()
 |- load_env_config + load_training_config + config_hash
 |- build_agent(tcfg, seed)          ActorCriticAgent, feature_scales SHARED with DQN/REINFORCE (D-032)
 |- for repeat in repeats:
 |   |- train_one_run(...)
 |   |   |- for episode in range(n_episodes):
 |   |   |   |- run_episode(env, agent, seed_base+episode, learn=True)
 |   |   |   |   |- per STEP: agent.act(obs)       sample from pi, no argmax
 |   |   |   |   |- per STEP: agent.update(...)    <== THE ENTIRE ALGORITHM RUNS HERE
 |   |   |   |                                         delta = r + gamma*v(s') - v(s)
 |   |   |   |                                         critic step, then actor step, then I <- gamma*I
 |   |   |   |- agent.end_episode()                <== only resets I + publishes TD errors
 |   |   |   |- log entropy, td_error_std
 |   |   |   |- every eval_every: greedy_diagnostic() on TRAIN-diag seeds (1-10)
 |   |- agent.save(...)
 |- run_episodes(_GreedyView(final_agent), cfg.seeds.eval)   <== FIRST AND ONLY eval-seed touch
 |- summarise + save_records + JSON + 3-panel PNG
```

**Contrast with Flow G, and it is the thing to notice.** In Flow G (REINFORCE) `agent.update()`
only *buffers* - forget `end_episode()` and nothing trains at all. Here `agent.update()` *is* the
learning, and forgetting `end_episode()` does not stop training; it silently leaves `I` decayed
from the previous shift, so every subsequent episode weights its early steps as though they were
late ones. **Same two method names, opposite failure modes.**

**Two extra tuning flows**, each on its own seed block (D-035), both measured on train-diagnostic
seeds with `_assert_no_eval_seeds` making the separation a runtime failure rather than a comment:

```
scripts/reinforce_clip_experiment.py       -> E-019, 3 clip values x 3 repeats x 1500 eps  (~3.7 min)
scripts/actor_critic_entropy_experiment.py -> E-020, 4 entropy values x 3 repeats x 80 eps (~6.7 min)
```

**And two analysis flows**, which report rather than tune:

```
scripts/variance_demo.py               drives all three agents itself, 3 seeds x 30 eps
                                       -> results/variance_demo/  (E-021, ~3 min, DONE)

scripts/compare_sample_efficiency.py   reads the TRAINERS' artefacts, runs nothing
                                       -> results/sample_efficiency/  (needs the full runs)
```

`compare_sample_efficiency.py` is the only Phase 4 script that touches no environment at all -
it consumes `episode_rewards`, `episode_steps` and `curves` from whatever the trainers left in
`results/`, and refuses (with a message) any artefact predating the `episode_steps` recording
rather than approximating the x-axis.

---

## Flow D — RLHF (Phase 5) 🟡 *(stages 1 and 2 built & tested — FEATURE_011 2026-09-04, FEATURE_012 2026-09-05; stage 3 not started)*

Three separate stages. Don't fuse them; each produces an artefact the next consumes.

**What changed in stage 1 as built, against the sketch below.** The entry point is
`scripts/generate_pairs.py`, not `build_pairs.py`, and it splits in two: generating
the EpisodeRecords needs agents and torch, while building pairs from them needs
neither (D-037). Pairs are written as **JSON files, not into SQLite** — the database
holds labels only, which keeps the irreplaceable data in one small file that is easy
to back up (CONSTRAINTS #19). And the pair set is written as **two** files, because
`pairs.json` must contain no policy name at all (D-038).

```
STAGE 1a — generate the episodes  (needs agents + torch; NOT BUILT YET)
scripts/generate_pairs.py
  └─ load each of the 9 policies in config rlhf.policies
  └─ runner.run_episodes(seeds = pair_seed_start .. +n_pair_seeds)   ← seed block 3000000
  └─ runner.save_records(...)            ──▶ results/rlhf/records/*.json

STAGE 1b — build the pairs  (pure data; no agent, no env, no torch)   ✅ BUILT
rlhf.pairs.load_records(results/rlhf/records/)
  └─ rlhf.pairs.build_pairs(records, policies=..., sampling_seed=...)
       └─ index by (agent_name, seed); refuse a mixed config_hash  (BUG_005)
       └─ eligible seeds = those on which EVERY policy ran
       └─ allocate target_pairs evenly over the 36 policy pairings
       └─ shuffle, number p0000.., seeded side-swap, round-robin double-labelling
       └─ rlhf.summary.summarise_episode(record) per side   ← reward stripped (D-039)
  └─ rlhf.pairs.write_pairs(...)
       ├──▶ results/rlhf/pairs.json      ← the UI reads this. NO policy names.
       └──▶ results/rlhf/pairs_key.json  ← names + `swapped`. Analysis only.

STAGE 2 — collect labels (human in the loop)   ✅ BUILT — FEATURE_012, 2026-09-05

**What changed here as built, against the sketch.** Not `web/labeller` — the page
lives in `src/soc_triage/labelling/`, because `tests/conftest.py` only ever puts
`src/` on the import path (D-... FEATURE_012 §5). The route is `POST /label`, not
`POST /pair/{id}/label`: the pair id travels in the JSON body instead of the URL,
which is what let the answer be sent with a single `fetch()` alongside the timer
elapsed. And `labeller_id` is never a request field at all (D-041) — it is bound
once at launch and the handler closes over it, so nothing in the body can name a
different person.

scripts/label_ui.py --labeller L1
  └─ config.load_training_config(...).rlhf   ← labellers, max_seconds_per_pair, ui_host, ui_port
  └─ labelling.queue.load_pairs(results/rlhf/pairs.json)   ← refuses pairs_key.json by name (D-038)
  └─ labelling.app.create_app(pairs, labeller_id, labellers, db_path, max_seconds)
       └─ uvicorn.run(app, host=rlhf.ui_host, port=rlhf.ui_port)

GET  /
  └─ labelling.queue.LabelQueue(pairs, labeller_id, labellers, answered=store.labels_by(labeller_id))
       └─ .next_pair()  ──▶ the first pair this person has not answered, or None
  └─ labelling.render.render_pair_page(pair, progress, max_seconds)   ← or render_done_page

POST /label   { pair_id, choice: left|right|tie, seconds }
  └─ queue.pair(pair_id)                      ← refuses a pair not assigned to this labeller
  └─ _clean_seconds(seconds, max_seconds)     ← cap re-applied server-side; NaN, negative → None (D-042, D-044)
  └─ rlhf.store.add_label(pair_id, labeller_id, choice, seconds_taken)
       └─ DuplicateLabelError on a refresh replay ──▶ swallowed; the page just moves on
  └─ 50 pairs are deliberately assigned to BOTH labellers ──▶ Cohen's κ (D-040)
       └─ store enforces UNIQUE(pair_id, labeller_id): one opinion per person
scripts/report_kappa.py                   ✅ BUILT
  └─ rlhf.agreement.agreement_between(store, A, B)  ──▶ κ + confusion matrix
       └─ prints "undefined", never a substituted number, when κ is 0/0

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

- **The scaffold's `actions:` YAML block was structurally invalid** (a sequence and a `bulk_close:` mapping key at the same indent level). PyYAML rejected the whole file on the loader's very first run. Action names now live under `actions.names`. Lesson: parse configs before trusting them — the docs looked fine for a whole session while the YAML was unloadable. (2026-08-13)
- **Never rewrite a source file through PowerShell's file cmdlets.** `Set-Content -Encoding utf8` on `scripts/train.py` re-encoded every non-ASCII character (each em dash became `â€”`) and prepended a BOM. Python still parsed it, so nothing failed — the damage was visible only on reading the file. Repaired by targeted replacement of the mangled sequences. Use the editor for source edits; reserve PowerShell for running things. (2026-08-17)
- **A reduced training run used to overwrite a full one's artefacts.** `--episodes 200` silently replaced a completed 20000-episode Q-table with a valid, correctly-shaped, wrong file. It surfaced later as unexplained coverage loss (121 states → 81) in `compare_agents.py` and first looked like a bug there. Fixed structurally: reduced runs now write to `results/smoke/` (D-018). (2026-08-17)
- **`python -c` with a PowerShell here-string loses its inner quotes.** Passing a multi-line Python snippet to `python -c` via `@'...'@` had PowerShell's native-command argument parsing strip the double quotes, so Python received `print(residual,` and died on an unclosed paren five lines in. The error points at the Python, but the bug is in the shell. Fix: write throwaway scripts to a file and run the file. This is the second quoting-related tooling failure in this project (the first being the stray zero-byte files created when written content contains a `>`), which is enough of a pattern to stop using inline shell snippets for anything non-trivial. (2026-08-17)
- **Calibration lives outside the flows above.** `scripts/calibrate_generator.py` calls `generator.generate_shift` directly — no env, no agent — on its own seed block (1000+), disjoint from train (1–10) and eval (101–105) seeds. Built and verified ✅. (2026-08-13)
