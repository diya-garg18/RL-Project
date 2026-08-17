# FLOW.md — How execution travels

> Field Guide habit #4. Bugs live in the gaps between files. This traces what calls what, in what order.
>
> **Status:** Flows B, C and C2 are built and verified. Flow A is partly built (env, runner, encoders — `scripts/train.py` still to come). Flows D and E are still planned. Each flow is marked ⬜ (not built) or ✅ (built and verified). Update the marker and correct the detail the moment a flow becomes real.

**Last updated:** 2026-08-16

---

## Flow A — One training run ✅ *(built & verified 2026-08-16 — FEATURE_004, E-008; entry point `scripts/train.py`)*

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

### Flow C3 — the Phase 2 correctness anchor (E-006) ✅ *(built & verified 2026-08-16 — FEATURE_002)*

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

- **The scaffold's `actions:` YAML block was structurally invalid** (a sequence and a `bulk_close:` mapping key at the same indent level). PyYAML rejected the whole file on the loader's very first run. Action names now live under `actions.names`. Lesson: parse configs before trusting them — the docs looked fine for a whole session while the YAML was unloadable. (2026-08-13)
- **Never rewrite a source file through PowerShell's file cmdlets.** `Set-Content -Encoding utf8` on `scripts/train.py` re-encoded every non-ASCII character (each em dash became `â€”`) and prepended a BOM. Python still parsed it, so nothing failed — the damage was visible only on reading the file. Repaired by targeted replacement of the mangled sequences. Use the editor for source edits; reserve PowerShell for running things. (2026-08-16)
- **A reduced training run used to overwrite a full one's artefacts.** `--episodes 200` silently replaced a completed 20000-episode Q-table with a valid, correctly-shaped, wrong file. It surfaced later as unexplained coverage loss (121 states → 81) in `compare_agents.py` and first looked like a bug there. Fixed structurally: reduced runs now write to `results/smoke/` (D-018). (2026-08-16)
- **`python -c` with a PowerShell here-string loses its inner quotes.** Passing a multi-line Python snippet to `python -c` via `@'...'@` had PowerShell's native-command argument parsing strip the double quotes, so Python received `print(residual,` and died on an unclosed paren five lines in. The error points at the Python, but the bug is in the shell. Fix: write throwaway scripts to a file and run the file. This is the second quoting-related tooling failure in this project (the first being the stray zero-byte files created when written content contains a `>`), which is enough of a pattern to stop using inline shell snippets for anything non-trivial. (2026-08-16)
- **Calibration lives outside the flows above.** `scripts/calibrate_generator.py` calls `generator.generate_shift` directly — no env, no agent — on its own seed block (1000+), disjoint from train (1–10) and eval (101–105) seeds. Built and verified ✅. (2026-08-13)
