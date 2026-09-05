# ARCHITECTURE.md — The system map

> Field Guide habit #6. This is the **shape** of the system, not the implementation detail. Read this at the start of every session so you don't re-derive the terrain. Update it whenever a module is added, removed, or its responsibility changes.

**Last updated:** 2026-08-17 — Phases 0, 1 and 2 built (Phase 2 closed unpassed, Phase 1 reopened — see `ROADMAP.md`). Everything in §3 marked as existing has been written and tested; the rest is still planned.

---

## 1. The shape in one picture

```
                        ┌──────────────────────────┐
                        │   config/*.yaml          │
                        │  (all tunable numbers)   │
                        └────────────┬─────────────┘
                                     │ loaded once at startup
                                     ▼
┌───────────────┐         ┌─────────────────────┐         ┌──────────────────┐
│  generator.py │────────▶│      env.py         │◀───────▶│    agents/       │
│ makes alerts  │ stream  │ SOCTriageEnv        │ s,a,r,s'│ random, fifo,    │
│ + hidden truth│         │ reset() / step()    │         │ severity, q_learn│
└───────────────┘         │ Gymnasium-style API │         │ dqn, reinforce...│
                          └──────────┬──────────┘         └────────┬─────────┘
                                     │ trajectory                  │
                                     ▼                             │
                          ┌─────────────────────┐                  │
                          │  runner.py          │◀─────────────────┘
                          │ episode loop,       │
                          │ seeding, logging    │
                          └──────────┬──────────┘
                                     │ EpisodeRecord (JSON)
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
          ┌──────────────┐  ┌────────────────┐  ┌──────────────┐
          │ evaluation/  │  │ rlhf/          │  │  results/    │
          │ metrics,     │  │ pair builder,  │  │ runs, plots, │
          │ baselines,   │  │ reward model,  │  │ q-tables,    │
          │ audit        │  │ label store    │  │ checkpoints  │
          └──────────────┘  └───────┬────────┘  └──────────────┘
                                    │ labelling pairs
                                    ▼
                          ┌─────────────────────┐
                          │  web/  (optional)   │
                          │ FastAPI + React     │
                          │ dashboard + labeller│
                          └─────────────────────┘
```

---

## 2. Modules and their single responsibility

| Module | Responsibility | Must NOT do |
|---|---|---|
| `src/soc_triage/config/` | Load + validate YAML config into typed frozen objects. A package since D-031: `validation.py` (ConfigError + shared checks), `environment.py`, `training.py`, and an `__init__.py` that re-exports every public name so callers import `soc_triage.config` exactly as before | Contain any numbers itself |
| `src/soc_triage/alerts.py` | The `Alert` dataclass and its hidden ground truth | Know about the agent |
| `src/soc_triage/generator.py` | Poisson arrivals, feature sampling, `is_true_incident` labelling | Know about rewards |
| `src/soc_triage/env.py` | The MDP: queue state, `reset()`, `step(action)`, reward, termination | Contain learning logic |
| `src/soc_triage/state.py` | Both state encoders: `discretise()` (576 states) and `featurise()` (~20 floats) | Mutate the environment |
| `src/soc_triage/agents/` | One file per algorithm, each exposing `act(state)` and `update(...)` | Touch the environment internals |
| `src/soc_triage/mrp_example.py` | The hand-solved 5-state MRP that validates the Bellman backup for **V** in `agents/dp.py` against an answer derived outside the code (FEATURE_001, D-013) | Be imported by anything in the main pipeline — it depends on `agents/dp`, never the reverse |
| `src/soc_triage/tiny_mdp.py` | The hand-solved 2-state MDP that validates the Bellman backup for **Q** in the Phase 2 learners against an answer derived outside the code (FEATURE_002, D-014) | Same rule: it is a check, so nothing in the main pipeline may depend on it |
| `src/soc_triage/agents/tabular.py` | `TabularAgent` — the Q-table, epsilon-greedy behaviour, per-episode decay, tie-breaking argmax and visit counts shared by all three learners (FEATURE_006). Extracted at the third implementation so each agent file is essentially just its update rule | Live in `agents/base.py` — baselines have no Q-table and no exploration schedule, and widening that interface would push dead members onto five agents |
| `src/soc_triage/agents/q_learning.py` | Off-policy TD control, S&B §6.5 (FEATURE_003, D-015). Converges to q\* | Know which environment it is in, or decay epsilon on its own — the episode boundary must be stated by the training loop |
| `src/soc_triage/agents/sarsa.py` | On-policy TD control, S&B §6.4 (FEATURE_006). Selects `a'` during `update` and commits to returning it from the next `act()` — the on-policy property depends on that commitment | Re-sample in `act()` when a commitment is outstanding; that silently produces an off-policy hybrid that still converges |
| `src/soc_triage/agents/monte_carlo.py` | First-visit MC control, S&B §5.4 (FEATURE_006). Buffers the episode in `update` and does all learning in `end_episode` | Learn anything mid-episode, or run on a task that never terminates |
| `src/soc_triage/agents/replay.py` | `ReplayBuffer` — a hand-written uniform ring buffer of five parallel numpy arrays (FEATURE_007). Breaks correlation between consecutive transitions and lets each be reused | Know anything about DQN, agents or the environment — it stores arrays and returns batches |
| `src/soc_triage/agents/dqn.py` | `QNetwork` + `DQNAgent` — Q-learning with an MLP over the 17-dim continuous state, plus replay and a target network, both switchable off for the required ablations (FEATURE_007, D-023 to D-026) | Decay epsilon on its own, or count `target_update_every` in environment steps — it counts GRADIENT steps, and with `train_freq: 4` the two differ fourfold |
| `src/soc_triage/agents/reinforce.py` | `ReinforceAgent` — Monte Carlo policy gradient over the same 17-dim state, with an optional learned value baseline (FEATURE_008, S&B 13.3/13.4). The first agent that learns a policy directly: no argmax in `act()` | Contain an epsilon schedule — it explores by sampling its own policy — or bootstrap the baseline, which would make it an actor-critic |
| `src/soc_triage/runner.py` | Run N episodes with a given agent + seed, emit `EpisodeRecord`s | Compute metrics |
| `src/soc_triage/evaluation/` | Metrics, baseline comparison tables, plots, the reward-hacking audit | Train anything |
| `src/soc_triage/rlhf/` | Phase 5a preference collection (FEATURE_011): `summary.py` renders an EpisodeRecord for a human to judge, `pairs.py` builds blinded same-seed pairs, `store.py` is the SQLite the labels land in, `agreement.py` is Cohen's kappa. `reward_model.py` (Bradley–Terry) is still to come in 5b. **Imports no agent, no env and no torch** — it reads EpisodeRecord dicts off disk, which is what keeps it runnable on a clone with no `results/` (D-037) | Show a labeller any reward number (D-039), or let a policy name reach `pairs.json` (D-038) |
| `src/soc_triage/labelling/` | The preference-labelling web page (FEATURE_012, Diya's box, brief §9): `queue.py` assigns and resumes, `render.py` draws the two panes as HTML, `app.py` is the two FastAPI routes. A **consumer** of `rlhf` only — imports `rlhf.store` and reads what `rlhf.pairs` writes; `rlhf` imports nothing back. Lives here rather than under `web/` because `tests/conftest.py` only ever puts `src/` on the path, and `web/` stays reserved for the Phase 6 React dashboard | Read `pairs_key.json` (D-038), let a request body set its own `labeller_id` (D-041), or trust a client-reported timer value past the cap without re-checking it server-side (D-042) |
| `web/` | React dashboard (Phase 6, not built) | Contain any RL logic, or duplicate the labelling UI — that already exists under `labelling/` |

**The rule this table encodes:** the environment never knows which agent is acting, and the agents never reach inside the environment. They communicate only through `(state, action, reward, next_state, done, info)`. Keep that boundary clean and every algorithm becomes swappable — which is the entire point, since we're implementing six of them.

---

## 3. Directory layout

```
RLPROJECT/                     ← D:\RLPROJECT on the current device
├─ README.md                   ← start here
├─ CLAUDE.md                   ← rules for the AI session (read first, every session)
├─ PROJECT_BRIEF.md            ← the idea, the MDP, the plan
├─ EXPLAIN.md                  ← plain-English living explanation (UPDATE EVERY SESSION)
├─ ROADMAP.md                  ← phase-by-phase task list
├─ ARCHITECTURE.md             ← this file
├─ CONSTRAINTS.md              ← what the AI must never do
├─ FLOW.md                     ← how execution travels
├─ HANDOVER.md                 ← where things stand right now (UPDATE EVERY SESSION)
├─ DECISIONS.md                ← why each choice was made (APPEND-ONLY)
├─ TEST_CHECKLIST.md           ← what "done" means
├─ ROLLBACK.md                 ← how to undo
├─ INTERVIEW_PREP.md           ← the functions you must know cold
├─ requirements.txt
├─ .gitignore
├─ config/
│   ├─ env_default.yaml        ← every environment number lives here
│   └─ training_default.yaml   ← every hyperparameter lives here
├─ docs/
│   ├─ features/               ← one FEATURE_xxx.md per feature (template inside)
│   ├─ bugs/                   ← one BUG_xxx.md per bug (template inside)
│   └─ experiments/
│       └─ EXPERIMENT_LOG.md   ← every training run: config, seed, result
├─ src/soc_triage/
│   ├─ __init__.py
│   ├─ config/
│   │   ├─ validation.py
│   │   ├─ environment.py
│   │   └─ training.py
│   ├─ alerts.py
│   ├─ generator.py
│   ├─ env.py
│   ├─ state.py
│   ├─ runner.py
│   ├─ mrp_example.py          ← hand-solved 5-state MRP; external check on V (dp.py)
│   ├─ tiny_mdp.py             ← hand-solved 2-state MDP; external check on Q (learners)
│   ├─ agents/
│   │   ├─ base.py             ← the Agent interface everything implements
│   │   ├─ baselines.py        ← random, fifo, severity, cheapest, oracle
│   │   ├─ dp.py               ← model estimation + value/policy iteration
│   │   ├─ monte_carlo.py
│   │   ├─ sarsa.py
│   │   ├─ q_learning.py
│   │   ├─ dqn.py
│   │   ├─ reinforce.py         ← Phase 4, built
│   │   └─ actor_critic.py
│   ├─ evaluation/
│   │   ├─ metrics.py
│   │   ├─ compare.py
│   │   ├─ plots.py
│   │   └─ audit.py            ← the reward-hacking experiments
│   ├─ rlhf/                    ← Phase 5a, FEATURE_011
│   │   ├─ summary.py          ← EpisodeRecord rendered for a human; no reward shown
│   │   ├─ pairs.py            ← blinded same-seed pairs from EpisodeRecords
│   │   ├─ store.py            ← SQLite label storage (gitignored, irreplaceable)
│   │   ├─ agreement.py        ← Cohen's kappa, by hand
│   │   └─ reward_model.py     ← Bradley–Terry MLP (5b, not built)
│   └─ labelling/               ← Phase 5a, FEATURE_012, Diya's box
│       ├─ queue.py             ← assignment (D-040), resume, progress
│       ├─ render.py            ← the two panes as HTML, both blinding guards
│       └─ app.py               ← GET / and POST /label
├─ tests/
├─ scripts/                    ← thin CLI entry points, no logic
│   ├─ generate_pairs.py       ← --survey measures which repeats collapse; --write
│   │                             builds the 300 pairs. The ONE place allowed to
│   │                             need both torch and results/ (FEATURE_011 §3)
│   └─ label_ui.py             ← serves labelling/app.py for one --labeller
├─ results/                    ← gitignored except .gitkeep
└─ web/                        ← React dashboard (Phase 6, not built)
```

---

## 4. Key data contracts

These three shapes are the spine of the system. Changing any of them is a **DECISIONS.md-worthy** event.

### `Alert`
```python
@dataclass(frozen=True)
class Alert:
    id: int
    arrival_time: float          # minutes into the shift
    severity: int                # 0..3  — vendor label, deliberately noisy
    asset_criticality: int       # 0..2
    verify_cost_min: int         # 5 | 10 | 20 | 40
    alert_type: str              # one of 6
    is_true_incident: bool       # HIDDEN — never passed to the agent
    deadline_min: float          # dwell budget; only meaningful if true incident
```

> `is_true_incident` and `deadline_min` are **ground truth**. They live on the object because the environment needs them to compute reward, but `state.py` must never encode them into an observation. This is the single easiest way to accidentally cheat, so there is a test for it (`test_no_ground_truth_leakage`).

### `StepInfo` (the `info` dict from `env.step`)
```python
{
  "action_name": str,
  "alert_investigated": Alert | None,
  "was_true_incident": bool | None,
  "delay_min": float | None,
  "bulk_closed": list[Alert],
  "time_consumed": float,
  "reward_breakdown": dict[str, float],   # for debugging + the dashboard
}
```

### `EpisodeRecord`
```python
{
  "run_id": str, "agent_name": str, "seed": int, "config_hash": str,
  "steps": [ {state_disc, state_cont, action, reward, info} ],
  "outcome": {
      "incidents_total": int, "incidents_caught": int,
      "incidents_missed": int, "critical_missed": int,
      "mttd_min": float, "wasted_minutes": float,
      "total_reward": float
  }
}
```
`EpisodeRecord` is what evaluation, the RLHF pair builder, and the dashboard all consume. Write it to JSON. It is the interchange format of the whole project.

---

## 4b. Phase 4 agents

Both Phase 4 learners are policy-gradient methods over the 17-dim continuous state, and they
share `_mlp` (defined in `reinforce.py`) so that a network built for one is literally the same
constructor as a network built for the other.

| module | responsibility | learns when |
|---|---|---|
| `agents/reinforce.py` | Monte Carlo policy gradient + optional learned baseline (S&B 13.3/13.4) | `end_episode()` - the whole algorithm |
| `agents/actor_critic.py` | one-step actor-critic, bootstrapped (S&B 13.5) | `update()` - every step; `end_episode()` only resets `I` |

**The seam between them is one term, not one file.** REINFORCE's update coefficient is
`G_t - b(s_t)` (the observed return); the actor-critic's is `r + gamma*v(s') - v(s)` (the
successor's estimate). Two networks is NOT the distinction - `reinforce.py` has two as well.
See D-034 and FEATURE_009.

Each has its own trainer, per D-025: `scripts/train_reinforce.py` carries a `--no-baseline`
ablation the other has no use for, and `scripts/train_actor_critic.py` logs TD-error spread and
policy entropy, which REINFORCE has no equivalent of.

Two tuning harnesses sit beside them, each with its own seed block so a tuning run can never
share alert streams with a reported one (D-035):
`scripts/reinforce_clip_experiment.py` (E-019) and
`scripts/actor_critic_entropy_experiment.py` (E-020).

Two **analysis** scripts complete the phase, and the distinction from the harnesses above is
worth keeping straight: the harnesses *choose a setting*, these *report a result*.

| script | box | reads | status |
|---|---|---|---|
| `scripts/variance_demo.py` | 4 | drives the agents directly, short budget | **done** (E-021) |
| `scripts/compare_sample_efficiency.py` | 3 | the trainers' JSON artefacts | built, needs the full runs |
| `scripts/aggregate_phase4.py` | - | the trainers' per-repeat JSON artefacts | **new 2026-09-04** |

`aggregate_phase4.py` is the Phase 4 counterpart to `aggregate_dqn.py`, needed because a
`--only-repeat` process writes one run per file and correctly refuses to call a single run a
result (CONSTRAINTS #3). It **re-evaluates nothing** - it applies `across_runs_summary` to
summaries the trainers already computed against the eval seeds - and it deliberately has no code
path that touches the environment, so it cannot drift into being a second, differently-configured
evaluation of the same policies. Files predating D-036 carry only the old single `eval_summary`
key and are refused rather than guessed at.

`variance_demo.py` samples the coefficient multiplying `grad ln pi` under three conditions and is
the only place the bias-variance trade is *measured* rather than described.
`compare_sample_efficiency.py` plots reward against **environment steps** - which is why all three
trainers now record `episode_steps`; the agents consume samples at very different rates per shift
(~88 vs ~47), so episodes are not a common currency.

---

## 5. Where each phase adds code

| Phase | New modules |
|---|---|
| 0 | `config`, `alerts`, `generator`, `env`, `state`, `runner`, `agents/base`, `agents/baselines`, `evaluation/metrics` |
| 1 | `agents/dp`, `mrp_example` |
| 2 | **all built ✅** — `tiny_mdp`, `agents/tabular`, `agents/q_learning`, `agents/sarsa`, `agents/monte_carlo`, `scripts/train.py`, `scripts/policy_table.py`, `scripts/compare_agents.py`, `scripts/ablations.py` |
| 3 | **all built ✅, no training result yet** — `agents/dqn`, `agents/replay`, `scripts/train_dqn.py`, `scripts/run_dqn_sweep.py`, `scripts/aggregate_dqn.py`, `scripts/compare_dqn_tabular.py`, `scripts/dqn_ablations.py` |
| 4 | 🟡 `agents/reinforce` **built + tested, unmeasured** (FEATURE_008, E-018), `scripts/train_reinforce.py`; still to come: `agents/actor_critic`, the sample-efficiency comparison, the variance demonstration |
| 5 | 🟡 **5a is code complete** (FEATURE_011, FEATURE_012) — `rlhf/summary`, `rlhf/pairs`, `rlhf/store`, `rlhf/agreement`, `labelling/queue`, `labelling/render`, `labelling/app`, `scripts/report_kappa.py`, `scripts/label_ui.py`, `scripts/generate_pairs.py`, `config.RLHFConfig`. The 300 pairs exist and load through the UI. Still to come: the 350 real judgements (human time, not code), `rlhf/reward_model` (5b), 5c re-training |
| 6 | `evaluation/audit`, `evaluation/plots` |

Nothing from a later phase should be needed to run an earlier phase. If Phase 5 breaks, Phase 2's results must still reproduce.

---

## 6. Tech stack and why

| Choice | Reason |
|---|---|
| Python 3.13 | Already installed on this machine |
| NumPy | Tabular Q-tables, all the array work |
| PyTorch | DQN, policy gradient, reward model. Pranav already uses it (EHCV). |
| Gymnasium-style API (`reset`/`step`) — **interface only, not the dependency** | Standard, recognisable on a resume, zero lock-in |
| pytest | Pranav's existing test discipline |
| YAML config | Keeps every magic number out of the code and in one reviewable place |
| SQLite | Preference labels; zero setup |
| FastAPI + React (Vite) | Matches both resumes; Phase 5+, kept optional |
| Groq / Llama | The plain-English justification line; Diya has shipped this twice |
| matplotlib | All plots for the report |

**Explicitly rejected:** Stable-Baselines3. The whole point is writing the algorithms by hand so both team members can reproduce them in an interview. It may be used *only* as an optional cross-check for PPO, and if so it must be recorded in `DECISIONS.md`.
