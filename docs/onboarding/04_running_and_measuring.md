# Document 4 — Running and Measuring

### How an episode becomes a number you can defend

> **Read Document 3 first.** This document assumes you know what the eleven agents are
> and what update rule each one runs. It does not add an algorithm — it adds the
> machinery that turns an algorithm into a reproducible measurement: the loop that drives
> agent and environment together, the five metrics, the two hand-solved fixtures that
> prove the machinery is honest, the 25 scripts that use all of it, and the 391 tests that
> hold it in place.
>
> **The promise of this document:** by the end you should be able to say, for any number
> in this project's results tables, *which script produced it, which seeds it ran on, and
> which test would fail if the pipeline that produced it broke.*

---

## Contents

| § | What it covers | File(s) |
|---|---|---|
| 1 | The map — how an episode becomes a row in a table | *(map)* |
| 2 | The runner — the loop nobody is allowed to put logic in | `src/soc_triage/runner.py` |
| 3 | The five metrics | `src/soc_triage/evaluation/metrics.py` |
| 4 | Proving the implementations are right | `src/soc_triage/tiny_mdp.py`, `src/soc_triage/mrp_example.py` |
| 5 | The 25 scripts | `scripts/` |
| 6 | The test suite | `tests/` (27 files) |
| 7 | The five rules of evidence, as they appear in code | *(cross-cutting)* |
| 8 | Things that will confuse you | *(traps)* |

---

## 1. The map — how an episode becomes a row in a table

Every number this project reports — every mean, every standard deviation, every line in
`results/baselines.md` — travels the same road:

```
   agent + environment                (Document 2, Document 3)
          |
          |  agent.act(obs), env.step(action), repeat until done
          v
   runner.py :: run_episode()
          |
          |  one dict per episode: run_id, agent_name, seed, config_hash,
          |  a full step-by-step trace, and the environment's own outcome
          v
   EpisodeRecord  (a plain dict — no class, see §2)
          |
     +----+----------------------+
     |                           |
     v                           v
evaluation/metrics.py     rlhf/ + labelling/          results/*.md, *.png
"the five headline        (Document 5 — a           (a script's own printed
 numbers, mean ± std       DIFFERENT consumer         table, always the FIRST
 over >=5 seeds"           of the same records)       thing a script writes)
```

**The one-sentence version of this document:** the runner produces a record that says
nothing about whether an episode went *well*, `evaluation/metrics.py` is the only place
that says whether it did, and every script under `scripts/` is a thin, disposable wrapper
that calls both in a particular order and prints or plots the result. Nothing downstream
of `metrics.py` has to change when an agent changes, and nothing in the runner has to
change when a metric changes. That separation is deliberate and it is the whole reason
this document can exist independently of Document 3.

---

## 2. The runner — `src/soc_triage/runner.py`

131 lines, and the module docstring states its own boundary before doing anything else:

> *"The runner computes no metrics (that's evaluation/) and holds no learning logic
> (that's agents/)."*

**Why that restraint matters.** A runner that also computed recall, or that special-cased
SARSA's action-commitment, would couple every future metric and every future agent to
this one file. Keeping it to "run the loop, write down what happened" is what let Phase 5
add a second consumer of `EpisodeRecord` (the RLHF pair builder, Document 5) without
touching this file at all.

### `_encode()` — the one place `obs_kind` is read

```python
def _encode(snap: EnvSnapshot, agent: Agent, cfg: EnvConfig) -> Any:
    """Give the agent the observation kind it declared. 'snapshot' is oracle-only."""
    if agent.obs_kind == "disc":
        return discretise(snap, cfg)
    if agent.obs_kind == "cont":
        return featurise(snap, cfg)
    if agent.obs_kind == "snapshot":
        return snap
    raise ValueError(f"unknown obs_kind '{agent.obs_kind}' on agent '{agent.name}'")
```

Document 3 §2 introduced `obs_kind` as the string that lets one loop drive eleven
different agents. This function is where that string is actually spent. Four lines,
and the `raise` on the last one is not defensive boilerplate — it is what stops a typo
in a new agent's `obs_kind` from silently handing it `None` and failing three files
downstream with a confusing shape error instead of failing here with a clear one.

### `run_episode()` — the loop itself

```python
snap = env.reset(seed)
obs = _encode(snap, agent, cfg)

steps: list[dict] = []
total_reward = 0.0
done = False
while not done:
    action = agent.act(obs)
    next_snap, reward, done, info = env.step(action)
    next_obs = _encode(next_snap, agent, cfg)

    if learn:
        agent.update(obs, action, reward, next_obs, done)

    steps.append({...})
    total_reward += reward
    snap, obs = next_snap, next_obs
```

This is the Document 2 §5 environment interface (`reset` / `step`) and the Document 3
§2 agent interface (`act` / `update`) wired together, and nothing else. The `learn` flag
is the entire difference between training and evaluation:

```
   learn=True   (training)     agent.update() IS called — the agent changes
   learn=False  (evaluation)   agent.update() is NEVER called — the agent is frozen

   Every "final evaluation" block in every trainer sets learn=False. Forgetting
   that would mean the agent kept learning DURING the number that is supposed to
   report what it had already learned — a moving target reported as a result.
```

**Every step is recorded, not summarised.** `steps` grows one dict per action, holding
the discretised state, the action, the reward, and a full `info` breakdown including
`reward_breakdown` and which alert (if any) was investigated. That is more detail than
any metric in §3 uses. It exists because Document 5's episode renderer needs the full
timeline to show a human labeller what happened, and the runner has no way to know in
advance which downstream consumer will need which field. Recording everything once, here,
is cheaper than teaching two different loops to walk the environment.

### `_alert_to_dict()` — ground truth goes IN the record, on purpose

```python
def _alert_to_dict(alert) -> dict:
    """Alert -> JSON-serialisable dict. Includes ground truth: EpisodeRecords are
    environment-side logs for evaluation and RLHF outcome rendering, never
    agent observations."""
    return dataclasses.asdict(alert)
```

**This looks like it violates Document 2 §8's ground-truth firewall. It does not,** and
the distinction is worth being able to state precisely in a viva: the firewall governs
what the *agent* sees during a step (`state.py`'s two encoders), never what gets *logged*
after the fact. An `EpisodeRecord` is written once an episode is already over, consumed
by `evaluation/metrics.py` and Document 5's tools — never fed back into `agent.act()`.
`is_true_incident` travels through this dict specifically so `metrics.py` can compute
recall and so Document 5's renderer can show a labeller which alerts were real. Keeping
the secret out of the record would make both of those impossible.

### `config_hash()` — every record can be traced to the config that produced it

```python
def config_hash(cfg_path: str | Path) -> str:
    text = Path(cfg_path).read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
```

Twelve hex characters of the config file's own SHA-256, stamped into every
`EpisodeRecord`. **The bug this catches, and it is not hypothetical:** Document 6 §5's
`BUG_005` is a config-hash mismatch that looked for a minute like two agents having been
measured on two different environments — the two hashes differed by one character inside
a YAML *comment*, a date correction that changed nothing about the simulation. The hash
covers the whole file byte-for-byte, comments included, which is why it fired on a change
that mattered not at all — and why the guard is kept anyway rather than loosened: an
over-sensitive alarm that occasionally needs a five-minute investigation is safer than an
under-sensitive one that misses the day the environment genuinely changes underneath a
result.

### `run_episodes()` and `save_records()` — nothing hidden

```python
def run_episodes(env, agent, seeds, cfg, cfg_hash="unhashed", learn=False) -> list[dict]:
    """Run one episode per seed. Explicit loop — no hidden parallelism."""
    records = []
    for seed in seeds:
        records.append(run_episode(env, agent, seed, cfg, cfg_hash, learn))
    return records
```

An ordinary Python `for` loop over seeds, and the docstring says so explicitly —
**"no hidden parallelism."** CONSTRAINTS #14 asks the *algorithms* to stay readable even
at a speed cost; this is the same principle applied one layer up. A script that needs
real parallelism (the DQN sweep, Document 4 §5) achieves it with separate OS processes,
each running this same simple loop, rather than by making the loop itself concurrent and
harder to reason about.

`save_records()` writes each record as its own `results/runs/<run_id>.json` — one file
per episode, human-inspectable, and gitignored (`results/` is never committed;
CONSTRAINTS #19).

---

## 3. The five metrics — `src/soc_triage/evaluation/metrics.py`

121 lines. The module docstring states the contract in one line:

> *"Consumes outcome dicts the runner produced; trains nothing, mutates nothing."*

This file has never called `env.step()` and never will. It reads what the runner already
wrote down. That is what makes it safe to change a metric's formula without any risk of
accidentally changing what an agent experiences during training.

### The five, and which direction is good

| Metric | What it means in SOC terms | Better is |
|---|---|---|
| `mttd_min` | Mean Time To Detect — minutes from arrival to investigation, over *caught* incidents | **lower** |
| `recall_at_deadline` | fraction of true incidents caught before their dwell deadline | **higher** — the headline number |
| `wasted_minutes` | analyst time spent investigating false positives | **lower** |
| `critical_misses` | crown-jewel (`asset_criticality == 2`) incidents missed | **lower** |
| `composite_cost_inr` | the whole shift converted to rupees under a stated cost assumption | **lower** |

### `MIN_RUNS_TO_REPORT = 5` — a protocol floor, not a tunable

```python
# CONSTRAINTS #3: "Never report a single run. Every headline number is mean ± std
# over at least 5 seeds." This is a protocol floor rather than a tunable, which is
# why it lives beside the metrics instead of in config/ — the same reasoning that
# puts MIN_EVAL_SEEDS in tests/test_eval_protocol.py.
MIN_RUNS_TO_REPORT = 5
```

**Worth noticing what this constant is *not*.** CONSTRAINTS #9 says every tunable number
lives in `config/*.yaml`. This one doesn't, and that is not an oversight — it is not a
knob anyone is meant to turn. A number in `config/` invites an edit; this number is a
floor on what counts as evidence at all, and it is checked by both `train_reinforce.py`
and `train_actor_critic.py` (§5) before either will print a result without a loud warning.

### `episode_metrics()` — one episode, five numbers

```python
total = outcome["incidents_total"]
recall = outcome["incidents_caught_in_time"] / total if total > 0 else 1.0
```

**Worked example.** Suppose a shift's `outcome` says 6 real incidents arrived, 4 were
caught before their deadline:

```
recall_at_deadline = 4 / 6 = 0.667
```

If a shift happens to contain *zero* real incidents, `recall` is defined as `1.0` — there
was nothing to miss, so the policy trivially caught everything there was to catch. That
edge case is rare (the incident rate is calibrated to ~3%, Document 2 §3) but not
impossible over 480 minutes, and returning `1.0` rather than raising or returning `None`
keeps the aggregate mean in §3.4 from being silently biased by dropping those shifts.

The composite cost is three stated assumptions added together:

```python
missed_cost += n_missed * cost_cfg.missed_incident_by_criticality[tier]   # by asset tier
delay_cost = mttd_min * incidents_caught * cost_cfg.detection_delay_per_min
composite = missed_cost + wasted_minutes * cost_cfg.wasted_analyst_minute + delay_cost
```

Document 2 §6 already covers where these rupee figures live and why they are declared
rather than hidden. The point worth repeating here: **no agent is ever trained on this
number.** It exists purely to translate a results table into a figure a non-technical
reader can react to, under assumptions a sceptical reader can substitute their own values
into and recompute.

### `summarise()` — mean ± std over one agent's episodes

```python
for name in METRIC_NAMES:
    values = [m[name] for m in per_episode if m[name] is not None]
    if values:
        summary[name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    else:
        summary[name] = {"mean": None, "std": None}
summary["mttd_undefined_episodes"] = sum(1 for m in per_episode if m["mttd_min"] is None)
```

**`mttd_min` is the one metric that can be undefined for a whole episode** — a shift that
caught nothing has no detection time to average. Filtering `None` out of the mean rather
than treating it as `0.0` is the difference between "this agent's average detection speed,
over the shifts where it detected anything" and a number that secretly rewards catching
nothing (a `0.0` mixed into a mean of positive minutes would *lower* it — the exact
direction that flatters a failing agent). `mttd_undefined_episodes` reports how many
episodes were dropped, so the reader can see the exclusion rather than trust it silently.

### `across_runs_summary()` — the aggregation CONSTRAINTS #3 is actually about

```python
def across_runs_summary(per_run_summaries: list[dict]) -> dict[str, dict]:
    """Mean and std ACROSS runs of each run's mean — never a single run.
    ...each run contributes ONE value (its own mean over the eval seeds), and the
    std reported is the spread of those values. A std computed over the pooled
    episodes instead would describe seed difficulty and would be much smaller —
    the mistake BUG_004 shipped."""
```

This is the function every trainer in §5 calls at the very end, and it is worth drawing
the distinction it protects as a picture, because the bug it names (`BUG_004`, Document 6
§5) is genuinely easy to reintroduce by accident:

```
   WRONG (what BUG_004 did):                RIGHT (what across_runs_summary does):

   Pool all 5 runs x 30 eval episodes        For EACH of the 5 runs, take its own
   into 150 numbers.                         mean over its 30 eval episodes -> 5 numbers.
   Take ONE mean and ONE std of the 150.     Take the mean and std of THOSE 5 numbers.

   The std describes: how much do shifts     The std describes: how much do
   differ from each other?                   independently trained RUNS differ
   (seed difficulty — small, ~50)            from each other? (run-to-run variance
                                              — the thing CONSTRAINTS #3 actually
                                              wants reported)
```

A run's own mean can itself be `None` — `summarise()` reports `mttd_min` as `None` when
a run's shifts caught nothing at all, and `across_runs_summary` drops that run from
*that metric's* average rather than counting it as zero. E-020's collapsed actor-critic
runs (Document 3 §12) scored recall `0.0000` — a real Phase 4 outcome this path has to
carry correctly, not an edge case to be papered over.

---

## 4. Proving the implementations are right — `tiny_mdp.py` and `mrp_example.py`

Document 3 §8 set this up without paying it off:

> *"on the tiny MDP: Q-learning converges to q_* itself (9.24e-14, E-007), while SARSA
> converges to `tiny_mdp.epsilon_soft_q(epsilon)`... Q-learning does not pay for its own
> exploration; SARSA pays."*

This section is that payoff, and `tiny_mdp.py`'s own docstring states why it has to exist
at all, better than a paraphrase would:

> *"The 576-state SOC MDP cannot be checked with a pen, so a Q-table computed on it can
> only ever be compared against another program. That catches disagreements, not shared
> mistakes: SARSA and Q-learning both converging to the same wrong number would look
> exactly like success."*

**Two objects, two Bellman equations, built in the order that makes them trustworthy.**

```
   mrp_example.py  (Phase 1)          tiny_mdp.py  (Phase 2)
   checks the backup for   V           checks the backup for   Q
   guards agents/dp.py                 guards the three tabular learners
   5 states, no actions                2 states, 2 actions
   S&B eq. 3.14 (v_pi)                 S&B eq. 3.20 (q_*)
```

Both are frozen with numbers derived by hand *first*, verified against code *second*,
and CONSTRAINTS #9 exempts them from living in `config/*.yaml` for a reason both
docstrings give explicitly (D-013, D-014): the pen-and-paper derivation in the companion
`docs/features/` document is correct only for these exact numbers, and moving them into
an editable YAML would let someone change a value and silently strand the derivation.

### `mrp_example.py` — the five-state MRP, four independent routes to one answer

Five states — `QUIET`, `BACKLOG`, `INVESTIGATING`, `CONFIRMED`, `MISSED` — with `CONFIRMED`
and `MISSED` absorbing. γ = 0.9, chosen (not the project's real 0.99) so the one
non-terminating value comes out as the exact rational `52/11`, verifiable by hand.

Four routes, all required to agree in `tests/test_mrp_bellman.py`:

| Route | Function | What it proves |
|---|---|---|
| 1. By hand | (paper, `docs/features/FEATURE_001_mrp_worked_example.md`) | the human got it right |
| 2. Closed form | `solve_linear` — `V = (I − γP)⁻¹R` | only possible because the state space is tiny |
| 3. Iterative | `evaluate_iteratively` — repeated Bellman backups | the general method that scales |
| 4. **The shipped solver** | `agents.dp.value_iteration`, on this MRP reshaped as a degenerate MDP | **this is the one that carries the weight** |

The docstring is explicit about why route 4 is the only one that actually matters for
the project's honesty claim:

> *"Route 4 is the one that carries the weight. Routes 2 and 3 only check this file
> against itself; route 4 checks the shipped Phase 1 solver against a number a human
> derived on paper."*

**The trick that makes route 4 possible: `as_degenerate_mdp()`.** An MRP has no actions;
`agents.dp.value_iteration` expects an MDP with a `max` over actions in its backup.
Give every action in the degenerate MDP the *same* transition and reward as the MRP's
one real transition, and the `max` over identical numbers collapses to just that number —
so value iteration on the degenerate MDP is provably solving the exact same equation as
plain iterative policy evaluation on the MRP. `scripts/run_mrp_example.py` prints all four
numbers side by side; the largest disagreement measured this session (see §5) is on the
order of `1e-14` — floating-point noise, not a real discrepancy.

### `tiny_mdp.py` — the two-state MDP, and why it is a *harder* fixture than it looks

Two states (`QUIET`, `BUSY`), two actions (`WAIT`, `WORK`), deterministic, **continuing**
— no terminal state. That last property is deliberate and the docstring names exactly
what it rules out:

> *"Continuing, not episodic. There is no terminal state, so a learner cannot get the
> right answer by averaging complete returns without ever bootstrapping. TD methods must
> actually use V(s') to pass."*

**The derivation, worked in full** (from `docs/features/FEATURE_002_tiny_mdp_qstar.md`):

```
V(QUIET) = +1 + 0.9 V(QUIET)     ->   0.1 V(QUIET) = 1   ->   V(QUIET) = 10
V(BUSY)  = +4 + 0.9 V(QUIET)     =   4 + 9   =   13

Q(QUIET, WAIT) = +1 + 0.9(10) = 10.0    <- optimal in QUIET
Q(QUIET, WORK) = -5 + 0.9(13) =  6.7
Q(BUSY,  WAIT) = -1 + 0.9(13) = 10.7
Q(BUSY,  WORK) = +4 + 0.9(10) = 13.0    <- optimal in BUSY
```

**The sanity check that is worth sitting with, because it is the single most common
misreading of a value function.** `V(BUSY) = 13` is *larger* than `V(QUIET) = 10`, even
though `BUSY` is the bad state. That does not mean "being backlogged is good" — it means
that from `BUSY`, one `WORK` step pays `+4` *and* returns you to `QUIET`, so you collect
that bonus **on top of** everything `QUIET` was already worth. You cannot farm the bonus
by deliberately returning to `BUSY` to collect it again: `Q(QUIET, WORK) = 6.7` is
exactly the arithmetic that forbids that exploit, and it is the tiny-MDP miniature of the
Phase 1 bulk-close hack (Document 3 §3, Document 1 §5) — a value function that let you
loop for profit here would have exactly the same shape of bug.

**Four design properties, each ruling out one specific broken learner:**

| # | Property | The broken learner it catches |
|---|---|---|
| 1 | Exact arithmetic — the answer is a number a human wrote, not a number a program produced | — |
| 2 | Continuing, not episodic | a learner that never actually bootstraps, and would still pass an episodic test |
| 3 | The optimal action **differs** between the two states | a learner hard-wired to `return 0` |
| 4 | Deterministic transitions | this fixture is not testing whether a learner can average out sampling noise |

**A fifth property fell out of the design rather than being asked for, and it is the best
viva question the fixture generates:** under the optimal policy the agent never leaves
`QUIET`, so `BUSY` is only ever reached by *exploring*. A learner with `epsilon` pinned to
`0` never sees half the MDP and fails on `Q(BUSY, ·)`. **The fixture demonstrates why
exploration is not optional, using the same object it uses to check correctness.**

**`MIN_ACTION_GAP = 2.3`, and the two reward designs that were rejected before this one.**
`docs/features/FEATURE_002_tiny_mdp_qstar.md` records that the first reward design gave
a margin of only 0.1 between the best and second-best action in `QUIET` — a learner
sitting within 1% of `q_*`, an entirely ordinary state after a finite number of episodes,
would flip the reported policy at random and the test would be flaky rather than strict.
The final design (`+1/−5/−1/+4`) buys margins of 3.3 and 2.3. **A test fixture needs its
correct answer to be *far* from its wrong answers, not merely different from them** — a
lesson worth carrying into any fixture written from here on, not just this one.

### `epsilon_soft_q()` — the target SARSA and Monte Carlo actually converge to

```python
"""Q-learning is off-policy and converges to q_* (S&B §6.5). SARSA (§6.4) and
first-visit Monte Carlo control (§5.4) are on-policy: they converge to
q_pi for the epsilon-greedy policy they are following, which is *not* q_*
whenever epsilon > 0. Grading them against HAND_COMPUTED_Q would mark a
correct implementation as broken."""
```

Iterates the expected-SARSA backup — the *exact expectation* over the epsilon-greedy
policy, not a sampled action — to convergence. At `epsilon = 0` it must collapse exactly
to `HAND_COMPUTED_Q`, and `test_epsilon_soft_q_collapses_to_q_star_as_epsilon_goes_to_zero`
is what ties this second, independent target back to the same pen-and-paper answer rather
than letting it float free as "whatever the code computes."

### `pad_actions()` — widening a 2-action MDP without touching the answer

`agents.dp.value_iteration` loops over a fixed `N_ACTIONS = 5`. To run the project's own
DP solver on this 2-action fixture (as a cross-check, mirroring what `as_degenerate_mdp`
does for the MRP), the action set has to be widened to 5 first. The chosen method
**duplicates the two real actions cyclically** (`0,1,0,1,0`) rather than inventing new
ones:

> *"the `max_a` in the Bellman backup then ranges over a multiset containing only copies
> of the genuine actions, so the maximum ... is unchanged. Padding with, say, zero-reward
> self-loops would have added an action worth 0 ... but would silently alter the answer
> in any MDP with negative values."*

This MDP has negative rewards (`−5`, `−1`), so the distinction is not academic here —
it is the difference between a padding trick that is provably safe and one that would
have quietly been wrong on exactly this fixture.

---

## 5. The 25 scripts — `scripts/`

**No logic lives here.** Every script in this directory is a thin driver: it loads
config, builds an agent or loads a saved one, calls `runner.py` and `evaluation/metrics.py`
in some order, and prints or plots the result. That is a deliberate design rule, not an
accident of how the project grew — it is what makes a script safe to read in isolation
without also having to audit it for hidden algorithm logic.

### The 25, grouped by purpose

| Group | Scripts |
|---|---|
| **Training entry points** | `train.py` (tabular), `train_dqn.py`, `train_reinforce.py`, `train_actor_critic.py` |
| **Evaluation & comparison** | `run_baselines.py`, `run_dp.py`, `compare_agents.py`, `policy_table.py`, `compare_dqn_tabular.py`, `compare_sample_efficiency.py`, `dp_collapse.py` |
| **Sweep infrastructure** | `run_dqn_sweep.py`, `aggregate_dqn.py`, `aggregate_phase4.py`, `dqn_ablations.py` |
| **Hyperparameter experiments** | `ablations.py`, `reinforce_clip_experiment.py`, `actor_critic_entropy_experiment.py`, `variance_demo.py` |
| **Calibration & derivation printers** | `calibrate_generator.py`, `run_mrp_example.py` |
| **Process tooling** | `commit_balance.py` |
| **RLHF pipeline** — owned by **Document 5**, listed here only for completeness | `generate_pairs.py`, `label_ui.py`, `report_kappa.py` |

The rest of this section gives the six scripts the spec calls out — `train.py`,
`run_baselines.py`, `run_dp.py`, `compare_agents.py`, `policy_table.py`,
`commit_balance.py` — their own subsection. The remaining 16 (outside the RLHF three) are
summarised in the table above; each carries the same three disciplines every trainer and
experiment script in this project shares, so once you have read one you can read any
other by its docstring alone:

1. **Its own dedicated seed block** (D-016), disjoint from every other script's, so no
   two experiments can ever share alert streams and no comparison can be confounded by
   a shared draw.
2. **The eval seeds are touched exactly once, at the very end**, after every training
   decision has already been made (CONSTRAINTS #2). `reinforce_clip_experiment.py` and
   `actor_critic_entropy_experiment.py` go further and assert this in code
   (`_assert_no_eval_seeds`) rather than trusting the convention.
3. **A reduced or smoke run is written to `results/smoke/`, never to the real output
   path**, so a quick sanity check can never be mistaken for, or silently overwrite, a
   real result (D-018 — a `--episodes 200` smoke test once did exactly that to a real
   20,000-episode Q-table, and the corruption only surfaced later as an unexplained drop
   in state coverage).

### `train.py` — Phase 2's training entry point

```python
AGENTS: dict[str, type[TabularAgent]] = {
    "q_learning": QLearningAgent,
    "sarsa": SarsaAgent,
    "monte_carlo": MonteCarloAgent,
}
```

One script drives all three tabular learners, selected by `--agent`. The docstring states
the four-step discipline every trainer in this project follows, and the comment on why
steps 2 and 4 are ordered the way they are is the sentence to remember:

> *"Evaluation-seed numbers are computed once, after every training decision has already
> been made, so nothing in this script can tune against them (CONSTRAINTS #2)."*

**`greedy_diagnostic()` switches exploration off and learning off, on purpose, and
restores both afterwards:**

```python
saved_epsilon = agent.epsilon
agent.epsilon = 0.0
try:
    rewards = [... learn=False ...]
finally:
    agent.epsilon = saved_epsilon
```

Two separate reasons stack here: `epsilon = 0.0` means the curve shows what the *learned*
policy is worth, not a partly-random one; `learn=False` means measuring the diagnostic
can never itself change what is being measured. The `try/finally` is what lets this
diagnostic run in the middle of training without permanently damaging the agent's own
exploration schedule.

**Each repeat gets its own agent seed AND its own slice of the training-seed block:**

```python
seed_base = train_seed_start + repeat_index * n_episodes
```

The comment explains why *both* have to vary:

> *"Varying only the agent seed would leave all five runs facing an identical alert
> stream, and the resulting std would understate the real variability."*

**The Q-table save guard.** `is_full_run` checks the episode count, `eval_every`, and
repeat count against the shipped config before deciding whether to write to `results/` or
`results/smoke/` — the D-018 guard from the table above, made concrete for the file that
was actually corrupted by it once (a stale 121-state Q-table quietly replaced by an
81-state one, discovered only because `compare_agents.py`'s coverage report changed for
no logged reason).

### `run_baselines.py` — the Phase 0 exit gate

Runs all five baselines on the eval seeds and checks the two conditions Document 1 §10
reports as Phase 0's headline finding:

```python
oracle_best = all(rewards["oracle_greedy"] > v for n, v in rewards.items() if n != "oracle_greedy")
informed_floor = min(recalls[n] for n in informed)          # severity_sort, oracle_greedy
uninformed_ceiling = max(recalls[n] for n in uninformed)    # random, fifo, cheapest_first
separated = informed_floor > uninformed_ceiling
```

**Worth noticing what is checked and what is not.** The gate is on *total reward* for the
oracle, not recall — the module comment cites the reason directly: recall@deadline
structurally favours severity-camping (D-007), so demanding the oracle also win on recall
would be asking an information advantage to beat a strategy tuned to the metric itself.
This is the same category of gate-writing lesson Document 1 §5 and Phase 1's D-012 make
from the other direction: pick the criterion that actually matches what the agent is
optimising, or the gate measures the gate-writer's assumption rather than the agent.

### `run_dp.py` — the Phase 1 pipeline, five steps in one file

```
1. estimate_model()      50,000 random-policy episodes -> P_hat, R_hat, visits
2. report coverage       how much of the 576-state space was ever seen
3. value_iteration()     converge, plot the delta curve
4. policy_iteration()    an INDEPENDENT cross-check — must agree >=95%
5. evaluate in the REAL environment, on the eval seeds
```

**Step 4 is a hard `raise`, not a warning:**

```python
if agreement < 0.95:
    raise SystemExit("VI and PI disagree beyond tie tolerance — one of them is wrong. STOP.")
```

Two independently-coded algorithms landing on the same policy is much stronger evidence
of correctness than either one alone (Document 3 §4) — and this script is where that
cross-check is actually enforced as a gate that stops the pipeline, rather than just a
number quoted in a report that a reader might not check.

**Step 5's final block reconstructs the full `Q` table from the converged `V`:**

```python
Q_dp[s, a] = R_hat[s, a] + tcfg.common.gamma * (P_hat[s, a] @ V_vi)
```

`value_iteration` returns only `V`; `compare_agents.py` (next subsection) needs `Q` to
compare against the tabular learners' own `Q` tables. Reconstructing it here — rather
than re-running the whole 50,000-episode estimation inside `compare_agents.py` — is why
that script can run in under a second.

### `compare_agents.py` — Q-learning, SARSA, Monte Carlo, and DP, compared honestly

The module docstring states the whole design in its first paragraph:

> *"Why 'over states actually visited' is the whole design. Both sides have a convention
> that fills in states they never saw, and both conventions look like opinions when
> printed."*

Document 3 §5 already named the convention on the tabular side (an unvisited state's
all-zero Q row makes `argmax` return action 0, the tie-break — not a decision); Document
3 §4 named DP's (an absorbing self-loop worth 0, D-011). This script is where both
conventions are prevented from silently inflating an "agreement" number:

```python
seen_a = visits_a.sum(axis=1) > 0
seen_b = visits_b.sum(axis=1) > 0
both = seen_a & seen_b
agree_all = float((policy_a == policy_b).mean())          # the misleading figure
agree_seen = float((policy_a[both] == policy_b[both]).mean())   # the honest one
```

**Both numbers are printed, deliberately, side by side**, with the misleading one labelled
as such in the script's own output. Document 1 §11 already quotes the consequence: "all
576" agreement measured 83–86%; restricted to states both agents actually visited, it
collapses to 22–44%. Printing only the honest figure would hide *how big* the artefact
was; printing both is what makes the lesson demonstrable rather than merely assertable.

### `policy_table.py` — turning 576 numbers nobody can read into a table a human can

The problem statement, from the module docstring:

> *"A 576-row Q-table is not a result anyone can read. This turns it into the question
> the project actually cares about: as the shift runs out, does the agent change its
> strategy — and in a way a human analyst would recognise?"*

**`decode()` and `encode()` are kept side by side and cross-checked against each other
before anything is printed:**

```python
def _self_check() -> None:
    for state_id in range(N_SEV * N_QLEN * N_AGE * N_TLEFT * N_CRIT):
        if encode(*decode(state_id)) != state_id:
            raise SystemExit(f"decode/encode disagree at state {state_id} — STOP")
```

The comment explains why this cheap check runs on *every* invocation rather than once in
a test: **"a transposed figure is the kind of error that survives review because the
table still *looks* plausible."** A packing bug that swapped, say, `queue_len` and
`oldest_age` would still print a well-formed table of five-action codes — nothing about
its shape would look wrong, and every cell would be lying about which situation it
describes.

**Every unvisited state prints `·`, never a guessed action** — the same discipline
`compare_agents.py` enforces, applied to the rendering rather than the comparison. The
script's own printed summary distinguishes two different tables that are easy to conflate:

| View | What it shows | The trap if you confuse them |
|---|---|---|
| 2a — the **learned** policy | of *visited* states in each time bucket, what the greedy (`argmax`) action is | this is what the agent would *do* |
| 2b — the **experience** (behaviour policy) | action share weighted by *visit count during training*, epsilon-greedy and partly random | mistaking this for the learned policy reports what the agent tried, not what it decided |

### `commit_balance.py` — a script that measures the team, not the model

Read in full in §5 above the fold — the module docstring's first paragraph states its
scope precisely:

> *"What this script is not. It does not, and must not, be used to attribute work to
> someone who did not do it. The point is to keep the real split even by handing over at
> the right time."*

**`author_counts()` uses `git log --use-mailmap`**, which collapses any identity a
`.mailmap` file maps together — so the same person committing under two email addresses
on two machines is still counted once. `HANDOVER_THRESHOLD = 3` is the number CONSTRAINTS
#24–26 fix as the point past which a hand-over is owed, and the script's closing message
is deliberately a *recommendation in prose*, not just a number:

```python
print(f"IMBALANCED. **{trail_name} should take the next block** — roughly "
      f"{needed}-{gap} commits to get back inside the threshold.")
```

Measured this session: **Pranav Upadhyay 71 commits (55.9%) to Diya Garg's 56 (44.1%), a
15-commit gap** — see the session preflight report earlier for the full per-phase
breakdown. This document's own three commits are Diya's contribution toward closing it.

---

## 6. The test suite — 27 files

`.\.venv\Scripts\python.exe -m pytest tests/ -q` was run this session:

```
391 passed in 777.18s (0:12:57)
```

391 — unchanged from the count HANDOVER.md already recorded, and consistent with this
document touching no source file. What follows is a map of what each file protects,
grouped the way the suite itself is grouped — by which layer of the pipeline it guards.

| File | What it protects |
|---|---|
| `conftest.py` | one shared `cfg` fixture — the real, calibrated config every other test runs against |
| `test_env.py` | environment determinism, reward-breakdown accounting, the bulk-close cap, step ordering |
| **`test_no_ground_truth_leakage.py`** | **the ground-truth firewall itself — own subsection below** |
| `test_eval_protocol.py` | **the 30-seed evaluation block's own integrity — own subsection below** |
| `test_mrp_bellman.py` | the five-state MRP anchor: all four routes to `V` must agree |
| `test_tiny_mdp.py` | the two-state MDP anchor itself, verified before any learner is graded against it |
| `test_tabular.py` | Q-learning: mechanics, the single-backup update rule, convergence to `q_*` |
| `test_on_policy.py` | SARSA and Monte Carlo: the same three groups, graded against `epsilon_soft_q` instead |
| `test_replay.py` | the replay buffer as a ring buffer: capacity, overwrite order, sampling shape, alignment |
| `test_dqn.py` | the DQN's two stabilisers and both required ablations, each pinned at a single backup |
| `test_dqn_analysis.py` | the sweep-aggregation *refusals* — `aggregate_dqn.py`'s guards against averaging incomparable runs |
| `test_reinforce.py` | REINFORCE: sampled (never argmaxed) actions, episode-boundary updates, the baseline is not a critic |
| `test_actor_critic.py` | actor-critic: per-step updates, genuine bootstrapping, the terminal-state cutoff — the mirror image of `test_reinforce.py` |
| `test_across_runs.py` | `across_runs_summary` — the exact aggregation `BUG_004` got wrong, pinned by hand arithmetic |
| `test_feature_scales.py` | the shared 17-column feature-scaling divisors (D-032) |
| `test_dqn_config.py`, `test_reinforce_config.py`, `test_actor_critic_config.py` | each agent's config section fails loudly at load time, not mid-training |
| `test_rlhf_config.py`, `test_rlhf_pairs.py`, `test_rlhf_store.py`, `test_rlhf_summary.py`, `test_rlhf_agreement.py`, `test_labelling_queue.py`, `test_labelling_render.py`, `test_labelling_app.py`, `test_generate_pairs.py` | **owned by Document 5** — the RLHF pipeline and labelling UI. Not re-described here; see Document 5 §3–§10. |

That accounts for all 27 files. Nine of them belong to the RLHF and labelling pipeline
and are Document 5's to explain in depth — listing them here without re-explaining
respects the ownership rule §0 of this set's spec sets out.

### `test_no_ground_truth_leakage.py` — THE integrity test

The module docstring does not hedge:

> *"THE integrity test (CONSTRAINTS.md #1). Never weaken, never skip. If either state
> encoding changes when only the hidden fields change, ground truth is leaking into
> observations and every result the agent produces is fraudulent."*

**The method is adversarial, not a spot check.** `_flip_hidden_fields()` builds a copy of
the current queue with `is_true_incident` inverted and `deadline_min` shifted by 123
minutes on *every* alert, then asserts both `discretise()` and `featurise()` produce
bit-identical output on the original and the flipped snapshot:

```python
assert discretise(snap, cfg) == discretise(flipped, cfg), (
    "discretise() changed when only hidden ground truth changed — LEAK"
)
assert np.array_equal(featurise(snap, cfg), featurise(flipped, cfg)), (
    "featurise() changed when only hidden ground truth changed — LEAK"
)
```

Walked across three full episodes of random actions (fixed seeds, so the test is
reproducible) rather than one hand-picked example — `assert checked > 100` is a floor
that guards the *walk itself*, not the property, catching the case where an early
`done` cut the loop short and left most of the state space unchecked.

**`test_snapshot_carries_no_precomputed_truth_fields` guards the future, not the
present.** It asserts `EnvSnapshot`'s field set exactly equals a hard-coded whitelist —
so if someone later adds, say, `n_true_in_queue` as a convenience field, this test fails
immediately rather than waiting for someone to notice the new field leaking through an
encoder that forgot to ignore it.

### `test_eval_protocol.py` — the evaluation protocol, enforced in code

The docstring states the stakes in one line: **"the evaluation seed block is not a
configuration detail — it is the measuring instrument."** Four tests, each pinning one
property the widened 30-seed block (D-019) has to keep forever:

```python
MIN_EVAL_SEEDS = 30   # set from measurement (E-008), not preference
```

| Test | What it refuses to let regress |
|---|---|
| `test_eval_block_is_large_enough_...` | fewer than 30 eval seeds — E-008 measured that 5 was too few to resolve ~100-reward effects against a ~218 per-seed std |
| `test_the_original_five_eval_seeds_are_still_in_the_block` | dropping 101–105 — every pre-2026-08-17 result must stay a sub-sample of the new block, not get orphaned |
| `test_eval_seeds_are_unique` | a duplicated seed, which would double-count one shift and silently shrink the reported std |
| `test_eval_and_train_blocks_stay_disjoint` | the widening accidentally reintroducing overlap — the loader already enforces this, but a test at the point someone is most likely to break it is worth the four lines |
| `test_eval_block_avoids_every_other_reserved_seed_range` | an eval seed landing inside calibration, DP estimation, or any learner's training block — "the single most damaging silent error available in this project, and one no test result would reveal" |

**Why this file exists at all, given the loader already checks disjointness.** The
loader's check (Document 2 §7) is a permanent, structural guard. This file is where the
*history* of why 30 and not 5 is asserted — it is the difference between "the code
happens to be correct" and "the code is correct and cannot regress to the mistake that
was already made once."

---

## 7. The five rules of evidence, as they appear in code

Document 1 §8.3 already lists these in prose. Here is where each one is actually
enforced, file and line, rather than merely stated:

**1. Never report a single run.**

```python
# src/soc_triage/evaluation/metrics.py
MIN_RUNS_TO_REPORT = 5
```
Checked explicitly by `train_reinforce.py` and `train_actor_critic.py` before either
prints a headline number:
```python
if len(sampled_per_run) < MIN_RUNS_TO_REPORT:
    print(f"\n  WARNING: {len(sampled_per_run)} run(s) only - NOT a reportable result ...")
```

**2. Training and evaluation seeds are disjoint, enforced in code.**

```python
# src/soc_triage/config/environment.py
overlap = set(cfg.seeds.train) & set(cfg.seeds.eval)
if overlap:
    raise ConfigError(f"train and eval seeds must be disjoint; both contain {sorted(overlap)}")
```
And, independently, at the point a hyperparameter sweep is most likely to slip and touch
the eval block by accident:
```python
# scripts/reinforce_clip_experiment.py / actor_critic_entropy_experiment.py / variance_demo.py
def _assert_no_eval_seeds(seeds, cfg) -> None:
    overlap = set(seeds) & set(cfg.seeds.eval)
    if overlap:
        raise ValueError(f"... would train on evaluation seeds {sorted(overlap)} — "
                          "tuning against the eval block is exactly what CONSTRAINTS #2 forbids")
```

**3. Same alert streams across policies — the paired comparison.** Not one line but a
convention followed by every trainer: `scripts/train.py`, `scripts/train_dqn.py`,
`scripts/train_reinforce.py` and `scripts/train_actor_critic.py` all evaluate on the exact
same `cfg.seeds.eval` tuple, so `compare_dqn_tabular.py` can compute a *paired* difference
per seed rather than an unpaired one — the difference between a ±52/±220 spread and a
comparison narrow enough to resolve.

**4. Never delete or overwrite an experiment result.**

```python
# scripts/aggregate_dqn.py — refuses to average runs that disagree on episode count,
# config hash, or ablation flags, rather than silently reporting a mean over a mixture.
for field in ("n_episodes", "config_hash", "no_replay", "no_target_network", "tag"):
    values = {r[field] for r in runs}
    if len(values) > 1:
        raise SystemExit(f"runs in {directory} disagree on '{field}': {sorted(values)} ...")
```
And the D-018 guard, in every trainer: a reduced run is written to `results/smoke/`,
never to the path a full run would occupy.

**5. A result that looks surprisingly good is a bug report until proven otherwise.**

`scripts/dp_collapse.py` exists entirely because of this rule — it tests, and does not
simply assert, the explanation offered for the DP policy's collapse from +305.9 to
−201.2 (E-014, E-015; Document 6 covers the full history). The script's own docstring
states the discipline: measure three correlations, not one, because a lone negative
correlation between "off-core share" and "DP reward" proves nothing on its own — it could
just mean off-core seeds are *harder* seeds for everyone. The control
(`corr(off-core share, severity-sort reward)`) is what turns a plausible story into a
tested one.

---

## 8. Things that will confuse you

| What you'll notice | Why it's like that |
|---|---|
| `EpisodeRecord`s hold `is_true_incident` even though agents never see it | The record is a log written *after* the episode, never fed back to `agent.act()`. Document 2 §8's firewall governs observations, not logs. |
| `MIN_RUNS_TO_REPORT` and `MIN_EVAL_SEEDS` aren't in `config/*.yaml` | They are protocol floors, not tunables — CONSTRAINTS #9 is about *knobs*, and these two are not meant to be turned. |
| Two scripts print "recall" numbers that don't match each other for the same agent | One is `across_runs` (mean of run-means, CONSTRAINTS #3's convention); watch for whether a script is reporting variance *across runs* or *across eval seeds* — they are printed side by side with a note precisely because they are not comparable. |
| `compare_agents.py` prints an "agree (all 576)" column that the script itself calls misleading | It is shown for contrast, deliberately, so the size of the artefact is visible rather than just asserted. |
| `policy_table.py` has two "policy" tables that look similar | View 2a is the learned (greedy) policy; view 2b is the training-time behaviour policy, epsilon-greedy and partly random. Confusing them means reporting exploration noise as a decision. |
| `run_dqn_sweep.py` and `dqn_ablations.py` check a memory *ceiling*, not a memory *floor* | A "keep N GB free" floor lets the scheduler launch until only N GB remains, which on a small machine is nearly 100% used — a ceiling on total usage scales with the machine instead. |
| Several experiment scripts (`ablations.py`, `reinforce_clip_experiment.py`, `variance_demo.py`) explicitly print "does not clear the noise floor" | This is the rule-5 discipline (§7) applied honestly: a sweep that cannot distinguish its own configurations says so, rather than reporting whichever config happened to look best. |
| `config_hash` is 12 hex characters of the *whole config file*, comments included | This is what let `BUG_005` (Document 6) fire on a one-character comment edit that changed nothing about the simulation — over-sensitive, kept anyway, because the alternative is missing a real drift. |
| `tests/test_no_ground_truth_leakage.py` flips `deadline_min` by an arbitrary `+123.0` | Any change to a hidden field must leave the encoders unmoved; the specific offset is arbitrary and unimportant — what matters is that it is nonzero. |

---

## Where to go next

- **Document 5 — RLHF and Labelling.** The second consumer of `EpisodeRecord` this
  document deliberately left unexplained: `rlhf/pairs.py`, `rlhf/summary.py`,
  `rlhf/store.py`, `rlhf/agreement.py`, the labelling UI, and the nine test files this
  document's §6 table pointed at without describing.
- **Document 6 — Decisions, Experiments, Results.** Every D-number and E-number cited in
  this document, in full, with the experiment that produced it — including `BUG_004`,
  `BUG_005`, E-008, E-014 and E-015, all named here but owned there.
- **Document 1 §11** for the honest-results summary this document's evidence machinery
  exists to make possible.

---

*Source of truth: this Markdown file. The `.docx` export is generated from it. If they
disagree, the Markdown is right.*
