# HANDOVER.md — Where things stand right now

> Field Guide habit #1 and #13. Read this first, every session. Rewrite it last, every session.
>
> This is **not** a changelog — it's a snapshot of the present. Overwrite stale entries rather than appending. (The permanent record lives in `DECISIONS.md` and `docs/experiments/EXPERIMENT_LOG.md`.)

---

## Snapshot

| | |
|---|---|
| **Last session** | 2026-08-25 (session 10, on Diya's PC) |
| **Model** | Claude Opus 5 |
| **Phase 0** | Closed. Gate **passes** on the 30-seed block. |
| **Phase 1** | **CLOSED as built-but-not-passed** (D-022, confirmed final by **D-033**). |
| **Phase 2** | **CLOSED as built-but-not-passed** (D-020, confirmed final by **D-033**). One work item is genuinely unfinished - see below. |
| **Phase 3** | **CLOSED as built-but-not-passed** (E-017, confirmed final by **D-033**). |
| **Phase 4** | **HALF BUILT.** REINFORCE (FEATURE_008) and actor-critic (FEATURE_009) both built and tested. Boxes 3 and 4 not written. **No full training run of either exists.** The shipped `entropy_coef` breaks the actor-critic. |
| **Phase 5** | Not started. This is the next block, and it is Pranav's. |
| **Repo state** | `C:\Users\Diya\Desktop\RL Project`, branch `master`. 10 session-10 commits, **pushed**. |
| **Tests passing** | **191/191** (`.\.venv\Scripts\python.exe -m pytest tests/ -q`). Wall time varies with machine load: **56 s to 8 min 49 s** observed the same session on the same code. Budget for the worst case. |
| **Blockers** | Nothing blocks building. **The big gate decision is TAKEN (D-033).** One new decision is owed - greedy vs sampled evaluation, E-019 section 3. |

---

## 🔑 STARTING THE NEXT SESSION - do these first, in order

> Rewritten 2026-08-25 at the end of session 10. Everything is pushed. The commit
> balance is **IMBALANCED, Diya ahead** - see "Whose turn is it". **Phase 5 is Pranav's.**

**1. Get the work.** Ten commits were pushed on 2026-08-25.

```powershell
cd <your RL-Project folder>
git checkout master          # the default branch is master, NOT main
git pull
git log --oneline -10        # newest should be: phase4: update HANDOVER ...
```

**2. Set up the environment** (first time only, or if `pytest` fails to import):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Tested on **Python 3.13.1** with `torch==2.13.0+cpu`, `numpy 2.5.2`, `pytest 9.1.1`.
`torch` is CPU-only **on purpose** - changing it is a dependency change needing
approval (CONSTRAINTS #8).

**3. Prove the repo is healthy before writing anything:**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q    # expect 191 passed
python scripts/commit_balance.py                  # expect: IMBALANCED, Diya ahead
```

**Suite timing is NOT stable, and the old "~8 min" figure was misleading.** The same
191 tests on the same commit took **56 s** once and **8 min 49 s** later the same
session, purely on machine load. Both anchors (REINFORCE 60x800 steps, actor-critic
40x200) dominate. Run the file you are touching while working; **run the whole suite
before every commit** - session 9 broke 16 tests by adding one dataclass field and not
doing that, and session 10 nearly repeated it for exactly the same reason.

**4. Read, in this order** - about 20 minutes:

| file | why |
|---|---|
| `CLAUDE.md` | the operating rules; read every session |
| this file | where things stand |
| `DECISIONS.md` **D-033** | the gate decision, finally taken. It unblocks the report. |
| `docs/experiments/EXPERIMENT_LOG.md` **E-019, E-020** | what Phase 4 measured, and the one thing that must run before any actor-critic number counts |
| `docs/features/FEATURE_009_actor_critic.md` | what the actor-critic is and why it is built the way it is |
| `DECISIONS.md` **D-034, D-035** | the design, and the two hyperparameters deliberately left alone |
| `ROADMAP.md` Phase 5 | where the next block starts |

**5. You do NOT need anything from `results/`.** It is gitignored and every artefact is
regenerable - commands under "Reproduce on this device".

**Still true: the 66 DQN run files are only on Pranav's machine.** Every number from them
is in E-017, so nothing is lost for the write-up, but the Phase 4 sample-efficiency
comparison needs DQN reward-vs-steps **curves**, and E-017 recorded final numbers, not
curves. If that comparison is built on Diya's machine the DQN sweep must be regenerated
first (~1.4 h at 5 parallel - read D-030). On Pranav's machine the runs already exist,
which makes his the cheaper place to do box 3.

---

## 📍 PHASE 4 - where it actually stands

**Built, tested, committed, pushed:**

- `agents/reinforce.py` + `scripts/train_reinforce.py` (FEATURE_008, session 9)
- `agents/actor_critic.py` + `scripts/train_actor_critic.py` (FEATURE_009, session 10), 18 tests
- `scripts/reinforce_clip_experiment.py` (E-019) - **run, complete**
- `scripts/actor_critic_entropy_experiment.py` (E-020) - built, smoke-verified, **NOT run**

**Not done:** ROADMAP boxes 3 (sample-efficiency comparison) and 4 (variance demonstration)
are not written, and **no full training run of any Phase 4 agent exists.** The boxes stay
unticked and that is correct - same treatment as Phase 3.

### Do this before anything else in Phase 4

**Run E-020 (~10 min) and set `entropy_coef` from it.**

```powershell
.\.venv\Scripts\python.exe scripts/actor_critic_entropy_experiment.py
```

The shipped `entropy_coef: 0.01` **demonstrably breaks the actor-critic**: the policy
saturates within five episodes (entropy 0.911 to 0.0003), the actor's gradient norm falls
to 0.00, and the greedy diagnostic sits on -515.4, the Phase 3 BULK_CLOSE collapse value.
The bonus contributes at most 0.016 against TD errors reaching 1410. **No actor-critic
number means anything until this is settled.** It was not changed in session 10 because a
20-episode eyeball is not a basis for choosing a hyperparameter (D-035).

### The two findings from E-019 that change what Phase 4 reports

**1. The sampled policy beats its own argmax, in all nine runs.** Positive sampled reward
(+6 to +112) against strongly negative greedy reads (-80 to -515) of the *same policy at
the same moment*. The mixture is the strategy; the argmax of a spread-out policy is a
different and worse policy. **Both trainers evaluate through a `_GreedyView`.** This is the
new decision owed.

**2. The degenerate greedy policy is not the clip's fault.** Seven of nine runs produced
exactly -515.4 or -78.7 (constant-action policies) at every clip value tested. That is why
the actor-critic has an entropy bonus rather than inheriting REINFORCE's exploration story.

### Compute budget for the next machine - measured, and E-018's projection was wrong

| agent | measured | 20000 episodes x 5 repeats |
|---|---|---|
| REINFORCE | **~0.016 s/episode** | **~27 min** (E-018 projected ~2 h - 4.5x pessimistic) |
| actor-critic | **~0.6 s/episode** | **~3.3 h per repeat** - 37x costlier, it updates every STEP |

The actor-critic is by far the most expensive item in Phase 4 and it decides the schedule.
Anything over ~10 min needs human approval (CLAUDE.md).

### The one genuinely unfinished pre-Phase-4 item

`ROADMAP.md` Phase 2, last box: **"SARSA and Monte Carlo measured against the tiny MDP, in
the same file."** Q-learning has its anchor (FEATURE_003); the other two learners never got
one. Small, and the only real gap behind us.

---

## ⚠️ READ THIS FIRST — the eval block was widened and most numbers moved

The two decisions left open last session were taken (**D-019**, **D-020**), and re-measuring **changed the project's conclusions**.

`seeds.eval` went from 5 seeds to 30 (101–130, the original five kept inside so old results remain a sub-sample). Every agent was re-measured (**E-014**). Nothing was deleted — E-002 to E-013 stand as recorded (CONSTRAINTS #4).

| agent | recall | reward (30 seeds) | reward std | reward on 5 seeds |
|---|---|---|---|---|
| oracle_greedy | **0.87** | **168.0** | ±232.9 | 214.1 |
| q_learning | 0.72 | 47.6 | **±52.0** | 270.9 |
| sarsa | 0.66 | 40.5 | **±49.4** | 324.1 |
| severity_sort | 0.84 | 40.4 | ±220.1 | 153.7 |
| monte_carlo | 0.70 | −16.4 | ±77.0 | 177.3 |
| **dp** | **0.23** | **−201.2** | ±438.5 | **+305.9** |

**Four consequences, in order of severity:**

1. **Phase 1 closes unpassed.** DP inverted from best reward of any agent (+305.9) to worst (−201.2), recall 0.43 → 0.23, falsifying D-012 criterion 2. *(The coverage explanation first offered here was tested and refuted — see the next section and E-015.)*
2. **Phase 2 fails its gate on every count.** Recall 0.66–0.72 vs 0.84, *and* the reward advantage is gone (47.6 / 40.5 vs 40.4, inside a ±220 spread). **The gate was deliberately not restated** — D-020 explains why that differs from Phase 1's legitimate D-012 amendment. Restating it on reward *consistency*, where the learners genuinely win, was the tempting option and was rejected as goalpost-moving.
3. **The reward-hacking narrative is restated, not abandoned.** Still true: the reward is exploitable and every agent trades recall away chasing it. No longer true: that the trade pays. That is a *stronger* case for Phase 5 — the objective is not merely misaligned, it is unstable.
4. **One new positive finding, invisible at 5 seeds:** the learners are ~4× more consistent shift-to-shift than the heuristics (±50 vs ±220). Nothing in the reward function values that, which is itself evidence for learning one from humans.

**Phase 0 still passes** its gate (oracle strictly best on total reward, 168.0 vs 40.4). But its amendment's *rationale* — "no honest greedy oracle can reliably out-recall severity-camping" — is weakened: on 30 seeds the oracle out-recalls it 0.87 to 0.84. Do not repeat that sentence as established.

**The methodological lesson, which outlasts every number above.** Every figure was computed correctly, reported with its standard deviation, and reproduced deterministically — and one had **the wrong sign**. E-002 printed ±218.7 beside a mean of 153.7 and nobody drew the inference. **Reporting a standard deviation is not the same as reading it.** Compare spread to effect size *before* believing the effect. `tests/test_eval_protocol.py` now enforces the seed-count floor.

---

## Phase 3 — CLOSED as built-but-not-passed. Read before claiming anything about the DQN.

**Every one of the six ROADMAP Phase 3 boxes is still unticked, and that is
correct.** The exit criterion — *"DQN matches or beats tabular Q-learning on the
same evaluation seeds, and the two ablations visibly destabilise training"* —
remains **unmeasured**.

### The first sweep is a negative result, not a mistake to hide

Twenty 20000-episode runs completed overnight and **every one collapsed**: the
agent chose BULK_CLOSE 99.4% of the time and caught **0.9%** of incidents
(recall 0.0086, reward -480 to -520 — worse than every baseline in the project
on recall, including fifo). The greedy diagnostic sat frozen at -515.4 from
episode 500 to episode 20000.

Cause: `F.huber_loss` was called with no `delta`, so torch's default of **1.0**
applied, while `env_default.yaml` prices burying a real incident at -150 and an
end-of-shift miss at -200. Those landed in Huber's linear regime, where a 150x
larger error produced a **1.014x** larger gradient (measured). The agent learned
the small frequent rewards perfectly and was told the catastrophes were rounding
errors.

**Read `docs/bugs/BUG_002` and `E-016` before touching the DQN.** The runs are
preserved at `results/dqn_runs/dqn_delta1_E016/` — do not delete them
(CONSTRAINTS #4) and do not let a sweep write over them; the corrected sweep
writes to `results/dqn_runs/dqn/`.

The thing to carry into a viva: **the loss curve converged beautifully the whole
time** (down to 0.04). Every structural check on the network passed — Q varied
across states, actions were well separated. Nothing errored. The only symptom
was a bad number in a results table.

### The sweep is finished — 46 runs, and the gate is not met

Completed 2026-08-19 10:30. Control **n=30**, each ablation **n=8** (not 15 — the
machine could not sustain it against the deadline; see D-030 and the note in
E-017). Full result in **E-017**. Headlines:

| | recall | reward | note |
|---|---|---|---|
| tabular q_learning | **0.73** | — | the thing to beat |
| DQN control (n=30) | 0.48 +- 0.19 | -46.9 +- 145.2 | **plateaued** — more episodes buy nothing |
| no replay (n=8) | **0.0000** | -520.5 | total collapse, all 8 seeds identical |
| no target network (n=8) | **0.588** | **+43.5** | **better than control**, ratio 2.5 |

1. **Gate NOT met and NOT restated.** Same treatment as D-012/D-020.
2. **Replay is load-bearing** — without it the agent does not learn at all.
3. **The target network is counterproductive** here, by a resolvable margin.
4. `dqn_ablations.py` reported the no-replay collapse as "NO clear
   destabilisation" because volatility, end-std and drawdown are all **0.00**
   for a flatlined policy — it reported the no-replay collapse as "NO clear
   destabilisation". **FIXED** (BUG_003): `is_collapse()` now gates every
   stability ratio on whether the condition learned anything at all, and a
   collapsed arm reports "LEARNING FAILED ENTIRELY" with no ratios. Re-run
   against the same 8 runs, it now gives the right verdict. The general rule
   for Phases 4-5: **any variance measure used as a gate needs a liveness check
   in front of it.**

### What is known about the fixed agent, and what is not

Verified end-to-end at 3000 episodes x 3 runs — **15% of the training budget**:

| | before (delta 1.0) | after (delta 200) |
|---|---|---|
| recall@deadline | 0.0086 | **0.48 +- 0.21** |
| total reward | -480 to -520 | -49.4 +- 136.6 |
| greedy curve | pinned at -515.4 | moves freely |

**The collapse is fixed. The agent is not yet good.** At that budget it is still
below severity_sort on recall (0.48 vs 0.84) and reward (-49 vs +40), still
volatile, and 16 of 90 eval episodes caught nothing. Whether 20000 episodes
closes the gap to tabular Q-learning (recall 0.73, reward 270.9) is exactly the
open question. **A third documented negative result is a plausible outcome and
would be the honest one.**

**Two things to decide with a human before reporting anything:**
1. Whether the ablations' shared seed block (both on 1200000) matters. Each is
   compared against the control on 1000000, so the comparison that counts is
   unconfounded; ablation-vs-ablation is paired as a side effect.
2. The `no_replay` batch-size confound (D-026) must appear in any write-up.

## Phase 3's two blocking decisions (historical — both were taken before it started)

**Phase 1 closes as "built, criterion falsified on better measurement"** (D-022, Pranav 2026-08-18). Gate deliberately **not** amended a second time: D-012's amendment fixed a category error in the criterion; this one would just be rewriting a criterion that came out false. Both Phase 1 and Phase 2 now close unpassed. Uncomfortable pair to present; the honest one.

**E-015 refuted E-014's explanation for DP's collapse.** Tested and dead:

| measure | result |
|---|---|
| off-core **state** share | **0.0% on all 30 seeds** |
| off-core **pair** share | **0.0% on all 30 seeds** |
| corr(severity reward, DP reward) | +0.085 (control — not seed difficulty) |

DP never leaves its estimated core, and **D-011's convention never fires at evaluation time.** Coverage and D-011 are both exonerated. On seed 128 DP loses 755 where severity-sort gains 233.

**Remaining explanation — untested, labelled as such:** `P̂`/`R̂` were counted under a *uniform-random* policy, but DP bulk-closes ~97% of the time, so the transitions following its own actions aren't the ones the model was built from, even though the states are familiar. **Distribution shift in the estimate, not gaps in it.** Test named in E-015: re-estimate from DP-policy rollouts, check whether the plan's predicted value matches measured reward. `scripts/dp_collapse.py` is the harness to extend.

**This sharpens D-004 for the report.** "Optimal for the estimated model" was read all along — including by me in E-014 — as being about *coverage*. It isn't. The gap is between the policy the model describes and the policy being planned.

## What a human still has to decide

*Items 1–3 (Phase 1's gate, investigating the DP collapse, Phase 3 timing) were all decided on 2026-08-18 — see the section above and D-022. Nothing now blocks Phase 3.*

1. **Diya's countersign** on D-012, D-019, D-020 and D-022. All four change what the report claims, and none has her sign-off yet. The two Phase 0 amendments both carried it; these should match that bar.
2. **Optional, not blocking:** run the distribution-shift test named in E-015 (re-estimate `P̂`/`R̂` from DP-policy rollouts). It would convert the last open "why" into a result. Worth doing before the report is written, not necessarily before Phase 3.

---

## Whose turn is it - read before starting work

**Measured 2026-08-25 (`python scripts/commit_balance.py`, D-021 / CONSTRAINTS #24-26).**
Run it again yourself rather than trusting this table - it is a snapshot and it goes stale
with the next commit.

> **IMBALANCED - Diya is ahead, past the threshold of 3. PRANAV SHOULD TAKE THE NEXT BLOCK.**
>
> Session 10 was Diya's and it was a full block of real work: the actor-critic agent and its
> 18 tests, its trainer, two experiment harnesses, a config split, and the documentation.
> That put Phase 4 entirely under one name.
>
> **Phase 5 is the correction, and it is the right one.** It is the largest phase in the
> roadmap (`rlhf/pairs.py`, the episode renderer, `rlhf/store.py`, the labelling UI, 300
> labelled pairs, Cohen's kappa, the reward model, the re-trained policies) and it is the
> phase the project is actually about. Pranav taking it in full brings the history back
> level and puts the differentiating work under the name that needs it.
>
> **Do not correct the balance by attributing Diya's work to Pranav** (CONSTRAINTS #24). A
> commit under a name is a claim that that person did the work and can walk an examiner
> through it. The balance is corrected by handing over the work, which is what is happening.

**What Pranav should know before starting Phase 5.** Nothing in Phase 5 depends on Phase 4
finishing - CONSTRAINTS #11 guarantees it. Phase 4's remaining boxes (the sample-efficiency
comparison and the variance demo) can be picked up later by either person, and the DQN runs
that box 3 needs already exist on Pranav's machine, which makes his the cheaper place for it
whenever it happens.

---

## Before the machine changes hands

Assume every session is the last one on this machine. All of these must be true (CONSTRAINTS #25):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q     # must pass
git status --porcelain                             # must be empty
git status -sb                                     # must show no "ahead"
git push                                           # <- WITHOUT THIS THERE IS NO HANDOVER
python scripts/commit_balance.py                   # report it, act on it
```

**`git push` is the step that is easiest to forget and the only one that is
fatal.** A clean tree with 20 unpushed commits looks perfect locally and gives
the other person nothing to pull. Check `git status -sb` says no `ahead` before
you close the session.
Plus: `HANDOVER.md` (this file) actually describes the current state, and no stray zero-byte files are staged (`docs/bugs/BUG_001`).

**Nothing in `results/` is ever needed to continue** — it is gitignored and fully regenerable by the commands below. If the other machine ever needs a file from `results/`, that is a bug in the scripts.

**Anything that exists only on one machine gets written down, not remembered** — install workarounds, tool versions, path quirks. The other person cannot see this terminal. Put it under "Watch out for".

## Reproduce on this device

```powershell
cd D:\RLPROJECT
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/ -q                    # expect 126 passed

python scripts/run_baselines.py                                   # fast
python scripts/run_dp.py                                          # ~2.5 min
python scripts/train.py --agent {q_learning,sarsa,monte_carlo}     # ~4 min each
python scripts/policy_table.py --agent <name>                      # box 6
python scripts/compare_agents.py                                   # box 5
python scripts/ablations.py                                        # ~4 min
```

**Phase 3 (DQN) — expensive, read the notes first:**

```powershell
# ONE run, to sanity-check the machine (~27 min solo on Pranav's i7-13650HX)
.\.venv\Scripts\python.exe scripts/train_dqn.py --only-repeat 0 --no-plot

# the full sweep: 30 control + 15 + 15 = 60 runs. Default is 8 parallel, but for
# a long unattended run use 5 -- see the memory notes below and D-030.
.\.venv\Scripts\python.exe scripts/run_dqn_sweep.py --max-parallel 5 --max-used-fraction 0.90

# then, in any order:
.\.venv\Scripts\python.exe scripts/aggregate_dqn.py --tag dqn
.\.venv\Scripts\python.exe scripts/compare_dqn_tabular.py
.\.venv\Scripts\python.exe scripts/dqn_ablations.py
```

**Phase 4 (REINFORCE) — measured at ~0.073 s/episode, so ~24 min per 20000-episode run:**

```powershell
# smoke test (~40 s) -- proves the trainer works end to end
.\.venv\Scripts\python.exe scripts/train_reinforce.py --episodes 300 --repeats 1 --eval-every 300 --no-plot

# the full run -- NEEDS HUMAN APPROVAL FIRST (>10 min, CLAUDE.md)
.\.venv\Scripts\python.exe scripts/train_reinforce.py --repeats 5

# the variance ablation (ROADMAP box 4), on its own seed block
.\.venv\Scripts\python.exe scripts/train_reinforce.py --no-baseline --repeats 5
```

**Delete `results/reinforce_runs/` after any smoke test.** Same trap as the DQN
(D-018): a 300-episode run left where a 20000-episode run writes gets averaged in,
and the stale file looks entirely valid.

**Measured costs, and every earlier number here was wrong at least once (D-024, D-030).** A 20,000-episode run is **~27 min** when the machine is healthy, and one training process costs **~940 MB of private commit** — not the 301 MB working set that an earlier note quoted, which understated it 3x and is why the memory guard used to launch too eagerly.

Parallelism was measured, not guessed: throughput peaks at 8 and *falls* after (4/6/8/10/12 = 28.8/39.0/48.6/47.3/47.6 runs per hour). **But use 5 for a long unattended sweep.** A benchmark on a freshly-rebooted idle machine projected 9.9 min/run; the real overnight sweep degraded to **195 min/run**. Benchmarks taken on a clean machine are not predictions about an 8-hour run — that is the same failure as D-024's compute probe.

`torch.set_num_threads(1)` is deliberate and is the *fastest* setting on this net (1 / 4 / 8 threads = 159 / 172 / 375 ms per episode) — do not "optimise" it. The RTX 4060 is unusable: `torch==2.13.0+cpu` is a CPU-only build and swapping it is a dependency change (CONSTRAINTS #8).

The sweep is **restartable and extendable**. Repeats are seeded by index alone, so re-running it adds only the missing indices, and `--control-runs 40` later would add repeats 30-39 without recomputing anything.

`results/` is gitignored and every artefact above is regenerable. The pipeline is deterministic under its seeds — re-runs reproduced E-004 and E-011 exactly, and the ablation sweep row-for-row. **If a re-run does not reproduce the logged numbers, something changed that shouldn't have.**

## Done — Phase 2 in full

All eight roadmap boxes:

- **`tiny_mdp.py`** (FEATURE_002, E-006, D-014) — 2-state MDP with a pen-and-paper `q_*`, built *before* any learner so a disagreement is unambiguously the learner's fault. Bellman residual 1.78e-15; mutation-checked.
- **Three learners, each test-first** — `q_learning.py` (FEATURE_003, S&B §6.5), `sarsa.py` and `monte_carlo.py` (FEATURE_006, §6.4 and §5.4), on a shared `tabular.py` base extracted at the third implementation.
- **`train.py`** (FEATURE_004) — dedicated training seed block per algorithm (D-016), diagnostics on train seeds, eval seeds read once at the end.
- **`policy_table.py`** (FEATURE_005) — marks unvisited states as `·` rather than letting the argmax tie-break invent a preference for 455 of 576 cells.
- **`compare_agents.py`** (E-011) — policy agreement 22–44% over commonly-visited states; the naive 83–86% figure is manufactured by states neither agent visited.
- **`ablations.py`** (E-012) — **none of α, γ or ε-decay clears the noise floor.** Reported as a negative result rather than filled in.

## Watch out for

**Added 2026-08-23 (session 9):**

- **Adding a field to a frozen config dataclass breaks every construction site of
  it, including the ones in tests.** `ablation_seed_start` was added to
  `ReinforceConfig` and all 16 REINFORCE tests died with `TypeError`, because
  `_rcfg()` builds the dataclass from an explicit field dict. It went unnoticed for
  three commits because only the two config test files were run afterwards. **Run
  the whole suite after any config dataclass change**, not the tests that look
  related.
- **The test suite is now ~8 minutes, up from ~1.** REINFORCE's tiny-MDP anchor is
  most of it. Budget for that before starting a tight edit-test loop.
- **`BUG_001`'s zero-byte stray files are still happening**, and heredocs are a new
  source of them: three appeared this session (`the`, `100000`, `28}`).
  `git status --porcelain` before every commit, and never `git add -A`.
- **Writing files with backslashes or box-drawing characters through a bash
  heredoc mangles them.** The venv path acquired a doubled segment once,
  doubled backslashes were silently eaten twice, and one heredoc failed
  outright with `unexpected EOF`. Use the editor for those files, or build
  the path with `chr(92)`. Same family as session 3's `Set-Content` em-dash
  damage.
- **Do not read anything into REINFORCE matching severity-sort exactly** without
  reading E-018 first. It is a genuine constant-policy collapse, not a scoring
  bug, and the distinction is the whole point.


**Added 2026-08-19 (session 8) — read these first, they cost the most time:**

- **A third-party memory tool nearly destroyed the second sweep too.** `Mem Reduct` runs at startup and was set to auto-clean whenever memory exceeded 90%, with **"Working set" among the regions it clears**. Clearing a working set calls `EmptyWorkingSet` on every process, so the trainers' resident pages are evicted and must instantly be faulted back in. Symptoms: ~96000 page faults/sec with almost no disk reads, CPU at 19%, and runs going from **27 min to 195 min**. Turning the auto-clean off took efficiency from 42% back to **95%**. If a long run mysteriously slows down, check for tools like this *before* blaming the code. (D-030.)
- **Never trust a benchmark taken on a freshly-rebooted idle machine.** An 800-episode benchmark projected 9.9 min per 20000-episode run. Reality was 27 min at best and 195 min at worst. Same failure as D-024: the right quantity, measured under the wrong conditions.
- **`huber_delta` must never go back to torch's default.** It is the single value that destroyed 20 completed runs (E-016, BUG_002). `config.py` now refuses anything below 50, and `tests/test_dqn.py::test_a_buried_incident_...` fails on the old behaviour. Do not "simplify" the explicit `delta=` argument away.
- **Do not let a sweep write over `results/dqn_runs/dqn_delta1_E016/`.** That directory holds the 20 collapsed runs and is the evidence for E-016 (CONSTRAINTS #4). The corrected sweep writes to `results/dqn_runs/dqn/`.
- **Child trainer output is invisible while a sweep runs.** `run_dqn_sweep.py` launches `train_dqn.py` without `-u`, so each `repeatN.log` stays empty until its run finishes. Progress can only be read from the scheduler log's `done` lines. Worth fixing; not fixed, because doing so mid-sweep would mean restarting it.
- **The machine's idle baseline is not stable across reboots.** One reboot came up at 2.2 GB used, the next at 11.2 GB — mostly Dell/NVIDIA/Alienware agents, MCP `node` processes and Claude Code itself. **Measure available memory immediately before launching**, do not assume a reboot bought you room.

**Added 2026-08-18 (Phase 3):**

- **The design spec's compute budget is retracted.** `docs/superpowers/specs/2026-08-18-dqn-design.md` §12 says a 20,000-episode run costs 4.6 min. It costs **~27 min** on a healthy machine (an earlier note in this file said 68 min, measured while ten processes fought over memory — also wrong; see D-030). The pre-design probe timed a fragment of the gradient step. Real numbers: 9.87 ms/gradient step, 0.709 ms/`act()`, 0.204 s/episode. See D-024 — the decision survived, its stated reason did not.
- **The RTX 4060 cannot be used.** `torch==2.13.0+cpu`; `torch.cuda.is_available()` is `False`. Switching builds is a dependency change and needs approval (CONSTRAINTS #8). It would also probably be *slower* — 19,461 parameters at batch 64 is far too small to amortise kernel-launch overhead.
- **Do not "fix" `torch.set_num_threads(1)`.** It is the fastest setting measured (1/4/8 threads = 159/172/375 ms per episode) *and* it is what makes ten-way process parallelism possible.
- **`Start-Job` does not survive between Claude Code tool calls** — each invocation is a fresh PowerShell process and the job dies with it. Use `Start-Process`, or the harness's own background mode, for anything long.
- **Delete `results/dqn_runs/` after any smoke test.** A 40-episode run left there can be averaged into a 20,000-episode sweep. `run_dqn_sweep.py` and `aggregate_dqn.py` both guard against it now, but the cheapest fix is not to create the mess.
- **Stray zero-byte files appeared repeatedly this session** (`10-min`, `10000`, `np.ndarray`, `tuple[float`, `{wall_min`, `5`, `None`, `1200000`) — BUG_001, apparently from shell-quoting artefacts. Sweep `git status --porcelain` before every commit. Names containing `[` or `{` need `Remove-Item -LiteralPath`.
- **`config.py` is 640 lines, over the 500-line limit** (CONSTRAINTS #12). It was already 539 before Phase 3, so the violation predates this work, but it is now the only file in the repo over the limit and it should be split.
- **`ROADMAP.md:101`** — the Phase 2 box "SARSA and Monte Carlo measured against [the tiny MDP], in the same file" is unticked while this file claims all 8 Phase 2 boxes are done. `tests/test_on_policy.py` appears to contain the work, in a different file than the box specifies. Doc reconciliation, not missing work.
- **`config.py`'s seed-block error message still says "eval (101-105)"** — the block has been 101–130 since D-019. Deliberately left alone so a Phase 3 commit did not touch Phase 2 behaviour.


- **Do NOT quote a number without checking whether it is 5-seed or 30-seed.** E-002 to E-013 are 5-seed. E-014 onward is 30-seed. The two differ enough to reverse a conclusion.
- **Do NOT "fix" SARSA or Monte Carlo to match `HAND_COMPUTED_Q`** (D-017). They are on-policy and correctly converge to `tiny_mdp.epsilon_soft_q(ε)`, which differs from q\* by >1.5 at ε = 0.1.
- **MC trains at `MC_HORIZON` (800), not `HORIZON` (200).** Not an inconsistency to tidy: MC computes a return from every timestep, so a 200-step truncation biases it down by up to 0.47.
- **`scripts/train.py` MUST call `agent.end_episode()` once per episode** (D-015). Forget it and ε stays at 1.0 forever, silently.
- **Any behavioural claim must be replicated across all three learners before being written down** (E-013). The Q-learning strategy-shift finding was monotonic, internally consistent and plausible — and SARSA reverses it.
- **A reduced training run writes to `results/smoke/`** (D-018). If a script reports coverage disagreeing with the logged experiments, suspect a stale artefact before suspecting the script.
- **Stray zero-byte files** appear whenever `>` occurs in written content, including Python return annotations — see `docs/bugs/BUG_001`. Mitigation: **`git add <explicit paths>`, never `git add -A`**, and sweep before committing. `Remove-Item` needs `-LiteralPath` for names with `[ ]`.
- **Never rewrite a source file through PowerShell's file cmdlets** — `Set-Content` mangled every em dash in `train.py` and added a BOM. Use the editor.
- **The default branch is `master`, not `main`.** The remote has one head.
- **Two modules in `src/` hold hard-coded numbers** — `mrp_example.py` (D-013) and `tiny_mdp.py` (D-014). Both look like CONSTRAINTS #9 violations and neither is. `tiny_mdp.py` uses γ = 0.9 deliberately, not the project's 0.99.
- Seed blocks: train-diag 1–10 · **eval 101–130** · calibration 1000–3099 · DP estimation 10000–59999 · q_learning 200000+ · sarsa 400000+ · monte_carlo 600000+ · ablations 800000+. Disjointness is enforced in `config.py`, not trusted to comments.

## Still owed by the humans

- **THE BIG ONE IS DONE.** The gate decision covering Phases 1-4 was taken by Diya on
  2026-08-25 and recorded as **D-033**: the criteria stay exactly as written, and failing
  phases close *built-but-not-passed*. Option (b) of the three this file listed. The report
  is no longer blocked on it, and Phase 4 will be judged against its criterion unchanged.
- **NEW, and it affects every Phase 4 number: greedy or sampled evaluation?** E-019
  section 3 found that in all nine runs the sampled policy earned positive reward while the
  greedy read of that same policy earned strongly negative reward. Both Phase 4 trainers
  evaluate through a `_GreedyView`, so every number they produce - including E-018's
  "REINFORCE has become severity-sort exactly" - describes the argmax rather than the agent.
  **A stochastic policy's argmax is not that policy.** Options: report the sampled policy,
  report the greedy one with the caveat stated, or report both. Take it before the full
  runs, not after seeing which scores better.
- **The REINFORCE gradient clip is settled** (E-019, D-035). It stays at 10.0 - not
  vindicated, but the between-value spread (74.4) is a third of the within-value spread
  (211.6), so nothing justifies moving it.
- **`entropy_coef` is NOT settled** and blocks the actor-critic. E-020 is built and costs
  ~10 min. See the Phase 4 section above.

- **The pen-and-paper derivations.** FEATURE_001's MRP and FEATURE_002's tiny MDP are both ticked because the derivation and its verification exist — not because Pranav and Diya can reproduce them cold, which is what the viva tests. FEATURE_002 is the likelier exam question: two states, four numbers, γ = 0.9.
- KPMG analyst contact for preference labels — longest-lead item in the project; Phase 5 cannot start without it.
- Report format / team-size confirmation from Dr. Kaur; target demo date.
