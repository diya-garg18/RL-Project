# TriageRL

**An RL agent that learns which security alert a human analyst should investigate next — trained first on a hand-written reward, then re-trained on a reward model learned from human preferences (RLHF).**

CS4148 Reinforcement Learning · VII Semester · Manipal University Jaipur
Diya Garg · Pranav Upadhyay  

---

## Start here

**If you're a new AI session:** read `CLAUDE.md` first. It has the session protocol. Then `HANDOVER.md` for where things stand, then `ROADMAP.md` for what's next.

**If you're a human who wants to understand the project:** read `EXPLAIN.md`. It assumes you know nothing and uses no unexplained jargon.

**If you want the full technical spec:** `PROJECT_BRIEF.md`.

---

## The documents

| File | What it's for | Updated |
|---|---|---|
| **`EXPLAIN.md`** | **Everything, in plain English. The one to read first.** | Every session |
| `PROJECT_BRIEF.md` | The idea, the MDP, the plan, the limitations | Rarely |
| `ROADMAP.md` | Phase-by-phase build order with exit criteria | As tasks complete |
| `ARCHITECTURE.md` | The system map — modules, data contracts, layout | When structure changes |
| `FLOW.md` | How execution travels between files | When paths change |
| `CONSTRAINTS.md` | What must never happen | Ask before editing |
| `HANDOVER.md` | Where things stand right now | Every session |
| `DECISIONS.md` | Why each choice was made (append-only) | Every decision |
| `TEST_CHECKLIST.md` | What "done" means — real commands, real outputs | When checks are added |
| `ROLLBACK.md` | How to undo | When risky work starts |
| `INTERVIEW_PREP.md` | The functions and answers you must know cold | Week 6 (read Week 1) |
| `docs/experiments/EXPERIMENT_LOG.md` | Every training run and its result | Every run |
| `docs/features/` | One file per feature, start to finish | Per feature |
| `docs/bugs/` | One file per bug, start to finish | Per bug |

The first ten follow the *AI Collaboration Field Guide* ("Don't just trust the AI. Trace it."). `EXPLAIN.md` and `INTERVIEW_PREP.md` are additions specific to this project — see `DECISIONS.md` D-006.

---

## Setup

```powershell
cd D:\RLPROJECT
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/ -q     # expect 76 passed
```

Python 3.13.1 confirmed on this machine. Verify the PyTorch install works before Phase 3 depends on it.

`results/` is gitignored. Regenerate it with `scripts/run_baselines.py` and `scripts/run_dp.py` (DP takes ≈ 2.1 min).

---

## Current status

**Phase 0** closed (2026-08-14) · **Phase 1 REOPENED** · **Phase 2 closed as built-but-not-passed** (2026-08-16) · **Phase 3** not started.

Built and tested: the simulator, the environment, both state encoders, five baselines, model-based Dynamic Programming, and all three tabular learners — Q-learning, SARSA and first-visit Monte Carlo — each verified against a hand-solved MDP before touching the real one.

Results on the **30-seed** evaluation block (widened from 5 on 2026-08-16 — see below):

| agent | recall@deadline | total reward | reward std |
|---|---|---|---|
| oracle_greedy | **0.87 ± 0.16** | **168.0** | ±232.9 |
| q_learning | 0.72 ± 0.04 | 47.6 | **±52.0** |
| sarsa | 0.66 ± 0.04 | 40.5 | **±49.4** |
| severity_sort | 0.84 ± 0.16 | 40.4 | ±220.1 |
| monte_carlo | 0.70 ± 0.05 | −16.4 | ±77.0 |
| dp | 0.23 ± 0.24 | −201.2 | ±438.5 |

**Phase 2 did not pass its exit criterion, and the gate was deliberately not restated to make it pass** (`DECISIONS.md` D-020). Every learner loses to a severity sort on recall and none reliably beats it on reward. What the learners *do* win on — invisible until the eval block was widened — is **consistency**: roughly four times less shift-to-shift variance than the heuristics. Nothing in the hand-written reward values that, which is itself an argument for learning a reward from humans.

### The finding the project is most defined by

**The evaluation protocol was too weak to support the project's own conclusions, and fixing it reversed one of them.** On 5 seeds the DP policy earned the highest reward of any agent (+305.9) and that was Phase 1's headline result. On 30 seeds it earns the **worst of any agent (−201.2)**. Phase 1 is reopened; E-004 is not deleted (`CONSTRAINTS.md` #4) but its gate assessment is withdrawn.

Every number involved was computed correctly, reported with its standard deviation, and reproduced deterministically — and one had the wrong sign. The very first baseline table printed ±218.7 beside a mean of 153.7 and nobody drew the inference. **Reporting a standard deviation is not the same as reading it.**

The reward-hacking story survives in restated form: the reward *is* exploitable and every agent trades recall away chasing it, but the trade does not reliably pay. An objective that is both gameable and unstable is a stronger argument for RLHF than one that is merely gameable. See D-012, D-019, D-020 and `EXPLAIN.md` Part 9.

**Also documented rather than deleted:** the hyperparameter ablations show *no* effect clearing the measurement noise (E-012), and a strategy-shift claim was **retracted** when it failed to replicate across algorithms (E-013).

See `HANDOVER.md` for the live picture.

---

## The idea in four lines

A security team gets thousands of alerts a day and can investigate a handful. Almost all are false alarms; a few are real breaches. Most teams just sort by the vendor's severity label, which knows nothing about their business, their critical servers, or how much time is left in the shift.

We frame it as an MDP — state = the queue situation and time remaining, actions = five triage strategies, reward = catch real incidents fast without burning analyst time — and learn a policy. Then, because the reward is genuinely un-writable by hand (how many wasted minutes equal one missed breach?), we learn the reward itself from human preference comparisons and audit the result for reward hacking.

---

## Layout

```
src/soc_triage/   simulator, environment, agents, evaluation, RLHF
config/           every tunable number
scripts/          thin CLI entry points
tests/            pytest
docs/             features, bugs, experiment log
results/          runs, plots, checkpoints (gitignored)
web/              FastAPI + React dashboard and labelling UI (Phase 5+)
```
