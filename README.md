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
.\.venv\Scripts\python.exe -m pytest tests/ -q     # expect 50 passed
```

Python 3.13.1 confirmed on this machine. Verify the PyTorch install works before Phase 3 depends on it.

`results/` is gitignored. Regenerate it with `scripts/run_baselines.py` and `scripts/run_dp.py` (DP takes ≈ 2.1 min).

---

## Current status

**Phase 0 complete** (2026-08-14) · **Phase 1 complete** (2026-08-16) · **Phase 2 — tabular model-free RL, in progress.**

Built and tested: the simulator, the environment, both state encoders, five baselines, and model-based Dynamic Programming (value iteration + policy iteration) with an external correctness check against a hand-solved 5-state MRP.

The headline finding so far is a negative one, and it is deliberate: the DP policy earns the **highest total reward of any agent (305.9)** while catching **fewer than half the real incidents (recall 0.43)**. It games the hand-written reward. We kept the exploitable reward rather than patching it, because a reward nobody can write correctly by hand is the entire argument for learning one from human preferences in Phase 5. See `DECISIONS.md` D-012 and `EXPLAIN.md` Part 9.

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
