# HANDOVER.md — Where things stand right now

> Field Guide habit #1 and #13. Read this first, every session. Rewrite it last, every session.
>
> This is **not** a changelog — it's a snapshot of the present. Overwrite stale entries rather than appending. (The permanent record lives in `DECISIONS.md` and `docs/experiments/EXPERIMENT_LOG.md`.)

---

## Snapshot

| | |
|---|---|
| **Last session** | 2026-08-14 (session 4) |
| **Model** | Claude Fable 5 |
| **Current phase** | Phase 1 (DP) — **code complete and run; exit gate blocked on one human decision** (below) |
| **Repo state** | Pushed to `origin/master` (github.com/diya-garg18/RL-Project), in sync. Phase 0 closed; Phase 1 DP built and evaluated. |
| **Tests passing** | 7/7 (`pytest tests/ -q`) |
| **Device switch** | Safe. Everything committed AND pushed. On the new device: `git clone`, recreate `.venv`, `pip install -r requirements.txt`. |

---

## Reproduce the environment on the new device

```
git clone https://github.com/diya-garg18/RL-Project.git
cd RL-Project
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # pinned; torch 2.13.0 works on Python 3.13
.\.venv\Scripts\python.exe -m pytest tests/ -q                   # expect 7 passed
```
`results/` is gitignored — regenerate with `scripts/run_baselines.py` and `scripts/run_dp.py` (DP run ≈ 2 min total).

## Done

- **Phase 0 complete** (gate approved by Diya; E-001 calibration, E-003 baselines, 7 tests)
- **Phase 1 DP complete as code** (E-004): 50k-episode model estimation (1.2 min), value iteration (converged Δ 9.95e-05, 1075 sweeps), policy iteration (**VI/PI agree 100%**), coverage report (133/576 states), convergence plot, evaluation on eval seeds
- Generator vectorised (38 min → 1.2 min for 50k episodes; recalibration re-confirmed 3.20% / r=0.321)
- D-011 (unvisited-state handling), E-004, EXPLAIN Parts 7+8, FLOW Flow C ✅, ROADMAP Phase 1 boxes — all current

## Broken / blocked

**Phase 1 exit gate needs a human decision before Phase 2 starts (E-004):**

> The DP policy **reward-hacks**. It scores the highest reward of anything (306 > oracle 214 > severity-sort 154) while being a *worse* triage system than severity-sort — recall 0.43 vs 0.87. It bulk-closes ~97% of the time as profitable idling and abandons 57% of real incidents. The hand-written reward genuinely rates this optimal. **This is the brief §3.5 deliberate trap, sprung two phases early by exact planning — the project's cleanest RLHF motivation, caught red-handed by our own code.**
>
> The original exit says "DP beats severity-sort on recall" — it doesn't, and *can't*, because DP optimises the (exploitable) reward, not recall. **Decision (same shape as the Phase 0 amendment): restate the Phase 1 exit on total reward + log the hack as a headline finding — OR treat the reward as a bug to patch (but the brief says the trap is intentional; patching deletes the Phase 5 motivation).** My recommendation: keep the reward, restate the gate, feature the hack. Do NOT start Phase 2 until Diya/Pranav decide.

## Next session should do

1. **Resolve the Phase 1 exit decision above.** Amend ROADMAP wording accordingly and declare Phase 1 closed.
2. Then **Phase 2 — tabular model-free RL**: `agents/monte_carlo.py`, `agents/sarsa.py`, `agents/q_learning.py` (all from scratch, ε-greedy, plain loops). `runner.run_episode(learn=True)` is already the training hook. `scripts/train.py` needs writing. Convergence comparison vs the DP Q-values; print the learned policy table; the tiny 2-state-MDP correctness test (`tests/test_tabular.py`).
3. **Expect the same recall tension in Phase 2** (E-003 implication 2 / E-004): Q-learning optimises the same exploitable reward, so it may also under-perform on recall while winning reward. Frame Phase 2's "beat severity-sort" claim on the objective, and watch for the bulk-close hack reappearing — that continuity (DP hacks → Q-learning hacks → RLHF fixes) is a strong report arc.

## Watch out for

- **`git add -A` twice now swept stray zero-byte files into commits** (shell-redirect artifacts named things like `If`, `uninformed`, `10`). Session 4 cleaned six of them. On the new device prefer `git add <explicit paths>`, and check `git status` for oddly-named 0-byte files before committing.
- The DP hack is REAL, not an estimated-model artefact — it reproduces in the true environment. Don't "fix" it thinking it's a bug (D-004 + D-011 caveats still apply to the DP policy's validity on unvisited states).
- Severity-sort (recall 0.87) is the honest opponent to beat on the *objective*, not on recall in isolation.
- Seeds: train 1–10, eval 101–105, calibration 1000–3099, DP estimation 10000–59999. Any new purpose gets a fresh disjoint block (config enforces train/eval disjointness in code).

## Open questions for the humans

1. **Phase 1 exit decision (blocks Phase 2) — see Broken/blocked.**
2. Pen-and-paper 5-state MRP for the report (ROADMAP Phase 1, last box) — a human task, not done.
3. KPMG analyst for preference labels — still open, needs lead time.
4. Report format / team-size confirmation from Dr. Kaur — still open.
5. Target demo date — still open.
