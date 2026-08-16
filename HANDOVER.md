# HANDOVER.md — Where things stand right now

> Field Guide habit #1 and #13. Read this first, every session. Rewrite it last, every session.
>
> This is **not** a changelog — it's a snapshot of the present. Overwrite stale entries rather than appending. (The permanent record lives in `DECISIONS.md` and `docs/experiments/EXPERIMENT_LOG.md`.)

---

## Snapshot

| | |
|---|---|
| **Last session** | 2026-08-16 (session 5) |
| **Model** | Claude Opus 5 |
| **Current phase** | **Phase 2 — tabular model-free RL. Not started.** Phases 0 and 1 both closed. |
| **Repo state** | Device migrated to `D:\RLPROJECT`. Committed locally; **not yet pushed** — see "Next session should do" #1. |
| **Tests passing** | **14/14** (`pytest tests/ -q`) — was 7, plus 7 new in `test_mrp_bellman.py` |
| **Blockers** | **None.** The Phase 1 gate decision that blocked session 4 is resolved (D-012). |

---

## Reproduce the environment on this device

```powershell
cd D:\RLPROJECT
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/ -q          # expect 14 passed
```
`results/` is gitignored — regenerate with `scripts/run_baselines.py` and `scripts/run_dp.py` (DP ≈ 2.1 min). Both were re-run on this device 2026-08-16 and reproduce the recorded numbers exactly: baselines match E-003, DP matches E-004 (VI Δ 9.95e-05, 1075 sweeps, VI/PI 100%, coverage 133/576).

## Done

- **Phase 0 complete** (2026-08-14, Diya-approved; E-001 calibration, E-003 baselines)
- **Phase 1 COMPLETE** (2026-08-16) — all six boxes ticked:
  - DP model estimation, value iteration, policy iteration, convergence plot, real-environment evaluation (E-004, session 4)
  - **The last box closed this session:** 5-state MRP hand-worked with explicit Bellman equations, verified against code (FEATURE_001, E-005). Four routes to the same value function — hand, closed form, iterative evaluation, and the shipped `agents/dp.value_iteration` — agreeing to **7.11e-15**.
- **D-012** — Phase 1 exit gate restated on total reward; reward hack kept deliberately and promoted to a headline finding.
- **D-013** — the MRP's constants live in code, not config; a narrow, documented exception to CONSTRAINTS #9.
- Stale documentation swept: `ROADMAP.md` "current phase" line, `TEST_CHECKLIST.md` Phase 0 wording (still said the amendments were awaiting sign-off — they were approved 2026-08-14), `EXPLAIN.md` Part 8 (E-002 table now marked superseded by E-003).
- New files: `src/soc_triage/mrp_example.py`, `tests/test_mrp_bellman.py`, `scripts/run_mrp_example.py`, `docs/features/FEATURE_001_mrp_worked_example.md`.

## Broken / blocked

**Nothing is blocked.** Phase 2 can start immediately.

Two things are *owed* but block nothing:

1. **Diya's countersign on D-012.** Both Phase 0 amendments carried her approval; this one currently has Pranav's only. Get it before the report cites the Phase 1 gate. If she disagrees and wants the reward patched instead, that is a CONSTRAINTS #15 change and invalidates E-002/E-003/E-004 — decide before Phase 2 training runs, not after.
2. **The pen-and-paper half of the MRP box.** The roadmap box is ticked because the derivation and its verification exist. It does **not** mean Pranav and Diya can each reproduce it unaided, which is what the viva actually tests. Tracked under `TEST_CHECKLIST.md` → "The human check".

## Next session should do

1. **Push to `origin/master`.** This session committed locally but did not push — the remote is still at `38ee87d`. `git push origin master`.
2. **Start Phase 2 — tabular model-free RL.** `agents/monte_carlo.py` (first-visit MC control), `agents/sarsa.py` (on-policy TD), `agents/q_learning.py` (off-policy TD with ε-decay). All by hand, plain loops, ε-greedy. `runner.run_episode(learn=True)` is already the training hook. `scripts/train.py` needs writing.
3. Then: learning curves over 5 seeds, convergence comparison against the DP Q-table (max-norm distance + policy agreement %), the printed policy table (headline viva figure), the α/γ/ε ablations, and `tests/test_tabular.py` with the hand-checkable 2-state MDP.

## Watch out for

- **Expect the bulk-close reward hack to reappear in Phase 2.** Q-learning maximises the same exploitable reward DP did. If it does recur, that is a *result*, not a bug — it demonstrates the pathology lives in the reward rather than in any one algorithm, which is the report's spine (DP hacks → Q-learning hacks → RLHF fixes). `ROADMAP.md` Phase 2's exit criterion is **flagged but deliberately not pre-emptively weakened** — run it first, decide on real numbers, exactly as Phases 0 and 1 did.
- **The DP reward number must never be quoted without its recall beside it** (D-012). 305.9 reward with 0.43 recall is the finding; 305.9 alone is misleading.
- **Stray zero-byte files are still appearing.** Two more this session (`This`, `V(QUIET)` — created 12:53 and 12:57, deleted before commit), same phenomenon session 4 hit. Root cause **not** confirmed; the pattern is that something in the tooling chain interprets a `>` inside written content as a shell redirect. Mitigation stands: **`git add <explicit paths>`, never `git add -A`**, and check `git status` for oddly-named 0-byte files before every commit.
- **`mrp_example.py` uses γ = 0.9, not the project's 0.99.** Deliberate (keeps the hand arithmetic an exact fraction, 52/11). Don't "fix" it to match config.
- **If `test_mrp_bellman.py` ever fails, fix `agents/dp.py` — never the expected values.** They came from a human with a pen; editing them to make the test pass destroys the only external correctness anchor Phase 1 has.
- Seeds: train 1–10, eval 101–105, calibration 1000–3099, DP estimation 10000–59999. Any new purpose gets a fresh disjoint block (config enforces train/eval disjointness in code).
- `README.md` still shows the old Desktop path in its Setup block and says "no code written yet" under Current status. Cosmetic, untouched this session, worth 2 minutes next time.

## Open questions for the humans

1. **Diya's countersign on D-012** (see Broken/blocked #1).
2. KPMG analyst for preference labels — still open, needs lead time. This is the longest-lead item in the project and Phase 5 cannot start without it.
3. Report format / team-size confirmation from Dr. Kaur — still open.
4. Target demo date — still open.
