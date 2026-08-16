# HANDOVER.md — Where things stand right now

> Field Guide habit #1 and #13. Read this first, every session. Rewrite it last, every session.
>
> This is **not** a changelog — it's a snapshot of the present. Overwrite stale entries rather than appending. (The permanent record lives in `DECISIONS.md` and `docs/experiments/EXPERIMENT_LOG.md`.)

---

## Snapshot

| | |
|---|---|
| **Last session** | 2026-08-16 (session 6) |
| **Model** | Claude Opus 5 |
| **Current phase** | **Phase 2 — tabular model-free RL.** 5 of 8 boxes done. Q-learning has now run on the real environment (E-008). **The Phase 2 exit criterion is NOT met, and two decisions are owed to the humans.** Phases 0 and 1 closed. |
| **Repo state** | `D:\RLPROJECT`, branch `master`. **Session 6's work is not yet committed** — see "Next session should do" #1. Last commit `4833775`. |
| **Tests passing** | **50/50** (`.\.venv\Scripts\python.exe -m pytest tests/ -q`, 2.34 s) — was 14; +13 `test_tiny_mdp.py`, +23 `test_tabular.py` |
| **Blockers** | **None technical.** Two decisions are owed by the humans before Phase 2 can close — see "Broken / blocked". |

---

## Reproduce the environment on this device

```powershell
cd D:\RLPROJECT
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/ -q          # expect 27 passed
```
`results/` is gitignored — regenerate with `scripts/run_baselines.py` and `scripts/run_dp.py` (DP ≈ 2.1 min). Both were re-run on this device 2026-08-16 (session 5) and reproduce the recorded numbers exactly: baselines match E-003, DP matches E-004 (VI Δ 9.95e-05, 1075 sweeps, VI/PI 100%, coverage 133/576). Not re-run in session 6 — nothing session 6 touched can affect them.

## Done

- **Phase 0 complete** (2026-08-14, Diya-approved; E-001 calibration, E-003 baselines)
- **Phase 1 complete** (2026-08-16, session 5; E-004 DP, E-005 the MRP anchor, D-012, D-013)
- **Phase 2 opened this session with the correctness anchor, built ahead of the learners** — FEATURE_002, E-006, D-014:
  - `src/soc_triage/tiny_mdp.py` — a 2-state, 2-action, deterministic, continuing MDP, γ = 0.9. Hand-derived `q_* = [[10.0, 6.7], [10.7, 13.0]]`, `V* = [10, 13]`, `π* = [WAIT, WORK]`.
  - `tests/test_tiny_mdp.py` — 13 tests. Bellman-optimality residual **1.78e-15**; `agents/dp.value_iteration` reproduces the hand answer.
  - **Mutation-checked:** injecting a 0.1 error moves the residual to ~0.10, thirteen orders of magnitude above the 1e-12 tolerance. The anchor demonstrably detects wrong answers.
  - `docs/features/FEATURE_002_tiny_mdp_qstar.md` — the full derivation and the two rejected designs.
- **Q-learning, built test-first** — FEATURE_003, E-007, D-015:
  - `src/soc_triage/agents/q_learning.py` — off-policy TD control (S&B §6.5). Explicit loops, own seeded RNG, no default hyperparameters.
  - `tests/test_tabular.py` — 20 tests. RED verified first (`ModuleNotFoundError`), then GREEN.
  - **Result:** reproduces `tiny_mdp.HAND_COMPUTED_Q` to **9.24e-14**; correct policy after **10 episodes**, machine precision by ~50; identical across 5 seeds.
  - `src/soc_triage/config.py` — `epsilon` and `q_learning` sections now loaded (they already existed in the YAML, unread), with range validation.
  - **Suspicious result investigated, not celebrated** (CONSTRAINTS #5): std across 5 seeds was exactly 0. Confirmed the seeds genuinely diverge early and converge to the same unique fixed point — correct for a deterministic MDP. Locked down in a test.
- **`scripts/train.py` and the first real learning run** — FEATURE_004, E-008, D-016:
  - Trains on a dedicated seed block (200000+, one fresh shift per episode), diagnoses on seeds 1–10, reads eval seeds 101–105 **exactly once, at the end**.
  - 5 runs × 20,000 episodes in **2.8 min**. Baseline rows reproduce E-002/E-004 exactly — the eval path is unchanged and still gives the same answers.
  - **Result: recall 0.73 ± 0.03, reward 270.9 ± 105.5, MTTD 22.0** vs severity_sort 0.87 / 153.7 / 23.0.

- **The readable policy table** — FEATURE_005, E-009: `scripts/policy_table.py` → `results/policy_table.md`. The strategy shift is real and **monotonic across all three time buckets** (severity-first 34.9% → 28.0% → 15.4%; bulk-close 25.3% → 36.0% → 46.2%). Coverage 121/576 states; the crunch column rests on **13 states**. The agent now records per-(s,a) visit counts purely so 455 unvisited states print as `·` rather than as a confident `PULL_HIGHEST_SEVERITY` via the argmax tie-break.
  - **Two readings of the shift, not separated by the data:** analyst-like escalation under deadline pressure, or the E-008 reward hack intensifying where end-of-shift miss charges bite. Both fit. Deliberately not resolved in favour of the flattering one.
  - The re-run needed to capture visit counts reproduced E-008 to the digit — a free determinism check.

## The two things a human has to decide

**1. The Phase 2 exit criterion is NOT met.** It requires beating severity-sort on recall *and* MTTD. Recall 0.73 < 0.87 — fails. MTTD marginally better, inside the spread. Reward wins clearly (270.9 vs 153.7) on the metric the gate does not use.

The gate is left **unmet and unamended on purpose**, per D-012: a criterion contradicted by measurement is decided by a human with real numbers in hand, not patched by whoever ran the experiment. The reward hack recurred exactly as D-012 predicted — **BULK_CLOSE is 62.3% of the learned policy's actions** (DP: ~97%). Two unrelated algorithms finding the same exploit is the evidence that the pathology is in the reward, and it is the strongest possible setup for Phase 5. That is a result, not a defect.

**2. The eval seed block is not representative — and this one is bigger than Phase 2.** Every agent, *including the oracle*, scores far better on seeds 101–105 than on seeds 1–10:

| agent | seeds 1–10 | seeds 101–105 | gap |
|---|---|---|---|
| random | −201.1 ± 218.9 | −155.7 ± 148.9 | +45 |
| severity_sort | −78.7 ± 325.5 | +153.7 ± 218.7 | **+232** |
| oracle_greedy | +94.0 ± 317.3 | +214.1 ± 207.6 | +120 |

Per-seed spread (±325 for severity_sort) is **several times larger** than the ~117-point effect E-008 reports. This affects **E-002, E-003, E-004 and E-008 alike** — every headline comparison in the project rests on 5 shifts from a distribution far noisier than the effects being measured. CONSTRAINTS #3's "at least 5 seeds" was honoured; 5 is simply too few for this environment, which nobody had measured until now.

**Nothing was changed in response.** Widening the eval block invalidates the comparability of every prior experiment and means re-running them. That is the humans' call. Ask Diya at the same time as the D-012 countersign — it is the same kind of decision.

- **Documentation sweep** (the user asked for docs to be updated as work lands, not at session end): `README.md` had the old Desktop path and still said "no code written yet"; `ARCHITECTURE.md` and `FLOW.md` still carried "no code exists yet" banners from 2026-08-13; `TEST_CHECKLIST.md` still said 7 tests. All corrected. Also fixed a docstring in `test_mrp_bellman.py` pointing at a filename that does not exist.

## Broken / blocked

**Nothing is blocked.**

Three things are *owed* but block nothing:

1. **Session 6's work is uncommitted.** Six new/modified files. Commit before anything else next session.
2. **Diya's countersign on D-012** (carried over from session 5). Both Phase 0 amendments carried her approval; D-012 has Pranav's only. Get it before the report cites the Phase 1 gate. If she wants the reward patched instead, that is a CONSTRAINTS #15 change and invalidates E-002/E-003/E-004 — decide before Phase 2 training runs, not after.
3. **The pen-and-paper half of both worked examples.** FEATURE_001's MRP and now FEATURE_002's tiny MDP are both ticked because the derivation and its verification exist. Neither means Pranav and Diya can reproduce them unaided, which is what the viva tests. Tracked under `TEST_CHECKLIST.md` → "The human check". FEATURE_002 is the more likely exam question of the two: two states, four numbers, γ = 0.9.

## Next session should do

1. **Commit session 6's work** — `git add` the paths explicitly (never `-A`, see below), message `phase2: hand-solved 2-state MDP anchor + tabular Q-learning + doc sweep`.
2. **Get the two decisions above answered** before spending compute on ablations. Box 7's α/γ/ε sweep is close to meaningless while the measurement noise is several times the effect size — that would be tuning against noise, and the results would have to be thrown away if the eval block changes.
3. **Settle the two readings of the strategy shift** with a per-action reward decomposition inside the crunch bucket (E-009). Cheap, needs no decisions, and it feeds the report's central argument — is the agent learning analyst-like escalation, or intensifying the reward hack? Highest-value remaining analysis.
4. **The DP convergence comparison** (box 5): max-norm distance between the Q-learning and DP Q-tables plus policy agreement %. Both tables already exist (`results/q_learning_Q.npy`, `results/dp_policy.npy`), and `scripts/policy_table.py` can be pointed at the DP policy for a cell-by-cell comparison.
5. Then `agents/sarsa.py` and `agents/monte_carlo.py` — verify each on `tiny_mdp` first (reuse `_train_on_tiny_mdp` in `tests/test_tabular.py`), then run through `scripts/train.py`. Both should use the D-016 seed block, offset per algorithm, or their numbers will not be comparable to Q-learning's.

## Watch out for

- **`scripts/train.py` MUST call `agent.end_episode()` once per episode** (D-015). Forget it and epsilon stays at 1.0 forever — the agent explores at random for the whole run and never converges, with no error message. This is the single most likely way to lose an afternoon in the next session.
- **Q-learning's tiny-MDP number is a unit test, not a finding.** 9.24e-14 says the update rule is textbook-correct. The real-environment result is E-008, and it is a much more mixed picture. Never put the tiny-MDP figure in the report as a result.
- **Do not read the two ± columns in `train.py`'s output table as comparable.** The `q_learning` row's spread is across training runs; the baseline rows' spread is across eval seeds. The script prints a warning about this; fixing the presentation is an open follow-up.
- **Nothing has been re-run since the eval-seed finding.** Every number currently in the repo predates it. If the humans widen the eval block, E-002 / E-003 / E-004 / E-008 all need re-running before any of them can be quoted against each other.
- **The tiny MDP's exploration trap.** Under `π*` the agent never leaves `QUIET`, so `BUSY` is reachable only by exploring. A learner tested with ε pinned to 0 will produce garbage for `Q(BUSY, ·)` — and the cause is the exploration schedule, not the update rule. Do not debug the update rule first.
- **Monte Carlo needs `tiny_mdp.HORIZON`.** The fixture is *continuing*, with no terminal state. MC must truncate at 200 steps (γ^200 ≈ 7e-10, negligible). **Truncation is not termination:** a TD learner must keep bootstrapping through the cut (`done=False`), or it learns that the world ends and drags every value toward the last reward it saw.
- **If `test_tiny_mdp.py` or `test_mrp_bellman.py` ever fails, fix the code — never the expected values.** They came from a human with a pen. Editing them to make a test pass destroys the only external correctness anchors the project has.
- **Expect the bulk-close reward hack to reappear in Phase 2.** Q-learning maximises the same exploitable reward DP did. If it recurs, that is a *result*, not a bug — it shows the pathology lives in the reward rather than in any one algorithm, which is the report's spine (DP hacks → Q-learning hacks → RLHF fixes). `ROADMAP.md` Phase 2's exit criterion is **flagged but deliberately not pre-emptively weakened** — run it first, decide on real numbers, exactly as Phases 0 and 1 did.
- **The DP reward number must never be quoted without its recall beside it** (D-012). 305.9 reward with 0.43 recall is the finding; 305.9 alone is misleading.
- **Stray zero-byte files: root cause now CONFIRMED — see `docs/bugs/BUG_001_stray_zero_byte_files.md`.** Eight appeared this session, each mapping by timestamp to a `>` in text being written. Four from Markdown (`0`, `6.8`, `There`, `Watch`) and — the new finding — four from **typed Python** (`list[int]`, `np.ndarray`, `expected`, `` ` ``), triggered by return annotations like `def draw() -> list[int]:`. The bug is not Markdown-specific, and since this project mandates type hints on every public function, **every future session will produce these.** Mitigation is unchanged and sufficient: **`git add <explicit paths>`, never `git add -A`**, and sweep before committing. All eight deleted. Note `Remove-Item` needs `-LiteralPath` for names containing `[ ]`.
- **Shell quoting bites in the other direction too.** A `python -c` snippet passed through a PowerShell here-string had its inner quotes stripped and died with a Python syntax error pointing five lines away from the real problem. Write throwaway scripts to a file and run the file. Logged in `FLOW.md` gotchas.
- **The default branch is `master`, not `main`.** The remote has exactly one head. "Push to main" means `master` here — don't create a second branch and split the history.
- **Two modules in `src/` hold hard-coded numbers** — `mrp_example.py` (D-013) and `tiny_mdp.py` (D-014). Both look like CONSTRAINTS #9 violations and neither is; the docstrings carry the pointers. `tiny_mdp.py` uses γ = 0.9, not the project's 0.99. Deliberate — it keeps `V(QUIET) = 10` exact. Don't "fix" it to match config.
- Seeds: train 1–10, eval 101–105, calibration 1000–3099, DP estimation 10000–59999. Any new purpose gets a fresh disjoint block (config enforces train/eval disjointness in code). The tiny MDP is deterministic and uses no seed at all.

## Open questions for the humans

1. **Diya's countersign on D-012** (see Broken/blocked #2).
2. KPMG analyst for preference labels — still open, needs lead time. Longest-lead item in the project; Phase 5 cannot start without it.
3. Report format / team-size confirmation from Dr. Kaur — still open.
4. Target demo date — still open.
