# HANDOVER.md — Where things stand right now

> Field Guide habit #1 and #13. Read this first, every session. Rewrite it last, every session.
>
> This is **not** a changelog — it's a snapshot of the present. Overwrite stale entries rather than appending. (The permanent record lives in `DECISIONS.md` and `docs/experiments/EXPERIMENT_LOG.md`.)

---

## Snapshot

| | |
|---|---|
| **Last session** | 2026-08-17 (session 6) |
| **Model** | Claude Opus 5 |
| **Phase 0** | Closed. Gate **passes** on the 30-seed block. |
| **Phase 1** | **CLOSED as built-but-not-passed** (D-022). Criterion 2 falsified; cause also refuted (E-015). |
| **Phase 2** | **CLOSED as built-but-not-passed** (D-020). All 8 boxes done; gate not met, deliberately not restated. |
| **Phase 3** | Not started, **cleared to begin**. Runs on Pranav's machine until the commit balance evens. |
| **Repo state** | `D:\RLPROJECT`, branch `master`. |
| **Tests passing** | **76/76** (`.\.venv\Scripts\python.exe -m pytest tests/ -q`, ~8.5 s) |
| **Blockers** | **None.** Both blocking decisions taken (D-022). Phase 3 may start. |

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

## Both blocking decisions are now taken — Phase 3 is cleared to start

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

## Whose turn is it — read before starting work

**Measured 2026-08-17 (`python scripts/commit_balance.py`, D-021 / CONSTRAINTS #24–26):**

| author | commits | share |
|---|---|---|
| Diya Garg | 17 | 70.8% |
| Pranav Upadhyay | 7 | 29.2% |

Per phase: **Phase 0** 12 (all Diya) · **Phase 1** 5 (Diya 3, Pranav 2) · **Phase 2** 4 (all Pranav).

> ⚠️ **IMBALANCED — Diya is 10 commits ahead. Phase 3 should run on Pranav's machine** until the gap is inside the 3-commit threshold, i.e. roughly the next 7–10 commits.
>
> Phase 3 (DQN) divides naturally into that many meaningful commits — network, replay buffer, target network, training loop, the two required ablations, docs — so no padding is needed. **Do not manufacture commits to close the gap**, and do not commit on the other person's behalf: the split must be real, because an examiner may ask either student to explain any commit under their name (CONSTRAINTS #24).

`.mailmap` collapses Diya's two author identities (personal + GitHub noreply), so these counts are accurate where a raw `git shortlog` would over-split her.

## Before the machine changes hands

Assume every session is the last one on this machine. All of these must be true (CONSTRAINTS #25):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q     # must pass
git status --porcelain                             # must be empty
git status -sb                                     # must show no "ahead"
python scripts/commit_balance.py                   # report it, act on it
```
Plus: `HANDOVER.md` (this file) actually describes the current state, and no stray zero-byte files are staged (`docs/bugs/BUG_001`).

**Nothing in `results/` is ever needed to continue** — it is gitignored and fully regenerable by the commands below. If the other machine ever needs a file from `results/`, that is a bug in the scripts.

**Anything that exists only on one machine gets written down, not remembered** — install workarounds, tool versions, path quirks. The other person cannot see this terminal. Put it under "Watch out for".

## Reproduce on this device

```powershell
cd D:\RLPROJECT
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/ -q                    # expect 76 passed

python scripts/run_baselines.py                                   # fast
python scripts/run_dp.py                                          # ~2.5 min
python scripts/train.py --agent {q_learning,sarsa,monte_carlo}     # ~4 min each
python scripts/policy_table.py --agent <name>                      # box 6
python scripts/compare_agents.py                                   # box 5
python scripts/ablations.py                                        # ~4 min
```

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

- **The pen-and-paper derivations.** FEATURE_001's MRP and FEATURE_002's tiny MDP are both ticked because the derivation and its verification exist — not because Pranav and Diya can reproduce them cold, which is what the viva tests. FEATURE_002 is the likelier exam question: two states, four numbers, γ = 0.9.
- KPMG analyst contact for preference labels — longest-lead item in the project; Phase 5 cannot start without it.
- Report format / team-size confirmation from Dr. Kaur; target demo date.
