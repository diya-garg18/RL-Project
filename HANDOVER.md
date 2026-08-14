# HANDOVER.md — Where things stand right now

> Field Guide habit #1 and #13. Read this first, every session. Rewrite it last, every session.
>
> This is **not** a changelog — it's a snapshot of the present. Overwrite stale entries rather than appending. (The permanent record lives in `DECISIONS.md` and `docs/experiments/EXPERIMENT_LOG.md`.)

---

## Snapshot

| | |
|---|---|
| **Last session** | 2026-08-14 (session 3) |
| **Model** | Claude Fable 5 |
| **Current phase** | Phase 0 — Foundation: **all tasks built and ticked; exit gate awaiting one human decision** (see below) |
| **Repo state** | 9 commits. Full simulator + 5 baselines + runner + metrics + 7 passing tests + baseline table (E-002). |
| **Tests passing** | 7/7 (`pytest tests/ -q`), including `test_no_ground_truth_leakage` |

---

## Done

- Everything on the ROADMAP Phase 0 checklist — all boxes ticked (see ROADMAP for the per-item list)
- Calibration gate PASSED and human-approved (E-001: 168.7 alerts/shift, 3.34% incidence, r=0.323)
- Baseline table PASSED oracle-strictly-best on mean recall (E-002: oracle 0.86 > severity-sort 0.85)
- Oracle debugged twice to get there — first version lost to severity-sort; see D-010 for the path-clearing design
- Reward-timing semantics fixed and documented (D-009)
- All session-end docs current: EXPLAIN Parts 7+8, DECISIONS through D-010, E-001/E-002, FLOW (Flow B ✅), TEST_CHECKLIST Phase 0 block updated to real commands

## In progress

Nothing mid-flight.

## Broken / blocked

**One open decision blocks formally closing Phase 0** (the code is done):

> The ROADMAP exit criterion says "random is clearly worst". Reality (E-002): **FIFO is worst by a mile** (recall 0.20 vs random's 0.46, MTTD 246 min). This is correct queueing behaviour, not a bug — always working the *oldest* alert in an overloaded queue means investigating things whose deadlines already expired. Proposed amendment: "oracle strictly best on mean recall; random and FIFO clearly at the bottom." **Needs Diya/Pranav sign-off before the criterion text is edited** (flagged in E-002, TEST_CHECKLIST, and EXPLAIN — nothing hidden).

## Next session should do

1. Get the exit-criterion decision above; update ROADMAP wording accordingly; declare Phase 0 complete.
2. Start **Phase 1 — DP**: `agents/dp.py` — estimate P̂/R̂ from 50k random-policy episodes, report state coverage, then value iteration + policy iteration from scratch (Sutton & Barto §4.1–4.4 docstrings per CLAUDE.md).
3. Runtime heads-up: 50k episodes × ~130 steps ≈ 6.5M steps. Time the estimation loop on 1k episodes first; if the full 50k projects to >10 min, ask before launching (CLAUDE.md rule).

## Watch out for

- **Oracle is an upper bound in expectation only** — it loses seed 101 to severity-sort by one incident (end-game timing). Never write "nothing can beat the oracle" without the caveat (D-010).
- **Severity-sort at 0.85 recall is the real opponent.** The learnable gap: its 14% missed incidents, 413 wasted min/shift, and its indifference to asset value and deadlines. Q-learning beating it will be a *small-margin* story on recall; the win likely shows more on MTTD, wasted minutes, and composite cost.
- Phase 1 DP: unvisited states need an explicit handling rule, logged in DECISIONS (ROADMAP requirement).
- `runner.run_episode(learn=True)` is the training hook — no separate loop needed for tabular agents.
- Calibration seeds 1000+/2000+/3000+ are burned for calibration; train 1–10, eval 101–105. Any new purpose gets a fresh disjoint block.
- FLOW.md Flow A ordering (arrivals after clock, before observation) is implemented and test-covered — don't "fix" it.

## Open questions for the humans

1. **Exit-criterion amendment above (blocks closing Phase 0).**
2. KPMG analyst for preference labels — still open, needs lead time.
3. Report format / team-size confirmation from Dr. Kaur — still open.
4. Target demo date — still open.
