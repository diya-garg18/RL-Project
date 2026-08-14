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

**One open decision blocks the Phase 0 gate wording — again, and for a better-understood reason (E-003):**

> After vectorising the generator (Phase 1 speed: 50k-episode estimation 37.6 → 1.3 min), the same seeds produce different alert streams. Recalibration still passes (rate 3.20%, r 0.321). But the baseline re-run + a 30-seed diagnostic showed **E-002's "oracle strictly best on recall" was 5-seed noise**: severity-sort wins recall robustly (0.826 vs 0.799 over 30 seeds) because ~64% of incidents carry severity 3; the oracle's truth advantage is decisive on **total reward** (145 vs 51) instead. Proposal awaiting Diya/Pranav: restate oracle dominance on total reward; keep the recall finding as a documented design feature. Phase 2's recall-based criterion flagged too (E-003 implication 2).

## Next session should do

1. Get the E-003 gate decision; amend ROADMAP Phase 0 exit wording accordingly.
2. Run **Phase 1 DP pipeline**: `agents/dp.py` is written and committed (estimation + VI + PI + DPAgent); `scripts/run_dp.py` still needs writing — estimation (~1.3 min for 50k), coverage report, convergence curve plot, VI-vs-PI agreement %, evaluation on eval seeds, DP row added to baselines table. Log as E-004, decide unvisited-state handling entry (already drafted as D-011 self-loop convention in dp.py docstring — needs a DECISIONS entry).

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
