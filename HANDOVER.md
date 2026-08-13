# HANDOVER.md — Where things stand right now

> Field Guide habit #1 and #13. Read this first, every session. Rewrite it last, every session.
>
> This is **not** a changelog — it's a snapshot of the present. Overwrite stale entries rather than appending. (The permanent record lives in `DECISIONS.md` and `docs/experiments/EXPERIMENT_LOG.md`.)

---

## Snapshot

| | |
|---|---|
| **Last session** | 2026-08-13 (session 2) |
| **Model** | Claude Fable 5 |
| **Current phase** | Phase 0 — Foundation (in progress, ~40%) |
| **Repo state** | git initialised, 6 commits + session-end docs commit. `config.py`, `alerts.py`, `generator.py` built and verified. **Generator calibration PASSED and human-approved.** |
| **Tests passing** | No pytest suite yet (next session starts it). All code verified by direct execution — see EXPERIMENT_LOG E-001. |

---

## Done

- `git init`; `.gitignore` verified against CONSTRAINTS #19; scaffold committed
- venv on Python 3.13.2; all deps install and import — **torch 2.13.0 works on 3.13** (HANDOVER risk retired); `requirements.txt` pinned (human-approved)
- `config.py` — typed frozen config loader; fails loudly with dotted key paths; **enforces train/eval seed disjointness in code** (CONSTRAINTS #2)
- `alerts.py` — frozen `Alert` dataclass per ARCHITECTURE §4, ground-truth warning in docstring
- `generator.py` — Poisson arrivals; truth model `base_rate × type_lift × severity_lift × asset_lift` (D-007); deterministic per (config, seed)
- **Calibration gate PASSED** (E-001): 168.7 alerts/shift, 3.34% incidence, r(sev,truth)=0.323; robust on two untouched seed blocks; Diya eyeballed and approved
- Fixed scaffold bug: `actions:` YAML block was invalid YAML; names now under `actions.names` (FLOW.md gotcha #1)
- DECISIONS D-007 (truth model), D-008 (time-of-day deferred) appended; EXPLAIN Parts 7+8 updated

## In progress

Nothing mid-flight. Clean stopping point at the ROADMAP calibration checkpoint.

## Broken / blocked

Nothing.

---

## Next session should do

Continue **Phase 0** in ROADMAP order:

1. `state.py` — `discretise(env_state) -> int` (0..575) and `featurise(env_state) -> np.ndarray` (~20 floats)
2. `env.py` — `SOCTriageEnv` with `reset(seed)` / `step(action)`, the 5 actions, reward from brief §3.5, 480-min termination
3. **Write `test_no_ground_truth_leakage` immediately after `state.py`** — not later (CONSTRAINTS #1; the docstrings in `alerts.py` point at it, so it must exist)
4. `agents/base.py`, `agents/baselines.py`, `runner.py`, `evaluation/metrics.py`, tests, baseline table — per ROADMAP

The Phase 0 exit criterion (baseline table, oracle strictly best / random worst) is still 3–4 chunks of work away.

---

## Watch out for

- **Bucket-boundary convention for `state.py`:** config boundaries `[10, 40, 100]` must mean `[0,10) [10,40) [40,100) [100,∞)` everywhere. Write one shared helper; an off-by-one here silently corrupts all 576 states.
- **FLOW.md Flow A ordering detail:** admit new arrivals *after* the clock advances, *before* building the next observation. Get it backwards and the agent acts on a stale queue — mysterious underperformance, not a crash.
- **Severity-sort will be a strong baseline** — by construction ~64% of true incidents carry severity 3 (E-001 / D-007 consequence). Don't panic when it looks good in the Phase 0 table; the learned-policy gap comes from the other third + asset value + time pressure.
- `deadline_min == 0.0` on false positives is a *convention* (meaningless there). `env.py` must only read deadlines for true incidents.
- Empty queue: `step` advances clock by `shift.empty_queue_wait_min` (5 min), reward 0 — already in config, easy to forget.
- Calibration seeds are 1000+/2000+/3000+ — if anything else ever wants a seed block, keep it disjoint from 1–10, 101–105, and these.

---

## Open questions for the humans

1. Is a working security analyst reachable through Diya's KPMG team for even 20–30 preference labels? Worth asking early — it materially raises the project's credibility, and the ask needs lead time.
2. Does Dr. Kaur want a written report, a presentation, or both? Group size is specified as 3–4 in the handout but this team is 2 — has that been confirmed as acceptable?
3. Target demo date, so the roadmap can be anchored to a real deadline?
