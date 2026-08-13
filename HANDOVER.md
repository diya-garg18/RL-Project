# HANDOVER.md — Where things stand right now

> Field Guide habit #1 and #13. Read this first, every session. Rewrite it last, every session.
>
> This is **not** a changelog — it's a snapshot of the present. Overwrite stale entries rather than appending. (The permanent record lives in `DECISIONS.md` and `docs/experiments/EXPERIMENT_LOG.md`.)

---

## Snapshot

| | |
|---|---|
| **Last session** | 2026-08-13 |
| **Model** | Claude Opus 5 |
| **Current phase** | Phase 0 — Foundation (not started) |
| **Repo state** | Documentation scaffold only. No code, no `git init` yet. |
| **Tests passing** | N/A — no tests exist |

---

## Done

- Project chosen, scoped, and fully specified in `PROJECT_BRIEF.md`
- All nine Field Guide documents created, plus `EXPLAIN.md` and `INTERVIEW_PREP.md`
- Directory structure, `requirements.txt`, `.gitignore`, and both config files created
- Roadmap broken into 7 phases with concrete exit criteria

## In progress

Nothing.

## Broken / blocked

Nothing.

---

## Next session should do

Start **Phase 0** in `ROADMAP.md`, in this order:

1. `git init` and make the first commit (the scaffold)
2. Create the venv and install `requirements.txt`
3. `config.py` → `alerts.py` → `generator.py`
4. **Stop at the generator calibration check.** Do not build the environment until the generator produces ~3% true-incident rate and a severity↔truth correlation in the 0.30–0.40 band, and those numbers are written into `EXPLAIN.md` Part 8.

Follow the session-start protocol in `CLAUDE.md`: state the plan, get approval, then implement.

---

## Watch out for

- **The generator calibration is the foundation of everything.** If severity ends up strongly predictive of truth, severity-sort becomes near-optimal and there is no project. If it ends up with no signal at all, the environment is pure noise and nothing can learn. The 0.30–0.40 target band matters — check it, don't assume it.
- **Ground-truth leakage is the easiest way to accidentally fake a great result.** Write `test_no_ground_truth_leakage` early, not late.
- `requirements.txt` pins nothing yet. Pin actual versions after the first successful install so the environment is reproducible.
- PyTorch on Python 3.13 — verify the install works before Phase 3 depends on it. If it fights, that's a decision point worth raising early (fall back to 3.12), not in week 3.
- Time budget: RLHF (Phase 5) needs ~100 minutes of *human* labelling time across two people. Book it in advance; it can't be rushed at the end.

---

## Open questions for the humans

1. Is a working security analyst reachable through Diya's KPMG team for even 20–30 preference labels? Worth asking early — it materially raises the project's credibility, and the ask needs lead time.
2. Does Dr. Kaur want a written report, a presentation, or both? Group size is specified as 3–4 in the handout but this team is 2 — has that been confirmed as acceptable?
3. Target demo date, so the roadmap can be anchored to a real deadline?
