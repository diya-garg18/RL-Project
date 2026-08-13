# ROLLBACK.md — The way back out

> Field Guide habit #9. Confidence to let the AI make bigger changes comes from knowing exactly how to reverse them.

---

## Standing safety net

- **Commit at the end of every session**, tagged `phase<N>: <what>`. A session's work is one or a few commits — never a week of uncommitted changes.
- **Tag every phase completion:** `git tag phase0-complete`. These are the known-good points to fall back to.
- `results/` is gitignored, so **archive it before risky work**: `Copy-Item results results_backup_<date> -Recurse`. Losing a 50k-episode DP model estimate to a bad refactor costs an hour of compute.

---

## Standard undo

```bash
git status                      # see what actually changed
git diff                        # read it (habit #10)
git checkout -- <file>          # discard one file's changes
git reset --hard HEAD           # discard ALL uncommitted changes — destructive, be sure
git reset --hard phase2-complete   # fall back to a known-good phase
```

Before any `--hard`: confirm nothing unstaged is worth keeping. If unsure, `git stash` first — it's reversible, `--hard` isn't.

---

## Risky operations and their specific reversals

| Operation | Risk | Reversal |
|---|---|---|
| Changing the MDP (state buckets / actions / reward) | Invalidates **every** prior result | Requires human approval first (`CONSTRAINTS.md` #15). Tag before starting. If it lands, all previous results must be re-run or explicitly marked as belonging to the old MDP. |
| Retuning the generator | Silently changes what every past number meant | Record old and new calibration values in `EXPLAIN.md` Part 8. Re-run Phase 0 baselines. Old results are void — mark them, don't delete them. |
| Refactoring `env.py` | Breaks reproducibility invisibly | Before: save an `EpisodeRecord` under a fixed seed. After: regenerate and diff. Identical or explain why not. |
| Deleting/overwriting experiment results | Loses the record | Prohibited (`CONSTRAINTS.md` #4). Append to `EXPERIMENT_LOG.md` and mark entries superseded instead. |
| Upgrading PyTorch / NumPy mid-project | Changes numerics under you | Pin versions in `requirements.txt` after the first good install. If an upgrade is needed, tag first and re-run one prior experiment to confirm results still match. |
| Bulk edits across many files | Unreviewable diff | Don't. One logical change per request (`CONSTRAINTS.md` #17). If it's already happened, `git reset --hard` and redo in pieces. |
| Dropping the preference-label DB | **Unrecoverable — that's ~100 minutes of human time** | Back up `rlhf.db` after every labelling session, to a location outside the repo. Never `DROP TABLE`. Migrations copy to a new table. |

---

## If a phase's results stop reproducing

Don't patch forward. Bisect:

1. `git log --oneline` — find the last commit where results were known good
2. `git stash` current work
3. `git checkout <that commit>` and re-run the phase's checklist block
4. If it reproduces there, the break is in between — `git diff <good>..HEAD -- src/` and read it
5. Log the cause in `docs/bugs/BUG_xxx.md` **before** fixing it, so the trace survives

---

## The one thing that can't be rolled back

**Human preference labels.** Compute is cheap and re-runnable; 300 human judgments are not. Treat `rlhf.db` as the most valuable file in the repository. Back it up outside the repo after every labelling session, and export a plain CSV copy as well — a CSV survives a corrupted SQLite file and a schema change alike.
