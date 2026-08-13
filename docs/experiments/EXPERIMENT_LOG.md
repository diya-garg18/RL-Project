# EXPERIMENT_LOG.md — Every training run

> **Append-only.** Never edit or delete an entry. If a run turns out to be invalid, mark it `SUPERSEDED` with the reason and add a new entry — don't erase it (`CONSTRAINTS.md` #4).
>
> This file is the evidence base for the report. Every number that appears in the report must be traceable to an entry here.

**Entry format:**

```
## E-nnn — <agent> — <date>
**Status:** valid | SUPERSEDED (reason)
**Config:** env_default.yaml @ <git sha> · training_default.yaml @ <git sha>
**Command:**
**Seeds:** train=[...] eval=[...]
**Runtime:**
**Result:** | metric | mean | std |
**Observations:** anything surprising
**Notes:** deviations from default config, known caveats
```

---

## Standing rules

1. **5 seeds minimum**, mean ± std. Never report a single run.
2. **Eval seeds are disjoint from train seeds.** Never tune against eval results.
3. Record the **config git sha**, not just the filename — configs change.
4. If a result is surprisingly good, log it *and* log what you did to check it wasn't a bug (`CONSTRAINTS.md` #5).
5. Negative results get entries too. An ablation that showed no effect is a finding.

---

## Runs

*No experiments yet. First entry will be the Phase 0 baseline comparison.*
