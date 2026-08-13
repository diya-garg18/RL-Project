# BUG_<nnn> — <short title>

> Field Guide habit #5. One file per bug, traced start to finish.
>
> **Write the first three sections BEFORE fixing anything.** A bug diagnosed after the fix is a bug you don't actually understand.
>
> Copy this template to `BUG_<nnn>_<slug>.md`.

**Status:** open | diagnosed | fixed | won't fix
**Severity:** blocks-phase | wrong-results | cosmetic
**Phase:** · **Found:** · **Fixed:**
**Model(s) used:**

---

## Symptom

What was observed. Exact output, exact command, exact seed.

```
<paste>
```

## Reproduction

The minimal steps to make it happen every time. If it's intermittent, say so and give the frequency — intermittent bugs in an RL project usually mean a seeding or ordering problem.

## Why this matters

Which results are invalidated by it. **If any published number is affected, mark that entry superseded in `docs/experiments/EXPERIMENT_LOG.md` — do not silently re-run and replace it** (`CONSTRAINTS.md` #4).

## Hypotheses

| # | Hypothesis | How to test it | Result |
|---|---|---|---|
| 1 | | | |

Rule out one at a time. Don't change two things at once — you'll learn nothing from the outcome.

## Root cause

What was actually wrong, and **why it produced that particular symptom**. If you can't connect cause to symptom, you haven't found the root cause yet — you've found a change that made the symptom go away.

## Fix

| File | What changed |
|---|---|
| | |

## Verification

Real commands, real output, showing the symptom is gone **and** nothing else broke.

```
$ pytest -q
<paste>
```

## Regression test added

Which test now catches this. If none, justify why — for anything in the `wrong-results` severity class, "no test" needs a very good reason.

## Plain-English summary

One paragraph, jargon-free, for `EXPLAIN.md`. Bugs that were interesting to find are worth mentioning in the viva — they demonstrate you understand the system rather than just running it.
