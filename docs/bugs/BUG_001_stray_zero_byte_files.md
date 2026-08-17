# BUG_001 — Stray zero-byte files appear in the repo root during doc-writing sessions

**Status:** root cause confirmed · mitigated by convention · not fixed at source
**Phase:** spans 1–2 · **First seen:** 2026-08-14 (session 4) · **Root cause confirmed:** 2026-08-17 (session 6)
**Model(s) used:** Claude Fable 5 (sessions 3–4, first observed), Claude Opus 5 (session 6, diagnosis)

---

## Symptom

Zero-byte files with odd names appear in `D:\RLPROJECT\` during sessions that write a lot of Markdown. They are untracked, empty, and harmless in themselves — the danger is that a careless `git add -A` commits them.

Observed across three sessions:

| Session | Date | Stray files |
|---|---|---|
| 4 | 2026-08-14 | *(logged in `HANDOVER.md` at the time, names not preserved)* |
| 5 | 2026-08-16 | `This`, `V(QUIET)` |
| 6 | 2026-08-17 | `0`, `6.8`, `There`, `Watch` *(doc sweep + FEATURE_002)* |
| 6 | 2026-08-17 | `` ` ``, `expected`, `list[int]`, `np.ndarray` *(Q-learning + `test_tabular.py`)* |

Sessions 4 and 5 both recorded the phenomenon and both recorded the root cause as **not confirmed**, with a standing hypothesis that "something in the tooling chain interprets a `>` inside written content as a shell redirect."

## Root cause — confirmed 2026-08-17

The hypothesis was right. Session 6 confirmed it by correlating each stray filename against the exact text written, with file creation timestamps:

| Stray file | Created | The text written that produced it | Where |
|---|---|---|---|
| `6.8` | 21:24:19 | `residual, Q(QUIET,WORK)  6.7 ->  6.8: 0.1000` | `FEATURE_002` |
| `Watch` | 21:26:05 | `> Watch for one specific false pass here:` | `TEST_CHECKLIST.md` |
| `0` | 21:26:46 | `demonstration of why ε > 0 is necessary` | `DECISIONS.md` D-014 |
| `There` | 21:28:37 | `> There are two of them now, and they check…` | `EXPLAIN.md` |

Four out of four, each timestamp landing inside the write that contained the matching text.

**The rule:** any `>` in written content that is followed by whitespace and then a token gets interpreted as a shell output redirect somewhere in the tooling chain, creating an empty file named after the following token. Session 5's `V(QUIET)` fits the same rule (`> V(QUIET)`), as does `This`.

The common triggers are all ordinary, legitimate content:

1. **Blockquote lines** — `> Watch out for…`. Every Field Guide document in this repo opens with one, which is why doc-heavy sessions produce the most strays.
2. **ASCII arrows in code output and tables** — `6.7 ->  6.8`.
3. **Comparison operators in prose** — `ε > 0`, `Δ > 1e-4`.
4. **Python return-type annotations** — `def draw() -> list[int]:`. Added 2026-08-17 after writing `q_learning.py` and `test_tabular.py` produced strays named `list[int]` and `np.ndarray`.

**Trigger 4 matters more than the others** and was not known when this file was first written. It means the bug is **not Markdown-specific** — it fires on ordinary typed Python, which this project requires on every public function (`CLAUDE.md` → Code → "Type hints on all public functions"). Any session writing typed Python will produce strays. That removes the last trace of an argument for "just avoid the trigger": the trigger is mandatory project style.

**Deleting a stray named `list[int]` needs `-LiteralPath`.** PowerShell treats `[` and `]` as wildcard characters, so `Remove-Item 'list[int]'` silently matches nothing and reports success. Use `Remove-Item -LiteralPath 'list[int]' -Force`.

## What is *not* established

The precise component doing the interpreting has **not** been identified. The correlation establishes that written content reaches a shell that performs redirection; it does not say which hook, wrapper, or harness layer is responsible, and no fix has been attempted at that level. Everything above the "Root cause" heading is measurement; the mechanism below the tooling boundary is inference from that measurement.

Also unexplained: the trigger is not *every* `>` in every write. Session 6 wrote many blockquote lines and produced four strays, not forty. Whatever samples the content does so incompletely. That inconsistency is itself a reason not to rely on "just avoid `>`" as the mitigation — see below.

## Mitigation (standing, and sufficient)

Do **not** try to avoid `>` in documentation. Blockquotes are correct Markdown, arrows are correct in output blocks, and comparison operators are correct in prose. Contorting the writing to dodge a tooling bug would damage the documentation to protect against an empty file.

Instead:

1. **`git add <explicit paths>`, never `git add -A` or `git add .`** — the strays are untracked, so explicit paths cannot pick them up. This is the actual protection.
2. **Check `git status` for oddly-named zero-byte files before every commit.** One-liner:
   ```powershell
   Get-ChildItem -Recurse -File | Where-Object { $_.Length -eq 0 -and $_.FullName -notmatch '\\\.venv\\|\\\.git\\|gitkeep' }
   ```
3. **Delete them before committing.** They carry no content — verify `Length -eq 0` first, then remove.

Rule 1 alone is sufficient; 2 and 3 are hygiene. `CONSTRAINTS.md` #19 already forbids committing junk, and this is a specific instance of it.

## Why this is written up rather than fixed

The bug produces zero-byte files outside `src/`, `tests/` and `config/`. It cannot corrupt code, cannot alter a result, and cannot affect a number in the report. Chasing it into the harness layer would cost more than it protects, and the mitigation is a one-line habit already in force.

It is documented because two prior sessions each spent effort re-noticing it and re-recording it as unexplained. That cost is now paid off: the next session that sees a file called `Watch` in the repo root can read this file, delete it, and move on.

## Follow-ups left open

- Identify the specific tooling component performing the redirection, **if** it ever causes damage beyond empty files in the repo root. Not worth doing pre-emptively.
- If a stray file ever appears with **non-zero** length, that is a different and much more serious bug — content is being written somewhere unintended. Stop and investigate rather than deleting.

## Plain-English summary

Every so often an empty file with a strange name — `Watch`, `There`, `6.8` — turns up in the project folder. We now know why: when the AI writes documentation containing a `>` character, something in the chain of tools between it and the disk mistakes that for the command-line symbol meaning "save output into a file", and creates an empty file named after whatever word came next. Since `>` is how you write a quoted paragraph in Markdown, and our documents are full of quoted paragraphs, it happens fairly often.

The files are empty and harmless. The only real risk is accidentally committing them into the project history, so the rule is to name files explicitly when committing rather than using the "add everything" shortcut, and to sweep for empty files first. We are not fixing the underlying tool, because it cannot damage any code or any result — it just leaves litter.
