# BUG_005 — `config_hash` changes when only a comment changes, so identical environments look different

**Found:** 2026-09-04, session 12, while first running `rlhf/pairs.py` against real records.
**Severity:** low for correctness, medium for confusion. No existing result is wrong.
**Status:** documented, **not fixed** — the fix is worse than the bug. See "Why it is not being fixed".

---

## Symptom

`rlhf.pairs.build_pairs` refuses to build pairs from records produced under more
than one config (`MixedConfigError`), because two episodes run against different
environments are not the same shift even on the same seed. Pointed at the 300
EpisodeRecords in `results/runs`, it raised:

```
records span 2 different configs (['0bfe79509f34', '679eaa992c7f']);
a pair's two sides must have run the same environment
```

That looked, for about a minute, like a serious integrity problem: it would mean
the Phase 2 comparison table (E-014) had compared agents measured on two
different environments.

## What it actually is

It is not. The split is:

| config hash | records | written |
|---|---|---|
| `0bfe79509f34` | 270 — `cheapest_first`, `dp`, `fifo`, `monte_carlo`, `oracle_greedy`, `q_learning`, `random`, `sarsa`, `severity_sort`, 30 eval seeds each | 2026-08-17 19:19–19:33 |
| `679eaa992c7f` | 30 — `dqn`, 30 eval seeds | 2026-08-19 01:08 |

Hashing `config/env_default.yaml` at each revision reproduces both values exactly:

```
ccdbd66^  0bfe79509f34   (4991 bytes)
ccdbd66   679eaa992c7f   (4991 bytes)
```

`ccdbd66` is *"docs: correct session 6 dates 2026-08-16 -> 2026-08-17"*, committed
2026-08-17 20:59 — after the nine agents ran and before DQN ran. Its entire diff
to this file is one character inside a **comment**:

```diff
-  # Widened from [101..105] to [101..130] on 2026-08-16 (D-019). Five seeds
+  # Widened from [101..105] to [101..130] on 2026-08-17 (D-019). Five seeds
```

Both files are 4991 bytes. No environment parameter changed. **The DQN records
and the tabular records describe the same environment**, and E-014 / E-017 are
unaffected.

## Root cause

`runner.config_hash` (`runner.py:40`) hashes the raw bytes of the YAML file:

```python
text = Path(cfg_path).read_text(encoding="utf-8")
return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
```

So it is a hash of **the file**, not of **the configuration**. Comments,
whitespace, key order and line endings all move it. A hash that changes when a
typo in a comment is fixed will, sooner or later, tell somebody their results are
incomparable when they are identical — which is exactly what it did here.

## Why it matters anyway

The failure mode is asymmetric and that is what makes it worth writing down.

- **False "different"** — what happened here. Loud, and resolvable in about five
  minutes with `git show <rev>:config/env_default.yaml`.
- **False "same"** — cannot happen. Identical bytes really are an identical
  config.

A hash that only errs toward "these might differ, go and check" is the safe
direction for a scientific-integrity guard to err in. That is the argument for
leaving it alone.

## Why it is not being fixed

The obvious fix is to hash the *parsed and normalised* config instead of the file
text. Rejected:

1. It would change the hash of every future record while every existing record
   keeps the old one, so the two would never compare equal again — turning a
   cosmetic false positive into a permanent one across the 08-17 boundary.
2. `config_hash` is a Phase 0 function that Phases 0–4 all wrote records with.
   Changing it to make a Phase 5 module happier is building backwards
   (CONSTRAINTS #18), and it is the shape of change CONSTRAINTS #4 exists to
   discourage.
3. The guard in `pairs.py` is **correct and stays**. It will not fire in normal
   use: `scripts/generate_pairs.py` produces every pair record in one run under
   one config, so they share a hash by construction. It fired here only because
   it was pointed at `results/runs`, which holds **eval-seed** records that
   FEATURE_011 §4 forbids building pairs from in the first place.

So the guard did its job twice over — it caught a genuine config split, and it
refused a record set that was the wrong input regardless.

## What to do when you next see `MixedConfigError`

1. Do not assume the results are wrong. Get the two hashes.
2. Find which revisions they correspond to:
   `git log --format="%h %ad %s" --date=iso -- config/env_default.yaml`, then
   for each candidate revision hash the blob:
   `git show <rev>:config/env_default.yaml` piped through sha256, first 12 hex.
3. `git diff <a> <b> -- config/env_default.yaml`. If the diff touches only
   comments, the environments are identical and the records are comparable —
   record that finding rather than re-running anything.
4. If the diff touches a real key, the records genuinely are not comparable and
   the older set must be re-run. Do not merge them.

## Verification

```
$ python -c "<hash config/env_default.yaml at ccdbd66^ and ccdbd66>"
ccdbd66^ ('0bfe79509f34', 4991)
ccdbd66  ('679eaa992c7f', 4991)

$ git show --stat --format="" ccdbd66 -- config/env_default.yaml
 config/env_default.yaml | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

Run 2026-09-04. Both observed record hashes accounted for; the only difference
between them is a date inside a comment.
