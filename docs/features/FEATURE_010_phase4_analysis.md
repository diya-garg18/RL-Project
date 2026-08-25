# FEATURE_010 - Phase 4's two analyses: sample efficiency and the variance demonstration

**Phase:** 4 (ROADMAP boxes 3 and 4)
**Status:** box 4 **DONE and measured** (E-021). Box 3 **built and smoke-verified, NOT YET RUN** -
it needs the full training runs, which are Pranav's machine.
**Date:** 2026-08-25
**Model:** Claude Opus 5

---

## Box 4 - `scripts/variance_demo.py` (done)

### What it measures

The one number the three algorithms genuinely disagree about: **the coefficient multiplying
`grad ln pi(a|s)`**. Everything else is held equal - same network shape, same input scaling
(D-032), same Adam.

| condition | coefficient | what it is |
|---|---|---|
| REINFORCE, no baseline | `G_t` | a whole episode of rewards summed |
| REINFORCE, with baseline | `G_t - v(s_t)` | the same, re-centred |
| actor-critic | `r + gamma*v(s') - v(s)` | ONE reward plus one estimate |

### The result (E-021), including the half that failed

| condition | coefficient std |
|---|---|
| REINFORCE (no baseline) | 146.94 |
| REINFORCE (with baseline) | **147.68** |
| actor-critic (TD error) | **30.89** |

**The textbook baseline reduction did not replicate: 1.00x.** Bootstrapping's did: **4.78x**.

### Why that is the more interesting outcome

The baseline is unbiased for *any* `b(s)` - the identity `E[b(s) * grad ln pi] = b(s) * grad 1 = 0`
does not care whether `b` is any good. It only reduces **variance** when `b(s)` is close to
`E[G_t | s]`. At 30 episodes of fresh shifts the value head has not learned that, so it subtracts
a number uncorrelated with the return and removes nothing. The theory is not contradicted; its
precondition is unmet.

Bootstrapping's reduction is different in kind - **structural, not earned**. The TD error never
performs the summation that creates the spread, so even a poor critic narrows the coefficient.

**The unit test that looks contradictory is not.**
`tests/test_reinforce.py::test_a_trained_baseline_centres_the_update_coefficients` feeds the *same*
episode 40 times and the coefficients collapse by more than half. That proves the code implements
the baseline correctly, under a condition the real environment never supplies. E-021 shows the
environment does not let it pay off in this budget. Both true; neither quotable alone.

### An honest note about how this was measured

At an 8-episode smoke the baseline claim read **CONFIRMED at 1.09x**. At 30 episodes it is 1.00x.
The apparent effect was noise - the E-013 pattern again. **The budget was not extended until the
claim agreed**, which is the wrong way to settle it; whether the baseline earns itself over a full
20000-episode run is left open and named as open.

---

## Box 3 - `scripts/compare_sample_efficiency.py` (built, not run)

### The x-axis is environment steps, and that required a code change first

Sample efficiency means experience consumed, so the x-axis is **environment steps, never
episodes**. The trainers recorded only per-episode reward, so all three now also record
`episode_steps`.

That is not pedantry. Measured on identical 4-episode smoke runs, the **actor-critic took 88.0
steps per shift against REINFORCE's 46.8** - nearly 2x. A 480-minute shift is a fixed amount of
wall-clock, not of experience: a bulk-closing policy makes far more decisions inside it than one
doing slow verifies. Plotting per episode would have handed the sample-hungry agent a free
advantage.

Done **before** the runs deliberately. Afterwards would have meant re-running everything or
shipping an estimated x-axis.

### It plots both readings and endorses neither

Two curves per agent: **sampled** (what the agent collected from its own stochastic policy) and
**greedy** (`argmax_a pi(a|s)` on the train-diagnostic seeds). E-019 found these disagree
violently - nine runs of nine, positive sampled against strongly negative greedy - and E-020 found
the same argmax degeneracy in the actor-critic at every entropy coefficient.

Which one Phase 4 reports is **a decision the humans owe**. The script plots both, labels both,
and prints that the choice is unresolved. Picking one silently would be taking that decision by
default, inside a script.

### It can say "never"

`steps_to_target` returns `None`, printed as **NEVER REACHED**, when an agent's smoothed curve
never reaches `severity_sort`'s 40.4 (E-014). Three phases have already ended below their
baselines; a table that could not express that would be unable to describe its own most likely
outcome.

### Verified without the full runs

Smoke-verified end to end: the no-artefacts path prints the training commands and exits; both
artefact layouts load (`repeat<N>.json` from D-027's parallel DQN pattern, and the Phase 4
trainers' aggregated files); NEVER REACHED fires correctly; the plot and JSON are written. An
artefact lacking `episode_steps` is **skipped with a message**, never approximated - a fabricated
x-axis is worse than no plot.

## What remains

The full training runs. Budget from E-020's measurements: REINFORCE ~0.016 s/episode (~27 min for
20000 x 5), actor-critic **~0.6 s/episode (~3.3 h per repeat)**, plus regenerating the DQN curves
E-017 never stored.
