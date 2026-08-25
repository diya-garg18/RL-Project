# FEATURE_009 - Actor-critic: bootstrapped policy gradient

**Phase:** 4
**Status:** built and tested (18 tests, tiny-MDP anchor on 3 seeds, arithmetic critic anchor).
**Exit criterion NOT measured** - no full training run yet. `entropy_coef` is now set from
E-020 (1.0) and the agent trains; the open question is the degenerate argmax, not the config.
**Date:** 2026-08-25
**Model:** Claude Opus 5
**Decisions:** D-034 (design), D-035 (why entropy_coef was not edited), D-033 (the gate)

---

## What and why

Phase 4's second learner, and the one that completes the arc the whole project has been walking.

- Phases 1-3 learn a **value** and act greedily on it (DP, Q-learning, SARSA, MC, DQN).
- REINFORCE (FEATURE_008) learns a **policy** directly, using the actual return.
- Actor-critic learns **both**, and uses the value estimate to reduce the variance of the policy
  update - at the price of bias.

## Roadmap link

Phase 4, box 2: *"`agents/actor_critic.py` - separate actor and critic heads"*.

## The update rule (S&B 2nd ed. 13.5)

Every step, not every episode:

```
delta = r + gamma*v(s',w) - v(s,w)             (if s' is terminal, gamma*v(s') = 0)
w     <- w     + alpha_w     * delta * grad v(s,w)
theta <- theta + alpha_theta * I * delta * grad ln pi(a|s,theta)
I     <- gamma * I                             (reset to 1 at the episode start)
```

## The one thing to get right, and the one thing to be caught out on

**Bootstrapping is what makes it an actor-critic. Not the network count.**

`reinforce.py` also has two networks - a policy and a learned baseline - and it is not an
actor-critic. The difference is a single term:

| | coefficient on `grad ln pi` | comes from |
|---|---|---|
| REINFORCE + baseline | `G_t - b(s_t)` | the return that **actually happened** |
| actor-critic | `r + gamma*v(s') - v(s)` | the successor's **estimate** |

REINFORCE must wait for the episode to end, because `G_t` does not exist before then.
Actor-critic never waits, because it replaces the rest of the episode with the critic's guess.

`tests/test_reinforce.py::test_the_baseline_is_not_a_critic` pins that REINFORCE does not
bootstrap. `tests/test_actor_critic.py::test_the_critic_bootstraps` pins that this does. The two
test files are deliberate mirror images, and the clearest statement of the difference the
codebase contains:

```
test_reinforce.py     test_update_alone_changes_no_parameters      (nothing happens mid-episode)
test_actor_critic.py  test_update_changes_parameters_immediately   (everything happens mid-step)
```

## What it buys and what it costs

- **Buys variance.** `G_t` sums ~50 noisy rewards; `r + gamma*v(s')` is one reward plus one
  estimate, so the coefficient stops swinging with every unlucky shift.
- **Costs bias.** `v(s')` is wrong early in training, and an update scaled by a wrong number is a
  wrong update. REINFORCE is unbiased and noisy; this is biased and quiet.
- **Buys online learning.** It improves *inside* an episode, where REINFORCE cannot improve until
  the shift is over.
- **Costs wall-clock.** Measured **~0.6 s/episode against REINFORCE's ~0.016** - roughly 37x -
  because it takes a gradient step per environment step rather than one per episode.

The bias-variance trade is visible in the test budgets, which is a satisfying place for theory to
show up: the tiny-MDP anchor needs **40 episodes x 200 steps** here against REINFORCE's
**60 x 800** - about 6x cheaper for the same guarantee.

## Three design choices (D-034)

**1. Two separate networks, not a shared trunk.** A shared trunk couples the actor's and critic's
gradients through weights they both own, and cannot be explained in five minutes (CONSTRAINTS
#13). Separate also keeps the comparison against REINFORCE's policy+baseline pair like-for-like.

**2. An entropy bonus, which S&B 13.5 does NOT have.** `entropy_coef: 0.0` recovers the textbook
update exactly, pinned by `test_zero_entropy_coefficient_is_the_textbook_update`. It exists
because E-018 found REINFORCE's greedy policy degenerate by 300 episodes and E-019 ruled out the
gradient clip. A policy-gradient method has no epsilon; only a term that pays for spread can hold
exploration open. Declared in the module docstring under its own heading.

**3. The bootstrap is dropped at the shift boundary.** `done=True` gives a target of `r` alone.
The shift ends at 480 minutes with the end-of-shift penalty already charged, so there is no future
to value. **Viva point:** bootstrapping through a *time limit* rather than a true terminal is a
real bias in many published implementations; here the shift end is a genuine episode end, so
treating it as terminal is correct rather than merely conventional.

## Tests (18)

| group | what it pins |
|---|---|
| policy | distribution sums to 1, actions are SAMPLED not argmaxed, no epsilon, seeded reproducibility |
| bootstrapping | `update()` moves parameters immediately; `delta = r + gamma*v(s') - v(s)`; the bootstrap is dropped at a terminal |
| arithmetic anchor | a self-looping state paying 1.0 has `v = 1 + gamma*v = 10` at gamma 0.9. A critic that did **not** bootstrap would converge to 1.0, so this is proof the target contains `v(s')` |
| the I accumulator | decays by gamma per step, resets to 1 at the episode boundary |
| entropy | coefficient 0 is the textbook exactly; a bonus measurably keeps the policy less peaked; the reported entropy really is the policy's entropy |
| tiny-MDP anchor | recovers `HAND_COMPUTED_POLICY` on seeds 0/1/2; the critic ranks BUSY above QUIET |

**Episode count is measured, not chosen** (table in the test file): 10 episodes gives 2/3 seeds,
20/30/40 give 3/3. 40 is kept for margin, on the reasoning FEATURE_008 gives - an intermittently
red anchor gets rerun until it passes, and at that point it has stopped testing anything.

## The bug the anchor caught, which is worth reading

The tiny-MDP fixture originally copied REINFORCE's 10x learning-rate scaling (`actor_lr` 0.01,
`critic_lr` 0.05). The anchor failed: **the critic collapsed to a constant**, returning 10.000 for
both states to seven digits, and the policy anchor failed on seed 1.

Not a bug in the agent - a bug in the fixture, and the reason is the difference between the two
algorithms. REINFORCE's baseline takes **one** gradient step per episode; this critic takes one
per **step**, which on the fixture is 200 per episode. It already gets two orders of magnitude
more updates, so raising its learning rate as well over-corrects. The actor keeps the 10x; the
critic runs at the shipped 0.005, and all three seeds then pass with `v(QUIET)` landing on
**10.000 exactly** - `V*(QUIET)` - and `v(BUSY)` ranging 11.2-16.5 around a `V*` of 13.0.

## Known state, stated plainly

**The configuration is now measured, and the agent trains.** `entropy_coef` was 0.01, which broke
it: the policy saturated within five episodes (entropy 0.911 to 0.0003, actor gradient norm to
0.00). E-020 diagnosed a scale mismatch - the bonus contributes at most `coef * ln 5` against TD
errors reaching 1410 - and its sweep set the value to **1.0**, bounded by collapse at 0.01 and 0.1
and by 97.6%-of-uniform entropy at 10.0.

**What is still open, and it is not small.** At *every* coefficient tested - including the ones
where the sampled policy stayed healthy - the **greedy read of the policy was constant
BULK_CLOSE**, scoring exactly -515.4 on all twelve runs. The entropy bonus fixes the policy; it
does not fix the argmax. Combined with E-019 section 3, where REINFORCE's sampled policy beat its
own argmax in nine of nine runs, that puts the evaluation protocol itself in question: both Phase
4 trainers report through a `_GreedyView`. **A decision is owed** before the full runs.

80 episodes is far too short to call the degenerate argmax permanent. The full run is the first
honest test of it.

**Still not measured:** the Phase 4 exit criterion. No full training run of this agent exists.
