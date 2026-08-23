# FEATURE_008 — REINFORCE: Monte Carlo policy gradient, with a baseline

**Phase:** 4
**Status:** in progress — config and tests first, then the agent.
**Date started:** 2026-08-23
**Model:** Claude Opus 5

---

## What and why

Every agent in this project so far learns a **value** and then acts greedily with respect to it: the tabular three, DP, and the DQN. REINFORCE learns the **policy directly**. It is the first agent here with no argmax anywhere in `act()`.

That is the reason it is on the syllabus and the reason it is worth building even if it loses. The value-based agents all share one structural assumption — that a good policy can be read off a value estimate by maximising — and the whole project so far has been measuring how that assumption behaves in this environment. Phase 4 removes it.

## Roadmap link

Phase 4, box 1: *"`agents/reinforce.py` — Monte Carlo policy gradient, with a baseline for variance reduction"*, and box 4: *"show REINFORCE's variance explicitly — it is high, and being able to say why is a strong interview answer."*

**Phase 4's exit criterion:** "all three learners train to a policy beating severity-sort, and the sample-efficiency plot shows a clear ordering you can explain."

⚠️ **That criterion is not yet safe to measure against.** Phases 1, 2 and 3 all closed built-but-not-passed against criteria written the same way, and `HANDOVER.md` records that a single human decision covering all three is still owed. severity_sort is recall 0.84 / reward 40.4; the Phase 3 DQN control reached 0.48 / −46.9. Building and testing REINFORCE does not depend on that decision. **Reporting a Phase 4 result does.** Taking the decision after seeing the number is exactly the failure mode this project exists to avoid.

## The update rule

Sutton & Barto 2nd ed. §13.3 (REINFORCE) and §13.4 (REINFORCE with Baseline):

```
G_t = r_{t+1} + gamma*r_{t+2} + ... + gamma^{T-t-1}*r_T        the actual return
theta <- theta + alpha * gamma^t * (G_t - b(s_t)) * grad_theta ln pi(a_t | s_t, theta)
```

Read it as three factors multiplying a direction:

- `grad ln pi(a_t|s_t)` — the direction in parameter space that makes the action **that was actually taken** more likely.
- `(G_t - b(s_t))` — how much better the episode went than expected. Positive: push that way. Negative: push the other way. **This is the entire algorithm.**
- `gamma^t` — see below.

`b(s_t)` is a learned state-value estimate `v̂(s_t, w)`, trained on the same returns by `w <- w + alpha_w * (G_t - v̂(s_t,w)) * grad_w v̂(s_t,w)`.

### Why the baseline reduces variance without adding bias

Subtracting any function of state alone leaves the gradient's expectation unchanged, because `E[b(s) * grad ln pi(a|s)] = b(s) * sum_a grad pi(a|s) = b(s) * grad 1 = 0`. What it changes is the spread. Without it, every action taken in a good shift is reinforced — including the bad ones — because `G_t` is large and positive for all of them. With it, only actions that did better than the state's own average get pushed up. **Same expectation, smaller variance**, which is the one sentence to have ready in a viva.

### REINFORCE-with-baseline is NOT actor–critic

S&B §13.5 is explicit about this and it is the most likely place to be caught out. The baseline here is used **only as a baseline** — it never appears inside the target. `G_t` is the full observed return, so nothing bootstraps. Actor–critic (the next commit block) replaces `G_t` with `r + gamma*v̂(s')`, which *does* bootstrap, and that is what buys it lower variance and costs it bias. Two networks is not what makes something an actor–critic; **bootstrapping is**.

### The `gamma^t` factor is implemented, and most implementations drop it

S&B's boxed algorithm includes it: the return from step `t` of a discounted problem is worth `gamma^t` from the episode's start, and dropping it optimises the undiscounted objective with discounted returns — a mismatch S&B flags in a footnote. Most published code drops it because it shrinks late-episode gradients toward zero and slows learning. This project keeps it, because the teaching constraint says the code should be the textbook's algorithm and a student should be able to say what it is for. With `gamma = 0.99` over the ~50-step shifts here, `gamma^t` reaches ~0.6 at the end of an episode — real but not crippling. If it turns out to matter empirically, that becomes an experiment with a number, not a silent edit.

## Approach

| Piece | Choice |
|---|---|
| Observation | `state.featurise` — the same 17 continuous columns as the DQN, scaled by the shared `features.scales` (D-032). Same input to both, so the sample-efficiency comparison compares algorithms. |
| Policy | MLP → 5 logits → softmax. Actions are **sampled** from it, never argmaxed. |
| Exploration | The policy's own stochasticity. **No epsilon anywhere** — this agent has no exploration schedule to decay, which is a visible structural difference from every earlier learner. |
| Baseline | A second MLP, `v̂(s,w)`, one scalar output, its own learning rate. Switchable off (`use_baseline`) — that switch is the variance demonstration box 4 asks for. |
| When it learns | `end_episode()`. `update()` can only buffer, exactly as in `monte_carlo.py`, and for the identical reason: the update needs `G_t`, which does not exist until the episode is over. |
| Gradient clipping | On, by global norm, as in the DQN. |

## Risk carried into the first run

Returns in this environment reach ±500. The policy-gradient step is proportional to `(G_t - b)`, so an unbaselined gradient here is two to three orders of magnitude larger than the ±1-scale returns most REINFORCE examples assume. Gradient clipping and the baseline are both load-bearing for that reason, not decorative. **The first short run is a diagnostic, not a result** — the Phase 3 lesson (E-016) is that this environment's reward magnitudes break defaults that are fine elsewhere, and the failure looked like a converged loss curve.

## Files touched

| File | Change |
|---|---|
| `config/training_default.yaml` | fill in the `reinforce:` block |
| `src/soc_triage/config/training.py` | `ReinforceConfig`, loader section, validation, seed-block check |
| `tests/test_reinforce.py` | new — written before the agent |
| `src/soc_triage/agents/reinforce.py` | new |

## Status log

- **2026-08-23** — feature doc written. Prerequisites landed first: `config.py` split into a package (D-031) and `feature_scales` promoted to a shared block (D-032) so REINFORCE and the DQN provably share input scaling.
