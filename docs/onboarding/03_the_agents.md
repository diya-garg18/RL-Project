# Document 3 — The Agents

### Every algorithm in the project, and what makes each one different

> **Read Document 2 first.** This document assumes you know what a state is in this
> project (an integer 0–575, or 17 floats), what the five actions are, and where the
> reward comes from.
>
> **The promise of this document:** by the end you should be able to say, for each
> algorithm, *what its update rule is, what one line separates it from its nearest
> neighbour, and what that difference costs or buys.* That is exactly what an examiner
> asks.

---

## Contents

| § | What it covers | File(s) |
|---|---|---|
| 1 | The family tree — one page, all eleven agents | *(map)* |
| 2 | The interface everything implements | `agents/base.py` |
| 3 | The five baselines, including the cheater | `agents/baselines.py` |
| 4 | Dynamic programming — planning with a model | `agents/dp.py` |
| 5 | The shared tabular machinery | `agents/tabular.py` |
| 6 | Monte Carlo — no bootstrapping at all | `agents/monte_carlo.py` |
| 7 | SARSA — on-policy TD | `agents/sarsa.py` |
| 8 | Q-learning — off-policy TD | `agents/q_learning.py` |
| 9 | The replay buffer | `agents/replay.py` |
| 10 | DQN — Q-learning with a network | `agents/dqn.py` |
| 11 | REINFORCE — learning the policy directly | `agents/reinforce.py` |
| 12 | Actor–critic — bootstrapped policy gradient | `agents/actor_critic.py` |
| 13 | The four questions you will be asked | *(revision)* |
| 14 | Every hyperparameter, and where it lives | `config/training_default.yaml` |

---

## 1. The family tree

Eleven agents. Here is how they relate, and where each one sits in the standard
taxonomy of RL:

```
                              AGENTS
                                |
        +-----------------------+------------------------+
        |                                                |
   NOT LEARNING                                      LEARNING
   (fixed rules)                                          |
        |                              +-----------------+------------------+
   baselines.py                        |                                    |
   - random                     LEARN A VALUE,                        LEARN THE
   - fifo                       ACT GREEDILY                          POLICY DIRECTLY
   - severity_sort                     |                                    |
   - cheapest_first          +---------+---------+              +-----------+---------+
   - oracle_greedy           |                   |              |                     |
     (cheats — upper       MODEL-BASED       MODEL-FREE      REINFORCE          ACTOR-CRITIC
      bound only)             |                   |          (§13.3/13.4)        (§13.5)
                            dp.py                 |          no bootstrap        bootstraps
                          (§4.3/4.4)              |
                          value iteration   +-----+------+------------+
                          policy iteration  |            |            |
                                       monte_carlo    sarsa      q_learning ---> dqn
                                        (§5.4)       (§6.4)        (§6.5)      (§16.5)
                                       no bootstrap  on-policy    off-policy   + network
                                                                                + replay
                                                                                + target net
```

**Two questions place any algorithm on this tree.** Learn them and the whole map
follows:

| Question | If yes | If no |
|---|---|---|
| **Does it bootstrap?** *(use its own estimate of a future state's value)* | DP, SARSA, Q-learning, DQN, actor–critic | Monte Carlo, REINFORCE |
| **Does it learn the value of the policy it's actually following?** | on-policy: SARSA, MC, REINFORCE, actor–critic | off-policy: Q-learning, DQN |

Bootstrapping trades **bias** for **variance**. Not bootstrapping is unbiased and noisy;
bootstrapping is biased and quiet. That single trade-off explains most of the differences
in this document.

### One line each

| Agent | Textbook | The one line that is the algorithm |
|---|---|---|
| `random` | — | `return rng.integers(0, 5)` |
| `fifo` | — | `return PULL_OLDEST` |
| `severity_sort` | — | `return PULL_HIGHEST_SEVERITY` |
| `cheapest_first` | — | `return PULL_CHEAPEST` |
| `oracle_greedy` | — | *"if any rule lands on a real incident, take it"* |
| `dp` value iteration | S&B §4.4 | `V(s) ← max_a [ R̂(s,a) + γ Σ P̂(s'|s,a) V(s') ]` |
| `dp` policy iteration | S&B §4.3 | evaluate → improve → repeat until the policy stops changing |
| `monte_carlo` | S&B §5.4 | `Q(s,a) ← Q(s,a) + α [ G_t − Q(s,a) ]` |
| `sarsa` | S&B §6.4 | `Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]` |
| `q_learning` | S&B §6.5 | `Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]` |
| `dqn` | S&B §16.5 | minimise `( r + γ max_a' Q_target(s',a') − Q_online(s,a) )²` |
| `reinforce` | S&B §13.3/4 | `θ ← θ + α γ^t (G_t − b(s_t)) ∇ ln π(a_t|s_t)` |
| `actor_critic` | S&B §13.5 | `δ = r + γ v(s') − v(s)`, then step both heads on δ |

---

## 2. The interface everything implements — `agents/base.py`

Thirty-four lines, and every agent in the project satisfies it.

```python
class Agent:
    name: str = "base"
    obs_kind: str = "disc"

    def act(self, obs) -> int: ...
    def update(self, obs, action, reward, next_obs, done) -> None: ...
    def save(self, path) -> None: ...
    def load(self, path) -> None: ...
```

### `obs_kind` — how the runner knows what to feed you

```
  obs_kind = 'disc'      →  runner calls state.discretise()  →  an int, 0..575
  obs_kind = 'cont'      →  runner calls state.featurise()   →  17 floats
  obs_kind = 'snapshot'  →  runner passes the raw EnvSnapshot (ORACLE ONLY)
```

This one string is what lets the runner drive a Q-table and a neural network with
identical code. The agent declares what it eats; the runner serves it.

### `update()` defaults to doing nothing

```python
def update(self, obs, action, reward, next_obs, done) -> None:
    """Learning hook. Baselines have nothing to learn — default is a no-op,
    so the runner never needs to special-case them (FLOW.md Flow B)."""
```

That comment names the real reason. If `update` raised `NotImplementedError`, the runner
would need `if isinstance(agent, Learner):` scattered through it. A no-op default means
**there is exactly one loop in the runner, and it works for all eleven agents.** That's
CONSTRAINTS #10 — the environment and runner must not know which algorithm they're
driving — held up by a two-line method body.

### The sanctioned exception, declared in the base class

```python
'snapshot' -> the raw EnvSnapshot (oracle ONLY — it is the point of an upper bound)
```

Document 2 §8 describes the ground-truth firewall. The one legal hole in it is declared
right here, in the interface, rather than being a special case buried in the runner.
Anyone reading `base.py` learns both the rule and its single exception at once.

---

## 3. The five baselines — `agents/baselines.py`

Before you can claim a learned agent is good, you need to know what "good" means. These
five give you the scale.

| Baseline | Rule | Its job in the report |
|---|---|---|
| `random` | uniform over 5 actions | **The floor.** Anything losing to this is broken. |
| `fifo` | always `PULL_OLDEST` | The naive human default: work the queue in order. |
| `severity_sort` | always `PULL_HIGHEST_SEVERITY` | **What the industry does. THE baseline to beat.** |
| `cheapest_first` | always `PULL_CHEAPEST` | Throughput strawman: maximise alerts closed. |
| `oracle_greedy` | cheats | **The ceiling.** Never presented as a result. |

The first four are three lines each. That's the point — they're not implementations, they
are *reference points*, and the simpler they are the harder they are to get wrong.

### Why `severity_sort` is the one that matters

It is what a real SOC does. If your learned agent can't beat "look at the scariest thing
first", you have built an expensive way to do nothing. And Document 2 §3 explains why
beating it is *possible but not trivial*: severity correlates with truth at only
r ≈ 0.32, so sorting by it captures real signal while leaving most of it on the table.

### `oracle_greedy` — the cheater, in detail

This is the most interesting file in the module and worth reading closely, because it
shows what "an upper bound" actually costs to build honestly.

**It reads `is_true_incident` directly.** That's the sanctioned CONSTRAINTS #1 exception.
But look at what it *can't* do with that knowledge:

```
  It knows which alerts are real.
  It CANNOT pick an alert directly — it still only has the same 5 actions.
```

So it has to work backwards from the rules. Its three-step logic:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. Would any of the four pull rules land on a real incident?     │
  │    → take that rule. Most urgent deadline first.                 │
  └─────────────────────────────────────────────────────────────────┘
                     │ no
                     v
  ┌─────────────────────────────────────────────────────────────────┐
  │ 2. Is a real incident in the queue that no rule currently        │
  │    reaches?                                                      │
  │    → CLEAR A PATH. Find the rule needing the fewest removals     │
  │      before that incident becomes its argmax, and pull along     │
  │      that path (each pull removes that path's top blocker).      │
  └─────────────────────────────────────────────────────────────────┘
                     │ no incident visible
                     v
  ┌─────────────────────────────────────────────────────────────────┐
  │ 3. Wait as cheaply as possible.                                  │
  │    Safe bulk-close (2 min) beats a cheapest pull (5 min).        │
  └─────────────────────────────────────────────────────────────────┘
```

**Step 1 requires it to simulate the environment's own tie-breaking.** It re-derives what
each of the four rules would select, using the same conventions as `env._select_alert` —
strict comparison, ties to the lowest id. If it got that wrong it would predict the wrong
alert and its "upper bound" would be too low.

**Step 2 is the clever part, and it had a bug worth learning from:**

```python
# Bulk-close is only useful if it actually removes blockers on the
# chosen path — it can never touch high-severity or high-criticality
# blockers, so blindly sweeping junk would loop forever on hygiene
# while the incident sits (the bug the first oracle version had).
```

The first version bulk-closed whenever it was safe. But bulk-close can only touch
severity ≤ 1 on criticality-0 assets — so if the alert blocking your path is a severity-3
alert, sweeping junk does nothing to reach the incident. The oracle happily did queue
hygiene forever while a real incident aged out. The fix: only bulk-close if it actually
removes a blocker on the chosen path.

**And the docstring is honest about what it still isn't:**

> *"Still not optimal (greedy, no lookahead over arrivals) — but no honest policy can see
> more than it sees."*

This is exactly the right way to describe an upper bound. It is *an* upper bound, not
*the* optimum. A true optimum would need to plan over future arrivals. Saying so keeps
the claim defensible.

---

## 4. Dynamic programming — `agents/dp.py`

### The problem DP has here

Classical DP needs the environment's **model**: `P(s'|s,a)` and `R(s,a)`. Do we have one?

> *"The true transition dynamics of the queue are not analytically available."*

You could in principle derive them — but the transition depends on the whole queue
composition, the arrival process, and which alert each rule would pick. It is not a
tractable derivation. So the project does the honest alternative: **estimate the model by
counting.**

```
  ┌──────────────────────────────────────────────────┐
  │ 1. estimate_model()                              │
  │    Run 50,000 episodes under a RANDOM policy.    │
  │    Count every (s, a) → s' transition.           │
  │    Average every (s, a) reward.                  │
  │                                                  │
  │    → P̂ (576, 5, 576)   R̂ (576, 5)               │
  └──────────────────────────────────────────────────┘
                       │
       ┌───────────────┴────────────────┐
       v                                v
  ┌────────────────────┐        ┌────────────────────┐
  │ 2. value_iteration │        │ 3. policy_iteration│
  │    S&B §4.4        │        │    S&B §4.3        │
  └────────────────────┘        └────────────────────┘
       │                                │
       └───────────────┬────────────────┘
                       v
            must agree on ≥95% of states
            (FLOW.md Flow C cross-check)
```

### The sentence that has to be said every time

> *"The resulting policy is optimal FOR THE ESTIMATED MODEL, not for the true
> environment — say exactly that, every time (D-004)."*

This is a small thing that separates a careful project from a sloppy one. "We computed
the optimal policy with dynamic programming" is false. "We computed the optimal policy
for a model estimated from 50,000 random-policy rollouts" is true, and immediately raises
the right follow-up question: *how good is that estimate?* — which is why `estimate_model`
also returns raw `visits` as "coverage evidence for the report".

### Unvisited state-actions — D-011

```python
else:
    P_hat[s, a, s] = 1.0  # unvisited: absorbing self-loop, reward 0 (D-011)
```

Some `(s,a)` pairs are never seen even in 50,000 episodes. What should the model say
about them? Three options:

| Option | Consequence |
|---|---|
| Uniform over all 576 next states | Invents dynamics. Fiction in the Bellman backups. |
| Optimistic value | Greedy policy would *prefer* the unknown — the worst outcome |
| **Absorbing self-loop, reward 0** | Value stays 0. Never preferred, never crashes. |

The docstring's reasoning:

> *"A state-action we never saw under 50k random episodes is essentially unreachable;
> pretending we know its dynamics would inject fiction into the Bellman backups, and a
> neutral self-loop keeps its value at 0 so the greedy policy never prefers the unknown."*

### Value iteration, written as the Bellman equation

```python
for _ in range(max_sweeps):
    delta = 0.0
    for s in range(n_states):
        best = -np.inf
        for a in range(N_ACTIONS):
            q_sa = R_hat[s, a] + gamma * (P_hat[s, a] @ V)  # Bellman expectation
            if q_sa > best:
                best = q_sa
        delta = max(delta, abs(best - V[s]))
        V[s] = best
    deltas.append(delta)
    if delta < theta:
        break
else:
    raise RuntimeError(f"value iteration did not converge in {max_sweeps} sweeps ...")
```

Compare to the textbook:

```
    V(s) ← max_a [ R̂(s,a) + γ Σ_s' P̂(s'|s,a) V(s') ]
                   └──────┘       └────────────────┘
                   R_hat[s,a]        P_hat[s,a] @ V
```

The `@` is a dot product — `P_hat[s, a]` is a 576-length probability vector, `V` is a
576-length value vector, and their dot product is exactly `Σ_s' P(s'|s,a) V(s')`. The
style note in the docstring is worth repeating:

> *"the vector `P[s, a] @ V` inside is the expectation term of the Bellman equation
> written directly, not an optimisation trick."*

**The `for...else` is doing real work.** Python's `else` on a `for` runs only if the loop
finished *without* `break`. So: converge → `break` → skip the `else`. Run out of sweeps →
`else` → raise. A non-converging run **fails loudly instead of silently returning a
half-converged value function** that would look like a result.

### Policy iteration — the same answer by a different road

```python
for rounds in range(1, max_sweeps + 1):
    # --- policy evaluation (§4.1): V ← V^π by iterative sweeps
    ...
    # --- policy improvement: greedy w.r.t. the evaluated V
    new_policy = greedy_policy(P_hat, R_hat, gamma, V)
    if np.array_equal(new_policy, policy):
        return V, policy, rounds
    policy = new_policy
```

The difference from value iteration in one picture:

```
 VALUE ITERATION                    POLICY ITERATION

 V ← max over actions               evaluate current policy to convergence
 V ← max over actions               improve policy greedily
 V ← max over actions               evaluate new policy to convergence
 ...until V stops moving            improve again
                                    ...until the POLICY stops changing
 Improvement is folded into
 every sweep.                       Two clearly separated phases.
```

**Why implement both?** Because they must agree, and that agreement is a free bug check:

> *"Cross-check (FLOW.md Flow C): must agree with value_iteration on ≥95% of states —
> disagreement beyond ties means one of them is wrong."*

Two independent implementations of "the optimal policy for this model" landing on the
same answer is much stronger evidence than either one alone. Ties (where two actions have
genuinely equal value) explain the missing 5%.

---

## 5. The shared tabular machinery — `agents/tabular.py`

Q-learning, SARSA and Monte Carlo differ in **one method**. Everything else is here.

### Why this file exists — and it isn't to save typing

> *"The point is not to save typing: it is that each agent file now contains essentially
> nothing but its own update rule, which is the part the students must be able to write
> from memory and the part an examiner will ask about. Reading `sarsa.py` next to
> `q_learning.py` shows the difference in a few lines rather than burying it in forty
> lines of shared boilerplate."*

That is a *pedagogical* justification for a refactor, and it's the right one for this
project. The result:

```
   tabular.py       Q-table, epsilon-greedy, argmax, decay, visits   (144 lines)
        │
        ├── q_learning.py    the update rule, and nothing else        (71 lines)
        ├── sarsa.py         the update rule + the a' commitment     (107 lines)
        └── monte_carlo.py   the update rule + the episode buffer    (105 lines)
```

It also notes it was extracted **after the third implementation appeared, not before** —
the rule of three. Abstracting after two examples usually produces the wrong abstraction.

### Why not put this in `base.py`?

> *"`base.Agent` is the interface every agent implements, baselines included, and
> baselines have no Q-table, no exploration schedule and nothing to decay. Widening that
> interface to suit three subclasses would push dead members onto five agents that do not
> want them (CONSTRAINTS #18)."*

A clean statement of a principle worth internalising: **an interface should carry what
all its implementors need, not what most of them need.**

### Zero init, not optimistic init

```python
# Zero init, not optimistic init. Optimistic values are a legitimate
# exploration technique but a different algorithm, and would change
# every number this project reports — it would need a DECISIONS entry.
self.Q = np.zeros((n_states, n_actions), dtype=np.float64)
```

Optimistic initialisation (starting Q high so every action looks worth trying) is a
standard trick. It is deliberately *not* used, and the comment explains that the reason
isn't that it's bad — it's that it would be **a different algorithm**, silently changing
every reported number.

### The visit counts — a result-integrity feature

```python
# Visit counts play no part in learning. They exist so the policy table
# can print "never seen" instead of silently reporting the argmax
# tie-break — see FEATURE_005, where 455 of 576 unvisited states would
# otherwise have rendered as a confident preference (E-009).
self.visits = np.zeros((n_states, n_actions), dtype=np.int64)
```

**Read that number again: 455 of 576 states were never visited.**

Without the counts, the policy table would have printed action 0 for all 455 of them —
because a row of zeros has its argmax at index 0 by the tie-break rule. It would have
looked like the agent had *decided* to `PULL_HIGHEST_SEVERITY` in 455 states. It had
decided nothing; it had never been there.

This is the most instructive small feature in the codebase. The counts contribute nothing
to learning. They exist purely so the project cannot accidentally report a tie-break as a
finding.

### The argmax tie-break, and why it's load-bearing

```python
def _argmax(self, state: int) -> int:
    best_action = 0
    best_value = -np.inf
    for action in range(self.n_actions):
        if self.Q[state, action] > best_value:   # strict >
            best_value = self.Q[state, action]
            best_action = action
    return best_action
```

An explicit loop instead of `np.argmax`, and the docstring says why:

> *"A zero-initialised table ties on every action at step one of every run, so an
> arbitrary rule would make runs unreproducible across seeds. Strict `>` keeps the first
> index — the same convention as `agents.dp.greedy_policy`, so the learned policies and
> the DP policy are comparable rather than merely similar."*

Two separate reasons, both good:
1. **Reproducibility** — a zero table ties everywhere at step 1.
2. **Comparability** — DQN, DP and all three tabular agents use the *same* rule, so a
   policy disagreement between them is a real disagreement, not two conventions.

### Epsilon decay — per episode, not per step (D-015)

```python
def end_episode(self) -> None:
    decayed = self.epsilon * self.epsilon_decay
    self.epsilon = decayed if decayed > self.epsilon_min else self.epsilon_min
```

The config says `decay: 0.9995  # per episode`. What if you applied it per step instead?

```
  Per episode (correct):   0.9995 ^ 20000 episodes  →  reaches the 0.05 floor gradually
  Per step   (the bug):    ~50 steps per shift, so within ONE shift epsilon has
                           decayed 50 times. Within a few hundred shifts it's on
                           the floor. The agent stops exploring almost immediately.
```

And the docstring names the reason this bug is expensive:

> *"the symptom (a learning curve that plateaus early) looks nothing like the cause."*

You'd see a flat learning curve and start tuning the learning rate. That's the general
shape of the worst bugs in RL: **the symptom points somewhere other than the cause.**

There's also a subtle note about why `done` can't be used to detect episode ends:

> *"It cannot be inferred from `done` either: the tiny MDP is continuing and never sets
> it."*

### The exploration floor

```yaml
epsilon:
  min: 0.05   # floor stays above zero: queue composition shifts during a shift,
              # so a fully greedy agent stops adapting.
```

Epsilon never reaches zero. In a stationary problem you'd anneal to 0 and take the
optimum. Here the queue at minute 30 and the queue at minute 450 are genuinely different
worlds, so a permanently greedy agent stops discovering.

---

## 6. Monte Carlo — `agents/monte_carlo.py`

**S&B §5.4. First-visit MC control.**

```
    G_t   = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...     the ACTUAL return
    Q(s,a) ← Q(s,a) + α [ G_t − Q(s,a) ]
```

### The defining property: no bootstrapping

> *"This is the one algorithm in Phase 2 that never uses its own estimate of a
> successor's value. It waits for the episode to finish and uses the return that actually
> happened."*

```
   TD methods (SARSA, Q-learning):        Monte Carlo:

   "This state is worth                   "This state is worth
    r + γ × (my guess about                r + γr' + γ²r'' + ...
    the next state)"                       — what actually happened"

   Biased (the guess is wrong             Unbiased (it's the real
   early on), low variance                 return), high variance
```

**High variance, concretely:** one unlucky shift — a crown-jewel incident missed, −500 —
moves the estimate for *every state-action pair visited during that shift*. The noise
doesn't average out within an episode; it contaminates the whole episode's updates.

### The structural consequence

```python
def update(self, obs, action, reward, next_obs, done) -> None:
    """Buffer the step. MC cannot learn anything yet."""
    self._episode.append((obs, action, float(reward)))
```

`update()` does nothing but record. All the learning is in `end_episode()`. As the
docstring says:

> *"That is not an implementation choice, it is what 'use the actual return' requires,
> and it means MC cannot learn at all on a task that never terminates."*

Note also the honest comment about the unused parameters:

> *"`next_obs` and `done` are unused ... They stay in the signature because the runner
> calls every agent identically — the runner must not need to know which algorithm it is
> driving (CONSTRAINTS #10)."*

### First-visit vs every-visit

If a state-action pair occurs three times in an episode, do you use all three returns, or
only the first?

```python
first_occurrence: dict[tuple[int, int], int] = {}
for t, (state, action, _) in enumerate(self._episode):
    if (state, action) not in first_occurrence:
        first_occurrence[(state, action)] = t
```

This project is **first-visit** (S&B §5.1). Both variants converge, but they differ on
finite data — and the test pins which one:

> *"from identical episode data they produce 2.615 and 2.8075 respectively."*

That's a good test design. It doesn't check "MC converges" (which both do); it checks a
number that only first-visit produces.

### The backwards walk

```python
G = 0.0
for t in range(len(self._episode) - 1, -1, -1):
    state, action, reward = self._episode[t]
    G = reward + self.gamma * G
    if first_occurrence[(state, action)] == t:
        self.Q[state, action] += self.alpha * (G - self.Q[state, action])
        self.visits[state, action] += 1
```

Walking backwards makes each return one multiply-add from the previous one:

```
   Step:      0      1      2      3      4  (end)
   Reward:   r0     r1     r2     r3     r4

   Backwards, G = r + γG :
     t=4:  G = r4
     t=3:  G = r3 + γ·r4
     t=2:  G = r2 + γ·r3 + γ²·r4
     t=1:  G = r1 + γ·r2 + γ²·r3 + γ³·r4
     t=0:  G = r0 + γ·r1 + γ²·r2 + γ³·r3 + γ⁴·r4

   O(T). Computing each one forwards from scratch would be O(T²).
```

### The buffer clear, and the bug it prevents

```python
self._episode.clear()
super().end_episode()  # the epsilon decay every learner shares (D-015)
```

> *"a leaked buffer folds the previous episode's rewards into this one's returns, a bug
> whose only symptom is 'MC is oddly noisy'."*

Again: the symptom does not point at the cause. There's a test named exactly
`test_monte_carlo_clears_its_buffer_between_episodes`.

---

## 7. SARSA — `agents/sarsa.py`

**S&B §6.4. On-policy TD control.**

```
    Q(s,a) ← Q(s,a) + α [ r + γ·Q(s',a') − Q(s,a) ]
                                    ↑
                        the action it WILL ACTUALLY TAKE
```

The name is the tuple: **S**tate, **A**ction, **R**eward, next **S**tate, next **A**ction.

### On-policy, in one sentence

> *"If epsilon-greedy is about to do something foolish, SARSA bootstraps off the foolish
> action's value and learns that this state is worth less than it would be under perfect
> play."*

SARSA evaluates the policy it's *actually following*, exploration and all. So it learns
that standing near a cliff is dangerous — not because the optimal action falls off, but
because it knows it will occasionally step randomly.

### The `a'` problem, and a genuinely elegant solution

Here is a real engineering problem. The interface is:

```python
def update(self, obs, action, reward, next_obs, done) -> None:
```

There is no `a'`. The runner doesn't know SARSA needs one, and widening the interface
would force ten other agents to supply something meaningless.

**The solution: SARSA picks `a'` itself during `update`, caches it, and the next `act()`
returns exactly that cached action.**

```python
def act(self, obs: int) -> int:
    if self._committed_action is not None:
        action = self._committed_action
        self._committed_action = None
        return action
    return self._epsilon_greedy(obs)
```

```python
def update(self, obs, action, reward, next_obs, done) -> None:
    if done:
        target = reward
        self._committed_action = None
    else:
        if self._committed_action is None:
            self._committed_action = self._epsilon_greedy(next_obs)
        next_action = self._committed_action
        self.last_bootstrap_action = next_action
        target = reward + self.gamma * self.Q[next_obs, next_action]
    ...
```

The flow:

```
   act(s)  ────────────>  a           (no commitment: choose fresh)
   env.step(a)  ───────>  r, s'
   update(s,a,r,s')  ──>  chooses a' with ε-greedy
                          bootstraps off Q[s', a']
                          COMMITS to a'
   act(s')  ───────────>  a'          (returns the commitment)
   env.step(a')  ──────>  ...
```

### The warning that makes this file worth reading twice

> *"The on-policy property depends entirely on that cache being honoured: if `act()`
> re-sampled instead, the agent would become a strange hybrid that **still converges and
> still passes every value test**."*

This is the archetype of the bug class this project is built to catch. Break the
commitment and SARSA still runs, still learns, still produces a sensible-looking curve,
and is **no longer SARSA**. You would report results for an algorithm that does not have
a name.

`test_sarsa_actually_takes_the_action_it_bootstrapped_off` is the tripwire.

### `force_next_action()` — a test hook that says why it exists

```python
def force_next_action(self, action: int) -> None:
    """Pin the next action, for tests that need a specific `a'`.
    ... instead of fishing for a seed that happens to produce it. A
    seed-hunted test breaks the moment the RNG stream shifts for an
    unrelated reason."""
```

Worth stealing as a habit. The alternative — hunting for a seed where epsilon-greedy
happens to explore at the right moment — produces a test that silently stops testing
anything the day someone adds an unrelated RNG call.

---

## 8. Q-learning — `agents/q_learning.py`

**S&B §6.5. Off-policy TD control.**

```
    Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
                                  ↑
                        the value of the BEST next action,
                        regardless of what it will actually do
```

### The one-line difference from SARSA

```
   SARSA:       target = r + γ · Q[s', a']            a' = what I will do
   Q-learning:  target = r + γ · max_a' Q[s', a']     the best I could do
```

That is the entire difference between the two files. Everything else is in `tabular.py`.

> *"So the agent explores with one policy and learns the value of another: the greedy
> one."*

### The measurable consequence — and how the project proves it

This is where the project does something better than assert:

> *"on the tiny MDP: Q-learning converges to q_* itself (9.24e-14, E-007), while SARSA
> converges to `tiny_mdp.epsilon_soft_q(epsilon)`, which at epsilon = 0.1 is lower by
> more than 1.5 in places. Q-learning does not pay for its own exploration; SARSA pays."*

There is a tiny hand-solvable MDP (`tiny_mdp.py`, covered in Document 4) with an
analytically known optimal Q. Q-learning converges to it to 14 decimal places. SARSA
converges to a *different, also analytically known* quantity — the ε-soft optimum.

**That's not "our implementation seems right". That's "our implementation converges to
the exact number the theory says it should."**

### The test that catches what convergence can't

```
test_update_bootstraps_off_the_max_not_the_behaviour_action
```

> *"pins this at a single backup, because the two agree at the optimum and a convergence
> test on this fixture cannot tell them apart."*

At the optimum, the greedy action *is* the best action, so `max_a' Q` and `Q[s',a']`
agree — and a convergence test would pass for both algorithms. So the test checks **one
backup** where they differ, not the endpoint where they don't. That is a real insight
about test design.

### The `done` warning, stated twice in the project

```python
"""`done` must mean *terminated*, never merely truncated. On termination
there is no successor, so the target is the reward alone; bootstrapping
past a terminal state invents value that does not exist. The mirror-image
bug is just as costly — passing done=True at an episode-length cutoff on
a continuing task teaches the agent the world ends, dragging every value
toward the last reward it happened to see."""
```

```
   TERMINATED                          TRUNCATED
   The episode genuinely ended.        You stopped it early.
   There is no future.                 The future exists; you just aren't watching.

   target = r                          target = r + γ·max Q(s')
   ✓ correct                           ✓ correct

   Confusing them in EITHER direction is a real bug.
```

In this project the shift end at 480 minutes is a **genuine terminal** — the end-of-shift
penalty is charged and nothing follows. So `done=True` is correct here. But the file says
it explicitly, because the same code copied into a time-limited continuing task would be
wrong.

---

## 9. The replay buffer — `agents/replay.py`

Written by hand (CONSTRAINTS #7). Mnih et al. (2015); S&B §16.5.

### It buys two distinct things

The docstring is explicit that these are separate and you should be able to name both:

**1. Decorrelation.**

```
   WITHOUT REPLAY — learning from consecutive transitions:

   step t   : queue = [43 alerts]  ─┐
   step t+1 : queue = [42 alerts]   │  nearly the same sample,
   step t+2 : queue = [41 alerts]   │  over and over
   step t+3 : queue = [40 alerts]  ─┘

   Each gradient step sees a batch that is effectively ONE sample.
   The network chases whatever the agent is doing right now.

   WITH REPLAY — uniform sampling from 100,000 stored transitions:

   batch = [ shift 1841 step 12,  shift   77 step 40,
             shift 9002 step  3,  shift 4410 step 28, ... ]

   Thousands of different shifts mixed into every batch.
```

**2. Sample reuse.** Each transition costs one environment step to generate and can then
be learned from many times.

### Five arrays, not a deque of tuples

```python
self._obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
self._action   = np.zeros(capacity, dtype=np.int64)
self._reward   = np.zeros(capacity, dtype=np.float32)
self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
self._done     = np.zeros(capacity, dtype=np.float32)
```

> *"Sampling then returns contiguous arrays ready for `torch.from_numpy` with no
> per-sample Python work, and the capacity semantics are two integers you can print
> (`_cursor`, `_size`) rather than container behaviour you have to trust."*

The ring buffer, drawn:

```
   capacity = 8, after 11 pushes:

   index:    0     1     2     3     4     5     6     7
           ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
           │ #8  │ #9  │ #10 │ #3  │ #4  │ #5  │ #6  │ #7  │
           └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                          ^
                       _cursor = 3   ← next write goes here,
                                        overwriting the OLDEST (#3)
                       _size = 8     ← full

   self._cursor = (i + 1) % self.capacity
   self._size   = min(self._size + 1, self.capacity)
```

### Two details with reasons

**`done` stored as float, not bool:**

```python
# `done` is stored as a float, not a bool, because it is used
# arithmetically in the target: y = r + gamma * (1 - done) * max Q'.
```

**Its own RNG:**

```python
# Own generator, not the global numpy one: two buffers with the same
# seed must sample identically regardless of what else ran first.
```

This pattern recurs across the whole project — **every source of randomness owns its own
seeded generator.** Environment, agent exploration, replay sampling: three separate
streams. Changing one cannot shift another, which is what makes two runs comparable.

### The alignment property

```python
idx = self._rng.integers(0, self._size, size=batch_size)
return (self._obs[idx], self._action[idx], self._reward[idx],
        self._next_obs[idx], self._done[idx])
```

> *"Indexing all five arrays with the same index array is what keeps a transition's five
> fields together — that alignment is the property
> `test_transitions_stay_aligned_across_the_five_arrays` exists to pin."*

If the five arrays ever fell out of alignment, you'd be training on `(state from
transition 400, action from transition 900, reward from transition 12)`. Complete
nonsense — and nothing would crash.

---

## 10. DQN — `agents/dqn.py`

**S&B §16.5. Q-learning with the table replaced by a network.**

### The claim, and it's the right one

```
    tabular:  Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
    DQN:      minimise  ( r + γ·max_a' Q_target(s',a') − Q_online(s,a) )²
              by gradient descent on the online network's weights
```

> *"The bracketed quantity is the same TD error in both. What changed is that `alpha` is
> no longer a step size on one table cell — the gradient step moves every weight, and
> therefore moves the estimate for *every* state at once. That generalisation is the
> entire reason to do this, and also the entire reason it is unstable without help."*

That last sentence is the whole of DQN in one line. Generalisation is the feature *and*
the problem.

### The two stabilisers

**1. Experience replay** — §9 above.

**2. The target network.** A frozen copy of the online network, refreshed by a hard copy
every 1000 *gradient* steps.

```
   WITHOUT A TARGET NETWORK:

     target = r + γ · max Q_online(s')
                          ↑
              the same network you're about to update

     You move the weights → the target moves → you chase it → repeat.
     The network is chasing a value it is itself changing.

   WITH A TARGET NETWORK:

     target = r + γ · max Q_target(s')
                          ↑
              a frozen snapshot, unchanged for 1000 gradient steps

     Now it's ordinary supervised regression against a fixed target.
     Every 1000 steps the target is refreshed and you regress again.
```

The docstring names the sentence to memorise:

> *"the target must not be a function of the parameters currently being updated."*

### Both stabilisers can be switched off, on purpose

```python
self.use_replay = not dcfg.no_replay
self.use_target_network = not dcfg.no_target_network
```

> *"Both are switchable off, because Phase 3's roadmap requires demonstrating what each
> is worth rather than asserting it."*

And it wasn't a formality. From the experiment log:

> **Without replay, the DQN does not learn at all** — identical results to four decimal
> places across eight seeds, the always-`BULK_CLOSE` collapse. *"Replay is not a
> refinement in this environment; it is load-bearing."*

Note also the `None` choice when the target network is off:

```python
else:
    # None rather than an unused copy, so any code path that reaches for
    # a target network under this ablation fails loudly instead of
    # silently bootstrapping off a stale network nobody updates.
    self.target = None
```

An unused copy would still be *usable*, and a stray reference would silently bootstrap
off a network frozen at initialisation. `None` makes that an `AttributeError`.

### `update()` — eight named steps

```python
# 1. count the environment step
self.env_steps += 1

# 2. store the transition, already scaled
self.buffer.push(scaled_obs, action, reward, scaled_next_obs, done)

# 3. wait for enough data, then train only every train_freq steps
if len(self.buffer) < self.dcfg.learning_starts: return
if self.env_steps % self.dcfg.train_freq != 0: return

# 4. assemble the batch
batch = self.buffer.sample(self.dcfg.batch_size)

# 5. the TD target — under no_grad
with torch.no_grad():
    best_next_value = bootstrap_net(next_obs_t).max(dim=1).values
    target = reward_t + self.gamma * (1.0 - done_t) * best_next_value

# 6. the prediction — Q(s,a) for the action actually taken
predicted = self.online(obs_t).gather(1, action_t.unsqueeze(1)).squeeze(1)
loss = F.huber_loss(predicted, target, delta=self.dcfg.huber_delta)

# 7. one optimiser step, with the gradient norm clipped
self.optimiser.zero_grad(); loss.backward()
grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.dcfg.grad_clip_norm)
self.optimiser.step()

# 8. refresh the target network on a GRADIENT-step schedule
self.gradient_steps += 1
if self.use_target_network and self.gradient_steps % self.dcfg.target_update_every == 0:
    self.target.load_state_dict(self.online.state_dict())
```

**Step 5's `no_grad`:** *"letting gradients flow into it is the bug that turns the
regression into a moving-target chase even WITH a target net."* You can have a target
network and still get this wrong.

**Step 5's `(1 - done)`:** the vectorised form of "on termination the target is the
reward alone". `done=1` zeroes the bootstrap term.

**Step 6's `gather`:** the batched form of `Q[s, a]` — picks one column per row. The
network outputs all five action values; `gather` selects the one for the action actually
taken.

**Step 8's counter:** *"counted in GRADIENT steps, not environment steps — with
train_freq=4 the two differ by a factor of four, and confusing them silently changes the
stabiliser this phase is measuring."*

### The Huber delta — the bug worth the whole section

```yaml
huber_delta: 200.0   # E-016. NOT torch's default of 1.0
```

Huber loss is quadratic for small errors and linear for large ones:

```
  loss
    |                      ╱          ← LINEAR: gradient is FLAT.
    |                    ╱               An error of 150 and an error of 1000
    |                  ╱                 produce the SAME gradient.
    |    ╲          ╱
    |      ╲___  ╱
    |         ╲╱ ← QUADRATIC: gradient scales with the error
    +----------┼-----------------  error
               δ
```

The intent is sensible: don't let one wild outlier blow up the network. **But it depends
entirely on where δ sits relative to your rewards.**

```
   δ = 1.0 (torch's default), with this project's rewards:

     error of      1  →  linear regime, gradient ≈ 1
     error of    150  →  linear regime, gradient ≈ 1     ← burying a real incident
     error of    200  →  linear regime, gradient ≈ 1     ← an end-of-shift miss
                                        ↑
              MEASURED RATIO: 1.014 for a 150× LARGER ERROR
```

Every catastrophe produced the same learning signal as a routine ±1 error. So the agent
never learned to fear them — and **20 runs collapsed to `BULK_CLOSE` 99.4% of the time,
recall 0.009.**

The experiment log's verdict on the original comment is the lesson:

> *The comment said Huber was chosen because it is "less sensitive to the large negative
> outlier rewards". That was exactly backwards: **those penalties are the signal, not
> outliers to be suppressed.***

**δ = 200.0** is chosen as *"the largest NAMED single-event penalty in env_default.yaml,
so every individual penalty stays quadratic and only the compound multi-miss tail (to
−1499) is linearised."* Not tuned — derived from the reward table.

And this is why `training_validation.py` now refuses a delta below the measured collapse
threshold. Document 2 §7: *a setting that quietly produces a wrong number needs a guard.*

### Reproducibility details

```python
torch.set_num_threads(1)
```
> *"Multi-threaded CPU kernels reorder float reductions, so two runs with the same seed
> can differ in the last bits."*

```python
torch.manual_seed(seed)
self.online = QNetwork(...)
```
> *"Seed torch before constructing the network: the weights are drawn at construction, so
> seeding afterwards would not make them reproducible."*

```python
state = torch.load(path, weights_only=True)
```
> *"a checkpoint is data, and torch.load's default unpickles arbitrary objects."*

### Feature scaling — fixed divisors, not running statistics

```python
def _scale(self, obs):
    """Not a learned or running normalisation: the divisors are fixed domain
    facts, so a state's encoding is identical in episode 1 and episode
    20000, and identical between training and evaluation."""
    return obs.astype(np.float32) / self._scales
```

Running normalisation (tracking the mean and std as you go) is common and would be wrong
here: the same queue would encode differently at different points in training, and
differently again at evaluation. Fixed domain divisors — 480 for age because that's the
shift length, 3.0 for severity because that's the top level — keep the encoding stable.

The config note explains the magnitude of the problem being solved:

> *"Measured spread across 888 observations was 470× (max_age_min reaches 473.6,
> frac_type_* stay in [0,1]); unscaled, the age columns dominate every gradient."*

---

## 11. REINFORCE — `agents/reinforce.py`

**S&B §13.3 and §13.4. Monte Carlo policy gradient.**

### The change of paradigm

> *"This is the first agent in the project that does not learn a value and then act
> greedily with respect to it. It learns the policy directly, and `act()` contains no
> argmax at all."*

```
   VALUE-BASED  (everything so far)      POLICY-GRADIENT  (this and §12)

   learn Q(s,a)                          learn π(a|s) directly
   act: argmax over Q                    act: SAMPLE from π
   explore: ε-greedy                     explore: the policy IS stochastic
```

### The update rule, factor by factor

```
    θ ← θ + α · γ^t · (G_t − b(s_t)) · ∇ ln π(a_t|s_t, θ)
              └───┘   └────────────┘   └───────────────┘
                1            2                3
```

| # | Factor | What it does |
|---|---|---|
| 3 | `∇ ln π(a_t|s_t)` | The direction that makes the action **actually taken** more likely. Not the best action — the taken one. |
| 2 | `(G_t − b(s_t))` | How much better the episode went than expected. Positive → push that way. Negative → push the other way. |
| 1 | `γ^t` | The step's own discount from the episode start. |

Factor 3 is the surprising one on first reading. The gradient doesn't know what the *best*
action was. It only knows what was done and whether things went well. **Multiply "make
this more likely" by "that went badly" and you get "make this less likely."** That's the
whole mechanism.

### Why a baseline reduces variance without adding bias

The proof, from the docstring, in one line:

```
    E[ b(s) · ∇ ln π(a|s) ] = b(s) · Σ_a ∇π(a|s) = b(s) · ∇1 = 0
```

Because `π` sums to 1 over actions, its gradient sums to 0, so subtracting any
state-dependent `b(s)` changes the expectation by exactly nothing.

What it *does* change:

> *"Without it, every action taken during a good episode is reinforced — including the
> bad ones — because G_t is large and positive for all of them. With it, only actions
> that did better than the state's own average keep pushing."*

```
   NO BASELINE, a good shift (all G_t ≈ +400):

     step 1: good action    → G=+400  → reinforce  ✓
     step 2: bad action     → G=+380  → reinforce  ✗  it was still positive!
     step 3: good action    → G=+390  → reinforce  ✓

   WITH BASELINE b(s) ≈ 385:

     step 1: G−b = +15  → reinforce  ✓
     step 2: G−b =  −5  → discourage ✓  correct
     step 3: G−b =  +5  → reinforce  ✓
```

### "This is NOT an actor–critic" — the most important paragraph in the file

REINFORCE-with-baseline has *two networks*: a policy and a value baseline. So does
actor–critic. They are still different algorithms.

```
   REINFORCE + baseline:  coefficient = G_t − b(s_t)
                                        └─┘
                                     the OBSERVED return.
                                     b(s) is subtracted from it.
                                     NOTHING BOOTSTRAPS.

   Actor–critic:          coefficient = r + γ·v(s') − v(s)
                                            └────┘
                                     the successor's ESTIMATE.
                                     THIS BOOTSTRAPS.
```

> ***"Two networks is not the criterion — bootstrapping is."***

`test_the_baseline_is_not_a_critic` fails if that ever changes. And `actor_critic.py`
carries the mirror-image test. Two tests, written as a pair, pinning a distinction that
is easy to blur.

### The `γ^t` factor — kept, though most implementations drop it

> *"S&B's boxed algorithm includes it; most published implementations drop it. Kept here,
> because the teaching constraint says the code should be the textbook's algorithm and a
> student should be able to say what the factor is for: a return earned at step t is
> worth γ^t from the episode's start, and dropping it optimises the undiscounted
> objective using discounted returns."*

And honestly quantified: *"With γ = 0.99 over ~50-step shifts it reaches ~0.6 by the end
of an episode — real but not crippling."*

### No epsilon anywhere

> *"There is no epsilon anywhere in this file, and that absence is deliberate rather than
> an omission. Value-based agents explore by sometimes ignoring their policy; a
> policy-gradient agent explores by *having* a stochastic policy and sampling from it."*

Which sets up the problem that motivates the entropy bonus in §12: **if nothing keeps π
spread out, exploration can vanish permanently.**

### `end_episode()` — five steps

```python
returns = self._returns()                                    # 1

if self.rcfg.use_baseline:                                   # 2
    baseline = self.value_net(observations).squeeze(-1).detach()
else:
    baseline = torch.zeros(len(self._episode))
advantages = returns_t - baseline

discounts = torch.tensor([self.gamma**t for t in range(...)])  # 3
coefficients = discounts * advantages

logits = self.policy_net(observations)                       # 4
log_probs = torch.log_softmax(logits, dim=-1)
chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
policy_loss = -(coefficients * chosen_log_probs).sum()
...
if self.rcfg.use_baseline:                                   # 5
    predicted = self.value_net(observations).squeeze(-1)
    value_loss = ((returns_t - predicted) ** 2).mean()
```

**Step 2's `.detach()`:** *"the value head is trained by its own loss in step 5, and
letting the policy's gradient flow into it would make one network chase two different
objectives."*

**Step 4's minus sign** — the docstring flags it as the single most dangerous character
in the file:

> *"the only thing that turns gradient ASCENT on the objective into the descent that
> torch optimisers perform. Losing it trains the agent to do the opposite of what worked,
> which produces a confidently wrong policy and no error."*

**The gradient clip is load-bearing:** *"returns in this environment reach ±500, so a
single unbaselined episode can otherwise produce a step large enough to destroy the
policy in one move."*

---

## 12. Actor–critic — `agents/actor_critic.py`

**S&B §13.5, the boxed episodic one-step algorithm.**

```
    δ = r + γ·v(s',w) − v(s,w)              (if s' is terminal, γ·v(s') = 0)
    w ← w + α_w · δ · ∇v(s,w)                the critic
    θ ← θ + α_θ · I · δ · ∇ln π(a|s,θ)       the actor
    I ← γ·I                                  reset to 1 each episode
```

**Every step, not every episode.** That's the headline difference.

### What it buys and what it costs

| | REINFORCE | Actor–critic |
|---|---|---|
| Coefficient | `G_t − b(s_t)` | `r + γv(s') − v(s)` |
| Made of | ~50 noisy rewards summed | one reward + one estimate |
| Bias | **none** | **yes** — `v(s')` is wrong early |
| Variance | **high** | **low** |
| Learns | only at episode end | **inside the episode** |

> *"REINFORCE is unbiased and noisy; this is biased and quiet."*

The online-learning point is worth separating out:

> *"this agent improves inside an episode, where REINFORCE cannot improve until the shift
> is over."*

A shift is ~50 steps. REINFORCE gets one update per shift; actor–critic gets fifty. That
is what makes the sample-efficiency comparison interesting.

### The entropy bonus — an addition to the textbook, declared as one

> *"S&B §13.5 has no entropy term. This one does, and `entropy_coef: 0.0` recovers the
> textbook algorithm exactly."*

```python
probs = torch.softmax(logits, dim=-1)
entropy = -(probs * log_probs).sum()
actor_loss = actor_loss - self.accfg.entropy_coef * entropy
```

Subtracted from the loss, so entropy is **maximised** — the policy pays a price for
collapsing onto one action.

**Why it's there** — this is the E-018 story:

> *"REINFORCE's greedy policy was degenerate — a single action in every state — by 300
> episodes, and nothing in its configuration resisted a policy sharpening early. E-019
> then ruled out the gradient clip as the cause. A policy-gradient method has no epsilon
> to hold exploration open, so the only thing that can keep π spread out is a term that
> pays for spread."*

And then the discipline:

> *"It is declared here rather than assumed, because an addition to a textbook algorithm
> that goes unmentioned is how a student ends up defending code they cannot name."*

The test `test_zero_entropy_coefficient_is_the_textbook_update` proves the claim that
`entropy_coef = 0` recovers S&B exactly.

### The `I` accumulator

The same `γ^t` idea as REINFORCE, in the form S&B's boxed algorithm states it: `I` starts
at 1, multiplies by γ each step, resets at the episode boundary. Kept for the same
reason — *"it is in the algorithm as stated, and a student should be able to say what it
is for."*

### `update()`, and the terminal-state comment worth memorising

```python
with torch.no_grad():
    value_now = self.critic_net(observation).squeeze(-1)
    bootstrap = (torch.zeros(()) if done
                 else self.gamma * self.critic_net(next_observation).squeeze(-1))
    td_error = reward + bootstrap - value_now
```

The comment:

> *"Worth knowing for a viva: bootstrapping through a TIME LIMIT rather than a true
> terminal is a real bias in many implementations. Here the shift end is a genuine
> episode end, not a truncation, so treating it as terminal is correct rather than merely
> conventional."*

That's the Q-learning `done` warning again, from the other direction — and this time with
the justification for *this* environment spelled out.

**`no_grad` on δ:** *"delta is a scalar COEFFICIENT everywhere it appears, never
something either network differentiates through."*

### The critic step, written as a squared error

```python
predicted = self.critic_net(observation).squeeze(-1)
target = predicted.detach() + td_error
critic_loss = (target - predicted) ** 2
```

The textbook writes `w ← w + α·δ·∇v(s,w)`. This writes a squared loss and lets torch
produce the gradient. The comment shows they're the same thing:

```
    d/dw [ (target − v)² / 2 ]  =  −(target − v) · ∇v  =  −δ · ∇v
```

Descending that is ascending `δ·∇v`. Identical update, one line instead of a hand-rolled
gradient.

**The actor's minus sign** gets the same warning as REINFORCE's, in the same words. It
appears twice in the codebase because it is the same trap twice.

---

## 13. The four questions you will be asked

If you understand these four answers you understand this document.

### Q1. What is the difference between SARSA and Q-learning?

**One term in the target.**

```
   SARSA:       r + γ·Q(s', a')        a' = the action it will actually take
   Q-learning:  r + γ·max_a' Q(s',a')  the best available, whatever it does
```

SARSA is **on-policy**: it learns the value of the policy it's following, exploration
included, so it accounts for the cost of its own mistakes. Q-learning is **off-policy**:
it explores with ε-greedy and learns the value of behaving greedily.

**The evidence in this project:** on the tiny MDP, Q-learning converges to `q_*` to
9.24e-14; SARSA converges to the ε-soft optimum, lower by more than 1.5 in places. That
gap is not error — it is the price of exploration, correctly accounted for.

### Q2. What is the difference between REINFORCE-with-baseline and actor–critic?

**Not the network count — both have two networks.** It is *bootstrapping*.

```
   REINFORCE:     G_t − b(s_t)          G_t is what ACTUALLY HAPPENED
   Actor–critic:  r + γv(s') − v(s)     v(s') is a GUESS
```

REINFORCE is unbiased and high-variance and can only learn at episode end. Actor–critic
is biased and low-variance and learns every step.

Two mirror-image tests pin it: `test_the_baseline_is_not_a_critic` and the actor–critic
test that asserts it does bootstrap.

### Q3. Why does DQN need a target network?

Because without one the regression target is computed from the network you are updating.
You move the weights, the target moves, you chase it forever.

> **The target must not be a function of the parameters currently being updated.**

And in this project the *replay* ablation turned out to matter even more: without replay
the DQN produced identical results to four decimal places across eight seeds — the
always-`BULK_CLOSE` collapse. **Replay is load-bearing here, not a refinement.**

### Q4. Why is Monte Carlo high-variance and TD low-variance?

MC uses the actual return — the sum of ~50 noisy rewards. One unlucky shift moves the
estimate for every state visited during it.

TD uses `r + γ·V(s')` — one reward plus one estimate. Much less noise per update, but the
estimate is wrong early, so it's biased.

```
   MC:  unbiased,  noisy    ← trusts reality, and reality is noisy
   TD:  biased,    quiet    ← trusts its own guess, and its guess is wrong at first
```

This is the central comparison of S&B chapters 5 and 6, and it is why the report runs all
three tabular methods rather than picking one.

---

## 14. Every hyperparameter, and where it lives

Nothing is hardcoded. `TabularAgent.__init__` takes every hyperparameter with **no
defaults**, and the reason is stated:

> *"a constructor default is a magic number in disguise."*

From `config/training_default.yaml`:

| Setting | Value | The reason given in the file |
|---|---|---|
| `common.gamma` | 0.99 | *"patient: end-of-shift misses are charged at the very end, so the agent must value distant consequences"* |
| `common.n_episodes` | 20,000 | the shared budget for every learner |
| `epsilon.start` | 1.0 | fully random at episode 1 |
| `epsilon.min` | 0.05 | *"queue composition shifts during a shift, so a fully greedy agent stops adapting"* |
| `epsilon.decay` | 0.9995 | **per episode** (D-015) |
| `q_learning.alpha` | 0.10 | |
| `sarsa.alpha` | 0.10 | |
| `monte_carlo.alpha` | 0.05 | lower — MC's updates are noisier |
| `monte_carlo.first_visit` | true | *"every-visit MC is not implemented; the loader rejects false"* |
| `dp.n_estimation_episodes` | 50,000 | random rollouts for P̂ / R̂ |
| `dp.max_sweeps` | 5,000 | *"a hard stop so a non-converging run fails loudly, not forever"* |
| `dqn.hidden_layers` | [128, 128] | |
| `dqn.lr` | 0.0005 | Adam |
| `dqn.replay_capacity` | 100,000 | *"~2000 shifts of history at ~50 steps/shift"* |
| `dqn.train_freq` | 4 | keeps the episode budget comparable to tabular: 4.6 min vs 18.3 |
| `dqn.target_update_every` | 1,000 | **gradient** steps |
| `dqn.huber_delta` | **200.0** | E-016 — the largest named single-event penalty |

### The seed blocks — D-016

Every learner gets its own disjoint block of episode seeds:

| Purpose | Seed block |
|---|---|
| train (evaluation of baselines) | 1–10 |
| **eval** | **101–130** |
| calibration | 1,000–3,099 |
| DP model estimation | 10,000–59,999 |
| q_learning training | 200,000+ |
| sarsa training | 400,000+ |
| monte_carlo training | 600,000+ |
| dqn training | 1,000,000+ |
| dqn ablations | 1,200,000+ |
| RLHF pair generation | 3,000,000+ |

And the reason each learner gets a *fresh shift per episode* rather than cycling the ten
train seeds:

> *"cycling the 10 train seeds 2000× would let the agent memorise ten alert streams
> instead of learning to triage."*

That's the difference between learning a policy and learning a lookup table for ten
specific nights.

---

## Where to go next

- **Document 4 — Running and Measuring.** How these agents get run in bulk, the tiny MDP
  that proves the implementations are correct, and how episodes become numbers you can
  defend.
- **Document 5 — RLHF and Labelling.** What replaces the hand-written reward.
- **Document 6 — Decisions, Experiments, Results.** Every D-number cited here, in full,
  with the experiment that produced it.
- **Document 1 §11** for the honest results — including the DQN losing to tabular, and
  the `reinforce@1` policy that turned out to be the best in the pool.

---

*Source of truth: this Markdown file. The `.docx` export is generated from it. If they
disagree, the Markdown is right.*
