# Document 2 — The World

### The simulated SOC, file by file

> **Read this after Document 1.** Document 1 told you *what* the project is. This one
> shows you the code that builds the world the agent lives in: where alerts come from,
> what the agent is allowed to see, how a shift plays out minute by minute, and where
> every number is written down.
>
> **Read it with the files open beside you.** Every section names the file and the line
> range it is describing.

---

## Contents

| § | What it covers | File |
|---|---|---|
| 1 | The five files, and how they fit together | *(map)* |
| 2 | What an alert is | `src/soc_triage/alerts.py` |
| 3 | Where alerts come from | `src/soc_triage/generator.py` |
| 4 | What the agent is allowed to see | `src/soc_triage/state.py` |
| 5 | The shift itself — the MDP | `src/soc_triage/env.py` |
| 6 | Where every number lives | `config/env_default.yaml` |
| 7 | The code that reads the numbers | `src/soc_triage/config/*.py` |
| 8 | The ground-truth firewall | *(cross-cutting)* |
| 9 | Things that will confuse you | *(traps)* |

---

## 1. The five files, and how they fit together

Everything in this document is one of five ideas. Here they are as a picture, in the
order the data actually moves:

```
                    config/env_default.yaml
                    (every number, no exceptions)
                              |
                              | read once at startup
                              v
                    soc_triage/config/*.py
                    (turns YAML into typed, checked objects)
                              |
                              | EnvConfig
                              v
    +---------------------------------------------------------------+
    |                                                               |
    v                                                               v
generator.py                                                     env.py
"here are all 170 alerts                                    "the shift clock,
 for this shift, decided                                     the queue, the
 up front from one seed"                                     reward arithmetic"
    |                                                               |
    | list[Alert]                                                   |
    v                                                               |
alerts.py                                                           |
"what one alert IS"  <----------------------------------------------+
                                       env holds the alerts and hands out
                                       a censored view of the queue
                                                                    |
                                                            EnvSnapshot
                                                                    |
                                                                    v
                                                              state.py
                                                    "translate the snapshot into
                                                     something an agent can eat"
                                                          /              \
                                                         /                \
                                              discretise()            featurise()
                                             one int, 0..575        17 floats
                                          (for Q-tables)          (for networks)
                                                         \                /
                                                          v              v
                                                            the agent
                                                        (Document 3)
```

**The one-sentence version of each file:**

| File | Lines | One sentence |
|---|---|---|
| `alerts.py` | 33 | An `Alert` is 8 frozen fields; two of them are secrets. |
| `generator.py` | 97 | Given a seed, produce the whole shift's alerts up front. |
| `state.py` | 180 | Turn "what's happening right now" into a number, or into 17 numbers. |
| `env.py` | 278 | Run the clock, serve the queue, pay the rewards. |
| `config/` | 4 files, 852 | Read the YAML, and refuse to load it if it's wrong. |

---

## 2. What an alert is — `alerts.py`

This is the smallest file in the project and one of the most important, because
**every other module agrees on this shape.** Change a field here and the generator,
the environment, both state encoders, and all of evaluation change with it.

### The eight fields

```python
@dataclass(frozen=True)
class Alert:
    id: int                      # unique per shift
    arrival_time: float          # minutes into the shift
    severity: int                # 0..3 — the vendor's guess
    asset_criticality: int       # 0=dev box, 1=standard, 2=crown jewel
    verify_cost_min: int         # analyst minutes: 5 | 10 | 20 | 40
    alert_type: str              # one of six names
    is_true_incident: bool       # *** HIDDEN FROM THE AGENT ***
    deadline_min: float          # *** HIDDEN FROM THE AGENT ***
```

Read as a table, with who is allowed to look:

| Field | Example | Agent sees it? | What it means in plain English |
|---|---|:---:|---|
| `id` | `47` | yes | The 48th alert of the shift. Also the tie-breaker everywhere. |
| `arrival_time` | `133.8` | yes | It landed 2h14m into the shift. |
| `severity` | `2` | yes | The tool said "medium". The tool is often wrong. |
| `asset_criticality` | `2` | yes | It's on a crown-jewel machine. |
| `verify_cost_min` | `20` | yes | Checking it will burn 20 analyst-minutes. |
| `alert_type` | `data_exfil_volume` | yes | The category of thing that fired. |
| `is_true_incident` | `True` | **NO** | Whether this is a real attack. |
| `deadline_min` | `310.4` | **NO** | How long it may sit before it counts as a breach. |

### Two design choices worth understanding

**`frozen=True` — an alert can never be edited.**

```python
@dataclass(frozen=True)
```

Python will raise an error if anything tries `alert.severity = 3`. The reasoning is in
the docstring and it is a good one:

> *"An alert is a fact about the world; nothing may rewrite it after generation. The
> queue changes, alerts do not."*

The *queue* is a list that shrinks as alerts are handled. The *alerts* themselves are
immutable facts. Keeping that distinction in the type system means a whole category of
bug — "something quietly modified an alert halfway through the episode" — cannot happen.

**The two hidden fields are in the object anyway.**

You might reasonably ask: if the agent can't see `is_true_incident`, why is it stored on
the alert at all? Why not keep a separate secret list?

Because *someone* has to know. The environment needs it to compute the reward when the
alert is investigated, and to work out at the end of the shift which real incidents were
missed. Putting it on the alert keeps it next to the thing it describes.

The cost of that convenience is that the secret is one attribute access away from
leaking. The project pays that cost deliberately and then defends it in three places —
see §8.

---

## 3. Where alerts come from — `generator.py`

### The job

> **Input:** a config, and one integer seed.
> **Output:** the complete list of every alert that will arrive during this 480-minute
> shift — all ~170 of them — decided before the shift starts.

### Why generate everything up front?

This is the single most important design decision in the file, and it exists to serve
the *comparison* work rather than the simulation itself.

```
   WHAT WE DO                              WHAT WE COULD HAVE DONE
   generate all alerts at reset            generate each alert as time passes

   seed 4201 --> [the same 170             seed 4201 --> alerts depend on when
                  alerts, always]                        the agent happened to act

   Policy A and Policy B both              Policy A and B see DIFFERENT alerts,
   face an identical shift.                because they act at different times.
   Any difference in outcome is            Any difference in outcome is
   the POLICY.                             policy + luck, tangled together.
```

That second world would make the whole RLHF phase impossible. Document 1 §6 explains
why a labeller comparing two shifts must be looking at the *same shift handled two
ways* — otherwise "left was better" might just mean "left got an easier night".

The docstring says it in one line:

> *"generating everything ahead of time keeps the stream a pure function of (config,
> seed), which is what makes paired policy comparisons possible."*

### How arrivals are timed

Alerts arrive as a **Poisson process** — the standard model for "independent events
happening at a steady average rate". You do not need the theory; you need one fact:

> In a Poisson process, the *gaps between consecutive events* follow an exponential
> distribution with mean `1/λ`.

So instead of asking "did an alert arrive this minute?" 480 times, the code draws the
gaps directly and adds them up:

```python
mean_gap = 1.0 / cfg.arrivals.rate_per_min          # 1 / 0.35 = 2.857 minutes
gaps = rng.exponential(mean_gap, size=chunk)        # [2.1, 0.4, 5.8, 3.0, ...]
arrivals_all = np.cumsum(gaps)                      # [2.1, 2.5, 8.3, 11.3, ...]
arrival_times = arrivals_all[arrivals_all < cfg.shift.length_min]
```

Visually:

```
  gaps:      2.1      0.4        5.8         3.0      1.2
         |--------|-------|-------------|----------|------| ...
  time:  0       2.1     2.5          8.3       11.3   12.5      ... 480
         ^        ^       ^             ^          ^      ^
      shift    alert 0  alert 1      alert 2   alert 3  alert 4
      starts

  Keep everything below 480. That's the shift.
```

The `while gaps.sum() < cfg.shift.length_min` loop handles the astronomically unlikely
case where the first chunk of random gaps doesn't reach the end of the shift. It draws
more. In practice it never runs — the chunk size is 50% larger than the expected need
plus 50 — but "never runs" and "cannot happen" are different, and the loop makes it the
second one.

**λ = 0.35 per minute × 480 minutes ≈ 168 alerts.** The measured average is 168.7.

### How each alert's features are drawn

Four independent draws, one per feature, each from a probability distribution written in
the YAML:

| Feature | Options | Probabilities | Meaning |
|---|---|---|---|
| `severity` | 0, 1, 2, 3 | 45%, 30%, 18%, 7% | Most alerts are low-severity noise |
| `asset_criticality` | 0, 1, 2 | 55%, 35%, 10% | Crown jewels are rare |
| `verify_cost_min` | 5, 10, 20, 40 | 35%, 35%, 20%, 10% | Most checks are quick |
| `alert_type` | six names | 25/20/10/10/20/15% | See table below |

### How truth is decided — the heart of the file

This is the mechanism that makes the whole project non-trivial. **Whether an alert is
real is not random-with-a-fixed-chance. It depends on the alert's features:**

```
P(true incident) = base_rate  ×  type_lift  ×  severity_lift[sev]  ×  asset_lift[crit]
```

In code:

```python
p_true = (
    cfg.incident.base_rate            # 0.0135
    * type_lift_arr                   # 0.8 .. 3.0 depending on alert_type
    * severity_lift[severities]       # [0.1, 0.35, 2.4, 15.0]
    * asset_lift[criticalities]       # [0.8, 1.2, 1.4]
)
is_true = rng.random(n) < np.minimum(p_true, _P_TRUE_CAP)
```

The lifts, from `config/env_default.yaml`:

| Alert type | Prior | Incident lift | Reading |
|---|---:|---:|---|
| `failed_login_burst` | 25% | ×0.8 | Common, usually noise |
| `malware_signature` | 20% | ×1.6 | Moderately suspicious |
| `data_exfil_volume` | 10% | **×3.0** | Rare and dangerous |
| `priv_escalation` | 10% | **×2.8** | Rare and dangerous |
| `phishing_click` | 20% | ×1.1 | Common, mildly suspicious |
| `unusual_process` | 15% | ×0.9 | Common, usually noise |

| Severity | Lift | | Asset | Lift |
|---|---:|---|---|---:|
| 0 (info) | ×0.10 | | 0 (dev box) | ×0.8 |
| 1 (low) | ×0.35 | | 1 (standard) | ×1.2 |
| 2 (medium) | ×2.4 | | 2 (crown jewel) | ×1.4 |
| 3 (high) | ×15.0 | | | |

**Worked example.** A `data_exfil_volume` alert, severity 2, on a crown jewel:

```
0.0135  ×  3.0  ×  2.4  ×  1.4  =  0.136   →  a 13.6% chance of being real
```

Against a `failed_login_burst`, severity 0, on a dev box:

```
0.0135  ×  0.8  ×  0.1  ×  0.8  =  0.00086 →  a 0.09% chance
```

That's a **158× difference** — and the agent can see every input to it.

### The cap, and why it exists

```python
_P_TRUE_CAP = 0.95
```

A severity-3 `data_exfil_volume` on a crown jewel computes to `0.0135 × 3.0 × 15.0 × 1.4
= 0.85`, which is under the cap. But the cap is there so that no combination of lifts,
now or after any future retuning, can ever produce a *certain* incident. The comment
says it plainly:

> *"even the worst-looking alert can be a false alarm."*

That's a statement about the real world. Analysts don't get certainty, so neither does
the agent.

### The severity correlation — the project's defining assumption

Here's the number this whole design exists to produce:

> **Pearson r(severity, is_true_incident) ≈ 0.32**

Which means: **severity is informative but weak.** If you sort by severity alone you do
better than random, and much worse than you could. The real signal lives in the
*combination* of type, asset criticality, and cost — exactly the combination a learned
policy can pick up and a single sort key cannot.

The YAML says it out loud, right where a future tuner will see it:

```yaml
# Note: the informative signal lives in alert_type + asset_criticality + verify_cost,
# NOT mainly in severity. That is the whole reason a learned policy can beat severity-sort.
```

**If someone "improves" the severity lifts until r = 0.9, the project stops being
interesting** — severity-sort would become near-optimal and there'd be nothing to learn.
That's why `target_severity_corr: [0.30, 0.40]` is a config value with a validation
check, not a comment.

### Deadlines

```python
deadlines = np.where(is_true, rng.uniform(120, 480, size=n), 0.0)
```

Real incidents get a random dwell budget between 120 and 480 minutes. **False positives
get `0.0`, which is meaningless and documented as meaningless** — nothing ever reads a
false positive's deadline. Storing `0.0` rather than `None` keeps the field a plain float
so the array stays numeric.

### One thing that was optimised, and why that's allowed

```python
# ... vectorised because profiling showed per-alert draws were 82% of episode
# runtime (CONSTRAINTS #14: optimise the simulator, not the algorithms).
```

The project's teaching rule says *prefer the clear implementation over the clever one*.
This file breaks that rule on purpose, and the constraint it cites explains the split:

> **Vectorise the simulator if needed; keep the algorithms readable.**

Nobody will be asked in a viva to write a Poisson generator from memory. Everybody will
be asked to write Q-learning from memory. So the simulator gets to be fast and the
learning code stays slow and legible. The final loop that builds the `Alert` objects is
still an ordinary `for` loop, because that part is about *shape*, not speed.

---

## 4. What the agent is allowed to see — `state.py`

### The problem this file solves

The environment knows everything: every alert in the queue, the exact clock, the hidden
truth. An agent cannot be handed that. Two reasons:

1. **Secrecy.** It must not see the hidden fields (that's the whole project).
2. **Shape.** A Q-table needs a single integer index. A neural network needs a
   fixed-length vector of floats. Neither can consume "a list of 43 alert objects".

So this file offers two translations of the same situation:

```
                        EnvSnapshot
              (the censored view of right now)
                    /              \
                   /                \
          discretise()            featurise()
                 |                      |
                 v                      v
        one integer, 0..575      17 floats
                 |                      |
                 v                      v
        Q-tables, DP,            DQN, REINFORCE,
        SARSA, Monte Carlo       actor-critic
```

### The `EnvSnapshot` — the censored view

```python
@dataclass(frozen=True)
class EnvSnapshot:
    queue: tuple[Alert, ...]        # alerts waiting right now
    time_now: float                 # minutes into the shift
    shift_length: float             # 480
    alerts_handled: int             # investigations completed
    incidents_confirmed: int        # of those, how many were real
```

Note the last field, and the comment beside it:

> `incidents_confirmed: int  # known AFTER investigating — not leakage`

This is worth pausing on because it looks like a violation and isn't. The agent is
allowed to know how many of the alerts *it already investigated* turned out to be real —
because in the real world, an analyst who investigates an alert finds out the answer.
That's the reward signal arriving, not a peek at the future. What it must never know is
the truth about an alert **still sitting in the queue**.

Also note `queue` is a `tuple`, not a `list`, and the dataclass is frozen. Together those
mean an agent cannot reach into the snapshot and modify the environment's queue through
it. The docstring is explicit:

> *"Frozen so encoders (or agents) cannot mutate the environment through it."*

### `bucket()` — the shared helper

```python
def bucket(value: float, boundaries: tuple[float, ...]) -> int:
    index = 0
    for boundary in boundaries:
        if value >= boundary:
            index += 1
    return index
```

Four lines, and the docstring explains why it's a shared function rather than inline
comparisons:

> *"One shared helper so every discretisation uses the identical convention (HANDOVER
> warned: an off-by-one here silently corrupts all 576 states)."*

**N boundaries produce N+1 buckets.** With `[10, 40, 100]`:

```
   value:  0 ....... 10 ....... 40 ....... 100 ....... ∞
  bucket:      0     |     1    |     2    |     3
              <10      10-39      40-99       100+
```

The convention is *"greater than or equal goes up"*. Exactly 10 lands in bucket 1, not 0.
Whether that's the right convention barely matters — what matters is that all three
bucketed features use the same one, which is guaranteed by there being only one
implementation.

### `discretise()` — one integer for the whole situation

Five features, each reduced to a small bucket index:

| Feature | Buckets | Boundaries | Meaning of the top bucket |
|---|:---:|---|---|
| `max_severity` in queue | 4 | *(the value itself)* | Something severity-3 is waiting |
| `queue_len` | 4 | 10, 40, 100 | 100+ alerts backed up |
| `oldest_age` | 4 | 30, 90, 180 min | Something has waited 3+ hours |
| `time_left` | 3 | 60, 240 min | Plenty of shift remaining |
| `max_asset_criticality` | 3 | *(the value itself)* | A crown jewel is in the queue |

**4 × 4 × 4 × 3 × 3 = 576 states.**

The packing is mixed-radix — the docstring's analogy is exactly right:

> *"like reading a 5-digit number where each digit has its own base"*

```python
state_id = max_severity
state_id = state_id * 4 + queue_len_bucket
state_id = state_id * 4 + age_bucket
state_id = state_id * 3 + time_left_bucket
state_id = state_id * 3 + max_criticality
```

Compare with how you read the decimal number 4,271:

```
   4271  =  ((( 4 ×10 + 2) ×10 + 7) ×10 + 1

   state =  ((((sev ×4 + qlen) ×4 + age) ×3 + tleft) ×3 + crit
                     ^          ^         ^          ^
                base 4      base 4    base 3     base 3
              (because     (because  (because   (because
               qlen has     age has   tleft has  crit has
               4 buckets)   4)        3)         3)
```

**The multiplier at each step is the number of buckets in the feature being added
next.** That's the whole rule. It guarantees every combination maps to a unique integer
and every integer 0–575 maps back to exactly one combination.

**Worked example.** Mid-shift, things are getting bad:

| Feature | Raw value | Bucket | Why |
|---|---|:---:|---|
| max severity | 3 | **3** | a high-severity alert is waiting |
| queue length | 52 alerts | **2** | 52 is in [40, 100) |
| oldest age | 95 min | **2** | 95 is in [90, 180) |
| time left | 200 min | **1** | 200 is in [60, 240) |
| max criticality | 2 | **2** | a crown jewel is waiting |

```
  3
  3 × 4 + 2 = 14
 14 × 4 + 2 = 58
 58 × 3 + 1 = 175
175 × 3 + 2 = 527      →  state 527
```

Every time the shift looks like that, the agent is in state 527 and looks up the same
five Q-values.

### The empty-queue convention

```python
else:
    # Empty-queue convention: all queue-derived features take their lowest bucket.
    max_severity = 0
    max_criticality = 0
    oldest_age = 0.0
```

An empty queue looks identical to a queue holding one severity-0 alert on a dev box that
just arrived. That's a genuine loss of information, accepted because the alternative —
a 577th "empty" state — would complicate the packing for a situation that is both rare
and low-stakes. It is written down here rather than left for someone to discover.

### `featurise()` — 17 floats

Same situation, much more detail. The columns, in order:

| # | Name | What it is |
|---:|---|---|
| 0 | `queue_len` | how many alerts are waiting |
| 1 | `mean_severity` | average severity in the queue |
| 2 | `max_severity` | highest severity in the queue |
| 3 | `mean_age_min` | average wait so far |
| 4 | `max_age_min` | longest wait so far |
| 5 | `mean_asset_criticality` | average asset tier |
| 6 | `max_asset_criticality` | highest asset tier |
| 7 | `mean_verify_cost_min` | average cost to check |
| 8–13 | `frac_type_0` … `frac_type_5` | **fraction** of the queue that is each alert type |
| 14 | `time_left_norm` | shift remaining, scaled to 0–1 |
| 15 | `alerts_handled` | investigations done |
| 16 | `incidents_confirmed` | of those, how many were real |

**What the vector has that the integer doesn't:**

```
  discretise() sees:                    featurise() sees:

  "something severity-3                 "the queue is 43 alerts, 12% of them
   is in the queue"                      data_exfil_volume, mean age 51 minutes,
                                         mean verify cost 14 minutes, and the
                                         worst thing in it is severity 3"
```

The discretised state throws away *composition*. It cannot distinguish "one nasty alert
buried in noise" from "a queue full of nasty alerts". The 17-float vector can — columns
8–13 are exactly that missing composition, expressed as fractions so the numbers don't
grow with queue length.

**This is the honest motivation for DQN in this project** — not "neural networks are
better", but "there is specific information the tabular state cannot represent, and here
it is". Document 1 §11 records how that turned out. (It did not beat tabular. That is a
result, not a failure.)

### `feature_scale_vector()` — the guard nobody would have thought to write

Neural networks want inputs on comparable scales. `queue_len` runs 0–100+; `time_left_norm`
runs 0–1. So the config carries a divisor per column, and this function orders them to
match `FEATURE_NAMES`.

The important part is what it refuses:

```python
missing = [name for name in FEATURE_NAMES if name not in provided]
unknown = [name for name in provided if name not in FEATURE_NAMES]
if missing or unknown:
    raise ValueError(...)
```

Both directions raise. The docstring explains why that matters more than it looks:

> *"Silently accepting a partial mapping would leave one column unscaled — the network
> would still train and the bug would surface only as a worse result."*

**This is the shape of the most dangerous class of bug in the whole project.** Nothing
crashes. Nothing is obviously wrong. You get a slightly worse number, you write it in
the results table, and you conclude something false about DQN. The check turns a silent
wrong answer into a loud error at startup.

There's also a small architectural note buried in the docstring — the check lives here
rather than in `config.py` because `config.py` cannot import `state.py` without a cycle
(`state.py` imports `EnvConfig`). Worth noticing as an example of *why* a piece of code
sometimes sits somewhere slightly surprising.

---

## 5. The shift itself — `env.py`

This is the MDP. It owns the clock, the queue, the hidden truth, and the reward
arithmetic. It has no idea which agent is talking to it — that's CONSTRAINTS #10, and
it's what lets the same environment serve a hand-written baseline and a neural network
without either knowing about the other.

### The interface

```python
env = SOCTriageEnv(cfg)
snap = env.reset(seed)                          # start a shift
snap, reward, done, info = env.step(action)     # take one action
outcome = env.episode_outcome()                 # after done: the truth
```

This is the Gymnasium convention — **the interface only, with no gymnasium import**.
Anyone who has used a standard RL library will recognise the shape immediately, and the
project takes on no dependency for it.

### The five actions

```python
PULL_HIGHEST_SEVERITY    = 0
PULL_OLDEST              = 1
PULL_MOST_CRITICAL_ASSET = 2
PULL_CHEAPEST            = 3
BULK_CLOSE_LOW_RISK      = 4
```

And immediately in `__init__`:

```python
if cfg.actions.names != expected:
    raise ValueError(f"action names/order mismatch with env constants: {cfg.actions.names}")
```

The action *order* is load-bearing across the entire codebase — Q-table columns, network
output heads, every results table. Somebody reordering the YAML list to be tidy would
silently remap every learned policy. This check makes that a startup crash.

### `reset(seed)` — beginning a shift

```python
self._pending  = generate_shift(self.cfg, seed)   # all ~170 alerts, sorted by arrival
self._queue    = []                               # nothing has arrived yet
self._clock    = 0.0
...
self._admit_arrivals()
```

Two lists, and the distinction between them is the core of the simulation:

```
   _pending                            _queue
   "generated, but the clock           "arrived — the agent can
    hasn't reached them yet"            see and act on these"

   [alert 12 @ 34.2 min ]              [alert 0 @ 2.1 min]
   [alert 13 @ 35.9 min ]   ------->   [alert 1 @ 2.5 min]
   [alert 14 @ 41.0 min ]  _admit_     [alert 2 @ 8.3 min]
   [...                  ]  arrivals   [...              ]

   The future.                         The present.
   Invisible to the agent.             Visible (minus the secrets).
```

`_admit_arrivals()` walks the front of `_pending` and moves anything whose
`arrival_time` has passed into `_queue`. Because `_pending` is sorted by arrival time,
it can stop at the first alert that hasn't arrived yet:

```python
while self._pending and self._pending[0].arrival_time <= self._clock:
    self._queue.append(self._pending.pop(0))
```

### `step(action)` — the ordering trap

The step order is documented at the top of the file because getting it wrong is subtle
and silent:

```
  1. resolve the action on the CURRENT queue
  2. advance the clock by the action's time cost
  3. admit alerts whose arrival_time has now passed
  4. build the next observation from the fresh queue
```

> *"Getting 3 and 4 backwards would show the agent a stale queue."*

Picture it. The agent investigates a 40-minute alert:

```
  t=100  agent picks an alert. Queue has 12 alerts.
         |
         | 1. resolve: remove that alert, compute reward
         |
  t=140  2. clock advances 40 minutes
         |
         | 3. admit: 14 alerts arrived during those 40 minutes
         |
         v
  t=140  4. observe: queue now has 11 + 14 = 25 alerts   ← correct

  If 3 and 4 were swapped, the agent would be told the queue has 11 alerts
  at t=140 — a view of the world 40 minutes out of date, and it would keep
  making decisions on it forever.
```

The whole `step` body, annotated:

```python
if not self._queue:
    # Empty queue: any action just waits.
    time_cost = self.cfg.shift.empty_queue_wait_min   # 5 minutes
    reward = 0.0
elif action == BULK_CLOSE_LOW_RISK:
    reward, time_cost = self._do_bulk_close(info, breakdown)
else:
    reward, time_cost = self._do_investigate(action, info, breakdown)

self._clock += time_cost          # step 2
self._admit_arrivals()            # step 3

if self._clock >= self.cfg.shift.length_min:
    self._done = True
    reward += self._end_of_shift_penalty(breakdown)

return self._snapshot(), reward, self._done, info    # step 4
```

**The empty-queue branch.** If nothing is waiting, every action does the same thing:
wait 5 minutes and earn nothing. There's no "wait" action in the action set because
there doesn't need to be — with an empty queue the choice is meaningless anyway. It is
the environment, not the agent, that handles the case.

### `_select_alert()` — the four pull rules

```python
best = self._queue[0]
for alert in self._queue[1:]:
    if action == PULL_HIGHEST_SEVERITY:
        better = alert.severity > best.severity
    elif action == PULL_OLDEST:
        better = alert.arrival_time < best.arrival_time
    elif action == PULL_MOST_CRITICAL_ASSET:
        better = alert.asset_criticality > best.asset_criticality
    else:  # PULL_CHEAPEST
        better = alert.verify_cost_min < best.verify_cost_min
    if better:
        best = alert
return best
```

An explicit linear scan, and the comment on the last line is the point:

```python
if better:  # note: equal keeps `best` (earlier id wins — queue is id-ordered)
```

**Strict `>` and strict `<`.** A tie leaves `best` alone, and since the queue is in
arrival order (which is id order), the earlier alert wins. That's not an accident of
implementation — it's what makes a run reproducible. With `>=`, ties would go to
whichever alert happened to be scanned last, and the same seed could produce different
episodes if anything about the queue ordering ever changed.

**Note what these four rules are not.** The agent does not choose *an alert*. It chooses
*a rule*, and the environment applies it. That's the design decision from Document 1 §4:
the action space stays at 5 regardless of queue size, and the resulting policy is
readable as English ("when the queue is long and old, pull oldest").

### `_do_investigate()` — where reward is computed

```python
alert = self._select_alert(action)
self._queue.remove(alert)

start_time = self._clock  # delay measured at investigation START (D-009)
self._investigated.append((alert, start_time))
self._alerts_handled += 1

if alert.is_true_incident:
    self._incidents_confirmed += 1
    delay = start_time - alert.arrival_time
    multiplier = self.cfg.asset_criticality.reward_multiplier[alert.asset_criticality]
    reward = (
        self.cfg.reward.true_incident_base                       # 100.0
        * math.exp(-delay / self.cfg.reward.true_incident_decay_min)   # decay over 120 min
        * multiplier                                             # 1.0 / 1.5 / 2.5
    )
else:
    self._wasted_minutes += alert.verify_cost_min
    reward = self.cfg.reward.false_positive_per_min * alert.verify_cost_min   # -1 × cost
```

**The decay curve** — this is the shape that makes speed matter:

```
 reward
  100 |*
      | *
   75 |   *
      |     *
   50 |        *                +100 × exp(-delay / 120)
      |            *
   25 |                 *
      |                        *  *
    0 |                                *  *  *  *  *
      +---+---+---+---+---+---+---+---+---+---+---+---
      0   60  120 180 240 300 360 420 480   delay (minutes)

  caught instantly ....... +100
  caught after  60 min ...  +61
  caught after 120 min ...  +37
  caught after 240 min ...  +14
  caught after 480 min ...   +2
```

Multiply by the asset multiplier (1.0 / 1.5 / 2.5) and you get the full picture: a
crown-jewel incident caught immediately is worth +250; the same incident caught four
hours late is worth +34.

**Why "delay measured at investigation start" is a decision (D-009).** You could argue
for measuring at the moment the investigation *finishes* — that's when the analyst
actually knows. The project measures at the start, because that's when the analyst
*commits* to it, and because the finish time is just start + verify_cost, which the
reward already penalises through the cost of the alert. Whichever you pick, it must be
written down: two different choices produce different numbers, and someone comparing
your results to a paper needs to know which one you used.

### `_do_bulk_close()` — the deliberate trap

```python
rules = self.cfg.actions.bulk_close
eligible = []
for alert in self._queue:  # queue is id-ordered (arrival order)
    if (alert.severity <= rules.max_severity              # <= 1
            and alert.asset_criticality <= rules.max_asset_criticality):   # == 0
        eligible.append(alert)
        if len(eligible) == rules.max_alerts:             # cap at 10
            break
```

Then each one is paid out:

```python
for alert in eligible:
    self._queue.remove(alert)
    self._bulk_closed.append(alert)
    if alert.is_true_incident:
        buried_penalty += self.cfg.reward.bulk_close_true_incident * multiplier   # -150 × mult
    else:
        fp_credit += self.cfg.reward.bulk_close_fp                                # +0.5
```

**The arithmetic of the trap:**

```
   Close 10 low-risk alerts, all false positives:
        10 × (+0.5)  =  +5.0        and it cost only 2 minutes

   Close 10, one of which was real (on a standard asset):
        9 × (+0.5)  +  (-150 × 1.5)  =  +4.5 - 225  =  -220.5
```

A **44-fold** difference. And the alerts it's allowed to close are *precisely* the ones
least likely to be real — severity ≤ 1 and criticality 0 gives a lift of about
`0.35 × 0.8 = 0.28`, so roughly a 0.4% incident rate. Bulk-closing is nearly always
right and occasionally catastrophic.

This is in the MDP on purpose. It's the situation where a policy that optimises the
average happily accepts a rare disaster — and it's exactly the kind of judgement that a
hand-written reward function struggles to express and a human labeller has an instant
opinion about. Document 1 §5 makes the argument; this is the code that creates the
opportunity.

### `_end_of_shift_penalty()` — the bill for what you never looked at

```python
untriaged = self._queue + self._pending  # pending: arrived too late to ever be seen
for alert in untriaged:
    if alert.is_true_incident:
        deadline_at = alert.arrival_time + alert.deadline_min
        if deadline_at <= self.cfg.shift.length_min:
            penalty += self.cfg.reward.end_of_shift_missed * multiplier   # -200 × mult
```

Three separate decisions are packed into those six lines:

**1. `_queue + _pending` — both count.** An alert you never got to and an alert that
arrived at minute 478 are both untriaged. The `_pending` half might feel unfair, but a
real incident arriving in the last two minutes of a shift *is* a real miss.

**2. The deadline test.** An incident whose dwell budget stretches past minute 480 is
**not** charged. The comment: *"the next shift's problem — no charge."* You are running
one shift, not the whole SOC. Penalising you for something the night shift can still
catch would be wrong.

**3. Bulk-closed incidents are not charged twice.** They were already charged −150×mult
at closure. This is D-009 and it's the kind of thing that is very easy to get wrong —
double-charging would make bulk-close look far worse than the reward table says it is,
and every result about it would be quietly untrustworthy.

### `episode_outcome()` — the honest scoreboard

Called after the episode ends. This is the *only* place the environment is allowed to
tell the truth about everything, because it's evaluation output, never an observation:

```python
if not self._done:
    raise RuntimeError("episode_outcome() only valid after the episode ends")
```

What it returns:

| Key | Meaning |
|---|---|
| `incidents_total` | how many real incidents the shift contained |
| `incidents_caught` | how many were investigated |
| `incidents_caught_in_time` | of those, how many before their deadline |
| `incidents_missed` | never investigated + buried |
| `incidents_buried_by_bulk_close` | the subset that were bulk-closed |
| `critical_missed` | missed incidents on crown jewels |
| `missed_by_criticality` | the miss count split by asset tier |
| `mttd_min` | mean time to detect, or `None` if nothing was caught |
| `wasted_minutes` | analyst time spent on false positives |

Two details worth catching:

**`mttd_min` is `None`, not `0.0`, when nothing was caught.** `0.0` would mean "detected
instantly" — the best possible score — for a shift that caught nothing at all. That is
the sort of thing that produces a beautiful, meaningless graph. `None` forces every
consumer to handle the case.

**`incidents_missed` includes buried ones.** `len(missed) + len(buried)`. A real incident
that was bulk-closed unread is missed. Reporting it any other way would flatter the
bulk-close action.

---

## 6. Where every number lives — `config/env_default.yaml`

> **CONSTRAINTS #9: No magic numbers in code. Every tunable number lives in
> `config/*.yaml`.**

The reason isn't tidiness. It's that a number in a YAML file **can be defended**. When
an examiner asks "why 0.35 arrivals per minute?", the answer is a line you can point at,
with a comment next to it and a git history behind it. When the number is buried in a
function, the honest answer is usually "I don't remember".

The file has a header that sets the rule and the ritual:

```yaml
# Values marked TUNE are starting guesses that Phase 0 calibration will adjust.
# When you change one, record the old and new value in EXPLAIN.md Part 8.
```

### The sections

| Section | What it controls | Notable values |
|---|---|---|
| `shift` | length and idle cost | 480 min, 5 min empty-queue wait |
| `arrivals` | how fast alerts come | λ = 0.35/min |
| `incident` | the truth model | base 0.0135, the three lift arrays, dwell 120–480 |
| `severity` | the noisy vendor label | 4 levels, prior [.45 .30 .18 .07] |
| `asset_criticality` | asset tiers | 3 levels, **reward multipliers 1.0 / 1.5 / 2.5** |
| `verify_cost_min` | investigation costs | 5 / 10 / 20 / 40 minutes |
| `alert_types` | six categories | name, prior, incident lift |
| `state_buckets` | the 576-state grid | the three boundary lists |
| `actions` | names (order matters!) and bulk-close rules | 10 alerts, 2 min, sev ≤1, crit 0 |
| `reward` | the six reward numbers | all of them admittedly arbitrary |
| `metrics` | rupee cost model | **evaluation only, never learned from** |
| `seeds` | train and eval blocks | 10 train, 30 eval, disjoint |

### The seeds block, and the story in its comment

```yaml
seeds:
  train: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  # Widened from [101..105] to [101..130] on 2026-08-17 (D-019). Five seeds
  # could not resolve the effects being reported: per-seed reward std is ~218
  # for severity-sort against differences of ~100, so the standard error of a
  # 5-seed mean was the same size as the findings (E-008)...
  eval:  [101, ..., 130]
```

This comment is worth reading twice, because it is a small lesson in evidence.

```
   Per-seed noise:  std ≈ 218
   Effect measured: ≈ 100

   With 5 seeds:   standard error = 218/√5  = 97   ← the same size as the effect.
                   You cannot tell a real difference from luck.

   With 30 seeds:  standard error = 218/√30 = 40   ← now the effect is visible.
```

And the last sentence of the comment is the discipline:

> *"The original five are KEPT INSIDE the block, so every pre-widening result is a
> sub-sample of the new one rather than being orphaned."*

They could have picked 30 fresh seeds. Then every result recorded before 2026-08-17
would be measured on a different set and be incomparable. By keeping 101–105 inside
101–130, the old numbers remain a subset of the new ones. That's CONSTRAINTS #4 —
*never delete an experiment result* — being enforced by a choice of integers.

### The `metrics` block — assumption, not learning

```yaml
metrics:
  composite_cost_inr:
    missed_incident_by_criticality: [50000, 150000, 500000]
    wasted_analyst_minute: 30
    detection_delay_per_min: 200
```

The comment above it:

> *"These rupee figures are a STATED ASSUMPTION for reporting, not learned."*

These numbers translate an episode into money for the results section. **No agent ever
sees them and no agent is ever trained on them.** They're a lens for presenting results,
declared openly so a reader who disagrees with ₹500,000 for a crown-jewel breach can
substitute their own figure and recompute. That is what makes an assumption honest
rather than hidden.

---

## 7. The code that reads the numbers — `src/soc_triage/config/`

Four files. The package once was a single 657-line module and was split in D-031:

```
  config/
    __init__.py            84   re-exports everything; explains the split
    validation.py          46   ConfigError + the three shared checks
    environment.py        327   env_default.yaml  -> typed frozen objects
    training.py           376   training_default.yaml -> typed frozen objects
    training_validation.py 405  the checks for the above
```

### The naming collision, admitted in the docstring

```
  config/                          <- YAML files, at the repo root
  src/soc_triage/config/           <- the Python that reads them
```

The `__init__.py` docstring points at this directly:

> *"Note the two meanings of the word 'config' in this repo, which the split makes
> easier to confuse rather than harder."*

That is a good habit worth copying: when a design has a wart, say so in the place
someone will hit it, rather than hoping they work it out.

### The split was a filing change, not an interface change

> *"the package re-exports every name the old module exported, so `from
> soc_triage.config import EnvConfig, load_training_config` keeps working exactly as
> before"*

Nothing outside the package had to change. That's what makes a refactor safe to
believe: if no caller changed, the existing tests are testing the *move* rather than a
rewrite.

`training_validation.py` says the same thing even more strictly:

> *"**Every check here was moved verbatim.** None was reworded, reordered or renamed,
> which is what makes the existing suite a real test of the move rather than a test of a
> rewrite."*

### `validation.py` — three checks, one error type

```python
_require(mapping, key, path)      # a missing key names its own dotted path
_check_prior(prior, n, path)      # must be n non-negative values summing to 1
_check_ascending(values, path)    # bucket boundaries must strictly ascend
```

The design rule is in the module docstring:

> *"A missing key that surfaces mid-training costs an evening; the same key named in a
> ConfigError costs a line."*

So `ConfigError` messages always carry the full dotted path — `'incident.base_rate'`,
not "invalid config". You can fix it without reading the loader.

### `_validate()` in `environment.py` — the cross-field checks

Everything `_validate` refuses, and what would happen if it didn't:

| Check | What it prevents |
|---|---|
| `shift.length_min > 0` | An episode that ends before it starts |
| `arrivals.rate_per_min > 0` | Division by zero computing the mean gap |
| `0 < base_rate < 1` | A probability that isn't one |
| `target_severity_corr` ascending in [0,1] | A meaningless calibration gate |
| lift array lengths match levels | Indexing past the end of `severity_lift` |
| lifts non-negative | Negative probabilities |
| every prior sums to 1 | A distribution that isn't one |
| `reward_multiplier` length matches levels | An `IndexError` deep in a reward calculation |
| bucket boundaries strictly ascend | Ambiguous, silently wrong bucketing |
| **exactly 5 actions** | Q-tables the wrong width |
| cost array length matches asset levels | A crash in the metrics report |
| **train ∩ eval seeds = ∅** | **The project's central methodological failure** |

That last one deserves its own box.

### The seed-disjointness check — a rule made physical

```python
# CONSTRAINTS.md #2: train/eval seed separation is enforced here, in code,
# not by convention. Overlapping seeds refuse to load at all.
overlap = set(cfg.seeds.train) & set(cfg.seeds.eval)
if overlap:
    raise ConfigError(f"train and eval seeds must be disjoint; both contain {sorted(overlap)}")
```

Four lines that carry the weight of the project's honesty claim.

If training seeds and evaluation seeds overlap, your agent has been tested on shifts it
was trained on. The number goes up. The result is worthless and *nothing about it looks
wrong* — no crash, no warning, just a flattering figure you'd have no reason to doubt.

Most projects handle this with a comment saying "remember to keep these separate". This
one refuses to start. The phrase in the comment is the point: **"enforced in code, not
by convention."** A convention is something a tired student breaks at 2am; a
`ConfigError` is not.

### `training_validation.py` — the criterion for what belongs there

The docstring states the test for whether a check belongs in this file, and it's a
genuinely useful idea:

> *"Raise ConfigError on any setting that would fail as a bad RESULT rather than as an
> error. ... An out-of-range alpha produces a diverging Q-table; a Huber delta below the
> measured collapse threshold produces an agent that ignores catastrophes; an entropy
> coefficient of the wrong sign produces a policy that deletes its own exploration."*

**The distinction: a setting that crashes doesn't need a guard, because you'll find out.
A setting that quietly produces a wrong number does.** The Huber delta example is not
hypothetical — it's the bug from Document 1 §11, where a too-small delta flattened every
catastrophe to the size of a routine error. It cost real time to find, and the check
exists so it can't happen twice.

---

## 8. The ground-truth firewall

The project's first constraint:

> **CONSTRAINTS #1: Never let ground truth leak into an observation.**

Here is every place that rule is defended, in the order data flows:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │ alerts.py — DECLARE                                              │
  │   The two secret fields carry a warning in the class docstring   │
  │   naming state.py as the enforcement point and the test as the   │
  │   tripwire. Whoever reads the Alert reads the rule.              │
  └──────────────────────────────────────────────────────────────────┘
                                 │
  ┌──────────────────────────────────────────────────────────────────┐
  │ env.py — CENSOR                                                  │
  │   _snapshot() builds EnvSnapshot from the queue. The Alert       │
  │   objects go in whole — the secrets are still physically there.  │
  │   Nothing is deleted; the contract is that nobody looks.         │
  └──────────────────────────────────────────────────────────────────┘
                                 │
  ┌──────────────────────────────────────────────────────────────────┐
  │ state.py — THE FIREWALL                                          │
  │   Both encoders read ONLY: severity, arrival_time,               │
  │   asset_criticality, verify_cost_min, alert_type.                │
  │   Neither reads is_true_incident or deadline_min.                │
  │   This is the last point before the agent.                       │
  └──────────────────────────────────────────────────────────────────┘
                                 │
  ┌──────────────────────────────────────────────────────────────────┐
  │ test_no_ground_truth_leakage — THE TRIPWIRE                      │
  │   Flips the hidden fields on every alert, re-runs both encoders, │
  │   asserts nothing changed. If a leak is ever introduced, this    │
  │   test fails.                                                    │
  │   CONSTRAINTS: "must never be weakened or skipped."              │
  └──────────────────────────────────────────────────────────────────┘
```

**Why this design rather than simply removing the fields?**

You could hand the agent a stripped copy of each alert with no secrets in it. It would be
safer. It would also mean building and copying a second object for every alert on every
step, and it would put the "what may the agent see" decision in a place that's easy to
change without noticing.

The project instead puts the secret next to the fact it describes, states the rule in the
docstring, enforces it at exactly one chokepoint, and tests the chokepoint. The test is
what makes that defensible: the rule isn't "we were careful", it's "here is a test that
fails if we weren't".

### The one legal exception

```python
# The only reader allowed to act on them is the `oracle` baseline, which exists
# to be an upper bound.
```

The `oracle_greedy` baseline reads `is_true_incident` directly and always investigates a
real incident if one is waiting. It is not a competitor; it is the ceiling — "what score
would a policy get if it had perfect knowledge?". Every other policy is measured against
it.

It is also excluded from the RLHF policy pool. If a labeller compared an oracle shift to
a real one, they'd be learning to recognise clairvoyance rather than good judgement.
Document 1 §7 and Document 5 cover that.

---

## 9. Things that will confuse you

A list of the things in this part of the codebase that look wrong and aren't.

| What you'll notice | Why it's like that |
|---|---|
| `is_true_incident` is right there on the `Alert` | Yes. See §8 — it's guarded at a chokepoint and tested, not hidden. |
| False positives have `deadline_min = 0.0` | Meaningless for them; nothing reads it. Keeps the array numeric. |
| Empty queue and "one quiet alert" are the same state | An accepted, documented loss (§4). |
| `generator.py` is vectorised, agents aren't | CONSTRAINTS #14 — speed for the simulator, legibility for the algorithms. |
| `_select_alert` uses strict `>` not `>=` | Ties must go to the earlier alert or runs stop being reproducible. |
| `mttd_min` can be `None` | `0.0` would read as "instant detection" for a shift that caught nothing. |
| A `metrics` block in the env config the agent never sees | Reporting assumption, declared openly (§6). |
| `soc_triage.config` vs `config/` | Two different things with the same name. Admitted in the docstring. |
| `training_validation.py` takes 11 loose arguments | So the moved checks could be moved *verbatim*. Explained in its docstring. |
| The bulk-close action is a trap | On purpose. It's the case the hand-written reward handles least convincingly. |

---

## Where to go next

- **Document 3 — The Agents.** Everything that consumes what this document produces:
  the baselines, DP, Monte Carlo, SARSA, Q-learning, DQN, REINFORCE, actor-critic.
- **Document 4 — Running and Measuring.** How episodes get run in bulk, and how results
  are turned into numbers you can defend.
- **Document 1 §11** if you haven't read the honest-results section yet. It is the part
  of this project most worth understanding.

---

*Source of truth: this Markdown file. The `.docx` export is generated from it. If they
disagree, the Markdown is right.*
