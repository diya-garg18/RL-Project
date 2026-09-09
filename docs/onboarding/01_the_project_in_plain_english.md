# Document 1 — The Project in Plain English

**TriageRL — an onboarding guide for someone who has never seen this project**

> **Who this is for.** Someone with no background in this project, and not much in
> reinforcement learning or cyber security either. By the end of this document you
> should be able to explain, out loud, what this project is, what problem it solves,
> how it solves it, and what it has actually found so far.
>
> **What this document does NOT do.** It does not walk through code. Documents 2 to 6
> do that, file by file. This one builds the picture the code fits into. Read it first.
> Nothing else will make sense without it.
>
> **How long.** About 45 minutes if you read carefully. Do not skim §4 — it is the
> part you will be asked about.

---

## Table of contents

| § | Section | Why you need it |
|---|---|---|
| 1 | What this project is, in one page | The elevator answer |
| 2 | The real-world problem | Why anyone should care |
| 3 | What reinforcement learning is | The one concept everything rests on |
| 4 | **Turning triage into an RL problem — the MDP** | **The heart of the project** |
| 5 | The reward function, and why it is a lie | The motivation for Phase 5 |
| 6 | RLHF — learning the reward from humans | What makes this project different |
| 7 | The simulator | Where the data comes from |
| 8 | How we measure anything | Metrics and the rules of evidence |
| 9 | Map of the repository | Where to find things |
| 10 | The six phases, and what each one found | Project history in one table |
| 11 | The honest results so far | Including the ones that did not work |
| 12 | Glossary | Every term, defined |

---

## 1. What this project is, in one page

**TriageRL teaches a computer to decide which security alert a human should look at next.**

That is the whole thing. Everything below is detail.

A slightly longer version:

> An RL agent that learns which security alert a human analyst should investigate
> next, trained first on a hand-written reward and then re-trained on a reward model
> learned from real human preference judgments (RLHF).

It is a university mini-project for **CS4148 Reinforcement Learning** at Manipal
University Jaipur, built by two students — **Pranav Upadhyay** and **Diya Garg** —
under **Dr. Arshpreet Kaur**. It is graded as a 10-mark mini project plus a 5-mark
viva, with an end-term demo.

That grading detail matters more than it looks, because it shapes every engineering
decision in the repository. **Both students must be able to write the core algorithms
from memory, on a whiteboard, under questioning.** So the code is deliberately written
to be *readable rather than clever*, every algorithm is written by hand instead of
imported from a library, and the project carries an unusual amount of documentation —
this document among it.

### The three things that make it unusual

Most student RL projects pick a problem, implement one algorithm, report a number that
went up, and stop. This one does three things differently.

**1. The problem genuinely needs RL.** It is not a classification problem in a costume.
Each decision consumes a limited budget and changes what comes next — that is the
definition of a sequential decision problem.

**2. The reward is genuinely un-writable by hand.** Nobody knows how many wasted
analyst-minutes equal one three-hour detection delay. There is no correct number. That
makes learning the reward from human preferences *necessary* rather than decorative.

**3. Negative results are reported, not buried.** Several things in this project did
not work. They are all written down, with the evidence. §11 lists them.

---

## 2. The real-world problem

### 2.1 What a SOC is

A **Security Operations Centre** (SOC) is the room — often virtual — where an
organisation watches its own computers for signs of attack. Security software
(firewalls, antivirus, intrusion detection systems, cloud monitors) constantly emits
**alerts**: "this login looks odd", "this server contacted an unusual address", "this
file matched a known-malware pattern".

Human **analysts** work through those alerts. Investigating one means actually looking:
pulling logs, checking whether the user was travelling, deciding whether it is real.

### 2.2 The numbers that create the problem

```
   Alerts arriving in one 8-hour shift    ~170
   Alerts one analyst can investigate     ~35
   ────────────────────────────────────────────
   Coverage                               ~20%

   Of those ~170 alerts, roughly           3%  are real intrusions
   → about 5 real incidents, hidden among ~165 false alarms
```

**Roughly 80% of the queue is never looked at.** That is not a failure of diligence;
it is arithmetic. There is not enough time.

So the question is not *"can we investigate everything?"* — we cannot. The question is:

> **Which 20% do we look at?**

Get that ordering right and the five real incidents are caught early. Get it wrong and
an attacker sits inside the network for hours or days while analysts work through false
alarms. Time matters enormously here: every hour a real intrusion sits untouched is
another hour of **attacker dwell time** — more data stolen, more systems reached.

### 2.3 What the industry actually does

The standard answer is **sort by severity** — work the alerts the security vendor
labelled "high" first, then "medium", and so on.

This is the baseline to beat, and it is weak for a specific reason:

> The severity label is produced by a product that has never seen your business, does
> not know which of your servers matter, and does not know how many analyst-minutes
> are left in the shift.

A "high severity" alert on a developer's test laptop is less urgent than a "medium" on
the payroll database. The vendor's label cannot know that. A human triage lead knows it
instinctively. **The question is whether an agent can learn it.**

The phenomenon of analysts drowning in low-value alerts has a standard industry name —
**alert fatigue** — which tells you this problem is real and unsolved.

### 2.4 Why this is a hard *sequential* problem

Three properties make it genuinely sequential rather than a one-shot decision:

| Property | What it means |
|---|---|
| **Actions consume a budget** | Investigating a slow alert costs 40 minutes you can never spend again |
| **Actions change the future** | The queue you face next depends on what you just did |
| **Outcomes are delayed and uncertain** | You do not learn whether an alert was real until you finish investigating it, and you never learn about the ones you skipped |

That combination — limited budget, state that evolves, delayed feedback — is exactly
the shape reinforcement learning is designed for.

---

## 3. What reinforcement learning is

If you already know RL, skip to §4. If you do not, this section is enough to follow the
whole project.

### 3.1 The core loop

Reinforcement learning is learning by trying things and seeing what happens. There are
only ever two participants:

```
        ┌───────────────────────────────────────────────┐
        │                                               │
        │            ┌─────────────┐                    │
        │            │    AGENT    │                    │
        │            │ (the learner)│                   │
        │            └──────┬──────┘                    │
        │                   │                           │
        │      state s      │      action a             │
        │   "here is the    │   "I choose to            │
        │    situation"     │    do this"               │
        │         ▲         ▼                           │
        │            ┌─────────────┐                    │
        │            │ ENVIRONMENT │                    │
        │            │  (the world)│                    │
        │            └──────┬──────┘                    │
        │                   │                           │
        │              reward r                         │
        │           "that was worth                     │
        │            +12 points"                        │
        └───────────────────┴───────────────────────────┘
                     repeat, thousands of times
```

1. The **environment** shows the agent a **state** — a description of the current
   situation.
2. The agent picks an **action**.
3. The environment returns a **reward** (a number saying how good that was) and a new
   state.
4. Repeat until the episode ends.

The agent's goal is **not** to maximise the next reward. It is to maximise the **total
reward over the whole episode**. That distinction is the entire subject. An action that
scores badly right now may set up something far better later — and an action that
scores well right now may be a trap.

### 3.2 The five words you need

| Term | Plain meaning | In this project |
|---|---|---|
| **Agent** | The learner making decisions | The triage system |
| **Environment** | Everything the agent acts on | The simulated SOC shift |
| **State** | A summary of the current situation | What the alert queue looks like right now |
| **Action** | A choice the agent can make | Which triage rule to apply |
| **Reward** | A number scoring what just happened | Points for catching real incidents, penalties for wasting time |
| **Episode** | One complete run, start to finish | One 8-hour analyst shift |
| **Policy** | The agent's strategy: state → action | "When the queue is long and time is short, do X" |

### 3.3 The one hard idea: exploration vs exploitation

An agent that always does whatever currently looks best will never discover something
better. An agent that always tries random things never benefits from what it learned.

Every algorithm in this project handles that tension somehow. The most common trick is
**ε-greedy** (epsilon-greedy): with probability ε, act randomly; otherwise act greedily.
Start with ε high (explore a lot), decay it over training (exploit what you learned).

You will see `epsilon`, `epsilon_decay` and `epsilon_min` throughout the config files.
That is what they mean.

### 3.4 The families of algorithm this project implements

All six are written by hand. Here is the map, so the file names mean something:

```
                    RL ALGORITHMS
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
  KNOW THE MODEL?                     DON'T KNOW IT
  (you have P and R)                  (learn from experience)
        │                                    │
  Dynamic Programming            ┌───────────┴────────────┐
  • value iteration              │                        │
  • policy iteration       VALUE-BASED              POLICY-BASED
   (Phase 1, dp.py)        learn "how good           learn the strategy
                            is each action"           directly
                                 │                        │
                    ┌────────────┼──────────┐       ┌─────┴──────┐
                    │            │          │       │            │
              Monte Carlo     SARSA    Q-learning  REINFORCE  Actor-Critic
              (Phase 2)     (Phase 2)  (Phase 2)   (Phase 4)   (Phase 4)
                                            │
                                          DQN
                                    (Phase 3 — the same
                                     idea, but a neural
                                     network instead of
                                     a table)
```

**The three-line summary of each:**

- **Dynamic programming** — if you already know exactly how the world works, you can
  compute the perfect answer by repeatedly improving your estimate. We do not know how
  the world works, so we *estimate* the model from 50,000 random runs first, then solve
  that. The answer is optimal *for the estimate*, not for reality. Being honest about
  that gap is part of the deliverable.
- **Monte Carlo** — play a whole episode, see the total reward, then credit every action
  you took along the way. Simple and unbiased, but you learn nothing until the episode
  ends.
- **SARSA / Q-learning** — update after *every step* instead of waiting for the end,
  using your own current estimate of what comes next ("bootstrapping"). The difference
  between them is subtle and important: SARSA learns the value of the policy it is
  actually following, Q-learning learns the value of the best policy regardless of what
  it is doing.
- **DQN** — Q-learning where the table is replaced by a neural network, so the state no
  longer has to be squeezed into a fixed set of buckets.
- **REINFORCE** — skip values entirely; directly nudge the probability of actions that
  led to good outcomes.
- **Actor-critic** — REINFORCE plus a second network that estimates value, used to
  reduce the wild variance REINFORCE suffers from.

Each has its own file under `src/soc_triage/agents/` and its own document section later.

---

## 4. Turning triage into an RL problem — the MDP

> **This is the most important section in this document.** An MDP (Markov Decision
> Process) is the formal description of an RL problem: states, actions, rewards, and
> how they connect. Getting this right *is* the project; everything else is
> implementation.

### 4.1 The episode

**One episode = one 8-hour analyst shift = 480 simulated minutes.**

Alerts arrive continuously throughout. The agent repeatedly chooses an action; each
action consumes simulated minutes; the episode ends when the clock hits 480.

### 4.2 The state — discretised version (576 states)

For the table-based algorithms (Phases 1 and 2), the situation is squeezed into
**five features**, each chopped into a few buckets:

| Feature | Buckets | The values |
|---|---|---|
| `max_severity_in_queue` | 4 | 0 = info, 1 = low, 2 = medium, 3 = high |
| `queue_len_bucket` | 4 | `[0,10)`, `[10,40)`, `[40,100)`, `[100,∞)` |
| `oldest_age_bucket` | 4 | `[0,30)`, `[30,90)`, `[90,180)`, `[180,∞)` minutes |
| `time_left_bucket` | 3 | `>240`, `[60,240]`, `<60` minutes |
| `max_asset_crit_in_queue` | 3 | 0 = dev box, 1 = standard, 2 = crown jewel |

**4 × 4 × 4 × 3 × 3 = 576 possible states.**

With 5 actions, the agent's entire brain is a **576 × 5 table = 2,880 numbers**.

That size is a deliberate design choice, and it buys something valuable: **you can print
the whole policy and read it.** `scripts/policy_table.py` does exactly that. A security
manager can look at the output and say "I agree with that" or "that is wrong" — which is
worth more in this domain than a slightly better score from something nobody can inspect.

**A worked example.** Suppose right now:

```
  the queue holds 55 alerts                     → queue_len_bucket   = 2   ([40,100))
  the worst severity in it is "high"            → max_severity       = 3
  the oldest alert has waited 100 minutes       → oldest_age_bucket  = 2   ([90,180))
  there are 45 minutes left in the shift        → time_left_bucket   = 2   (<60)
  the most critical asset involved is a         → max_asset_crit     = 2
      crown-jewel server
```

Those five numbers get combined into a single integer between 0 and 575. That integer
is the state. The agent looks up that row in its table, sees five numbers (one per
action), and picks the best one.

### 4.3 The state — continuous version (~17 numbers)

For DQN and the policy-gradient methods (Phases 3 and 4), the same situation is
described **without bucketing** — about 17 raw floating-point numbers: queue length,
mean and max severity, mean and max age, mean and max asset criticality, mean verify
cost, the fraction of the queue that is each of the 6 alert types, normalised time
remaining, alerts handled so far, true incidents confirmed so far.

**Why keep both?**

| Version | What it proves |
|---|---|
| Discretised (576) | That we understand tabular RL, and it gives a policy a human can read |
| Continuous (~17) | The honest motivation for function approximation — bucketing throws information away, and a neural network does not have to |

Bucketing genuinely destroys information. Two situations that a human would treat
completely differently — a queue of 41 alerts and a queue of 99 — land in the same
bucket and are, to a tabular agent, *literally the same state*. The continuous version
can tell them apart.

### 4.4 The actions — five, and this is the key design decision

**The agent does not pick an alert. It picks a rule.**

| # | Action | What it does |
|---|---|---|
| 0 | `PULL_HIGHEST_SEVERITY` | Investigate the highest-severity alert |
| 1 | `PULL_OLDEST` | Investigate the longest-waiting alert |
| 2 | `PULL_MOST_CRITICAL_ASSET` | Investigate the alert on the most business-critical asset |
| 3 | `PULL_CHEAPEST` | Investigate the fastest-to-verify alert (clear backlog) |
| 4 | `BULK_CLOSE_LOW_RISK` | Auto-close up to 10 low-severity, low-criticality alerts (costs 2 min) |

> **This is the single most important design decision in the project.** If you remember
> one thing from this document, remember this and the reason for it.

**Why not just "pick alert number N"?**

Because the queue changes size constantly — it might hold 3 alerts or 300. That would
give a **variable-sized action space**, and:

- it **breaks tabular Q-learning outright** (the Q-table needs a fixed number of columns),
- it makes DQN awkward (the output layer would have to change size),
- and the learned policy becomes unreadable ("in state 412, pick alert #187" tells a
  human nothing).

By making the action *a choice of triage strategy* rather than *a choice of alert*:

- the action space stays **fixed at 5**, whatever the queue size,
- the Q-table stays small enough to print,
- and the policy is human-readable: *"when time is short and the queue is huge, bulk-close."*

**What we gave up:** expressiveness. The agent can never say "investigate that specific
odd-looking alert". It can only pick among five rules. In this domain that trade is
worth it, because an uninspectable policy is not deployable in security work anyway.

**Two details that matter:**

- **Ties are broken deterministically by alert ID**, so runs reproduce exactly across
  seeds. Without that, two runs with the same seed could diverge.
- **Empty queue** → the clock advances 5 minutes and the reward is 0. The agent cannot
  investigate nothing.

**The trap built into action 4.** `BULK_CLOSE_LOW_RISK` shrinks the queue fast and earns
small positive rewards. An agent that learns to spam it looks productive and achieves
nothing. That is **reward hacking**, and catching it is a named deliverable in Phase 6 —
not an accident we hope to avoid, but a thing we go looking for deliberately.

---

## 5. The reward function, and why it is a lie

### 5.1 The hand-written reward (version 1)

| Event | Reward |
|---|---|
| Investigate a **true incident** | `+100 × exp(−delay/120) × asset_multiplier` |
| Investigate a **false positive** | `−1 × verify_cost_min` (so −5 to −40) |
| Bulk-close a false positive | `+0.5` each (queue hygiene) |
| Bulk-close a **true incident** | `−150 × asset_multiplier` |
| End of shift: a true incident never triaged past its deadline | `−200 × asset_multiplier` |

where `asset_multiplier` = **1.0 / 1.5 / 2.5** for criticality 0 / 1 / 2, and `delay`
is the minutes between an alert arriving and being investigated.

**Reading the first line in English:** catching a real incident is worth 100 points,
multiplied by how quickly you caught it, multiplied by how important the machine was.
The `exp(−delay/120)` term means the value decays smoothly with delay — catch it
instantly and you get the full 100; catch it 120 minutes later and you get about 37;
catch it after 240 minutes and about 13.

### 5.2 Now the uncomfortable part

**Every one of those numbers is made up.**

Why is a missed incident worth −200 and not −180, or −5,000? Why does a caught incident
decay with a 120-minute half-life rather than 90? Why is bulk-closing a false positive
worth +0.5?

There is no principled answer. Someone chose them because they felt reasonable.

This means the agents in Phases 1 through 4 are **optimising our arbitrary opinion**
about the trade-off between wasted analyst time and missed breaches. They are extremely
good at optimising it. That does not make the opinion right.

> This is not a flaw to apologise for — it is the entire motivation for Phase 5, and it
> should be said out loud in the viva.

### 5.3 So what do you do instead?

You could try asking a security expert to write down the numbers. They cannot. Ask an
experienced analyst "how many wasted minutes equal one three-hour detection delay?" and
you will get a shrug — the question is not answerable in that form.

But ask them something else and they answer instantly:

> *"Here are two shifts, same alerts, handled differently. Which one went better?"*

**Humans are bad at assigning numbers and good at making comparisons.** That single
observation is what Phase 5 is built on.

---

## 6. RLHF — learning the reward from humans

**RLHF** = Reinforcement Learning from Human Feedback. It is the technique behind how
modern chat assistants are tuned, and it is 7 lectures of this course's syllabus.

### 6.1 The idea in one diagram

```
  STEP 1                STEP 2                  STEP 3
  ──────                ──────                  ──────

  Run different    →    Show a human    →    Train a model to    →   Re-train the
  policies on the       two shifts           predict which one       agent using the
  SAME alert            side by side.        a human would            LEARNED reward
  stream.               They click            prefer.                 instead of the
                        left / right                                  made-up one.
                        / can't tell.
                                              This is the
  300 pairs             300 labels            REWARD MODEL
                                              (Bradley–Terry)
```

### 6.2 Why the pairs share an alert stream

Both sides of a comparison are **the same shift, handled by two different policies**.
Same alerts, same arrival times, same hidden truth.

That is deliberate. If the two sides had different alert streams, a labeller could not
tell whether one side did better because the *policy* was better or because it simply
got an easier shift. Holding the stream fixed removes that confusion entirely. In
statistical terms it is a **paired comparison**, and it is a variance-reduction
technique worth being able to explain.

### 6.3 What the labeller actually sees

Each side shows a compact **timeline of actions** plus **outcome cards**: which
incidents were caught and when, which were missed, how many minutes were wasted.

**The labeller is shown the ground truth** — which alerts were really incidents. That
is intentional. They are judging *outcomes*, not guessing which alerts were real. The
job is "which of these two shifts went better", not "play the game yourself".

**The labeller is never shown a reward number.** Not the total, not per-step, not
anywhere. If they saw the hand-written reward they would simply agree with it, and the
whole exercise would collapse into re-learning the numbers we were trying to replace.
Every reward field is stripped out in code, and two tests exist purely to enforce it.

**The labeller is never shown which policy is which.** No policy names reach the
labelling file. Otherwise "oh, this is the DQN" biases the judgement.

### 6.4 The targets

| Target | Number | Why |
|---|---|---|
| Labelled pairs | **300** | Enough to train a small reward model |
| Pairs labelled by **both** Pranav and Diya | **50** | To compute inter-annotator agreement |
| Estimated human time | **~100 minutes each** | ~20 seconds per pair |

### 6.5 Cohen's κ — and why a bad result is still a result

The 50 double-labelled pairs let us compute **Cohen's kappa (κ)**, a standard measure of
how much two people agree *beyond what you would expect from chance alone*.

```
        raw agreement − chance agreement
  κ  =  ───────────────────────────────
              1 − chance agreement
```

- κ = 1.0 → perfect agreement
- κ = 0.0 → agreeing exactly as often as two people guessing randomly would
- κ around 0.6–0.8 → conventionally "substantial"

Why subtract chance? Because if both labellers just clicked "left" 90% of the time, they
would agree 80%+ of the time while sharing no actual judgement. κ strips that out.

> **If κ comes out low, that is a finding, not a failure.** It would mean "good triage"
> is genuinely ill-defined even between two people who built the system — which is
> important, publishable-in-a-report information about the problem, and it must be
> reported rather than quietly re-labelled until it looks better.

### 6.6 The reward model (Phase 5b — not yet built)

Once the labels exist, they train a small neural network `r̂(state, action)` using the
**Bradley–Terry** model:

```
                        exp( Σ r̂(s,a) over shift A )
  P(A preferred to B) = ─────────────────────────────────────
                        exp( Σ over A ) + exp( Σ over B )
```

In words: **score every step of each shift, add the scores up, and the shift with the
higher total should be the one the human preferred.** Train by pushing the model's
predicted preference towards the human's actual click (binary cross-entropy loss).

It is about 40 lines of PyTorch, and you should be able to write that loss on a
whiteboard.

### 6.7 Then what (Phase 5c)

Re-train the Phase 2 and 3 agents with `r̂` in place of the hand-written reward, and ask
two questions:

1. **Do humans prefer the RLHF policy** over the hand-reward policy, on *fresh* pairs
   nobody has labelled before? This is the headline result.
2. **How do the two compare on ground truth** (incidents missed, MTTD)?

And the interesting case: **if the RLHF policy is worse on ground truth but preferred by
humans, that is a genuinely important result and gets reported, not buried.** It would
mean human preference and actual security outcomes diverge — which is exactly the
failure mode people worry about with RLHF in general.

---

## 7. The simulator — where the data comes from

**There is no public dataset of real analyst triage decisions.** So the environment is
simulated. That is documented as the project's primary external-validity limitation
rather than hidden.

### 7.1 How alerts are generated

Alerts arrive as a **Poisson process** — the standard model for "events happening
independently at a steady average rate" — with λ ≈ 0.35 per minute, giving ~170 alerts
per 480-minute shift.

Each alert carries:

```
id                  a number, used for deterministic tie-breaking
arrival_time        minutes into the shift
severity            0–3, the vendor's label — deliberately noisy
asset_criticality   0–2, how much the machine matters
verify_cost_min     5, 10, 20 or 40 minutes to investigate
alert_type          one of 6 categories
is_true_incident    HIDDEN from the agent
deadline_min        dwell-time budget; only meaningful if it is a true incident
```

### 7.2 The assumption everything rests on

`P(true incident)` ≈ **3% base rate**, adjusted by alert type, asset criticality and
time of day. And critically:

> **Severity is only weakly correlated with truth** (target Pearson r ≈ 0.3–0.4), while
> the *combination* of type + asset criticality + verify cost is substantially more
> informative.

**This is the assumption that makes the project possible, and it must be stated
plainly.** If severity were a perfect predictor of truth, severity-sort would already be
optimal and there would be nothing to learn. Because it is weak but not useless, an
agent that learns to combine the *other* signals can beat it.

It is a defensible assumption — the poor quality of vendor severity labels is widely
reported in industry — but it **is** an assumption that was deliberately built in, and
the report says so.

The calibration was checked and passed on 2026-08-13: **168.7 alerts per shift, 3.34%
incidence, r = 0.323**, robust across two untuned seed blocks.

### 7.3 The one rule that must never be broken

`is_true_incident` and `deadline_min` are **ground truth**. The environment needs them
to compute reward. The agent must **never** see them, directly or through any proxy.

This is the single easiest way to accidentally cheat — and cheating here would silently
invalidate every result in the project. So there is a dedicated test,
`test_no_ground_truth_leakage`, that must never be weakened or skipped.

**The one exception:** the `oracle` baseline sees the hidden labels *on purpose*. It
exists to show the theoretical ceiling — the best any policy could possibly do. It is
named `oracle`, it lives in `baselines.py`, and it is never presented as something we
achieved.

---

## 8. How we measure anything

### 8.1 The baselines

You cannot claim an agent is good without something to compare it to.

| Baseline | Why it exists |
|---|---|
| **Random** | The floor. Anything that cannot beat random is broken. |
| **FIFO** | First in, first out — the naive human default |
| **Severity-sort** | **What the industry actually does. This is the one to beat.** |
| **Cheapest-first** | Throughput-maximising strawman — clears the most alerts, catches little |
| **DP-on-estimated-model** | The Phase 1 reference |
| **Oracle-greedy** | Sees the hidden labels. The ceiling. Nothing can beat it. |

### 8.2 The metrics

| Metric | What it means |
|---|---|
| **MTTD** | Mean Time To Detect — average minutes from arrival to investigation, over incidents that were caught |
| **Recall@deadline** | % of true incidents caught before their dwell deadline. **The headline number.** |
| **Wasted minutes** | Analyst time spent on false positives |
| **Critical misses** | Count of crown-jewel (`asset_criticality = 2`) incidents missed |
| **Composite cost (₹)** | A single money figure under a clearly stated cost assumption |

A subtlety worth understanding: **MTTD and recall can disagree.** A policy that catches
only the three easiest incidents very quickly gets a beautiful MTTD and a terrible
recall. That is why recall is the headline and MTTD is reported alongside it, never
alone.

### 8.3 The rules of evidence — non-negotiable

These are enforced in `CONSTRAINTS.md` and in code:

1. **Never report a single run.** Every headline number is mean ± std over **at least 5
   seeds**. A single lucky run is not a result.
2. **Training seeds and evaluation seeds are disjoint, enforced in code, not by
   convention.** Never tune a hyperparameter by looking at evaluation results.
3. **Same alert streams across policies** (paired comparison) — a variance-reduction
   technique.
4. **Never delete or overwrite an experiment result** because a later run looked better.
   If a run was wrong, mark it wrong and say why. Do not erase it.
5. **A result that looks surprisingly good is a bug report until proven otherwise.**

That fifth rule has already earned its place in this project — twice. See §11.

---

## 9. Map of the repository

### 9.1 The shape of the system

```
                        ┌──────────────────────────┐
                        │   config/*.yaml          │
                        │  (all tunable numbers)   │
                        └────────────┬─────────────┘
                                     │ loaded once at startup
                                     ▼
┌───────────────┐         ┌─────────────────────┐         ┌──────────────────┐
│  generator.py │────────▶│      env.py         │◀───────▶│    agents/       │
│ makes alerts  │ stream  │ SOCTriageEnv        │ s,a,r,s'│ random, fifo,    │
│ + hidden truth│         │ reset() / step()    │         │ severity, q_learn│
└───────────────┘         └──────────┬──────────┘         │ dqn, reinforce...│
                                     │                    └────────┬─────────┘
                                     │ trajectory                  │
                                     ▼                             │
                          ┌─────────────────────┐                  │
                          │  runner.py          │◀─────────────────┘
                          │ episode loop,       │
                          │ seeding, logging    │
                          └──────────┬──────────┘
                                     │ EpisodeRecord (JSON)
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
          ┌──────────────┐  ┌────────────────┐  ┌──────────────┐
          │ evaluation/  │  │ rlhf/          │  │  results/    │
          │ metrics,     │  │ pair builder,  │  │ runs, plots, │
          │ audit        │  │ label store    │  │ checkpoints  │
          └──────────────┘  └───────┬────────┘  └──────────────┘
                                    │ pairs.json
                                    ▼
                          ┌─────────────────────┐
                          │  labelling/         │
                          │ the FastAPI page    │
                          │ humans click on     │
                          └─────────────────────┘
```

### 9.2 The one rule this shape encodes

> **The environment never knows which agent is acting, and the agents never reach inside
> the environment.** They communicate only through
> `(state, action, reward, next_state, done, info)`.

Keep that boundary clean and every algorithm becomes swappable — which is the whole
point, because six of them are implemented.

### 9.3 Where things live

| Path | What is in it |
|---|---|
| `config/*.yaml` | **Every tunable number in the project.** No magic numbers in code. |
| `src/soc_triage/alerts.py` | The `Alert` dataclass and its hidden ground truth |
| `src/soc_triage/generator.py` | Poisson arrivals, feature sampling, incident labelling |
| `src/soc_triage/env.py` | The MDP itself — queue, `reset()`, `step()`, reward, termination |
| `src/soc_triage/state.py` | Both state encoders: `discretise()` (576) and `featurise()` (~17 floats) |
| `src/soc_triage/agents/` | One file per algorithm |
| `src/soc_triage/runner.py` | Run N episodes with a seed, emit `EpisodeRecord`s |
| `src/soc_triage/evaluation/` | Metrics and (Phase 6) the reward-hacking audit |
| `src/soc_triage/rlhf/` | Phase 5a — pair building, label storage, Cohen's κ |
| `src/soc_triage/labelling/` | The web page humans actually click on (Diya's work) |
| `src/soc_triage/tiny_mdp.py` | A hand-solved 2-state MDP that checks the learners against an answer derived on paper |
| `src/soc_triage/mrp_example.py` | A hand-solved 5-state MRP that checks the DP code the same way |
| `scripts/` | Thin command-line entry points. **No logic lives here.** |
| `tests/` | 391 tests, all passing |
| `docs/features/`, `docs/bugs/` | One file per feature and per bug, start to finish |
| `docs/experiments/EXPERIMENT_LOG.md` | Every training run: config, seed, result |
| `results/` | **Gitignored.** Everything in it is regenerable. |
| `web/` | Reserved for the Phase 6 React dashboard. Not built. |

### 9.4 The documentation set, and what each file is for

This project follows a documentation discipline ("Don't just trust the AI. Trace it.").
The docs are not decoration — they are the mechanism by which two students stay able to
explain, in an interview, code that an AI helped write.

| File | Purpose |
|---|---|
| `README.md` | Start here |
| `CLAUDE.md` | The operating rules for AI sessions. Read first, every session. |
| `PROJECT_BRIEF.md` | The idea, the MDP, the plan — the *why and what* |
| `EXPLAIN.md` | Plain-English living explanation of everything. Updated every session. |
| `ROADMAP.md` | Phase-by-phase task list with checkboxes |
| `ARCHITECTURE.md` | The system map — module responsibilities |
| `CONSTRAINTS.md` | What must never happen. 26 hard rules. |
| `FLOW.md` | How execution travels between files |
| `HANDOVER.md` | Where things stand *right now*. Rewritten every session. |
| `DECISIONS.md` | Why each choice was made. **Append-only.** 46 entries. |
| `TEST_CHECKLIST.md` | What "done" means, with real commands |
| `ROLLBACK.md` | How to undo |
| `INTERVIEW_PREP.md` | The functions both students must know cold |
| **`docs/onboarding/`** | **This guide — documents 1 to 6** |

### 9.5 The two students

| Person | Owns |
|---|---|
| **Pranav** | All RL algorithms written from scratch, the Bradley–Terry reward model, the evaluation harness, the reward-hacking audit |
| **Diya** | The environment simulator and alert generator, the labelling UI, the React dashboard, the LLM justification layer |
| **Both** | The labelling sessions (must be both, for κ), the report, the presentation |

The git history is part of what gets evaluated, so commits are deliberately balanced
between them — and balanced by *handing over the work at the right time*, never by
committing on the other person's behalf. `scripts/commit_balance.py` reports the split
and flags when it drifts past 3 commits.

---

## 10. The six phases, and what each one found

| Phase | What was built | Status | The headline finding |
|---|---|---|---|
| **0** | Simulator, config, env, baselines, metrics | ✅ Done | Calibration passed: 168.7 alerts/shift, 3.34% incidence, r = 0.323 |
| **1** | Model estimation from 50k rollouts, value iteration, policy iteration | ✅ Done | Only 133 of 576 states are ever visited. VI and PI agree 100%. |
| **2** | Monte Carlo, SARSA, Q-learning | ✅ Done | **SARSA scores highest of any agent in the project** (reward 324.1 ± 81.6) |
| **3** | DQN, replay buffer, target network | ⚠️ Closed as *built but not passed* | **DQN loses to tabular** (recall 0.48 vs 0.73) |
| **4** | REINFORCE, actor-critic, variance analysis | ✅ Done | REINFORCE is ~32× more sample-efficient than DQN here |
| **5a** | Pair building, labelling UI, Cohen's κ code | ✅ **Code complete — no labels collected yet** | Pending: the 300 human judgements |
| **5b** | Bradley–Terry reward model | ⬜ Not started | Blocked on labels existing |
| **5c** | Re-train policies on the learned reward | ⬜ Not started | The headline result of the whole project |
| **6** | Reward-hacking audit, dashboard, report | ⬜ Not started | — |

**Where the project stands right now:** Phase 5a is code complete. Every piece of
software needed to collect preferences exists, is tested, and has been run end to end on
real data. **The 300 pairs exist. Nothing has been labelled.** The next step is not a
coding task — it is roughly 100 minutes each of two humans clicking left or right.

---

## 11. The honest results so far

> The rule in this project is that **negative results get documented, not deleted**.
> This section is why the project is worth more than its score.

### 11.1 The DQN lost, and that is reported

Phase 3's gate was "DQN beats tabular on the richer state". It did not.

- **Recall: DQN 0.48 vs tabular 0.73.**
- Reward is **not resolvable** — the paired |mean|/SEM was 1.42, meaning the difference
  is smaller than the noise. Saying "DQN was worse on reward" would be over-claiming, so
  it is not said.

But the phase's *premise* did hold: in **21 of 42 visited buckets**, the DQN chose a
different action for situations the discretisation lumps together. So the continuous
state really does carry information the buckets destroy — the network just did not turn
that into a better policy at this budget.

The phase was closed as **"built but not passed"**, which is an honest category most
projects do not have.

### 11.2 The Huber delta bug — a one-line setting that broke a phase

The DQN uses Huber loss, which has a parameter `delta` controlling where it switches
from squared to linear error. It was left at PyTorch's default of **1.0**, while the
project's penalties run from **−150 to −1,499**.

The effect: **every catastrophe was flattened to the size of a routine error.** Missing a
crown-jewel incident and mildly wasting time looked equally bad to the network. Twenty
training runs collapsed to spamming BULK_CLOSE.

It is now 200.0, with a loader guard that refuses anything below 50. Written up as
`BUG_002`.

**The lesson worth carrying:** a single default value, never questioned, silently
destroyed a phase's worth of results while every test still passed.

### 11.3 The target network made things worse

Standard DQN teaching says the target network stabilises learning. Here, **removing it
improved recall 0.481 → 0.588 and reward −46.9 → +43.5**.

Reported as measured, not as expected.

### 11.4 The replay buffer turned out to be load-bearing

The opposite surprise: with replay switched off, **all 8 runs scored recall 0.0000.**
Not "worse" — completely dead. Replay is not a refinement in this setting; without it
the agent learns nothing at all.

### 11.5 REINFORCE's baseline did not reduce variance

Textbooks say subtracting a learned baseline reduces the variance of the policy-gradient
estimate. Measured here:

| Condition | Coefficient std |
|---|---|
| REINFORCE, no baseline | 146.94 |
| REINFORCE, with baseline | 147.68 |
| Actor-critic (bootstrapped) | 30.89 |

**The baseline reduction did not replicate — 1.00×.** The value head is not accurate
enough at this training budget for it to pay off. Bootstrapping's **4.78×** reduction is
structural and did show up.

This is a negative result on the half of the theory everybody quotes, and it makes a
better interview answer than a confirmation would have.

### 11.6 The ablations that did not clear the noise floor

Phase 2's learning-rate, γ and ε-decay sweeps: **none of the three cleared the noise
floor.** Between-config spread was smaller than or comparable to within-config spread in
every sweep — the default config alone produced 75, −34 and 47 on different seeds.

Reported as a negative result rather than filled in with the best-looking row.

### 11.7 The convergence comparison that looked good and was not

Comparing the Q-learning policy against the Phase 1 DP solution gives **83–86% policy
agreement** if you compare all 576 states. That number is meaningless: it is manufactured
by the hundreds of states *neither* agent ever visited, where both fall back to the same
convention.

Over states that are actually visited, agreement is **22–44%**.

The honest number is the smaller one, and it is the one reported.

### 11.8 The config-hash false alarm

Pointing the pair builder at the run archive raised `MixedConfigError` on all 300
records — which looked, for a minute, like Phase 2 having compared agents measured on
two *different* environments. That would have been serious.

It was not. The two config hashes differ by **one character inside a comment** — a date
typo corrected from 2026-08-16 to 2026-08-17. Both files 4,991 bytes. No environment
parameter changed at all. The hash covers the *file*, not the settings.

Written up as `BUG_005`, and deliberately **not fixed** — the guard being over-sensitive
is safer than it being under-sensitive.

**The lesson:** an alarm firing is evidence about the alarm first, and the system second.

### 11.9 The clone that was not a clone

While choosing which training runs to put in front of labellers, three REINFORCE repeats
appeared identical to the `severity_sort` baseline on the evaluation seeds — same recall
to four decimal places. The obvious move was to discard all three as duplicates.

Measuring them on the actual pair seeds showed something different:

```
COLLAPSED:  reinforce@2  ==  reinforce@4  ==  severity_sort   (identical on all 12 seeds)
NOT A CLONE: reinforce@1 → recall 0.8947 — the highest in the entire pool,
             above severity_sort's 0.8530
```

**A rule chosen in advance would have thrown away the best policy in the project.**
Measuring instead of assuming is what caught it.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Action** | One of the 5 triage rules the agent can apply |
| **Actor-critic** | Policy gradient plus a value network, updated every step by bootstrapping |
| **Agent** | The decision-maker being trained |
| **Alert** | One notification from security software; may or may not be a real incident |
| **Alert fatigue** | The industry term for analysts drowning in low-value alerts |
| **Asset criticality** | 0–2: how much the affected machine matters (dev box → crown jewel) |
| **Baseline (RL)** | A value subtracted from the return to reduce variance |
| **Baseline (comparison)** | A simple policy used as a yardstick, e.g. severity-sort |
| **Bootstrapping** | Updating an estimate using another estimate rather than the final outcome |
| **Bradley–Terry** | The model turning two shift-scores into a probability that one is preferred |
| **Bulk close** | Action 4 — auto-close up to 10 low-risk alerts for 2 minutes |
| **CO1–CO5** | Course Outcomes — the syllabus objectives each phase maps to |
| **Cohen's κ** | Agreement between two labellers, corrected for chance |
| **Discretise** | Squeeze the continuous situation into one of 576 buckets |
| **Dwell time** | How long an attacker sits undetected. The thing we are minimising. |
| **Episode** | One complete 8-hour shift, 480 simulated minutes |
| **ε-greedy** | Act randomly with probability ε, greedily otherwise |
| **EpisodeRecord** | The JSON record of one complete episode. The project's interchange format. |
| **False positive** | An alert that turns out not to be a real incident (~97% of them) |
| **Featurise** | Turn the situation into ~17 raw numbers, no bucketing |
| **FIFO** | First in, first out |
| **γ (gamma)** | Discount factor — how much future reward is worth versus immediate |
| **Ground truth** | `is_true_incident` and `deadline_min`. Hidden from the agent. Always. |
| **Huber loss** | A loss that is squared for small errors, linear for large ones |
| **MDP** | Markov Decision Process — the formal statement of an RL problem |
| **Monte Carlo** | Learn only from complete episodes and their actual returns |
| **MRP** | Markov Reward Process — an MDP with no choices; used as a hand-checkable test |
| **MTTD** | Mean Time To Detect |
| **Off-policy** | Learning about one policy while following another (Q-learning) |
| **On-policy** | Learning about the policy you are actually following (SARSA) |
| **Oracle** | The baseline that cheats by design, to show the ceiling |
| **Paired comparison** | Comparing policies on identical alert streams to cut variance |
| **Poisson process** | The standard model for independent events at a steady average rate |
| **Policy** | The strategy: a mapping from state to action |
| **Q-table** | The 576 × 5 grid of "how good is this action in this state" |
| **Recall@deadline** | % of true incidents caught before their deadline. The headline metric. |
| **REINFORCE** | Monte Carlo policy gradient |
| **Replay buffer** | Stored past transitions, resampled to break correlation |
| **Reward hacking** | Scoring well while behaving badly — the thing Phase 6 hunts for |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **SARSA** | On-policy TD control, named for the tuple (s, a, r, s', a') |
| **Seed** | The number fixing the random stream, so runs reproduce exactly |
| **Severity** | 0–3, the vendor's label. Deliberately a weak predictor of truth. |
| **SOC** | Security Operations Centre |
| **State** | The description of the current situation the agent sees |
| **Target network** | A frozen copy of the Q-network used to compute update targets |
| **TD (temporal difference)** | Learning from the gap between successive estimates |
| **Value iteration** | Repeatedly applying the Bellman optimality backup until values settle |
| **Verify cost** | Minutes needed to investigate one alert: 5, 10, 20 or 40 |

---

## What to read next

| If you want to… | Read |
|---|---|
| Start labelling pairs today | **Document 0 — Labelling Handbook** |
| Understand the simulated world and its config | **Document 2 — The World** |
| Understand the six learning algorithms | **Document 3 — The Agents** |
| Understand how experiments are run and measured | **Document 4 — Running and Measuring** |
| Understand the RLHF machinery in detail | **Document 5 — RLHF and Labelling** |
| Know why every choice was made | **Document 6 — Decisions, Experiments, Results** |

---

*Part of the TriageRL onboarding set. Sources: `PROJECT_BRIEF.md`, `ARCHITECTURE.md`,
`ROADMAP.md`, `CONSTRAINTS.md`, `HANDOVER.md`, `DECISIONS.md`,
`docs/experiments/EXPERIMENT_LOG.md`, and the code itself. Every number quoted here was
taken from a project document that recorded a measured run — none are estimates.*
