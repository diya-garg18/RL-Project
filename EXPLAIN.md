# EXPLAIN.md — The whole project, in plain English

> **This is the file you read when you need to understand anything about this project.**
>
> It assumes you know nothing. No jargon without a translation. If something here can only be understood by someone who already understands it, it is written wrong — rewrite it.
>
> **Rule for every AI session:** whenever you add or change something, add or change the plain-English explanation here too, in the same session. A change that isn't explained here isn't finished.
>
> **Rule for the humans:** if you read a section here and still can't explain that part of the project out loud to a friend, the section is wrong. Say so and get it rewritten. That is the whole point of this file (Field Guide habit #15 — own the mental model).

**Last updated:** 2026-08-13 · Phase 0 in progress (generator built and calibrated) · written by Claude Fable 5

---

## Part 1 — What are we actually building, and why?

### The situation

Imagine an emergency room, but for computers.

A company runs security software that watches everything — logins, downloads, odd network traffic. Whenever something looks off, it fires an **alert**. A company might get 2,000 of these a day.

The problem: they have maybe 3 people to look at them, and each alert takes 10–20 minutes to check properly. So they can honestly investigate maybe 60 out of 2,000.

And here's the cruel part. Almost all of those alerts are nothing — someone logged in from a café, someone downloaded a big file for a normal reason. But hidden in there, maybe 2 or 3 a day, is a **real attacker**. Every hour that one sits unread in the queue, the attacker is inside the network taking more.

So it's an ER doctor with 200 people in the waiting room and time to see 6. Who do you see next? Get the order wrong and someone dies in the waiting room.

### What companies do today

Sort by the "severity" label the security software attached. High first. That's it — that's the whole strategy at most companies.

It's a weak strategy, because that label was decided by a software vendor who has never seen this company, doesn't know which servers actually matter, and doesn't know the analyst has 40 minutes left in their shift.

### What we're building

An AI that learns a better order — by practising.

### Why this is a reinforcement learning problem and not something simpler

Three reasons, and you should be able to give all three:

1. **Decisions affect the future.** Investigating one alert burns 20 minutes you can never use on another. This is not "classify each alert independently" — the choices are linked through a shared budget.
2. **Feedback is delayed.** You don't find out whether skipping an alert was a mistake until much later, possibly at the end of the shift. RL is specifically the tool for delayed consequences.
3. **There's no labelled "correct answer" to copy.** Nobody has a dataset of perfect triage orders to learn from. The agent has to discover a good strategy by trying and being scored.

---

## Part 2 — How does the learning work?

Think of it as a video game the AI plays over and over.

**The game:** You're the analyst. It's an 8-hour shift. Alerts pile into your queue. You pick one, spend time checking it, find out whether it was real or nothing, then pick the next. Shift ends. Score gets counted.

**The scoring:**

| What happened | Score |
|---|---|
| Caught a real attack | Big points — and *more* points the faster you caught it |
| Wasted 15 minutes on a false alarm | Small penalty |
| Auto-closed something that turned out to be real | Big penalty |
| A real attack sat in the queue all shift, never looked at | **Huge** penalty — that's the breach |

The AI plays this shift tens of thousands of times. Early on it's terrible — basically random. But it slowly notices patterns:

> *"When the queue is huge and I'm low on time, chasing the scary high-severity alerts actually loses me points — they're usually noise and they eat 20 minutes each. I score better clearing quick ones and watching the critical servers."*

That noticing-and-remembering is the learning. In our case it's mostly **Q-learning**, which is just: a big table where the AI writes down "in this kind of situation, this move tends to work out well," and keeps nudging those numbers as it gets more experience.

---

## Part 3 — The one clever design choice

The obvious design: the AI picks one alert out of the 400 waiting. That's 400 possible moves, and the number changes every second. Messy to code, messy to explain, and it breaks the simple table-based methods entirely.

**Our design:** the AI doesn't pick an *alert*. It picks a **strategy to use right now** — one of five:

1. Take the scariest-looking one (highest severity)
2. Take the one that's been waiting longest
3. Take the one on the most important server
4. Take a quick easy one (clear the backlog)
5. Mass-close a batch of obviously-harmless ones

Always 5 choices, no matter how big the queue gets.

**Why this matters so much:** the result is *readable by a human*. You can print out what the AI learned — "queue flooded + low time left → use strategy 4" — and a security manager could look at that table and agree or disagree. In this domain, that readability is worth more than the extra flexibility we gave up.

If you are asked one design question about this project in an interview, it will probably be this one.

---

## Part 4 — The human feedback part (the bit that makes this special)

Here's a genuine problem. Look back at the scoring table in Part 2. **How many points is a 3-hour delay worth?**

Nobody knows. Is catching a real attack 3 hours late worse than wasting 40 analyst-minutes on junk? There's no correct number. And if we just make numbers up, our AI is optimising *our made-up opinion* — the entire project would rest on a guess.

So instead of guessing, **we ask humans.**

We build a small web page showing two shifts side by side — same flood of alerts, two different AIs handling them. The person just clicks: *"the left one handled that better."* No numbers. Just a preference — which is something humans are genuinely good at, unlike assigning point values.

We collect about 300 of these clicks from ourselves, some classmates, and ideally a working security analyst.

Then we train a small model whose only job is to **predict which one a human would pick**. That model becomes our new scoring system. Now the AI is being trained toward *what real people think good triage looks like*, instead of toward numbers we invented on a Tuesday afternoon.

**This is how ChatGPT was trained to be helpful.** Same technique, shrunk to something two students can actually build. It's called **RLHF** — Reinforcement Learning from Human Feedback. It's 7 lectures of the course syllabus, and almost no student project ever actually does it.

---

## Part 5 — The part that makes us look genuinely good

Once the AI is chasing a *learned* score, it can start cheating — finding a loophole that scores brilliantly but behaves badly.

The obvious one here: strategy 5 (mass-close). Closing things makes the queue shrink fast, which *looks* like great work, while quietly burying a real attack.

So we go hunting for exactly that, on purpose. And whatever we find, **we write it down in the report instead of hiding it.**

This is called a **reward hacking audit**, and it's a real research concern in AI safety, not something we invented for marks. Pranav has done this once before — on ChurnLens he found his own pipeline was leaking test data and retracted a better-looking result for the honest one. Doing it once looks like luck. Doing it twice is a pattern.

---

## Part 6 — Words you'll hear, translated

| Word | What it actually means |
|---|---|
| **Agent** | The AI making decisions |
| **Environment** | The simulated shift — the queue, the clock, the alerts |
| **State** | A summary of the current situation ("queue is huge, 30 min left, one critical server involved") |
| **Action** | One of our 5 strategies |
| **Reward** | Points scored for what just happened |
| **Policy** | The AI's strategy — "in situation X, do Y". The thing we're trying to learn. |
| **Episode** | One complete 8-hour shift, start to finish |
| **MDP** (Markov Decision Process) | The formal name for "states, actions, rewards, and how they connect". The maths version of the game. |
| **Markov** | "The current state tells you everything you need — you don't need the full history." |
| **Q-value** | "How good is taking action A in situation S, counting all future consequences?" |
| **Bellman equation** | The rule saying a situation's value = immediate reward + value of wherever you land next |
| **γ (gamma) / discount** | How much we care about the future vs right now. 0.99 = very patient. |
| **ε (epsilon) / exploration** | How often the AI tries something random instead of its current best guess — so it doesn't get stuck on a mediocre strategy it found early |
| **Dynamic Programming** | Solving for the perfect strategy by calculation — only possible if you know exactly how the world works |
| **Model-free** | Learning by trial and error instead, without knowing the rules in advance. Q-learning is model-free. |
| **DQN** | Q-learning where a neural network replaces the table, so it can handle situations too detailed to list out |
| **Experience replay** | Storing past experiences and re-learning from them in random order, so training is stable |
| **Target network** | A frozen copy of the network used as a reference, so the AI isn't chasing its own constantly-moving estimate |
| **Policy gradient** | Learning the strategy directly instead of learning value estimates first |
| **RLHF** | Learning the scoring system itself from human preferences instead of writing it by hand |
| **Reward model** | The small model that predicts what a human would prefer. Our learned scorer. |
| **Bradley–Terry** | The specific maths for turning "A beat B" comparisons into scores. Same idea as chess Elo ratings. |
| **Reward hacking** | The AI finding a loophole that scores well but is actually bad |
| **Baseline** | A simple method we compare against, to prove the fancy method was worth it |
| **MTTD** | Mean Time To Detect — average delay before a real attack was found. Lower is better. |
| **Seed** | A number fixing the randomness, so a run can be repeated exactly |

---

## Part 7 — What's built so far

*This section grows every session. Newest at the top. For each thing built, answer all four: **what**, **where**, **why**, **how**.*

### Session 3 — 2026-08-14 — Claude Fable 5

**What:** The rest of the Phase 0 simulator: the game itself (`env.py`), the two ways of describing a situation to the AI (`state.py`), the five comparison strategies (`agents/baselines.py`), the loop that runs full shifts (`runner.py`), the scorekeeping (`evaluation/metrics.py`), a 7-test suite including the anti-cheating test, and the first results table.

**Where:** `src/soc_triage/` (env, state, runner, agents/, evaluation/), `tests/`, `scripts/run_baselines.py`, table in `results/baselines.md`.

**Why / how, per piece:**

- **`state.py`** answers "what does the AI get to see?" Two answers: a single number 0–575 (situation bucketed five ways — worst severity present, queue size, oldest wait, time left, most valuable server involved) for the table-based methods; and a 17-number vector (averages, extremes, queue composition) for the neural methods. One shared `bucket()` helper so every boundary works identically.
- **`env.py`** is the game. It owns the clock, the queue, and the hidden answers. Each turn: apply the chosen strategy to the current queue, spend the minutes, let newly-arrived alerts in, score the move. Three scoring details the brief left open are now fixed and explained (DECISIONS D-009): delay is counted when you *start* investigating; end-of-shift punishment only for real incidents whose deadline actually expired during the shift; a bulk-closed real incident is punished once, not twice.
- **The anti-cheating test** (`test_no_ground_truth_leakage`) walks a whole episode, flips the hidden truth on every alert, and proves both state encodings don't change by a single bit. A second test locks the snapshot's field list so nobody can quietly add a leaky field later.
- **`baselines.py`** — the five reference strategies, including the two that matter: **severity-sort** (the industry default, the one to beat) and the **oracle** (allowed to see the hidden answers — the cheating ceiling).
- **`runner.py` + `metrics.py`** — run shifts, write every episode to JSON, compute the five report metrics including a rupee-denominated composite cost whose assumptions are config values, not buried constants.

**The honest story of this session — the oracle needed debugging twice.** First version lost to severity-sort (0.72 vs 0.85 recall): a real incident that never becomes any rule's top pick can sit untouched all shift, and the oracle also had a loop where it endlessly tidied junk instead of clearing a path to a blocked incident. Fixed by "path-clearing": find the rule the incident is *closest* to topping, and pull along that path until it surfaces. Final: oracle 0.86, severity-sort 0.85. Even a cheating baseline is code, and code has bugs — we found ours by refusing to accept a nonsensical result (CONSTRAINTS #5 in action, on ourselves).

### Session 2 — 2026-08-13 — Claude Fable 5

**What:** The project's first real code: the config loader, the Alert record, the alert generator, and the calibration script that proves the generator behaves. Also: git repository created (6 commits), Python environment installed and version-pinned.

**Where:** `src/soc_triage/config.py`, `alerts.py`, `generator.py`; `scripts/calibrate_generator.py`; tuned numbers in `config/env_default.yaml`.

**Why each piece exists:**

- **`config.py`** reads the YAML file of tunable numbers and turns it into Python objects that cannot be modified after loading. If a number is missing or nonsense (probabilities that don't sum to 1, training and evaluation seeds overlapping), it refuses to start and tells you exactly which line of the YAML to fix. The seed-overlap check is our scientific-integrity rule #2 enforced by code rather than by trust.
- **`alerts.py`** defines what one alert *is*: eight facts (when it arrived, how severe it looks, which machine, how long it takes to check, what kind it is...) plus the two hidden answers — is it real, and how long until it becomes a breach. The hidden answers are for the scoring system only. The AI never sees them.
- **`generator.py`** manufactures one 8-hour shift of alerts from a seed number. Same seed, exact same shift, every time — which is what lets us later show two different strategies the *identical* day and compare fairly. Arrivals follow a Poisson process (the standard maths for "independent events at a steady average rate" — same model as calls hitting a call centre). Whether an alert is *real* is a weighted coin flip: base rate ~1.35%, multiplied up or down by the alert's type, its severity, and how important the machine is.
- **`calibrate_generator.py`** generates 100 shifts and measures whether the simulated world matches the world we claimed to build (see Part 8). We tuned the config until it did, and checked the result on two further batches of 100 shifts that were never used for tuning.

**How (the one formula to remember):**
`P(real) = base_rate × type_lift × severity_lift × asset_lift` — capped at 95% so nothing is ever a certainty.

**Honest note:** the original scaffold's YAML had a formatting bug (a list and a named key at the same indent level, which YAML forbids) — the very first run of the loader caught it. Fixed by nesting the action names under `actions.names`. The five actions themselves are unchanged.

### Session 1 — 2026-08-13 — Claude Opus 5

**What:** Nothing runnable yet — this session set up documentation and project structure only.

**Where:** All the `.md` files in the project root, plus empty `src/`, `tests/`, `config/`, `docs/` folders.

**Why:** The project will be built across many AI sessions over ~6 weeks. Each new session starts with no memory of the last one. These documents are the memory. Without them, session 8 would rebuild things session 3 already decided, and — more importantly — neither student would be able to explain the code in an interview six months later.

**How:** Followed the *AI Collaboration Field Guide*, which prescribes nine documents. Added this file (`EXPLAIN.md`) as a tenth, because all nine of the Field Guide's documents are technical, and there was no document meant for "explain it to me like I know nothing".

---

## Part 8 — Results so far

*Every headline number goes here in plain English as soon as it exists. Format: what we measured, what we got, what it means, and what could be wrong with it.*

### First baseline table — 2026-08-14 (E-002, eval seeds 101–105, mean over 5 shifts)

| strategy | % of real incidents caught in time | avg detection delay |
|---|---|---|
| **oracle (cheats)** | **86%** | 41 min |
| severity-sort (industry default) | 85% | 37 min |
| random | 46% | 79 min |
| cheapest-first | 47% | 55 min |
| FIFO (oldest first) | **20%** | 246 min |

**What it means:** the world behaves sensibly — the cheater wins, and sorting by severity is genuinely strong here (predicted consequence of our calibration: most real incidents carry the top severity label). The gap the learning agent must exploit is the 14% of incidents severity-sort misses, plus its wasted time and delays.

**Two honest caveats, written before anyone asks:**
1. **FIFO scored *below random*.** Not a bug — with time to work only ~20% of the queue, always taking the *oldest* alert means investigating things whose deadlines already passed (4-hour average delay). It's the cleanest illustration of why triage exists. The roadmap's exit line "random is clearly worst" was written before this was understood; the amendment is awaiting Diya/Pranav's sign-off.
2. **The oracle is a ceiling on average, not on every single day.** On one of the five evaluation shifts it caught one fewer incident than severity-sort — an incident arrived 16 minutes before closing time while the oracle was mid-investigation. Greedy, no crystal ball for arrivals. Say "upper bound in expectation".

### Generator calibration — 2026-08-13 — PASSED (checked by Diya)

**What we measured:** 100 simulated shifts (seeds 1000–1099, disjoint from training and evaluation seeds), ~16,900 alerts pooled.

| Number | Result | Target | Verdict |
|---|---|---|---|
| Alerts per shift | 168.7 | ~170 | ✓ |
| True-incident rate | 3.34% | 2.5–3.5% | ✓ |
| Real incidents per shift | 5.6 ± 2.6 | — | plausible |
| Pearson r (severity ↔ real) | 0.323 | 0.30–0.40 | ✓ |

**Robustness:** re-measured on two fresh 100-shift batches never used during tuning — rate 3.13% / r 0.311 (seeds 2000s) and rate 3.27% / r 0.317 (seeds 3000s). Stable, not a lucky sample.

**Final tuned values** (in `config/env_default.yaml`): `base_rate` 0.03 → 0.0135, `severity_lift` [0.1, 0.35, 2.4, 15.0], `asset_lift` [0.8, 1.2, 1.4]. The base rate had to drop because the lift multipliers, averaged over the alert population, inflate the effective rate by roughly 2.2×.

**What it means:** the simulated SOC now matches the story we tell about it — ~170 alerts a day of which ~5–6 are real, and the vendor severity label is *somewhat* informative but far from sufficient.

**What could be wrong with it / what it forces:** to reach r ≥ 0.30 with only ~3% of alerts being real, the maths forces real incidents to concentrate in the top severity: P(real) climbs from 0.2% (severity 0) to ~30% (severity 3), and about two-thirds of real incidents arrive at severity 3. Consequence: severity-sort will be a genuinely decent baseline — the remaining one-third of incidents hiding at lower severities, plus asset value and time pressure, are the gap a learned policy can exploit. This is a *built-in assumption*, not a discovery (see Part 9, item 2).

Planned first measurement (Phase 0 exit): the six baseline strategies compared on the same alert streams. Expectation — the oracle (which cheats by seeing the hidden answers) should be clearly best, and random clearly worst. If that doesn't happen, the simulator is broken.

---

## Part 9 — Things we know are wrong or unproven

*Honest limitations. Add to this list the moment one is discovered — never at the end.*

1. **The whole environment is simulated.** We have no real security-team data. Every result depends on our invented world being reasonable.
2. **We deliberately built in the assumption that severity labels are a weak predictor** of whether an alert is real. It's grounded in what the industry reports, but it *is* the mechanism that lets our AI beat severity-sorting. If severity were a perfect predictor, sorting by it would already be optimal and there'd be no project. This has to be stated in the report.
3. **Our human preference labels will come mostly from students**, not working analysts. So we're learning *our* idea of good triage, not a professional's.
4. **One analyst, one shift.** Real teams have multiple people, shift handovers, and alerts that group together into a single incident. We model none of that.
