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

### Session 6 (continued) — 2026-08-16 — Claude Opus 5 — turning the agent's knowledge into a picture

Everything the agent learned lives in a table of 2,880 numbers. Nobody can read that. So we turned it into a picture: for every situation the agent might face, which of its five options does it pick — laid out so you can see how its answer changes as the end of the shift approaches.

**There's a clear pattern, and it moves steadily in one direction.** Early in the shift, the agent mostly works alerts by severity, worst first — that's its choice in 35% of situations. By the final hour that's dropped to 15%. Meanwhile mass-dismissing batches of alerts climbs from 25% to 46%. Both trends move the same way across all three time periods, which makes a fluke unlikely.

**You can read that two ways, and we've written down both.** Generously: a real analyst under time pressure also triages more aggressively, so the agent has learned something human-like. Cynically: the scoring system only charges you for missed incidents at the very end of the shift, so emptying the queue cheaply right beforehand is a good way to dodge the bill — meaning this is the same loophole from earlier, getting worse exactly where it pays best. Both fit the evidence. We noted what would settle it rather than picking the version that flatters the project.

**One caveat that matters.** The agent only ever encounters about a fifth of the possible situations, and the final-hour column is based on just **13** of them. The trend across three time periods is reassuring, but the most interesting part of the headline figure has thirteen data points behind it, and we say so.

**And one thing that nearly went badly wrong.** For the four-fifths of situations the agent has never seen, it has no opinion at all. But the way the code breaks a tie means "no opinion" comes out looking identical to "definitely work the most severe alert first." Printed as-is, **79% of the picture would have been a confident recommendation the agent never made** — in a figure meant for a report and an exam. The agent now keeps a tally of how often it's been in each situation, purely so the picture can say "never been here" instead, with a test to keep it honest.

That's worth dwelling on, because nothing was broken. The tie-breaking code is correct and has a good reason to exist. It only became a lie at the moment it was *displayed*. Correct code can still produce a dishonest figure.

### Session 6 (continued) — 2026-08-16 — Claude Opus 5 — the first real result, and it's a mixed one

The learning agent finally met the real simulated SOC. It practised on 20,000 shifts, five separate times over, and only looked at the five "exam" shifts once, right at the end — so nothing it learned could be secretly tuned to the exam.

**What happened.** It earns far more reward than the sensible rule-of-thumb baseline: 271 against 154. But it catches **fewer** real incidents — 73% against 87%. The Phase 2 target was "beat the baseline at catching incidents", and it didn't. We are not moving the target to make it look like it did.

**Why it happened** is the interesting part, and we predicted it two phases ago. Looking at what the agent actually *does*, 62% of its actions are hitting the bulk-close button — dismissing batches of alerts as low-risk. That is the same shortcut the Phase 1 planner discovered, except that planner used it 97% of the time and this one uses it 62%. Two completely different algorithms, one that plans with a map and one that learns by trial and error, independently found the same loophole.

That is worth more than a passing grade would have been. It's direct evidence that the problem is **the reward we wrote by hand**, not any particular algorithm — which is exactly the argument for the human-feedback phase later in the project. If only the planner had cheated, you could blame the planner. When everything cheats, you have to blame the rules.

**And then we found something bigger, by accident.**

The agent was scoring much better on the exam shifts than on the practice ones. The obvious worry is that it had memorised the practice shifts, so we checked — and it hadn't. We tested the *baselines*, which learn nothing at all and can't memorise anything, on both sets of shifts. They showed the same gap. The rule-of-thumb baseline scores −79 on the practice shifts and +154 on the exam shifts. Even the theoretical best-possible agent gains 120 points just by switching sets.

**The five exam shifts are simply easier than the practice shifts.** And worse: shifts vary so much from one to the next that the random noise is several times larger than the differences we've been reporting. When we say "271 beats 154", the shift-to-shift wobble on that baseline is about ±325.

This is not a Phase 2 problem. It touches every comparison in the project so far, including the Phase 1 results we already wrote up. We have deliberately changed nothing, because fixing it means re-running everything and that is a decision for Pranav and Diya, not for the AI that found it.

There's a general lesson here that's worth more than the result. We only found this because a number looked *odd*, not because a test failed. Everything was passing. The project's rule that a surprising result is a bug report until proven otherwise is what turned "huh, that's a bit high" into the most important finding of the session.

### Session 6 (continued) — 2026-08-16 — Claude Opus 5 — the first algorithm that actually learns

Everything built before this point either planned using a map it was handed, or followed a fixed rule and never improved. This is the first agent that gets better from experience.

It's called **Q-learning**, and the idea is small enough to say in a sentence. The agent keeps a big table: one row per situation, one column per action, each cell holding its current guess at how good that action is in that situation. Every time it acts and sees what happens, it nudges one cell toward "the reward I just got, plus my own best guess about what comes next." That single line of arithmetic is the whole algorithm — everything else in the file is bookkeeping.

The clever part is the phrase *my own best guess about what comes next*. While it's learning, the agent deliberately does random things now and then, so it stumbles into options it would otherwise never try. But when it updates the table, it assumes it will play **perfectly** from the next step onward — not randomly. So it explores like an amateur and learns like an expert. That's what "off-policy" means, and it's the single difference from SARSA, the algorithm we write next. The two are easy to confuse, so we wrote a test whose only job is to make confusing them impossible.

**We wrote the tests before the algorithm, and watched them fail.** That ordering isn't ceremony. A test written afterwards passes on its first run, which tells you only that it matches what the code does — not that it would have caught the code being wrong.

On the two-state practice problem from earlier in the session, it reproduces the pen-and-paper answer to fourteen decimal places. More interestingly, it works out the right *strategy* after about ten practice runs, while the numbers themselves take about fifty runs to settle. Behaviour gets there before belief does — which is a genuinely useful thing to know, and a good viva answer.

**Two honest caveats.**

First: this has never been run on the real problem. Two situations versus 576, no randomness versus a lot of it. Passing here proves the arithmetic is right. It proves nothing about whether the agent is any good at triaging alerts.

Second, and more interesting: one result looked *too* good. Every random seed gave byte-identical answers — zero variation, which is normally the signature of randomness that isn't actually switched on. So we stopped and checked instead of celebrating. It turned out to be correct: the practice problem is completely predictable and has exactly one right answer, so different random routes all arrive at the same destination. We proved the routes really were different (the agents took visibly different action sequences early on) before accepting it. That check is now a permanent test, so nobody has to redo the investigation.

There was also a smaller lesson. The first version of the test said "close enough is within 1%." Measuring showed the real accuracy was about a trillion times better than that, meaning the test would have happily accepted a genuinely broken algorithm. Tightened. The general point: work out how accurate the thing actually is, then set the bar — don't pick a number that sounds reasonable and write it into a test.

### Session 6 — 2026-08-16 — Claude Opus 5 — Phase 2 started: the measuring stick

Phase 2 is where the project stops planning and starts *learning*. Three learning algorithms get written by hand — Monte Carlo, SARSA and Q-learning. Each one produces a big table of numbers saying "in this situation, this action is worth this much."

The problem is that the table has 576 rows and 5 columns, and nobody can check 2,880 numbers by hand. So how would we ever know an algorithm is right? The obvious answer — write all three and see if they agree — does not work. Three algorithms built on the same misunderstanding will agree with each other perfectly and all be wrong together. Agreement is not correctness.

So this session built a measuring stick first, before any of the learners.

It is a miniature version of the same problem, shrunk until a person can solve it with a pen: **two** situations instead of 576, and **two** things you can do instead of five. The two situations are a calm queue and a backlogged one. The two things you can do are wait, or work an alert. Doing the arithmetic by hand gives four numbers — how good each choice is in each situation — and, importantly, the best move is *different* in the two situations: wait when things are calm, work when there's a backlog. That difference matters, because it means a lazy algorithm that always picks the same action can't accidentally pass the test.

Those four numbers are now frozen in the code as the right answer. From here on, every learning algorithm has to reproduce them before we believe a word it says about the real problem.

Two details worth knowing:

**We checked the check.** A test that can't fail is worthless — it just makes you feel safe. So we deliberately typed wrong numbers in and confirmed the test noticed. It did, by a factor of roughly ten trillion. The correct answer registers as "off by 0.0000000000000018" (that's just the computer's rounding), and a wrong answer registers as "off by 0.1". There is no wrong answer that sneaks through.

**We got the design wrong twice first, and the reason is interesting.** The first two versions of the miniature problem were arithmetically neat, but the best action only beat the second-best by about 1%. That sounds fine and is actually terrible: a learning algorithm that's within 1% of the right answer is completely normal, and it would pick the wrong action about half the time, at random. The test would have failed sometimes and passed other times — and a test like that gets ignored rather than trusted. The third design gives the best action a much wider lead. The lesson generalises: a test needs its right answer to be *far away* from the wrong ones, not merely different from them.

Also swept a batch of stale documentation this session — the README still had the old computer's file path and claimed no code was written yet, and two other documents still described the project as an empty scaffold.

### Session 5 — 2026-08-16 — Claude Opus 5 — Phase 1 closed

**What:** Two things — a hand-solvable five-state example that proves our Bellman maths is right, and the decision that finally closed Phase 1.

**Where:** `src/soc_triage/mrp_example.py`, `tests/test_mrp_bellman.py`, `scripts/run_mrp_example.py`, `docs/features/FEATURE_001_mrp_worked_example.md`.

**Why the example was needed.** Session 4 left a gap nobody had named. Value iteration and policy iteration agreed with each other on 100% of states — but they share the same Bellman expression, so agreement proves they're *consistent*, not that they're *correct*. Two methods built on one wrong equation agree perfectly and are both wrong. There was no check against anything outside our own code.

**How we closed it.** Built the same problem at five states instead of 576 — small enough to solve with a pen in five minutes — worked the answer out by hand, then made the *real* solver reproduce it. It did, to fifteen decimal places. Details in Part 10; the interesting bit is that a quiet queue turns out to be worth more than a stale backlog, which is the whole intuition behind value functions in one comparison.

**The decision.** Phase 1's pass mark said "the DP agent must beat severity-sort at catching incidents". It can't, and it never could — it optimises reward, and catching incidents isn't what reward measures. We restated the criterion on total reward, kept the broken reward function deliberately, and promoted the reward hack from "problem" to "headline result". Reasoning in Part 9 and in D-012. Approved by Pranav; **Diya still needs to countersign**, since both Phase 0 amendments had her sign-off and this one should match that bar.

**What we did NOT do:** touch the reward function, or pre-emptively weaken Phase 2's pass mark even though it has the same weakness. Phase 2 gets run first and judged on real numbers, the same way this one was.

### Session 4 — 2026-08-14 — Claude Fable 5 — Phase 1 (Dynamic Programming)

**What:** The first "solve the MDP properly" phase — plus the first time our own agent cheated, which is a *good* thing that happened two phases early.

**Where:** `src/soc_triage/agents/dp.py`, `scripts/run_dp.py`, `config/training_default.yaml` (new `dp:` block), `results/dp_convergence.png`.

**The idea, plainly.** We don't know the exact rules of how the queue evolves, so we *learn them by watching*: run 50,000 random shifts, tally "from situation S, doing A, how often did we land in situation S′, and what reward came" — that gives an estimated map of the world (`P̂`, `R̂`). Then two classic algorithms squeeze the best strategy out of that map:
- **Value iteration** — repeatedly ask each situation "what's the best I can expect from here?" until the numbers stop moving. Converged after 1075 passes.
- **Policy iteration** — a different route to the same destination. It agreed with value iteration on **100%** of situations, which is the cross-check that says neither is buggy.

Both are written from scratch with plain loops (the assignment's whole point). The one number every reader should hold: a situation's value = *best action's* (immediate reward + how good the next situation is likely to be). That's the Bellman equation.

**The honest, important part — our agent found a loophole.** The DP policy scored the *highest reward of anything so far* (306, beating even the cheating oracle's 214) — and yet it's a **worse triage system than plain severity-sort**. It catches only 43% of real incidents (severity-sort catches 87%). How? It discovered that "bulk-close a batch of junk" earns a few tiny points and only costs 2 minutes, so it does that ~97% of the time as a way to *idle profitably*, occasionally darting out to grab an obvious high-severity incident, and simply lets more than half of all real incidents rot in the queue. The maths of our hand-written reward genuinely says this is optimal. No SOC manager on earth would agree.

**Why this is the single most valuable thing to happen yet.** The whole project's thesis (brief §3.5) is that hand-written rewards are secretly broken and you can't tell until something optimises them ruthlessly. We now have *proof, from our own code*: perfect planning against our reward produces a policy that games it. That is the airtight motivation for Phase 5 (learning the reward from human preferences instead). We didn't argue it — we caught it red-handed. Full write-up in EXPERIMENT_LOG E-004; it means the Phase 1 "beat severity-sort on recall" exit line needs a human decision before we move on (recorded in ROADMAP).

**One speed fix this session:** the alert generator was 82% of DP's runtime, so its per-alert random draws were replaced with vectorised batch draws (same distributions — recalibration re-confirmed 3.20% / r=0.321). 50k-episode estimation dropped from a projected 38 minutes to 1.2. The *learning* code stays plain loops; only the simulator was sped up (that's the rule — CONSTRAINTS #14).

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

### DP policy — 2026-08-14 (E-004) — the reward-hacking result

| strategy | real incidents caught in time | avg detection delay | total reward (what it optimises) |
|---|---|---|---|
| **DP (solved our reward exactly)** | **43%** | 6 min | **306** ← highest |
| oracle (cheats, sees answers) | 77% | 16 min | 214 |
| severity-sort (industry default) | 87% | 23 min | 154 |

**What it means:** the two right-hand columns disagree on purpose, and *that disagreement is the finding*. DP wins the reward column decisively and loses the "actually good triage" column badly. It reward-hacks: bulk-close ~97% of the time as profitable idling, snipe a few obvious incidents, ignore the rest. **Our hand-written reward is provably game-able — demonstrated by our own optimal planner, not asserted.** This is the strongest possible motivation for the RLHF phase and belongs in the report as a headline, not a footnote.

**Caveat to state:** the DP policy is optimal *for the estimated model* (built from 50k random rollouts, covering 133 of 576 states), not for the true environment — though we confirmed the hack reproduces in the true environment too, so it's real, not an estimation artefact.

### First baseline table — 2026-08-14 (E-002, eval seeds 101–105, mean over 5 shifts)

> **Superseded by E-003** (same agents, 30 seeds instead of 5). Kept because we don't delete results (CONSTRAINTS #4) and because *why* it was superseded is instructive: the oracle's 86%-vs-85% recall win over severity-sort below turned out to be five-seed noise. At 30 seeds severity-sort actually leads on recall (0.826 vs 0.799), and the oracle's real advantage shows up on total reward (145 vs 51). Read the numbers below as history, and E-003 as the truth.

| strategy | % of real incidents caught in time | avg detection delay |
|---|---|---|
| **oracle (cheats)** | **86%** | 41 min |
| severity-sort (industry default) | 85% | 37 min |
| random | 46% | 79 min |
| cheapest-first | 47% | 55 min |
| FIFO (oldest first) | **20%** | 246 min |

**What it means:** the world behaves sensibly — the cheater wins, and sorting by severity is genuinely strong here (predicted consequence of our calibration: most real incidents carry the top severity label). The gap the learning agent must exploit is the 14% of incidents severity-sort misses, plus its wasted time and delays.

**Two honest caveats, written before anyone asks:**
1. **FIFO scored *below random*.** Not a bug — with time to work only ~20% of the queue, always taking the *oldest* alert means investigating things whose deadlines already passed (4-hour average delay). It's the cleanest illustration of why triage exists. The roadmap's exit line "random is clearly worst" was written before this was understood; **that amendment was approved by Diya on 2026-08-14** and the criterion now reads "random and FIFO sit clearly at the bottom on recall". Reconfirmed at 30 seeds in E-003 (FIFO 0.141).
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

**What could be wrong with it / what it forces:** to reach r ≥ 0.30 with only ~3% of alerts being real, the maths forces real incidents to concentrate in the top severity: P(real) climbs from 0.2% (severity 0) to ~30% (severity 3), and about two-thirds of real incidents arrive at severity 3. Consequence: severity-sort will be a genuinely decent baseline — the remaining one-third of incidents hiding at lower severities, plus asset value and time pressure, are the gap a learned policy can exploit. This is a *built-in assumption*, not a discovery (see Part 11, item 2).

### Bellman check — 2026-08-16 (E-005)

**What we measured:** whether the equation inside our dynamic-programming code is the textbook one.

Until this point our only evidence was that two of our own methods agreed with each other — which proves consistency, not correctness. Two methods built on the same wrong equation would agree perfectly and both be wrong. So we built a five-state toy version of the problem, small enough to solve with a pen, worked out the answer by hand, and demanded the real code reproduce it.

| state | value, by hand | value, from our code |
|---|---|---|
| quiet queue | 4.727272… (exactly 52/11) | 4.727273 |
| stale backlog | −4 | −4.000000 |
| actively investigating | +20 | +20.000000 |
| incident caught | 0 | 0.000000 |
| incident missed | 0 | 0.000000 |

Agreement to 15 significant figures (largest gap 7.11e-15 — floating-point dust).

**What it means:** the maths in `agents/dp.py` is genuinely the Bellman equation from the textbook, verified against a human's paper answer rather than against itself. Full derivation in Part 10.

**What it doesn't mean:** it says nothing about whether our *estimated* 576-state model is accurate — that's a sampling question, still bounded by the 133-of-576 coverage figure and the caveats above. This check covers the solver, not the data it's fed.

---

## Part 9 — Reward hacking, and why we kept the bug

*This is the most important thing Phase 1 produced. If you read one section, read this one.*

**The short version.** We wrote a reward function by hand — points for catching real incidents quickly, penalties for wasting time and for missing things. Then we used dynamic programming to compute the *mathematically perfect* strategy for that reward. The perfect strategy turned out to be a terrible security practice, and we did not fix it on purpose.

**What the agent found.** One of the five actions is "bulk-close" — sweep up to ten obviously-junk alerts off the queue in one go. It's there to model the hygiene work real analysts do. We gave it a small positive reward (+0.5 per junk alert closed) because clearing noise genuinely is useful, and it costs 2 minutes.

Do the arithmetic the way a machine does: bulk-close is a small *guaranteed profit* that also passes time. Investigating an alert costs 5–30 minutes and usually turns up nothing, because 97% of alerts are false alarms. So the optimal policy is: bulk-close over and over as paid waiting, occasionally snipe an alert that looks obviously high-severity, and simply never work the rest of the queue.

That policy earns **306 reward points** — more than the oracle that cheats by seeing the hidden answers (214), and double the industry-standard severity-sort (154). It also **abandons 57% of the real intrusions**. No security manager on earth would accept it.

**Why we're certain it's real and not a coding error.** We checked before we celebrated, which is a rule in this project (CONSTRAINTS #5). Three checks: the reward arithmetic reconciles exactly against per-step breakdowns; the behaviour reproduces in the *true* environment, not just the estimated model the agent was planning against; and the exploit follows from three reward rules we can each point at (bulk-close pays, misses are only charged if the deadline expires during the shift, and instant catches pay maximum). The reward function is doing precisely what we wrote. We just wrote something we didn't mean.

**Why we didn't fix it.** This is the part that surprises people. Our project brief planned for a gameable reward from the start — because the honest answer to "how many wasted analyst-minutes equal one missed breach?" is *nobody knows*. There is no correct number. Any reward we write by hand will be wrong in some direction, and a sufficiently good optimiser will find that wrongness and exploit it.

That is the entire argument for the RLHF phase: instead of guessing the reward, learn it from humans comparing pairs of shifts and saying which they'd rather have happen. Patching the bulk-close loophole would just move the exploit somewhere we hadn't thought to look, and it would delete the motivation for the most interesting half of the project.

So the reward stays, the hack goes in the report as a headline result, and Phase 5 is where we try to fix it properly.

**What we changed instead.** The Phase 1 pass/fail criterion originally said "the DP agent must beat severity-sort at catching incidents." That was the wrong test — we'd asked a reward-maximiser to win at a measurement it isn't trying to maximise. We restated the criterion on total reward (the thing it actually optimises) and now always report the catch-rate beside it, so the reward number can never be quoted alone and mislead someone. That's D-012.

**What to watch for next.** Every future agent — Q-learning, DQN, REINFORCE — optimises this same reward. Expect the same hack to reappear. If it does, that's not three failures; it's one finding confirmed three times, and it's evidence the problem lives in the reward rather than in any particular algorithm.

---

## Part 10 — The worked examples (how we know our maths is right)

> There are two of them now, and they check different things. The five-state one (below) checks the maths behind *planning* — working out how good each **situation** is. The two-state one (at the end of this part) checks the maths behind *learning* — working out how good each **action in each situation** is. The first guards Phase 1's code; the second guards Phase 2's.

### The five-state worked example

**The problem.** Our dynamic-programming code solves a problem with 576 states. Nobody can check 576 states by hand. We had two methods (value iteration and policy iteration) that agreed with each other 100% — but they share the same underlying equation, so if that equation were wrong, both would be confidently, identically wrong.

**The fix.** Build the smallest possible version of the same problem and solve it on paper.

Five states describing an alert's life: **quiet queue**, **stale backlog**, **being investigated**, **caught**, **missed**. From quiet, things stay quiet half the time, or drift into backlog or investigation. From backlog you catch the incident 40% of the time. From investigation you catch it 80% of the time — that contrast is the point. Caught and missed are endings.

Rewards in miniature mirror the real ones: a minute of clock costs a little; catching pays (+30 if fast, +20 if late); missing costs −20.

**The Bellman equation**, which is the one idea underneath every method in this project:

> *The value of being somewhere = what you get immediately + (discount) × the average value of wherever you end up next.*

Written out: `V(s) = R(s) + γ · Σ P(s'|s) · V(s')`

Applying it by hand takes about five minutes. Endings are worth 0 (nothing more happens). Backlog is worth −4, investigation +20. Quiet is the interesting one, because from quiet you might stay quiet — so its value appears on both sides of its own equation and you have to solve for it:

```
V(quiet) = 2.6 + 0.45 × V(quiet)   ⟹   0.55 × V(quiet) = 2.6   ⟹   V(quiet) = 52/11 ≈ 4.73
```

**The result worth understanding.** A quiet queue is worth **more** (+4.73) than a stale backlog (−4), even though sitting quiet costs you a point every minute and sitting on a backlog costs nothing directly. Why? Because from quiet there's a decent chance of ending up investigating, which is worth +20 — whereas a backlog is a coin-flip that lands on "missed" more often than "caught". The value of a situation is about where it *leads*, not what it costs right now. That single comparison is what a value function is for, and it's the intuition every algorithm in Phases 2–4 is built on.

**The verification.** We then fed this same five-state problem to the actual 576-state solver (by giving it five identical actions, so its "pick the best action" step has nothing to choose between and collapses to the plain Bellman equation). It returned 4.727273, −4, +20, 0, 0 — our paper answer, to fifteen decimal places.

So the equation in our code is the textbook equation. Verified from outside, not by the code agreeing with itself.

*Full derivation: `docs/features/FEATURE_001_mrp_worked_example.md`. Run it yourself: `python scripts/run_mrp_example.py`.*

---

### The two-state worked example (added session 6, for Phase 2)

The five-state example above answers "how good is it to be *here*?" That was the right question for Phase 1. Phase 2's algorithms answer a harder one: "how good is it to *do this particular thing* from here?" — one number per situation-and-action pair. The five-state example cannot check that, because it has no actions in it at all. Hence a second, differently-shaped example.

**The setup.** Two situations and two choices.

| | If you WAIT | If you WORK |
|---|---|---|
| **Queue is calm** | stay calm, **+1** | you waste effort and a backlog builds, **−5** |
| **Queue is backlogged** | it stays backlogged, **−1** | you clear it and things are calm again, **+4** |

Future rewards count for 90% of present ones (that's the "discount" — a point tomorrow is worth 0.9 today).

**The answer, by hand.** Being in a calm queue and staying there earns +1 every step, forever, each one worth 90% of the last. That's a geometric series and it adds to exactly **10**. Being backlogged and clearing it earns +4 now and then puts you in the calm situation, so it's worth 4 + 0.9 × 10 = exactly **13**. From there the four action-values fall out:

| | WAIT | WORK |
|---|---|---|
| **Calm** | **10.0** ← best | 6.7 |
| **Backlogged** | 10.7 | **13.0** ← best |

**The bit that looks wrong.** Being backlogged (13) scores *higher* than being calm (10), even though backlogged is obviously the worse place to be. This trips up nearly everyone the first time and it's worth understanding, because it is the single most common misreading of what these numbers mean.

The number is not a rating of how nice the situation is. It's the total future earnings available from there. From the backlog, one clean-up pays +4 **and** hands you the calm queue with all of its future earnings still intact — so you collect the bonus *on top of* everything the calm queue was already worth. What you cannot do is farm it: going back to the backlog on purpose costs −5, which is more than the 3.3 you'd gain. The arithmetic forbids the loop, which is exactly why the correct strategy is "stay calm."

That last check was not optional, incidentally. Phase 1's big finding was that our main reward system *can* be gamed by an agent that games it (see Part 9). So the miniature example was checked for the same weakness before we agreed to trust it. It doesn't have one.

**The trap hidden in it.** If you follow the best strategy, you stay in the calm situation and never visit the backlogged one — so you never learn anything about it. The only way a learning algorithm ever sees half of this problem is by occasionally doing something other than the best-known thing on purpose. That's called exploration, and this tiny example demonstrates in four numbers why it isn't optional. Any algorithm we test with exploration switched off will fail here, and it'll fail for that reason rather than because its maths is wrong — which is a genuinely confusing thing to debug, so it's written down in the test checklist.

---

## Part 11 — Things we know are wrong or unproven

*Honest limitations. Add to this list the moment one is discovered — never at the end.*

1. **The whole environment is simulated.** We have no real security-team data. Every result depends on our invented world being reasonable.
2. **We deliberately built in the assumption that severity labels are a weak predictor** of whether an alert is real. It's grounded in what the industry reports, but it *is* the mechanism that lets our AI beat severity-sorting. If severity were a perfect predictor, sorting by it would already be optimal and there'd be no project. This has to be stated in the report.
3. **Our human preference labels will come mostly from students**, not working analysts. So we're learning *our* idea of good triage, not a professional's.
4. **One analyst, one shift.** Real teams have multiple people, shift handovers, and alerts that group together into a single incident. We model none of that.
