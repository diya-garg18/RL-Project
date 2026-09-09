# Document 0 — The Labelling Handbook

**How to collect the 300 preference judgements that Phase 5 exists to produce**

> **Read Document 1 first** if you have not. This handbook assumes you know what the
> project is and why human preferences are being collected. It answers a narrower
> question: *what do I actually do, and how do I not get it wrong?*
>
> **Time needed:** about 100 minutes per person, and it does not have to be in one
> sitting. The page remembers where you stopped.
>
> **Who does this:** Pranav (`L1`) and Diya (`L2`). Both of you. That is not optional —
> the whole κ measurement depends on two people independently judging the same 50 pairs.

---

## 1. Why you are doing this at all

The reward function this project has been optimising so far is **made up**. Catching a
real incident is worth 100 points; missing one costs 200. Nobody knows if that ratio is
right. Nobody *can* know — ask an experienced security analyst "how many wasted minutes
equal one missed breach?" and there is no answer to give.

So we stop asking for numbers and start asking for comparisons:

> *"Here are two shifts. Same alerts, handled two different ways. Which went better?"*

You will answer that question 175 times. Those 175 answers each (350 judgements total,
300 distinct pairs) become the training data for a reward model that replaces the made-up
numbers with something learned from actual human judgement.

**This is the point of the entire project.** Phases 1 through 4 built agents that
optimise our opinion. Phase 5 replaces our opinion with your judgement. Everything before
this was preparation.

---

## 2. Before you start — the five-minute setup

### 2.1 Get the pair file

The 300 pairs live in `results/rlhf/pairs.json`, which is **gitignored** — so it is not
in a fresh clone. You either copy it across from the machine that has it, or regenerate
it:

```powershell
python scripts/generate_pairs.py --write
```

That takes under a minute and is deterministic: the same records always produce
byte-identical pairs.

> ### ⚠️ The one thing that can destroy your work
>
> **Do NOT regenerate `pairs.json` once labelling has started.**
>
> Labels are stored against `pair_id`. A rebuild renumbers the pairs, so every label you
> have already collected would silently point at a *different comparison than the one you
> judged*. Nothing would error. The data would just be wrong.
>
> The script refuses to overwrite an existing `pairs.json` without `--force` for exactly
> this reason. **If you ever find yourself about to type `--force`, stop and ask.**

### 2.2 Launch the page

Pranav:
```powershell
python scripts/label_ui.py --labeller L1
```

Diya:
```powershell
python scripts/label_ui.py --labeller L2
```

You will see something like:

```
labelling as L1 — 300 pairs loaded, answers going to results\rlhf\labels.db
open http://127.0.0.1:8000/ — ctrl-c to stop
```

Open that address in a browser. That is it.

**If you both want to label on the same machine at the same time**, give the second one
its own port:

```powershell
python scripts/label_ui.py --labeller L2 --port 8001
```

### 2.3 Why the labeller id is on the command line and not on the page

There is no name box on the screen, and there never will be. Two reasons:

1. **The database has nowhere to put a real name.** Labels are anonymous by design
   (`CONSTRAINTS #23`) — an opaque id, a choice, a timestamp, nothing else.
2. **A stale value in a text box is a silent disaster.** If Diya sat down at a browser
   still showing `L1`, her 175 judgements would be recorded as Pranav's. Cohen's κ would
   then be comparing one person against themselves, come out beautifully high, and be
   completely meaningless.

The id is fixed when the server starts and the browser cannot override it. If you type an
id that is not in the config, the launcher refuses to start rather than writing rows
under an id nobody recognises.

---

## 3. What you are looking at

Each screen shows **two shifts side by side** — left pane and right pane — plus a
progress line and three buttons.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Which shift went better?                          pair 14 of 175    │
├───────────────────────────────┬──────────────────────────────────────┤
│            LEFT               │              RIGHT                   │
│                               │                                      │
│  Timeline of actions          │   Timeline of actions                │
│  ┌─────────────────────────┐  │   ┌──────────────────────────────┐   │
│  │ 000  PULL_HIGHEST_SEV   │  │   │ 000  PULL_OLDEST             │   │
│  │      sev 3, crit 2,     │  │   │      sev 1, crit 0,          │   │
│  │      40 min             │  │   │      10 min                  │   │
│  │ 040  BULK_CLOSE (7)     │  │   │ 010  PULL_OLDEST             │   │
│  │ 042  PULL_HIGHEST_SEV   │  │   │      ...                     │   │
│  │      ...                │  │   │                              │   │
│  └─────────────────────────┘  │   └──────────────────────────────┘   │
│                               │                                      │
│  Outcome                      │   Outcome                            │
│   • caught 4 of 6 incidents   │    • caught 2 of 6 incidents         │
│   • crown-jewel missed: 0     │    • crown-jewel missed: 2           │
│   • mean time to detect 34m   │    • mean time to detect 19m         │
│   • wasted 180 minutes        │    • wasted 95 minutes               │
│                               │                                      │
├───────────────────────────────┴──────────────────────────────────────┤
│              [ LEFT ]      [ TIE ]      [ RIGHT ]                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.1 Both sides are the same shift

This is the most important thing to understand about the display.

**The two panes are the same alert stream.** Same alerts, same arrival times, same hidden
truth about which ones were real. The only difference is the policy that handled them.

That is deliberate. If the two sides had *different* alert streams, you could never tell
whether one side did better because it was smarter or because it simply got an easier
shift. Holding the stream fixed removes that ambiguity completely.

So when you compare, you are genuinely comparing **decisions**, not luck.

### 3.2 You are shown the ground truth

The display tells you which alerts were really incidents and which were false alarms.

**This is intentional and it is not cheating.** You are judging *outcomes* — "did this
shift go well?" — not playing the game yourself and guessing which alerts were real. The
agent never sees ground truth; you do, because you are the judge, not a player.

### 3.3 You are never shown a reward number

Not the total, not per-step, not a breakdown. Every reward field is stripped out in code
before it reaches the page, and there are two tests whose only job is to fail if one ever
reappears.

**Why:** if you could see the hand-written reward, you would simply agree with it. The
model trained on your preferences would then rediscover `config/env_default.yaml` at
considerable expense, and Phase 5 would have measured nothing.

### 3.4 You are never told which policy is which

No policy names reach the pair file. You do not know whether you are looking at the DQN,
SARSA, or the plain severity-sort baseline.

**Why:** "oh, this one is the neural network" is a bias. Blinding removes it.

There *is* a `pairs_key.json` file that maps pairs back to policy names — needed later for
analysis — and the labelling page **refuses to open it by name**. That guard exists
because the realistic version of this mistake is not malice, it is copying the wrong file
into the wrong folder.

---

## 4. How to judge — the actual guidance

### 4.1 The question

> **"If I were the security manager for this organisation, which of these two shifts would
> I rather my team had worked?"**

That is the whole question. Not "which followed a better rule", not "which looks more
sophisticated" — which *outcome* would you rather have had.

### 4.2 What generally matters, roughly in order

| Consideration | Why it matters |
|---|---|
| **Real incidents caught vs missed** | This is the job. A shift that misses real intrusions has failed regardless of how tidy it looks. |
| **Crown-jewel incidents missed** | Missing an incident on a critical asset is much worse than missing one on a dev box. |
| **How fast incidents were caught** | Every hour undetected is another hour of attacker dwell time. |
| **Wasted minutes** | Time on false positives is time not spent on real ones — but it is the *least* important of these. |

### 4.3 The judgement calls that are genuinely hard

You will hit these. There is no official answer, and **your honest opinion is the data we
want** — this is precisely why the reward could not be written by hand.

**"Left caught 4 incidents but wasted 180 minutes. Right caught 2 and wasted 95."**
More catches, more waste. Probably left — catching real intrusions is the job — but if
right caught the two *critical* ones and left caught four trivial ones, reconsider.

**"Left has a beautiful MTTD of 19 minutes but only caught 2 of 6."**
Be suspicious of a great MTTD. Catching only the three easiest incidents very quickly
produces a lovely average and a terrible outcome. Recall usually matters more.

**"Both caught 3 of 6, both missed one crown jewel, times are similar."**
That is a tie. Press tie.

### 4.4 When to press TIE

Press **tie** when you genuinely cannot separate them, or when the two shifts are good
and bad in ways that honestly balance out.

**Do not use tie as an escape hatch when a pair is merely hard.** If you lean even
slightly, record the lean — a weak signal is still signal, and the reward model can use
it. But an artificially forced choice on a genuine coin-flip is noise, and noise is
worse than an honest tie.

There is no scoring on your consistency. Nobody is grading your clicks. **The most useful
thing you can do is answer honestly and quickly.**

### 4.5 Speed

Aim for roughly **20 seconds per pair**. The page times you, but that timer is
diagnostic — it is not a target and nothing is penalised.

Do not agonise. Your first instinct on "which shift went better" is the judgement we are
trying to capture; a two-minute deliberation is a different, more analytical thing, and
it is not what a security manager glancing at two shift reports would produce.

### 4.6 Do not discuss pairs with each other while labelling

**This is the one rule that matters for the science.**

50 of your pairs are the *same* pairs. Cohen's κ measures how much two people
**independently** agree. If you compare notes, calibrate on each other, or label sitting
at the same desk talking, κ measures your conversation, not your judgement — and it will
come out flatteringly high and be worthless.

Label separately. Compare afterwards. The disagreements are the interesting part.

---

## 5. Practical matters

### 5.1 How your 175 pairs are chosen

```
   300 pairs total
   │
   ├── 50 pairs  ──────────────►  BOTH of you label these
   │                              (this is what κ is computed on)
   │
   └── 250 pairs ──► dealt alternately ──►  125 to L1
                                        └─► 125 to L2

   Each person: 50 shared + 125 own = 175 judgements
   Together:    350 judgements over 300 distinct pairs
```

The 50 shared pairs are **not marked on screen**. You cannot tell which ones they are,
which is the point — you would label them differently if you knew.

### 5.2 Stopping and resuming

Close the tab whenever you like. Ctrl-C the server. Come back tomorrow.

When you reopen the page it is already at the next pair you have not answered. "Where I
got to" is a fact stored in the database, not in the running process, so nothing is lost
by stopping — even abruptly.

### 5.3 If you double-click, or refresh after answering

Nothing bad happens. The first answer stands and the page moves on. The database has a
uniqueness rule preventing the same person labelling the same pair twice, and the app
treats hitting it as success rather than an error — interrupting your session over work
that was never at risk would be worse than useless.

**You cannot change an answer once submitted.** There is deliberately no edit and no
delete path anywhere in the storage layer. If you realise you misclicked, note the pair
number and mention it — do not try to fix it yourself.

### 5.4 Where your answers go

`results/rlhf/labels.db` — a SQLite file.

> ### ⚠️ This file is irreplaceable
>
> Every other artefact in `results/` can be regenerated by re-running a script. **This one
> cannot.** It is 350 human judgements that took two people 100 minutes each to produce.
>
> It is gitignored (`CONSTRAINTS #19`), so it will not be committed. **Back it up
> manually** — copy it somewhere safe after each sitting.

### 5.5 Common problems

| Symptom | What it means |
|---|---|
| `cannot start: no pair file at ...` | `pairs.json` has not been generated or copied. See §2.1. |
| `unknown labeller 'L3'` | Only `L1` and `L2` are configured. Check the spelling. |
| The page says you are done, but the count is short | You are looking at *your* 175, not all 300. The other labeller has their own. |
| Port already in use | The other labeller's server is running. Use `--port 8001`. |
| A refusal mentioning `pairs_key.json` | You pointed it at the answer key. Use `pairs.json`. |

---

## 6. What happens to your answers afterwards

### 6.1 Cohen's κ, from the 50 shared pairs

```
        raw agreement − chance agreement
  κ  =  ────────────────────────────────
              1 − chance agreement
```

Run it with:

```powershell
python scripts/report_kappa.py
```

**Why subtract chance?** Suppose you both pressed "left" 90% of the time out of habit.
You would agree over 80% of the time while sharing no actual judgement whatsoever. The
subtraction strips that out, leaving only agreement that exceeds coincidence.

The conventional reading (Landis & Koch, 1977 — a rule of thumb, not a law of nature):

| κ | Conventional label |
|---|---|
| < 0 | poor — worse than chance |
| 0.0 – 0.2 | slight |
| 0.2 – 0.4 | fair |
| 0.4 – 0.6 | moderate |
| 0.6 – 0.8 | substantial |
| > 0.8 | almost perfect |

> **If κ comes out low, that is a result — not a failure, and not something to re-label
> until it improves.**
>
> A low κ would mean that "good triage" is genuinely ill-defined even between the two
> people who built the system. That is a real, reportable finding about the problem
> domain, and `PROJECT_BRIEF.md` §6.2 commits in advance to reporting it. Deciding what
> counts as a good result *before* seeing the number is what stops it becoming a number
> we tuned.

The code also refuses to flatter you. If you both used only one category throughout, the
formula is 0/0 — and rather than return a comforting 1.0, it returns "undefined" with the
reason attached. Two people who always press the same button agree by construction; the
data says nothing about whether they would agree anywhere else.

### 6.2 Then the reward model (Phase 5b)

Your 300 labelled pairs train a small neural network `r̂(state, action)` using the
**Bradley–Terry** model:

```
                        exp( Σ r̂(s,a) over shift A )
  P(A preferred to B) = ─────────────────────────────────
                        exp( Σ over A ) + exp( Σ over B )
```

In words: score every step of both shifts, add up each side, and the side with the higher
total should be the one you picked. Train it by pushing its prediction towards your actual
click.

80% of the pairs train it; 20% are held out to check it did not merely memorise.

### 6.3 Then the headline result (Phase 5c)

Re-train the agents using `r̂` instead of the invented reward, and ask: **do humans prefer
the RLHF policy over the hand-reward policy on fresh pairs nobody has labelled before?**

That graph is the point of the entire project, and it does not exist until you label.

---

## 7. The one-page checklist

```
BEFORE
  □ pairs.json exists (do NOT regenerate if labelling has begun)
  □ launched with the right --labeller id
  □ Document 1 read, so you know what you are judging

DURING
  □ compare outcomes, not tidiness
  □ real incidents caught > everything else
  □ crown-jewel misses weigh heavily
  □ suspicious of a great MTTD with poor recall
  □ tie only for genuine ties, not for hard ones
  □ ~20 seconds each, first instinct
  □ NOT discussing pairs with the other labeller

AFTER
  □ back up results/rlhf/labels.db
  □ when both are done: python scripts/report_kappa.py
  □ record the κ number in docs/experiments/EXPERIMENT_LOG.md
  □ report it whatever it is
```

---

*Part of the TriageRL onboarding set. Sources: `scripts/label_ui.py`,
`src/soc_triage/labelling/{app,queue,render}.py`, `src/soc_triage/rlhf/{summary,agreement,store}.py`,
`config/training_default.yaml`, `PROJECT_BRIEF.md` §6, decisions D-037 to D-046.
Document 5 covers the same machinery from the code side.*
