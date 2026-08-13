# INTERVIEW_PREP.md — What you must know cold

> The project exists to be explained. This file is the exam.
>
> Work through it in Week 6, but **read it in Week 1** so you build with the right end in mind.

---

## Part A — The four functions to write from memory

An interviewer says "you mentioned Q-learning — write the update." You need to produce these on a whiteboard, without notes, and explain every symbol.

### 1. The Q-learning update

```python
def update(self, s, a, r, s_next, done):
    """Off-policy TD control. Sutton & Barto 2nd ed., §6.5.

    The target uses max_a' Q(s', a') — the value of the BEST next action —
    regardless of what the policy will actually do next. That 'regardless'
    is exactly what makes Q-learning off-policy.
    """
    best_next = 0.0 if done else np.max(self.Q[s_next])
    td_target = r + self.gamma * best_next
    td_error  = td_target - self.Q[s, a]
    self.Q[s, a] += self.alpha * td_error
```

**Be ready for:** Why `0.0` when done? (No future after a terminal state.) What's the difference from SARSA? (SARSA uses `Q[s_next, a_next]` — the action actually taken — making it on-policy.) What is γ doing? What happens if α is too large? Where's the Bellman equation here? (The TD target *is* the sampled right-hand side of the Bellman optimality equation.)

### 2. ε-greedy action selection

```python
def act(self, s):
    """Explore with probability epsilon, otherwise exploit the current best.

    Without exploration the agent locks onto the first decent strategy it
    finds and never discovers better ones — in our env, that's usually
    'always pull highest severity'.
    """
    if self.rng.random() < self.epsilon:
        return self.rng.integers(self.n_actions)
    return int(np.argmax(self.Q[s]))

def decay_epsilon(self):
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

**Be ready for:** Why decay it? Why a floor instead of decaying to zero? What else could you use? (Softmax/Boltzmann, UCB, optimistic initialisation.) What breaks if ε = 0 from the start?

### 3. The experience replay buffer

```python
class ReplayBuffer:
    """Fixed-size circular buffer of transitions.

    Two jobs:
      1. Breaks temporal correlation — consecutive steps in one shift are
         highly correlated, and SGD assumes roughly i.i.d. samples.
      2. Reuses each experience many times, which matters because
         environment steps are the expensive resource.
    """
    def __init__(self, capacity, rng):
        self.buf, self.capacity, self.pos, self.rng = [], capacity, 0, rng

    def push(self, s, a, r, s_next, done):
        if len(self.buf) < self.capacity:
            self.buf.append(None)
        self.buf[self.pos] = (s, a, r, s_next, done)
        self.pos = (self.pos + 1) % self.capacity   # overwrite oldest

    def sample(self, batch_size):
        idx = self.rng.choice(len(self.buf), batch_size, replace=False)
        return [self.buf[i] for i in idx]
```

**Be ready for:** Why not just train on the most recent transitions? What's prioritised experience replay and when would you use it? What does the target network do that replay doesn't? (Replay fixes correlated *inputs*; the target network fixes a *moving target* — you're regressing toward a value the same network keeps changing.)

### 4. The Bradley–Terry preference loss (the RLHF core)

```python
def preference_loss(reward_model, traj_a, traj_b, label):
    """Human said one trajectory was better. Turn that into a gradient.

    Model each trajectory's score as the sum of learned per-step rewards.
    Then P(A preferred over B) = sigmoid(R_A - R_B), and we fit that
    probability to the human's actual choice with cross-entropy.

    label: 1.0 if the human preferred A, 0.0 if B.
    Same maths as Elo ratings in chess.
    """
    R_a = reward_model(traj_a).sum()
    R_b = reward_model(traj_b).sum()
    logit = R_a - R_b
    return F.binary_cross_entropy_with_logits(logit, label)
```

**Be ready for:** Why sums, not averages? (Longer trajectories would be penalised by averaging; and returns are what we actually care about — though note this does bias toward longer trajectories, a known issue worth mentioning.) Why comparisons rather than asking for scores directly? (Humans are consistent at ranking, wildly inconsistent at absolute scoring.) Why `binary_cross_entropy_with_logits` rather than sigmoid-then-BCE? (Numerical stability.) What is this reward model *actually* learning — and what happens when the policy moves somewhere the labels never covered?

---

## Part B — Questions you will be asked

**Q1. Why is this reinforcement learning and not supervised learning?**
No labelled dataset of correct triage orders exists. Actions consume a shared time budget so decisions are coupled, and consequences are delayed — you don't learn that skipping an alert was wrong until much later, sometimes only at end of shift. That's the definition of a sequential decision problem.

**Q2. Walk me through your state space.**
Five discretised features — max severity in queue, queue length, age of oldest alert, time left in shift, max asset criticality — giving 576 states. Deliberately small so tabular methods work and the learned policy stays human-readable. Phase 3 drops the bucketing for a ~20-dim continuous vector and uses DQN, which is the honest motivation for function approximation.

**Q3. Why are actions triage *rules* rather than individual alerts?** *(most likely design question — see `DECISIONS.md` D-002)*
"Pick one of N alerts" is a variable-sized action space with N in the hundreds. It breaks tabular Q-learning and complicates DQN. Fixing five rule-choices keeps the action space constant, keeps every syllabus algorithm applicable unmodified, and makes the policy readable by a human expert. We gave up expressiveness for interpretability — a deliberate trade in a domain where a security manager has to trust the output.

**Q4. Why Q-learning and not just DQN everywhere?**
576 states is small enough that a table is sufficient, converges faster, and — critically — is *inspectable*. We can print the policy and read it. DQN earns its place only in Phase 3, where the state is continuous and tabulating it is impossible. Using DQN on the discretised version would be strictly worse: slower, less stable, less interpretable, no benefit.

**Q5. How did you handle exploration vs exploitation?**
ε-greedy with exponential decay to a floor. Kept the floor above zero so the agent keeps a small amount of exploration late in training — the queue distribution shifts over a shift, and a fully greedy agent stops adapting.

**Q6. What's RLHF and why did you need it?**
Our hand-written reward contains invented numbers — is a 3-hour detection delay worse than 40 wasted analyst-minutes? There's no correct answer, so a hand-written reward encodes *our* guess. Instead we showed humans two shift replays and asked which was handled better, collected 300 comparisons, and fit a reward model with a Bradley–Terry loss. Then we optimised against the learned reward. Same technique that aligns ChatGPT, at student scale.

**Q7. How do you know your agent isn't cheating?**
We assumed it might be and went looking. Four checks: bulk-close exploit frequency; reward-model train vs held-out gap; state-visitation overlap between the new policy and the policies our labels actually covered (an out-of-distribution check — if the policy moves somewhere the reward model never saw labelled data, its reward estimates there are unfounded); and inter-annotator agreement via Cohen's κ. *[Then state what you actually found.]*

**Q8. What's the weakest part of this project?**
The environment is simulated, and we deliberately made vendor severity a weak predictor of ground truth. That assumption is industry-grounded but it *is* the mechanism by which our agent beats severity-sort. If severity were highly predictive, sorting by it would already be near-optimal. We state this in the report rather than burying it.

> **Answer Q8 honestly and directly.** Interviewers ask it to see whether you understand your own work or are selling it. Naming your project's real weakness — unprompted and precisely — is the single strongest signal you can give.

**Q9. What would you do with more time?**
Multi-analyst coordination (the queue becomes a shared resource, which turns it into a multi-agent problem), alert correlation into incidents, and real practitioner labels at scale.

**Q10. How did you evaluate it?**
Six baselines including an oracle upper bound and the industry-standard severity-sort. Five seeds, mean ± std, never a single run. Paired comparisons — every policy faces identical alert streams, which cuts variance. Training and evaluation seeds are disjoint and enforced in code.

---

## Part C — Division of expertise

Both of you must be able to explain **everything**. But own your half deeply enough to go three questions deep:

**Pranav** — algorithm internals, the reward model, evaluation methodology, the audit. Be ready to defend the DP-on-estimated-model choice (D-004) and to explain why value iteration and policy iteration agreeing is a correctness test.

**Diya** — the MDP framing, the environment design and its assumptions, the RLHF data pipeline, the dashboard. Be ready to explain why severity is a weak signal in our generator and why that's both defensible and a limitation.

---

## Part D — The 60-second version

> Security teams get thousands of alerts a day and can investigate a fraction of them. Almost all are false alarms, but a few are real breaches, and every hour a real one waits is more attacker dwell time. Most teams just sort by the vendor's severity label, which knows nothing about their business.
>
> We modelled it as an MDP — the state is the queue situation and time left, the actions are five triage strategies, the reward is catching real incidents fast while not wasting analyst time — and trained agents from Q-learning up to DQN and actor–critic.
>
> The interesting part is that the reward is genuinely un-writable. How many wasted minutes equal one missed breach? So instead of guessing, we showed people two shift replays and asked which went better, collected 300 comparisons, and learned a reward model from them — RLHF, the same technique used to align ChatGPT. Then we audited the result for reward hacking, because a learned reward can be gamed, and *[what you found]*.
>
> Against the industry-standard severity-sort baseline, we cut mean time-to-detect by *[X]*% across five seeds.

Practice this out loud until it's 60 seconds and doesn't sound recited.
