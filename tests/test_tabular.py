"""Phase 2 — tabular Q-learning, measured against the hand-solved 2-state MDP.

ROADMAP Phase 2. The correctness anchor these tests grade against was built and
verified first (FEATURE_002, E-006): `tiny_mdp.HAND_COMPUTED_Q` is a human's
pen-and-paper answer, cross-checked against `agents/dp.value_iteration`. So when
a test here fails, the learner is wrong — not the expected values.

Three groups, deliberately ordered:

  1. mechanics  — the Q-table, epsilon schedule, and action selection
  2. update rule — single hand-computed updates, including the one assertion
                   that distinguishes Q-learning from SARSA
  3. convergence — the real thing: train on the tiny MDP, reproduce q_*

Group 2 matters more than it looks. Group 3 can pass with a subtly wrong update
(a slightly-off backup still converges to something close on a 2-state problem),
so the single-step tests pin the arithmetic exactly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soc_triage.agents.q_learning import QLearningAgent  # noqa: E402
from soc_triage.config import load_training_config  # noqa: E402
from soc_triage.tiny_mdp import (  # noqa: E402
    BUSY,
    GAMMA,
    HAND_COMPUTED_POLICY,
    HAND_COMPUTED_Q,
    HORIZON,
    N_TINY_ACTIONS,
    N_TINY_STATES,
    QUIET,
    WAIT,
    WORK,
    step,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "training_default.yaml"


def _toy_agent(**overrides) -> QLearningAgent:
    """A 2-state agent with everything pinned, for arithmetic that must be exact."""
    kwargs = dict(
        n_states=N_TINY_STATES,
        n_actions=N_TINY_ACTIONS,
        alpha=0.5,
        gamma=0.9,
        epsilon_start=0.0,  # greedy: no RNG in the way of a hand-checked number
        epsilon_min=0.0,
        epsilon_decay=1.0,
        seed=0,
    )
    kwargs.update(overrides)
    return QLearningAgent(**kwargs)


# ---------------------------------------------------------------------------
# 1. Mechanics
# ---------------------------------------------------------------------------


def test_q_table_shape():
    """The real problem is 576 states x 5 actions (ARCHITECTURE.md §2)."""
    agent = QLearningAgent(
        n_states=576, n_actions=5, alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9995, seed=0,
    )
    assert agent.Q.shape == (576, 5)


def test_q_table_starts_at_zero():
    """Zero init, not optimistic init.

    Optimistic initialisation is a legitimate exploration technique, but it is a
    *different* algorithm and would change the numbers this project reports. If
    it is ever adopted it needs a DECISIONS.md entry, so pin the default here.
    """
    assert np.count_nonzero(_toy_agent().Q) == 0


def test_visits_are_counted_per_state_action():
    """The agent records how often it has updated each (s,a).

    Needed for honesty in the policy table, not for learning. Of the 576 states,
    most are never reached; their Q rows stay all-zero and `argmax` falls to the
    tie-break, action 0. Printed without a visit count that renders as a
    confident wall of PULL_HIGHEST_SEVERITY which the agent never actually
    chose — a figure that would mislead a reader and an examiner equally.
    """
    agent = _toy_agent()
    assert agent.visits.shape == (N_TINY_STATES, N_TINY_ACTIONS)
    assert agent.visits.sum() == 0

    agent.update(QUIET, WAIT, 1.0, QUIET, False)
    agent.update(QUIET, WAIT, 1.0, QUIET, False)
    agent.update(BUSY, WORK, 4.0, QUIET, False)

    assert agent.visits[QUIET, WAIT] == 2
    assert agent.visits[BUSY, WORK] == 1
    assert agent.visits[QUIET, WORK] == 0
    assert agent.visits[BUSY, WAIT] == 0


def test_unvisited_states_are_reported_as_unvisited_not_as_action_zero():
    """A state the agent never saw must be distinguishable from one it chose 0 for.

    This is the assertion that keeps the headline policy table honest.
    """
    agent = _toy_agent()
    agent.update(QUIET, WAIT, 1.0, QUIET, False)

    assert agent.visits[QUIET].sum() > 0
    assert agent.visits[BUSY].sum() == 0, "BUSY was never visited and must read as such"
    assert agent.greedy_policy()[BUSY] == 0, "argmax still returns 0 — which is why the count matters"


def test_epsilon_decays_geometrically_and_stops_at_the_floor():
    """One decay per episode, floored. Config calls it 'decay: per episode'."""
    agent = _toy_agent(epsilon_start=1.0, epsilon_min=0.25, epsilon_decay=0.5)
    assert agent.epsilon == pytest.approx(1.0)
    agent.end_episode()
    assert agent.epsilon == pytest.approx(0.5)
    agent.end_episode()
    assert agent.epsilon == pytest.approx(0.25)
    agent.end_episode()
    assert agent.epsilon == pytest.approx(0.25), "epsilon fell through its floor"


def test_epsilon_does_not_decay_on_update():
    """Decay is per EPISODE, not per step.

    Getting this wrong is silent and catastrophic: at 0.9995 per step instead of
    per episode, epsilon hits its floor within one shift and the agent stops
    exploring almost immediately. The learning curve would look like a plateau,
    not a bug.
    """
    agent = _toy_agent(epsilon_start=1.0, epsilon_min=0.0, epsilon_decay=0.5)
    for _ in range(10):
        agent.update(QUIET, WAIT, 1.0, QUIET, False)
    assert agent.epsilon == pytest.approx(1.0)


def test_greedy_action_is_the_argmax_when_epsilon_is_zero():
    agent = _toy_agent()
    agent.Q[QUIET] = [1.0, 5.0]
    assert agent.act(QUIET) == WORK


def test_ties_break_toward_the_lower_action_index():
    """Reproducibility across seeds depends on deterministic tie-breaking.

    Same rule as `agents.dp.greedy_policy` (strict `>`, first index wins). With
    a zero-initialised table every action ties on the very first step, so this
    is not a rare edge case — it is step one of every run.
    """
    assert _toy_agent().act(QUIET) == 0


def test_epsilon_one_explores_both_actions_reproducibly():
    """Pure exploration must actually reach both actions, and repeat given a seed."""
    def draw() -> list[int]:
        agent = _toy_agent(epsilon_start=1.0, seed=7)
        return [agent.act(QUIET) for _ in range(50)]

    picks = draw()
    assert set(picks) == {WAIT, WORK}, "epsilon=1 never explored one of the actions"
    assert draw() == picks, "same seed produced a different exploration sequence"


def test_greedy_policy_reads_out_of_the_q_table():
    agent = _toy_agent()
    agent.Q[QUIET] = [9.0, 1.0]
    agent.Q[BUSY] = [1.0, 9.0]
    assert np.array_equal(agent.greedy_policy(), np.array([WAIT, WORK]))


# ---------------------------------------------------------------------------
# 2. The update rule, one hand-computed step at a time
# ---------------------------------------------------------------------------


def test_single_update_matches_hand_arithmetic():
    """Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]

    With Q = 0, alpha = 0.5, gamma = 0.9, r = 1, max_a' Q(s',a') = 10:
        target = 1 + 0.9(10) = 10
        Q      = 0 + 0.5(10 - 0) = 5.0
    """
    agent = _toy_agent()
    agent.Q[BUSY] = [1.0, 10.0]
    agent.update(QUIET, WAIT, reward=1.0, next_obs=BUSY, done=False)
    assert agent.Q[QUIET, WAIT] == pytest.approx(5.0)


def test_update_bootstraps_off_the_max_not_the_behaviour_action():
    """THE test that separates Q-learning from SARSA (S&B §6.5 vs §6.4).

    Q-learning is off-policy: the target uses `max_a' Q(s',a')` regardless of
    which action the behaviour policy would actually take next. Above, the max
    is Q(BUSY,WORK) = 10 while Q(BUSY,WAIT) = 1.

        Q-learning target: 1 + 0.9(10) = 10.0  ->  Q = 5.0
        SARSA-style target: 1 + 0.9(1)  =  1.9  ->  Q = 0.95

    Without this assertion, `agents/sarsa.py` could be copy-pasted into
    `q_learning.py` and every convergence test would still pass — the two agree
    at the optimum, and this fixture is small enough that they converge to the
    same place. They are different algorithms and the report compares them, so
    the difference has to be pinned at the update, not at the outcome.
    """
    agent = _toy_agent()
    agent.Q[BUSY] = [1.0, 10.0]
    agent.update(QUIET, WAIT, reward=1.0, next_obs=BUSY, done=False)
    assert agent.Q[QUIET, WAIT] == pytest.approx(5.0)
    assert agent.Q[QUIET, WAIT] != pytest.approx(0.95), "this is the SARSA target"


def test_terminal_update_does_not_bootstrap():
    """done=True means no successor: target is the reward alone.

    Bootstrapping through a terminal state invents value that does not exist and
    is one of the two classic tabular bugs. The other is its mirror — see the
    truncation test below.
    """
    agent = _toy_agent()
    agent.Q[BUSY] = [1.0, 10.0]
    agent.update(QUIET, WAIT, reward=2.0, next_obs=BUSY, done=True)
    assert agent.Q[QUIET, WAIT] == pytest.approx(1.0)  # 0 + 0.5(2 - 0)


def test_alpha_zero_learns_nothing():
    """A dial that does nothing at 0 is a dial wired to the right place."""
    agent = _toy_agent(alpha=0.0)
    agent.update(QUIET, WAIT, reward=100.0, next_obs=BUSY, done=False)
    assert agent.Q[QUIET, WAIT] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Convergence on the hand-solved MDP
# ---------------------------------------------------------------------------


def _train_on_tiny_mdp(agent: QLearningAgent, n_episodes: int, seed: int = 0) -> QLearningAgent:
    """Run the agent on tiny_mdp for n_episodes truncated episodes.

    NOTE the `done=False` at the horizon. The tiny MDP is *continuing* — the
    cut is truncation, not termination, so the agent must keep bootstrapping
    through it. Passing done=True here would teach it the world ends every 200
    steps and collapse every value toward the last reward seen.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        state = int(rng.integers(0, N_TINY_STATES))  # vary the start so neither state is starved
        for _ in range(HORIZON):
            action = agent.act(state)
            next_state, reward = step(state, action)
            agent.update(state, action, reward, next_state, done=False)
            state = next_state
        agent.end_episode()
    return agent


def test_q_learning_converges_to_the_hand_derived_q_star():
    """THE Phase 2 test. Learned Q-table vs a human's pen-and-paper answer.

    If this fails, fix the agent — never the expected values. They come from
    docs/features/FEATURE_002_tiny_mdp_qstar.md and are independently confirmed
    by agents/dp.value_iteration (E-006).
    """
    agent = QLearningAgent(
        n_states=N_TINY_STATES, n_actions=N_TINY_ACTIONS,
        alpha=0.1, gamma=GAMMA,
        epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.99,
        seed=0,
    )
    _train_on_tiny_mdp(agent, n_episodes=500)

    # 1e-9, not the 1e-2 a stochastic problem would need. The tiny MDP is
    # deterministic, so the TD target carries no sampling noise and a constant
    # alpha converges *exactly* — measured 9.24e-14 at 500 episodes, and
    # already 9.24e-14 by episode 50. A loose tolerance here would let a
    # genuinely wrong backup through.
    max_error = np.abs(agent.Q - HAND_COMPUTED_Q).max()
    assert max_error < 1e-9, f"max |Q - q*| = {max_error:.4e}\nlearned:\n{agent.Q}\nq*:\n{HAND_COMPUTED_Q}"


def test_q_learning_recovers_the_optimal_policy():
    """WAIT when QUIET, WORK when BUSY — the behaviour, not just the numbers.

    Separate from the value test on purpose: values converge slowly, policies
    converge fast, and it is the policy the report shows.
    """
    agent = QLearningAgent(
        n_states=N_TINY_STATES, n_actions=N_TINY_ACTIONS,
        alpha=0.1, gamma=GAMMA,
        epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.99,
        seed=0,
    )
    _train_on_tiny_mdp(agent, n_episodes=500)
    assert np.array_equal(agent.greedy_policy(), HAND_COMPUTED_POLICY)


def test_convergence_is_reproducible_across_runs_with_the_same_seed():
    """Same seed, same table, exactly. Required by CONSTRAINTS #2's spirit:
    a result nobody can reproduce is not a result."""
    def run():
        agent = QLearningAgent(
            n_states=N_TINY_STATES, n_actions=N_TINY_ACTIONS,
            alpha=0.1, gamma=GAMMA,
            epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.99,
            seed=42,
        )
        return _train_on_tiny_mdp(agent, n_episodes=100, seed=42).Q

    assert np.array_equal(run(), run())


def test_different_seeds_explore_differently_but_reach_the_same_fixed_point():
    """Characterisation test, written after investigating a suspicious result.

    All 5 seeds converged to *identical* Q-tables (max error 9.24e-14, std
    exactly 0). Under CONSTRAINTS #5 that is a bug report until proven
    otherwise — the obvious explanation being that the seed is ignored and all
    five runs are the same run.

    It is not. Early trajectories genuinely diverge (different action sequences,
    different intermediate tables); they simply converge to the same place. That
    is correct for a *deterministic* MDP: the TD target carries no sampling
    noise, so the update has a unique fixed point and exploration order changes
    only how fast it is reached, not where. Locked down here so the next person
    to notice the zero variance does not have to re-run the investigation —
    and so that a genuinely broken seed would now fail loudly.
    """
    def train(seed: int, n_episodes: int) -> np.ndarray:
        agent = QLearningAgent(
            n_states=N_TINY_STATES, n_actions=N_TINY_ACTIONS,
            alpha=0.1, gamma=GAMMA,
            epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.99, seed=seed,
        )
        return _train_on_tiny_mdp(agent, n_episodes=n_episodes, seed=seed).Q

    early = [train(seed, 3) for seed in range(5)]
    for other in early[1:]:
        assert not np.allclose(early[0], other), "seeds are not actually being used"

    converged = [train(seed, 500) for seed in range(5)]
    for other in converged[1:]:
        assert np.allclose(converged[0], other, atol=1e-9)


def test_zero_exploration_starves_the_unvisited_state():
    """The trap documented in FEATURE_002, asserted so nobody rediscovers it at 1am.

    Under the optimal policy the agent never leaves QUIET, so with epsilon = 0
    and a zero-initialised table it takes WAIT forever and never observes BUSY.
    Q(BUSY, ·) stays exactly zero. This is a property of the MDP, not a defect
    in the agent — the failure it produces looks like a broken update rule, so
    it is worth having a test that says otherwise out loud.
    """
    agent = QLearningAgent(
        n_states=N_TINY_STATES, n_actions=N_TINY_ACTIONS,
        alpha=0.1, gamma=GAMMA,
        epsilon_start=0.0, epsilon_min=0.0, epsilon_decay=1.0,
        seed=0,
    )
    state = QUIET  # start in QUIET and stay greedy: the agent never has reason to leave
    for _ in range(HORIZON * 50):
        action = agent.act(state)
        next_state, reward = step(state, action)
        agent.update(state, action, reward, next_state, done=False)
        state = next_state

    assert np.count_nonzero(agent.Q[BUSY]) == 0, "BUSY was reached without exploration"
    assert agent.Q[QUIET, WAIT] == pytest.approx(10.0, abs=1e-2), "QUIET should still converge"


# ---------------------------------------------------------------------------
# 4. Config wiring — no magic numbers in code (CONSTRAINTS #9)
# ---------------------------------------------------------------------------


def test_training_config_exposes_the_q_learning_and_epsilon_sections():
    """Phase 2's hyperparameters must come from config/training_default.yaml.

    They already exist in the YAML; before this phase the loader simply did not
    read them (it parsed `common` and `dp` only, by design — no building ahead).
    """
    cfg = load_training_config(CONFIG_PATH)
    assert cfg.q_learning.alpha == pytest.approx(0.10)
    assert cfg.q_learning.train_seed_start == 200_000
    assert cfg.epsilon.start == pytest.approx(1.0)
    assert cfg.epsilon.min == pytest.approx(0.05)
    assert cfg.epsilon.decay == pytest.approx(0.9995)


def test_config_rejects_an_out_of_range_learning_rate(tmp_path):
    """alpha outside (0, 1] is not a tuning choice, it is a typo.

    Fail loudly at load time rather than producing a diverging Q-table that
    someone spends an afternoon debugging as an algorithm bug.
    """
    from soc_triage.config import ConfigError

    broken = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "q_learning:\n  alpha: 0.10", "q_learning:\n  alpha: 1.5"
    )
    path = tmp_path / "broken.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_training_config(path)


def test_config_rejects_a_training_seed_block_that_collides_with_other_blocks(tmp_path):
    """Seed blocks must stay disjoint, and that is enforced in code (CONSTRAINTS #2).

    Q-learning trains on its own block starting at 200000 (D-016). A start below
    100000 would run into the DP estimation block (10000-59999) — and a training
    run that silently overlaps an evaluation or estimation block is the exact
    failure this project's seed discipline exists to prevent. It must fail at
    load, not be noticed in a results table three sessions later.
    """
    from soc_triage.config import ConfigError

    broken = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "train_seed_start: 200000", "train_seed_start: 20000"
    )
    path = tmp_path / "broken.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_training_config(path)


def test_config_rejects_an_epsilon_floor_above_its_start(tmp_path):
    """A floor above the start means epsilon would *rise* toward it — incoherent."""
    from soc_triage.config import ConfigError

    broken = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "  min: 0.05", "  min: 1.5"
    )
    path = tmp_path / "broken.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_training_config(path)
