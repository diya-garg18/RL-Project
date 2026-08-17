"""Phase 2 — SARSA and first-visit Monte Carlo, the two ON-POLICY learners.

Split from `tests/test_tabular.py` (which keeps Q-learning) for two reasons: the
file was approaching the 500-line limit in CONSTRAINTS #12, and these two agents
share something Q-learning does not — **they do not converge to q_star**.

Both follow an epsilon-greedy policy and evaluate *that* policy, so their fixed
point is `tiny_mdp.epsilon_soft_q(epsilon)`, not `HAND_COMPUTED_Q`. At
epsilon = 0.1 those differ by more than 1.5 in places, so grading them against
q_* would mark a correct implementation as badly broken. The soft target is
itself anchored: at epsilon = 0 it reproduces the pen-and-paper q_* exactly
(`test_epsilon_soft_q_collapses_to_q_star_as_epsilon_goes_to_zero`).

The distinguishing tests are the point of this file. All three tabular agents
converge to nearly the same policy on the tiny MDP, so a convergence test alone
cannot tell them apart — `sarsa.py` could be a copy of `q_learning.py` and pass.
Each algorithm therefore has one test that pins its update rule at a single
hand-computed backup.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soc_triage.agents.monte_carlo import MonteCarloAgent  # noqa: E402
from soc_triage.agents.sarsa import SarsaAgent  # noqa: E402
from soc_triage.config import load_training_config  # noqa: E402
from soc_triage.tiny_mdp import (  # noqa: E402
    BUSY,
    GAMMA,
    HAND_COMPUTED_POLICY,
    HORIZON,
    MC_HORIZON,
    N_TINY_ACTIONS,
    N_TINY_STATES,
    QUIET,
    WAIT,
    WORK,
    epsilon_soft_q,
    step,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "training_default.yaml"
EVAL_EPSILON = 0.05  # the floor both learners settle at in the convergence tests


def _greedy_kwargs(**overrides) -> dict:
    """Pinned, exploration-free settings so a single backup is exact arithmetic."""
    kwargs = dict(
        n_states=N_TINY_STATES,
        n_actions=N_TINY_ACTIONS,
        alpha=0.5,
        gamma=0.9,
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        seed=0,
    )
    kwargs.update(overrides)
    return kwargs


def _train(agent, n_episodes: int, horizon: int, seed: int = 0):
    """Truncated episodes on the continuing tiny MDP.

    `done=False` at the horizon: this is truncation, not termination, and the TD
    learner must keep bootstrapping through the cut.

    The two learners get **different horizons**, which is not an arbitrary knob.
    A TD method only ever needs the next step, so HORIZON=200 costs it nothing.
    Monte Carlo computes a return from every timestep, so at t=199 it sees one
    reward and no future at all — measurably biasing it downward by up to 0.47
    at HORIZON=200 and 2.75 at HORIZON=50. `MC_HORIZON` is set where that bias
    drops below the constant-alpha noise the TD learners already carry; see the
    measurements in `tiny_mdp.py`.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        state = int(rng.integers(0, N_TINY_STATES))
        for _ in range(horizon):
            action = agent.act(state)
            next_state, reward = step(state, action)
            agent.update(state, action, reward, next_state, done=False)
            state = next_state
        agent.end_episode()
    return agent


def _make(cls, alpha: float = 0.01, seed: int = 0):
    return cls(
        n_states=N_TINY_STATES, n_actions=N_TINY_ACTIONS,
        alpha=alpha, gamma=GAMMA,
        epsilon_start=1.0, epsilon_min=EVAL_EPSILON, epsilon_decay=0.99, seed=seed,
    )


# Trained once per module, not once per test: the value assertion and the policy
# assertion are separate behaviours but they do not need separate training runs,
# and each run costs seconds.
@pytest.fixture(scope="module")
def trained_sarsa():
    return _train(_make(SarsaAgent), n_episodes=3000, horizon=HORIZON)


@pytest.fixture(scope="module")
def trained_monte_carlo():
    return _train(_make(MonteCarloAgent), n_episodes=1500, horizon=MC_HORIZON)


# ===========================================================================
# SARSA — on-policy TD control (S&B §6.4)
# ===========================================================================


def test_sarsa_update_uses_the_action_it_will_actually_take():
    """THE test that separates SARSA from Q-learning.

        SARSA:      Q(s,a) <- Q(s,a) + alpha [ r + gamma * Q(s',a') - Q(s,a) ]
        Q-learning: Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    With Q(BUSY) = [1.0, 10.0] and the agent greedy, the next action it takes
    from BUSY is WORK (value 10), so here the two agree. The difference shows
    when the behaviour policy would NOT take the best action — which is what
    the next test constructs.

    This one pins the arithmetic:
        target = 1 + 0.9 * Q(BUSY, WORK) = 1 + 9.0 = 10.0
        Q(QUIET, WAIT) = 0 + 0.5 * 10.0 = 5.0
    """
    agent = SarsaAgent(**_greedy_kwargs())
    agent.Q[BUSY] = [1.0, 10.0]
    agent.update(QUIET, WAIT, reward=1.0, next_obs=BUSY, done=False)
    assert agent.Q[QUIET, WAIT] == pytest.approx(5.0)


def test_sarsa_bootstraps_off_a_worse_action_when_exploration_picks_one():
    """When the behaviour policy explores, SARSA must follow it into the bad action.

    Forced to epsilon = 1.0 with a seed seeking WAIT from BUSY, the target
    becomes Q(BUSY, WAIT) = 1.0, not the max of 10.0:

        SARSA:      1 + 0.9 * 1.0  = 1.9   ->  Q = 0.95
        Q-learning: 1 + 0.9 * 10.0 = 10.0  ->  Q = 5.00

    Without this assertion `sarsa.py` could be a copy of `q_learning.py` — the
    two converge to nearly the same policy on this fixture, so no convergence
    test would notice.
    """
    agent = SarsaAgent(**_greedy_kwargs(epsilon_start=1.0))
    agent.Q[BUSY] = [1.0, 10.0]

    # Drive the next-action choice deterministically rather than fishing for a
    # seed: the agent must use whatever action it is actually about to take.
    agent.force_next_action(WAIT)
    agent.update(QUIET, WAIT, reward=1.0, next_obs=BUSY, done=False)

    assert agent.Q[QUIET, WAIT] == pytest.approx(0.95)
    assert agent.Q[QUIET, WAIT] != pytest.approx(5.0), "this is the Q-learning target"


def test_sarsa_actually_takes_the_action_it_bootstrapped_off():
    """The on-policy invariant: the a' used in the update IS the next action taken.

    SARSA needs a' at update time, but `Agent.update` is not given one. The
    agent therefore selects a' during `update` and caches it for the next
    `act()`. If that cache leaked — if `act()` re-sampled instead — the agent
    would silently become a strange off-policy hybrid that still converges and
    still passes every value test. This asserts the two agree.
    """
    agent = SarsaAgent(**_greedy_kwargs(epsilon_start=1.0, seed=3))
    agent.Q[BUSY] = [1.0, 10.0]
    agent.update(QUIET, WAIT, reward=1.0, next_obs=BUSY, done=False)
    assert agent.act(BUSY) == agent.last_bootstrap_action


def test_sarsa_terminal_update_does_not_bootstrap():
    agent = SarsaAgent(**_greedy_kwargs())
    agent.Q[BUSY] = [1.0, 10.0]
    agent.update(QUIET, WAIT, reward=2.0, next_obs=BUSY, done=True)
    assert agent.Q[QUIET, WAIT] == pytest.approx(1.0)  # 0 + 0.5(2 - 0)


def test_sarsa_converges_to_the_epsilon_soft_values_not_to_q_star(trained_sarsa):
    """SARSA's fixed point is q_pi for the policy it follows, NOT q_*.

    Graded against HAND_COMPUTED_Q this agent would appear to be wrong by more
    than 1.5 in places. That gap is not error — it is the price of exploring,
    and it is the whole content of the on-policy/off-policy distinction.

    Tolerance measured, not guessed: over 8 seeds at this configuration the
    worst error is 0.100 (mean 0.075), so 0.15 leaves headroom without
    admitting a wrong backup. The residual is constant-alpha noise — SARSA
    random-walks around its fixed point with amplitude proportional to
    sqrt(alpha), which is why more episodes do not shrink it but a smaller
    alpha does (0.113 -> 0.080 -> 0.041 for alpha 0.05 -> 0.01 -> 0.002).
    """
    target = epsilon_soft_q(EVAL_EPSILON)
    max_error = np.abs(trained_sarsa.Q - target).max()
    assert max_error < 0.15, f"max |Q - q_soft| = {max_error:.4f}\n{trained_sarsa.Q}\n{target}"


def test_sarsa_recovers_the_optimal_policy(trained_sarsa):
    """Values differ from q_*; the POLICY must not.

    Depends on `test_epsilon_soft_q_preserves_the_optimal_policy_at_this_epsilon`
    — at this epsilon the action ranking is unchanged, so the correct answer is
    still [WAIT, WORK]. Correct on 8/8 seeds when measured.
    """
    assert np.array_equal(trained_sarsa.greedy_policy(), HAND_COMPUTED_POLICY)


# ===========================================================================
# First-visit Monte Carlo control (S&B §5.4)
# ===========================================================================


def test_monte_carlo_learns_nothing_until_the_episode_ends():
    """MC needs the complete return, so `update` can only buffer.

    This is the structural difference from both TD methods and the reason MC
    cannot learn on a task that never terminates. Nothing may change in Q
    before `end_episode()`.
    """
    agent = MonteCarloAgent(**_greedy_kwargs())
    agent.update(QUIET, WAIT, 1.0, BUSY, False)
    agent.update(BUSY, WORK, 2.0, QUIET, False)
    assert np.count_nonzero(agent.Q) == 0, "MC updated mid-episode"

    agent.end_episode()
    assert np.count_nonzero(agent.Q) > 0, "MC did not update at the episode boundary"


def test_monte_carlo_return_matches_hand_arithmetic():
    """One episode, three steps, computed by hand.

    Episode: (QUIET,WAIT,r=1) -> (BUSY,WORK,r=2) -> (QUIET,WAIT,r=3), gamma=0.9

        G(t=0) = 1 + 0.9(2) + 0.81(3) = 1 + 1.8 + 2.43 = 5.23
        G(t=1) =     2 + 0.9(3)       = 4.7

    FIRST-visit: (QUIET,WAIT) appears at t=0 and t=2, and only t=0 counts.
    With alpha = 0.5 from zero:
        Q(QUIET,WAIT) = 0.5 * 5.23 = 2.615
        Q(BUSY,WORK)  = 0.5 * 4.70 = 2.350
    """
    agent = MonteCarloAgent(**_greedy_kwargs())
    agent.update(QUIET, WAIT, 1.0, BUSY, False)
    agent.update(BUSY, WORK, 2.0, QUIET, False)
    agent.update(QUIET, WAIT, 3.0, QUIET, False)
    agent.end_episode()

    assert agent.Q[QUIET, WAIT] == pytest.approx(2.615)
    assert agent.Q[BUSY, WORK] == pytest.approx(2.350)


def test_monte_carlo_is_first_visit_not_every_visit():
    """THE test separating first-visit from every-visit MC (S&B §5.1).

    Continuing the episode above: an every-visit implementation would also
    apply G(t=2) = 3.0 to (QUIET,WAIT), giving
        0.5 * 5.23 = 2.615, then 2.615 + 0.5(3.0 - 2.615) = 2.8075.
    `config/training_default.yaml` sets `first_visit: true`, so 2.615 is correct
    and 2.8075 is the bug.
    """
    agent = MonteCarloAgent(**_greedy_kwargs())
    agent.update(QUIET, WAIT, 1.0, BUSY, False)
    agent.update(BUSY, WORK, 2.0, QUIET, False)
    agent.update(QUIET, WAIT, 3.0, QUIET, False)
    agent.end_episode()

    assert agent.Q[QUIET, WAIT] == pytest.approx(2.615)
    assert agent.Q[QUIET, WAIT] != pytest.approx(2.8075), "this is every-visit MC"


def test_monte_carlo_clears_its_buffer_between_episodes():
    """A leaked buffer would fold the previous episode's rewards into this one's
    returns — a bug that shows up only as 'MC is oddly noisy'."""
    agent = MonteCarloAgent(**_greedy_kwargs())
    agent.update(QUIET, WAIT, 1.0, QUIET, False)
    agent.end_episode()
    first = agent.Q[QUIET, WAIT]

    agent.update(BUSY, WORK, 4.0, QUIET, False)
    agent.end_episode()
    assert agent.Q[QUIET, WAIT] == pytest.approx(first), "episode 1 was replayed"


def test_monte_carlo_converges_to_the_epsilon_soft_values(trained_monte_carlo):
    """Same on-policy target as SARSA, reached by averaging returns rather than
    bootstrapping.

    Tolerance is 0.40 against SARSA's 0.15, and both were measured rather than
    guessed (MC: worst 0.272 over 8 seeds, mean 0.125). The extra slack is the
    method, not the test: MC uses whole-episode returns and is the
    higher-variance estimator, which is precisely the bias/variance trade S&B
    chapters 5 and 6 are about.

    Note this trains at MC_HORIZON, not HORIZON — see `_train`.
    """
    target = epsilon_soft_q(EVAL_EPSILON)
    max_error = np.abs(trained_monte_carlo.Q - target).max()
    assert max_error < 0.40, (
        f"max |Q - q_soft| = {max_error:.4f}\n{trained_monte_carlo.Q}\n{target}"
    )


def test_monte_carlo_recovers_the_optimal_policy(trained_monte_carlo):
    """Correct on 8/8 seeds when measured — the policy is far more robust than
    the values, the same ordering seen for Q-learning in E-007."""
    assert np.array_equal(trained_monte_carlo.greedy_policy(), HAND_COMPUTED_POLICY)


# ===========================================================================
# Shared behaviour and config wiring
# ===========================================================================


@pytest.mark.parametrize("cls", [SarsaAgent, MonteCarloAgent])
def test_both_agents_track_visits_for_the_policy_table(cls):
    """Same display-honesty requirement as Q-learning (FEATURE_005): an unvisited
    state must be distinguishable from one where action 0 was chosen."""
    agent = cls(**_greedy_kwargs())
    agent.update(QUIET, WAIT, 1.0, QUIET, False)
    agent.end_episode()
    assert agent.visits[QUIET, WAIT] >= 1
    assert agent.visits[BUSY].sum() == 0


@pytest.mark.parametrize("cls", [SarsaAgent, MonteCarloAgent])
def test_both_agents_decay_epsilon_per_episode(cls):
    """D-015 applies to every learner, not just Q-learning."""
    agent = cls(**_greedy_kwargs(epsilon_start=1.0, epsilon_min=0.25, epsilon_decay=0.5))
    agent.end_episode()
    assert agent.epsilon == pytest.approx(0.5)
    agent.end_episode()
    agent.end_episode()
    assert agent.epsilon == pytest.approx(0.25)


def test_training_config_exposes_the_sarsa_and_monte_carlo_sections():
    cfg = load_training_config(CONFIG_PATH)
    assert cfg.sarsa.alpha == pytest.approx(0.10)
    assert cfg.monte_carlo.alpha == pytest.approx(0.05)
    assert cfg.monte_carlo.first_visit is True


def test_every_learner_has_its_own_training_seed_block():
    """D-016 requires each algorithm to train on shifts the others did not see.

    Not cosmetic. If SARSA and Q-learning trained on identical alert streams,
    a difference between their results could be the algorithms or could be that
    one got a luckier draw, and there would be no way to tell. Separate blocks
    make the comparison in ROADMAP box 5 mean something.
    """
    cfg = load_training_config(CONFIG_PATH)
    starts = {
        "q_learning": cfg.q_learning.train_seed_start,
        "sarsa": cfg.sarsa.train_seed_start,
        "monte_carlo": cfg.monte_carlo.train_seed_start,
    }
    assert len(set(starts.values())) == 3, f"seed blocks collide: {starts}"

    # Each run consumes n_episodes * repeats seeds from its start. With the
    # configured 20000 episodes and 5 repeats that is 100000 per algorithm, so
    # the blocks must be at least that far apart or two algorithms would overlap
    # partway through — a collision that would never surface as an error.
    span = cfg.common.n_episodes * 5
    ordered = sorted(starts.values())
    for low, high in zip(ordered, ordered[1:]):
        assert high - low >= span, (
            f"seed blocks {low} and {high} are {high - low} apart but each run "
            f"consumes {span} seeds"
        )
