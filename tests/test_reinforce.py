"""REINFORCE: the update rule, the baseline, and an anchor on the tiny MDP.

Written before the agent. The structure mirrors tests/test_dqn.py: mechanics
first (does it compute what S&B 13.3 says it computes), then the anchor (does
it actually learn the known-optimal policy on a fixture with a pen-and-paper
answer).

Two properties get their own tests because losing either would leave something
that still trains, still logs a curve, and is no longer REINFORCE:

  * actions are SAMPLED from the policy, never argmaxed -- the policy is the
    thing being learned, and an argmax in act() silently deletes exploration;
  * the update happens at the episode boundary, because G_t does not exist
    before then.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soc_triage.agents.reinforce import ReinforceAgent  # noqa: E402
from soc_triage.config import ReinforceConfig, load_training_config  # noqa: E402
from soc_triage.tiny_mdp import (  # noqa: E402
    GAMMA,
    HAND_COMPUTED_POLICY,
    MC_HORIZON,
    N_TINY_ACTIONS,
    N_TINY_STATES,
    step,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "training_default.yaml"


def _rcfg(**overrides) -> ReinforceConfig:
    """The shipped config, with the test's overrides applied.

    Starting from the real config rather than from invented values means a test
    cannot pass against settings the project does not actually use.
    """
    shipped = load_training_config(CONFIG_PATH).reinforce
    fields = {
        "hidden_layers": shipped.hidden_layers,
        "activation": shipped.activation,
        "lr": shipped.lr,
        "use_baseline": shipped.use_baseline,
        "baseline_lr": shipped.baseline_lr,
        "grad_clip_norm": shipped.grad_clip_norm,
        "train_seed_start": shipped.train_seed_start,
        "ablation_seed_start": shipped.ablation_seed_start,
    }
    fields.update(overrides)
    return ReinforceConfig(**fields)


def _agent(seed: int = 0, obs_dim: int = 4, n_actions: int = 3, **overrides) -> ReinforceAgent:
    return ReinforceAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        rcfg=_rcfg(**overrides),
        gamma=0.99,
        seed=seed,
        feature_scales=np.ones(obs_dim),
    )


def _obs(obs_dim: int = 4) -> np.ndarray:
    return np.linspace(0.1, 1.0, obs_dim)


# ---------------------------------------------------------------------------
# 1. The policy
# ---------------------------------------------------------------------------


def test_the_policy_is_a_probability_distribution():
    agent = _agent()
    probs = agent.action_probabilities(_obs())
    assert probs.shape == (3,)
    assert np.all(probs > 0.0)
    assert probs.sum() == pytest.approx(1.0)


def test_actions_are_sampled_not_argmaxed():
    """An argmax here would still train and still log a curve. It would just
    never explore, and the agent would stop being a policy-gradient method."""
    agent = _agent()
    obs = _obs()
    drawn = {agent.act(obs) for _ in range(200)}
    assert len(drawn) > 1, "act() returned one action 200 times -- that is an argmax"


def test_sampling_frequencies_follow_the_policy():
    agent = _agent()
    obs = _obs()
    probs = agent.action_probabilities(obs)
    counts = np.zeros(3)
    for _ in range(4000):
        counts[agent.act(obs)] += 1
    assert np.max(np.abs(counts / 4000 - probs)) < 0.03


def test_same_seed_gives_identical_action_sequences():
    obs = _obs()
    first, second = _agent(seed=7), _agent(seed=7)
    assert [first.act(obs) for _ in range(50)] == [second.act(obs) for _ in range(50)]


def test_the_agent_has_no_epsilon():
    """Structural, not cosmetic. Exploration comes from the policy's own
    stochasticity; an epsilon schedule bolted on would make this a different
    algorithm that still reports itself as REINFORCE (see FEATURE_008)."""
    agent = _agent()
    assert not hasattr(agent, "epsilon")


# ---------------------------------------------------------------------------
# 2. The update rule (S&B 13.3)
# ---------------------------------------------------------------------------


def _feed(agent: ReinforceAgent, rewards: list[float], obs_dim: int = 4) -> None:
    """Push a synthetic episode through update() with fixed observations."""
    obs = _obs(obs_dim)
    for i, reward in enumerate(rewards):
        agent.update(obs, i % agent.n_actions, reward, obs, done=(i == len(rewards) - 1))


def test_update_alone_changes_no_parameters():
    """G_t does not exist until the episode ends, so nothing can be learned
    mid-episode. Same structural constraint as monte_carlo.py."""
    agent = _agent()
    before = agent.policy_parameters()
    _feed(agent, [1.0, -2.0, 3.0])
    after = agent.policy_parameters()
    assert np.array_equal(before, after)


def test_end_episode_moves_the_policy():
    agent = _agent()
    before = agent.policy_parameters()
    _feed(agent, [1.0, -2.0, 3.0])
    agent.end_episode()
    assert not np.array_equal(before, agent.policy_parameters())


def test_returns_are_the_backwards_discounted_sums():
    agent = _agent(use_baseline=False)
    _feed(agent, [1.0, -2.0, 3.0])
    agent.end_episode()

    gamma = 0.99
    expected = [
        1.0 + gamma * (-2.0) + gamma**2 * 3.0,
        -2.0 + gamma * 3.0,
        3.0,
    ]
    assert agent.last_returns == pytest.approx(expected)


def test_the_gamma_to_the_t_factor_is_applied():
    """S&B 13.3 includes gamma^t and most implementations drop it. This test
    pins which one this project is, so the choice cannot be edited away by
    accident -- see FEATURE_008 for why it is kept."""
    agent = _agent(use_baseline=False)
    _feed(agent, [1.0, -2.0, 3.0])
    agent.end_episode()

    gamma = 0.99
    weights = np.array([gamma**t for t in range(3)])
    assert agent.last_coefficients == pytest.approx(weights * np.array(agent.last_returns))


def test_without_a_baseline_the_coefficient_is_the_raw_return():
    agent = _agent(use_baseline=False)
    _feed(agent, [5.0, 5.0, 5.0])
    agent.end_episode()
    assert np.all(np.array(agent.last_returns) > 0)
    # Every coefficient positive means every action taken is pushed UP, including
    # the bad ones. That is exactly the problem the baseline exists to fix.
    assert np.all(np.array(agent.last_coefficients) > 0)


def test_the_buffer_is_cleared_between_episodes():
    """A leaked buffer folds the previous shift's rewards into this one's
    returns. The only symptom is 'REINFORCE is oddly noisy'."""
    agent = _agent(use_baseline=False)
    _feed(agent, [1.0, 1.0])
    agent.end_episode()
    _feed(agent, [7.0])
    agent.end_episode()
    assert len(agent.last_returns) == 1
    assert agent.last_returns[0] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 3. The baseline (S&B 13.4) -- the variance claim, measured
# ---------------------------------------------------------------------------


def test_a_trained_baseline_centres_the_update_coefficients():
    """The variance-reduction mechanism, checked rather than asserted.

    Feed the same episode repeatedly. Without a baseline the coefficient stays
    the full return, so every action is reinforced by the same large positive
    number forever. With one, the value head learns that return and the
    coefficients collapse toward zero -- only actions that beat the state's own
    average keep pushing.
    """
    rewards = [10.0] * 8

    without = _agent(seed=1, use_baseline=False)
    for _ in range(40):
        _feed(without, rewards)
        without.end_episode()

    with_baseline = _agent(seed=1, use_baseline=True)
    for _ in range(40):
        _feed(with_baseline, rewards)
        with_baseline.end_episode()

    plain = float(np.mean(np.abs(without.last_coefficients)))
    baselined = float(np.mean(np.abs(with_baseline.last_coefficients)))
    assert baselined < 0.5 * plain, f"baseline barely moved it: {baselined:.3f} vs {plain:.3f}"


def test_the_baseline_is_not_a_critic():
    """S&B 13.5. The baseline is subtracted from the FULL observed return; it
    never appears inside the target. If someone bootstraps it, last_returns
    stops equalling the actual discounted rewards and this fails."""
    agent = _agent(use_baseline=True)
    _feed(agent, [2.0, 4.0])
    agent.end_episode()
    gamma = 0.99
    assert agent.last_returns == pytest.approx([2.0 + gamma * 4.0, 4.0])


# ---------------------------------------------------------------------------
# 4. The anchor -- the tiny MDP with a pen-and-paper optimal policy
# ---------------------------------------------------------------------------


def _one_hot(state: int) -> np.ndarray:
    vector = np.zeros(N_TINY_STATES, dtype=np.float64)
    vector[state] = 1.0
    return vector


def _train_on_tiny_mdp(agent: ReinforceAgent, n_episodes: int, seed: int = 0) -> ReinforceAgent:
    """MC_HORIZON, not HORIZON, for the same reason monte_carlo.py uses it:
    REINFORCE computes a return from every timestep, so a 200-step truncation
    biases the late-episode returns downward."""
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        state = int(rng.integers(0, N_TINY_STATES))
        for _ in range(MC_HORIZON):
            action = agent.act(_one_hot(state))
            next_state, reward = step(state, action)
            agent.update(_one_hot(state), action, reward, _one_hot(next_state), done=False)
            state = next_state
        agent.end_episode()
    return agent


def _tiny_agent(seed: int) -> ReinforceAgent:
    return ReinforceAgent(
        obs_dim=N_TINY_STATES,
        n_actions=N_TINY_ACTIONS,
        rcfg=_rcfg(hidden_layers=(32, 32), lr=0.01, baseline_lr=0.05),
        gamma=GAMMA,  # tiny_mdp's own 0.9, not the project's 0.99
        seed=seed,
        feature_scales=np.ones(N_TINY_STATES),  # one-hot needs no scaling
    )


# The episode count is MEASURED, not chosen. Seeds 0/1/2, varying only n_episodes,
# asking whether the greedy policy equals HAND_COMPUTED_POLICY:
#
#     episodes    seed0   seed1   seed2   correct
#           10    False    True    True     2/3
#           20     True    True    True     3/3
#           30     True    True    True     3/3
#           45     True    True    True     3/3
#           60     True    True    True     3/3
#
# 20 already succeeds on all three. 60 is kept for margin — it costs ~9s and the
# failure it guards against is a flaky anchor, which would be worse than slow:
# an intermittently-red anchor gets rerun until it passes, and at that point it
# has stopped testing anything.
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reinforce_finds_the_hand_computed_optimal_policy(seed):
    """The anchor. tiny_mdp's optimal policy is WAIT in QUIET, WORK in BUSY,
    derived by hand in FEATURE_002 before any learner existed. A policy-gradient
    method that cannot recover two actions on a two-state MDP has a broken
    update rule, and no result on the real environment would mean anything."""
    agent = _train_on_tiny_mdp(_tiny_agent(seed), n_episodes=60)
    greedy = np.array(
        [int(np.argmax(agent.action_probabilities(_one_hot(s)))) for s in range(N_TINY_STATES)]
    )
    assert np.array_equal(greedy, HAND_COMPUTED_POLICY)
