"""Actor-critic: bootstrapping, the two heads, and an anchor on the tiny MDP.

Written before the agent, and deliberately structured to mirror
tests/test_reinforce.py — because the *differences* between these two files are
the whole point of having both algorithms.

REINFORCE and actor-critic look almost identical on paper. Both learn a policy
directly, both sample actions, neither has an epsilon. Three tests here fail if
someone builds the second one as a copy of the first:

  * `test_update_changes_parameters_immediately` — actor-critic learns EVERY
    STEP. REINFORCE cannot learn until the episode ends, because G_t does not
    exist before then. tests/test_reinforce.py asserts the exact opposite, and
    that pair of opposing tests is the clearest statement of the difference the
    codebase contains.
  * `test_the_critic_bootstraps` — the target is r + gamma*v(s'), NOT the
    observed return. Two networks is not what makes an actor-critic;
    bootstrapping is. reinforce.py has `test_the_baseline_is_not_a_critic`
    pinning that it does not bootstrap; this is that test's mirror image.
  * `test_the_bootstrap_stops_at_a_terminal_state` — v(s') must be dropped when
    the episode ends. Keeping it means the agent values a state after the shift
    is over, which is a bias no curve reveals.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soc_triage.agents.actor_critic import ActorCriticAgent  # noqa: E402
from soc_triage.config import ActorCriticConfig, load_training_config  # noqa: E402
from soc_triage.tiny_mdp import (  # noqa: E402
    GAMMA,
    HAND_COMPUTED_POLICY,
    HORIZON,
    N_TINY_ACTIONS,
    N_TINY_STATES,
    step,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "training_default.yaml"


def _accfg(**overrides) -> ActorCriticConfig:
    """The shipped config with the test's overrides applied.

    Starting from the real config rather than from invented values means a test
    cannot pass against settings the project does not actually use.
    """
    shipped = load_training_config(CONFIG_PATH).actor_critic
    fields = {
        "hidden_layers": shipped.hidden_layers,
        "activation": shipped.activation,
        "actor_lr": shipped.actor_lr,
        "critic_lr": shipped.critic_lr,
        "entropy_coef": shipped.entropy_coef,
        "grad_clip_norm": shipped.grad_clip_norm,
        "train_seed_start": shipped.train_seed_start,
        "entropy_experiment_seed_start": shipped.entropy_experiment_seed_start,
    }
    fields.update(overrides)
    return ActorCriticConfig(**fields)


def _agent(seed: int = 0, obs_dim: int = 4, n_actions: int = 3, **overrides) -> ActorCriticAgent:
    return ActorCriticAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        accfg=_accfg(**overrides),
        gamma=0.99,
        seed=seed,
        feature_scales=np.ones(obs_dim),
    )


def _obs(obs_dim: int = 4, scale: float = 1.0) -> np.ndarray:
    return np.linspace(0.1, 1.0, obs_dim) * scale


# ---------------------------------------------------------------------------
# 1. The policy — shared with REINFORCE, and it must stay shared
# ---------------------------------------------------------------------------


def test_the_policy_is_a_probability_distribution():
    agent = _agent()
    probs = agent.action_probabilities(_obs())
    assert probs.shape == (3,)
    assert np.all(probs > 0.0)
    assert probs.sum() == pytest.approx(1.0)


def test_actions_are_sampled_not_argmaxed():
    agent = _agent()
    obs = _obs()
    drawn = {agent.act(obs) for _ in range(200)}
    assert len(drawn) > 1, "act() returned one action 200 times -- that is an argmax"


def test_the_agent_has_no_epsilon():
    """Structural. Exploration comes from the policy's own stochasticity plus the
    entropy bonus; an epsilon schedule bolted on would make this a different
    algorithm that still reports itself as actor-critic."""
    assert not hasattr(_agent(), "epsilon")


def test_same_seed_gives_identical_action_sequences():
    obs = _obs()
    first, second = _agent(seed=7), _agent(seed=7)
    assert [first.act(obs) for _ in range(50)] == [second.act(obs) for _ in range(50)]


def test_reseed_restores_the_action_stream_whatever_preceded_it():
    """Mirror of the REINFORCE test, for the same reason (D-036).

    Phase 4 reports the sampled policy, so the evaluation draws must be pinned
    to a seed rather than inherited from wherever training left the stream.
    """
    obs = _obs()
    agent = _agent(seed=7)
    reference = [agent.act(obs) for _ in range(50)]

    agent.reseed(7)
    assert [agent.act(obs) for _ in range(50)] == reference

    for _ in range(1234):
        agent.act(obs)
    agent.reseed(7)
    assert [agent.act(obs) for _ in range(50)] == reference, (
        "reseed did not restore the stream after intervening sampling"
    )


def test_reseed_touches_the_sampling_stream_and_nothing_else():
    """Sampling only — never the actor or the critic. Evaluation has to measure
    the agent that trained, not one perturbed on its way to being measured."""
    agent = _agent(seed=3)
    before_actor = [p.detach().numpy().copy() for p in agent.actor_net.parameters()]
    before_critic = [p.detach().numpy().copy() for p in agent.critic_net.parameters()]

    agent.reseed(99)

    for old, new in zip(before_actor, agent.actor_net.parameters()):
        np.testing.assert_array_equal(old, new.detach().numpy())
    for old, new in zip(before_critic, agent.critic_net.parameters()):
        np.testing.assert_array_equal(old, new.detach().numpy())


# ---------------------------------------------------------------------------
# 2. Bootstrapping — the thing that makes this an actor-critic (S&B §13.5)
# ---------------------------------------------------------------------------


def test_update_changes_parameters_immediately():
    """The mirror image of test_reinforce.py::test_update_alone_changes_no_parameters.

    Actor-critic's target is available the instant the step completes, so it
    learns online. If this ever passes for REINFORCE or fails here, the two
    algorithms have been confused for one another.
    """
    agent = _agent()
    before = agent.policy_parameters()
    agent.update(_obs(), 1, 5.0, _obs(scale=0.5), done=False)
    assert not np.array_equal(before, agent.policy_parameters())


def test_the_critic_bootstraps():
    """delta = r + gamma*v(s') - v(s), computed against the critic's OWN values
    read immediately before the update.

    This is the single line that separates actor-critic from REINFORCE-with-
    baseline. REINFORCE subtracts a baseline from the full observed return and
    nothing bootstraps; here the successor's estimated value IS the target.
    """
    agent = _agent()
    obs, next_obs = _obs(), _obs(scale=0.5)

    value_now = agent.state_value(obs)
    value_next = agent.state_value(next_obs)
    agent.update(obs, 1, 5.0, next_obs, done=False)

    expected = 5.0 + 0.99 * value_next - value_now
    assert agent.last_td_error == pytest.approx(expected, rel=1e-5)


def test_the_bootstrap_stops_at_a_terminal_state():
    """v(s') is dropped when done=True. Keeping it values a state after the shift
    has ended — a bias that produces no error and shows up nowhere in a curve."""
    agent = _agent()
    obs, next_obs = _obs(), _obs(scale=0.5)

    value_now = agent.state_value(obs)
    agent.update(obs, 1, 5.0, next_obs, done=True)

    assert agent.last_td_error == pytest.approx(5.0 - value_now, rel=1e-5)


def test_the_critic_converges_to_the_hand_computable_value():
    """A fixture with a pen-and-paper answer, the same idea as tiny_mdp.

    One state that always returns to itself for a reward of 1.0 has, by the
    Bellman equation, v = 1 + gamma*v, so v = 1/(1-gamma) = 10 at gamma = 0.9.
    A critic that does not bootstrap cannot reach 10 from single-step rewards of
    1.0 — it would converge to 1. This is the arithmetic proof that the target
    contains v(s').
    """
    agent = _agent(obs_dim=1, n_actions=2, critic_lr=0.01)
    agent.gamma = 0.9
    obs = np.array([1.0])
    for _ in range(4000):
        agent.update(obs, 0, 1.0, obs, done=False)
    assert agent.state_value(obs) == pytest.approx(10.0, abs=0.5)


def test_the_gamma_to_the_t_factor_decays_within_an_episode_and_resets():
    """S&B §13.5 carries an `I` accumulator: I <- gamma*I after every step, reset
    to 1 at the episode boundary. Same factor REINFORCE applies as gamma^t, and
    kept here for the same reason — it is in the boxed algorithm."""
    agent = _agent()
    obs = _obs()

    assert agent.discount_accumulator == pytest.approx(1.0)
    agent.update(obs, 0, 1.0, obs, done=False)
    assert agent.discount_accumulator == pytest.approx(0.99)
    agent.update(obs, 0, 1.0, obs, done=False)
    assert agent.discount_accumulator == pytest.approx(0.99**2)

    agent.end_episode()
    assert agent.discount_accumulator == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. The entropy bonus — this project's addition, not S&B's
# ---------------------------------------------------------------------------


def test_zero_entropy_coefficient_is_the_textbook_update():
    """With entropy_coef = 0 the entropy term must contribute nothing at all.

    Checked by running the identical update on two identically-seeded agents,
    one built at 0.0. If the term leaked in through anything other than its
    coefficient, the two policies would diverge.
    """
    obs, next_obs = _obs(), _obs(scale=0.5)

    textbook = _agent(seed=3, entropy_coef=0.0)
    also_textbook = _agent(seed=3, entropy_coef=0.0)
    for agent in (textbook, also_textbook):
        agent.update(obs, 1, 5.0, next_obs, done=False)

    assert np.array_equal(textbook.policy_parameters(), also_textbook.policy_parameters())

    bonused = _agent(seed=3, entropy_coef=0.5)
    bonused.update(obs, 1, 5.0, next_obs, done=False)
    assert not np.array_equal(textbook.policy_parameters(), bonused.policy_parameters())


def test_the_entropy_bonus_keeps_the_policy_spread_out():
    """The claim the bonus exists for, measured rather than asserted.

    Feed the same rewarded action over and over. Without a bonus the policy
    sharpens onto it; with one, the entropy term pays to stay spread, so the
    resulting distribution is measurably less peaked. E-018 found REINFORCE
    degenerate at 300 episodes with nothing resisting exactly this.
    """
    obs = _obs()

    def peak_after_training(entropy_coef: float) -> float:
        agent = _agent(seed=5, entropy_coef=entropy_coef)
        for _ in range(300):
            agent.update(obs, 1, 10.0, obs, done=False)
        return float(np.max(agent.action_probabilities(obs)))

    assert peak_after_training(0.5) < peak_after_training(0.0)


def test_the_reported_entropy_is_the_policy_entropy():
    """Diagnostic correctness. The variance demonstration and the trainer both
    read last_entropy; if it reported something else the plots would be wrong
    and nothing would fail."""
    agent = _agent()
    obs = _obs()
    probs = agent.action_probabilities(obs)
    agent.update(obs, 0, 1.0, obs, done=False)
    expected = float(-(probs * np.log(probs)).sum())
    assert agent.last_entropy == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# 4. The anchor — the tiny MDP with a pen-and-paper optimal policy
# ---------------------------------------------------------------------------


def _one_hot(state: int) -> np.ndarray:
    vector = np.zeros(N_TINY_STATES, dtype=np.float64)
    vector[state] = 1.0
    return vector


def _tiny_agent(seed: int) -> ActorCriticAgent:
    """The tiny-MDP fixture, and the one place it deliberately DIFFERS from
    test_reinforce.py's.

    REINFORCE's tiny agent scales both learning rates 10x over the shipped
    config (lr 0.001 -> 0.01, baseline_lr 0.005 -> 0.05). Copying that scaling
    here breaks the anchor: at critic_lr 0.05 the critic collapses to a CONSTANT
    function, returning 10.000 for both states to seven digits, and the policy
    anchor fails on seed 1.

    The reason is the difference between the two algorithms, not a tuning
    accident. REINFORCE's baseline takes ONE gradient step per episode;
    actor-critic's critic takes one per STEP, which on this fixture is 200 per
    episode. It already receives two orders of magnitude more updates, so raising
    its learning rate as well over-corrects. The actor keeps the 10x (it needs
    the anchor to converge in a test-sized budget); the critic runs at the
    shipped 0.005.
    """
    return ActorCriticAgent(
        obs_dim=N_TINY_STATES,
        n_actions=N_TINY_ACTIONS,
        accfg=_accfg(hidden_layers=(32, 32), actor_lr=0.01, critic_lr=0.005),
        gamma=GAMMA,  # tiny_mdp's own 0.9, not the project's 0.99
        seed=seed,
        feature_scales=np.ones(N_TINY_STATES),  # one-hot needs no scaling
    )


def _train_on_tiny_mdp(agent: ActorCriticAgent, n_episodes: int, seed: int = 0):
    """HORIZON (200), not MC_HORIZON (800).

    REINFORCE needed the longer horizon because it builds a return from every
    timestep, so truncation biases the late-episode returns downward. Actor-critic
    bootstraps: its target is one reward plus an estimate, and truncation costs it
    nothing. That the same anchor is four times cheaper here is the practical face
    of the bias-variance trade the two algorithms make.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        state = int(rng.integers(0, N_TINY_STATES))
        for _ in range(HORIZON):
            action = agent.act(_one_hot(state))
            next_state, reward = step(state, action)
            agent.update(_one_hot(state), action, reward, _one_hot(next_state), done=False)
            state = next_state
        agent.end_episode()
    return agent


# The episode count is MEASURED, not chosen. Seeds 0/1/2 at actor_lr 0.01 /
# critic_lr 0.005, varying only n_episodes, asking whether the greedy policy
# equals HAND_COMPUTED_POLICY:
#
#     episodes    seed0   seed1   seed2   correct
#           10     True   False    True     2/3
#           20     True    True    True     3/3
#           30     True    True    True     3/3
#           40     True    True    True     3/3
#
# 20 already succeeds on all three; 40 is kept for margin against a flaky anchor,
# for the reason test_reinforce.py gives — an intermittently-red anchor gets
# rerun until it passes, and at that point it has stopped testing anything.
#
# Note the budget against REINFORCE's: 40 episodes x 200 steps here against 60 x
# 800 there, a 6x smaller anchor for the same guarantee. Bootstrapping is why.
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_actor_critic_finds_the_hand_computed_optimal_policy(seed):
    """The anchor. tiny_mdp's optimal policy is WAIT in QUIET, WORK in BUSY,
    derived by hand in FEATURE_002 before any learner existed. An actor-critic
    that cannot recover two actions on a two-state MDP has a broken update rule,
    and no result on the real environment would mean anything."""
    agent = _train_on_tiny_mdp(_tiny_agent(seed), n_episodes=40)
    greedy = np.array(
        [int(np.argmax(agent.action_probabilities(_one_hot(s)))) for s in range(N_TINY_STATES)]
    )
    assert np.array_equal(greedy, HAND_COMPUTED_POLICY)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_critic_ranks_the_two_tiny_states_correctly(seed):
    """HAND_COMPUTED_V is [10.0, 13.0] — BUSY is worth more than QUIET.

    The ORDER is asserted, not the values. Under function approximation on an
    on-policy sample the critic estimates v for the policy it currently has, and
    that only equals V* once the policy is optimal; demanding 10.0 and 13.0 to a
    tolerance would be asserting a convergence this budget does not buy.

    Measured at the fixture's settings: v(QUIET) lands on 10.000 — V*(QUIET)
    exactly — on every seed, while v(BUSY) ranges 11.2 to 16.5 around a V* of
    13.0. Asserting the order is the claim those numbers support.
    """
    agent = _train_on_tiny_mdp(_tiny_agent(seed), n_episodes=40)
    quiet, busy = agent.state_value(_one_hot(0)), agent.state_value(_one_hot(1))
    assert busy > quiet
