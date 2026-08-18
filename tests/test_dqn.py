"""Phase 3 — the DQN agent, its two stabilisers, and both required ablations.

Structured like `test_tabular.py`, and for the same reason:

  1. mechanics   — obs_kind, feature scaling, epsilon schedule, update gating
  2. stabilisers — the target network and gradient clipping, each pinned at a
                   SINGLE backup rather than by convergence
  3. ablations   — replay off and target network off, also at a single backup
  4. convergence — the anchor: train on tiny_mdp and reproduce q_*

Group 2 and 3 matter more than they look. A target network that silently
tracks the online network still trains and still converges on an easy problem;
the only symptom is the instability that the `no_target_network` ablation is
supposed to demonstrate, which would then be indistinguishable from an ablation
that was never wired up at all. Both are therefore asserted against a target
value computed by hand in the test, not against a training curve.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.agents.dqn import DQNAgent, QNetwork  # noqa: E402
from soc_triage.config import load_training_config  # noqa: E402
from soc_triage.tiny_mdp import (  # noqa: E402
    GAMMA,
    HAND_COMPUTED_POLICY,
    HAND_COMPUTED_Q,
    HORIZON,
    N_TINY_ACTIONS,
    N_TINY_STATES,
    step,
)

CONFIG_PATH = ROOT / "config" / "training_default.yaml"

OBS_DIM = 3
N_ACTIONS = 5
SCALES = np.array([1.0, 2.0, 4.0], dtype=np.float64)


def _dcfg(**overrides):
    """The shipped DQN config with fields overridden for a fast unit test.

    Built by `replace` on the real loaded config rather than a hand-made object,
    so a field added to DQNConfig cannot silently go untested here.
    """
    base = load_training_config(CONFIG_PATH).dqn
    return replace(base, **overrides)


def _agent(seed: int = 0, scales: np.ndarray = SCALES, **overrides) -> DQNAgent:
    return DQNAgent(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        dcfg=_dcfg(**overrides),
        gamma=0.99,
        epsilon_start=0.0,  # greedy by default so act() is deterministic
        epsilon_min=0.0,
        epsilon_decay=1.0,
        seed=seed,
        feature_scales=scales,
    )


def _obs(value: float = 1.0) -> np.ndarray:
    return np.full(OBS_DIM, value, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Mechanics
# ---------------------------------------------------------------------------


def test_obs_kind_is_cont():
    """The runner dispatches on obs_kind; 'disc' here would feed the network a
    bucket id and every state would collapse to one number."""
    agent = _agent()
    assert agent.obs_kind == "cont"
    assert agent.name == "dqn"


def test_features_are_scaled_before_the_network():
    """Measured spread across featurise()'s columns is 470x (D-023). If scaling
    is skipped the network still trains, so nothing errors — the age columns
    simply dominate every gradient and the result is quietly worse."""
    agent = _agent(scales=SCALES)
    raw = np.array([1.0, 4.0, 8.0], dtype=np.float64)
    expected_input = torch.tensor([[1.0, 2.0, 2.0]], dtype=torch.float32)

    with torch.no_grad():
        expected = agent.online(expected_input).numpy().ravel()
    assert np.allclose(agent.q_values(raw), expected)

    # And the unscaled vector must NOT give the same answer, or the test above
    # would pass with scaling removed entirely.
    with torch.no_grad():
        unscaled = agent.online(
            torch.from_numpy(raw.astype(np.float32)).unsqueeze(0)
        ).numpy().ravel()
    assert not np.allclose(agent.q_values(raw), unscaled)


def test_epsilon_decays_once_per_end_episode_and_floors():
    """Per EPISODE, not per step (D-015). train.py must call end_episode()."""
    agent = DQNAgent(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS, dcfg=_dcfg(), gamma=0.99,
        epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.5,
        seed=0, feature_scales=SCALES,
    )
    assert agent.epsilon == 1.0
    agent.update(_obs(), 0, 0.0, _obs(), False)  # a step must NOT decay it
    assert agent.epsilon == 1.0

    agent.end_episode()
    assert agent.epsilon == pytest.approx(0.5)
    agent.end_episode()
    assert agent.epsilon == pytest.approx(0.25)
    for _ in range(20):
        agent.end_episode()
    assert agent.epsilon == pytest.approx(0.1)  # floored, never below


def test_no_learning_before_learning_starts():
    """Training on a buffer of 3 transitions overfits the first shift the agent
    ever saw, and the damage is done before any curve is plotted."""
    agent = _agent(learning_starts=50, train_freq=1)
    for _ in range(49):
        agent.update(_obs(), 0, 1.0, _obs(), False)
    assert agent.gradient_steps == 0
    agent.update(_obs(), 0, 1.0, _obs(), False)
    assert agent.gradient_steps == 1


def test_learns_only_every_train_freq_steps():
    """train_freq=4 is what keeps the 20000-episode budget affordable (D-024).
    If it were ignored, a run costs 18.3 min instead of 4.6."""
    agent = _agent(learning_starts=1, train_freq=5)
    for _ in range(20):
        agent.update(_obs(), 0, 1.0, _obs(), False)
    assert agent.env_steps == 20
    assert agent.gradient_steps == 4  # fires at env_steps 5, 10, 15, 20


def test_same_seed_gives_identical_parameters():
    a, b, c = _agent(seed=3), _agent(seed=3), _agent(seed=4)
    for p, q in zip(a.online.parameters(), b.online.parameters()):
        assert torch.equal(p, q)
    assert any(
        not torch.equal(p, q)
        for p, q in zip(a.online.parameters(), c.online.parameters())
    )


# ---------------------------------------------------------------------------
# 2. The stabilisers
# ---------------------------------------------------------------------------


def test_target_network_does_not_move_between_hard_updates():
    """The classic DQN implementation bug is a target that silently tracks the
    online network — it trains, it looks fine, and it produces exactly the
    instability the no_target_network ablation is supposed to demonstrate."""
    agent = _agent(target_update_every=10_000, learning_starts=1, train_freq=1)
    target_before = [p.clone() for p in agent.target.parameters()]
    online_before = [p.clone() for p in agent.online.parameters()]

    for i in range(50):
        agent.update(_obs(float(i % 3 + 1)), i % N_ACTIONS, 1.0, _obs(2.0), False)

    assert agent.gradient_steps > 0, "no gradient steps ran; the test proves nothing"
    for p, q in zip(agent.target.parameters(), target_before):
        assert torch.equal(p, q), "target moved without a hard update"
    assert any(
        not torch.equal(p, q)
        for p, q in zip(agent.online.parameters(), online_before)
    ), "the online network did not move either — the test is vacuous"


def test_hard_update_makes_target_equal_online():
    agent = _agent(target_update_every=5, learning_starts=1, train_freq=1)
    for i in range(5):
        agent.update(_obs(float(i % 3 + 1)), i % N_ACTIONS, 1.0, _obs(2.0), False)
    assert agent.gradient_steps == 5
    for p, q in zip(agent.target.parameters(), agent.online.parameters()):
        assert torch.equal(p, q)


def test_gradient_clipping_fires_on_a_large_loss():
    """A single outlier reward (the brief's missed-incident penalty is large and
    negative) can otherwise produce one gradient step that destroys the network.

    `last_grad_norm` is the norm BEFORE clipping, which is what makes it
    possible to assert that clipping actually engaged rather than that a small
    gradient happened to stay under the threshold.
    """
    tight = _agent(seed=1, learning_starts=1, train_freq=1, grad_clip_norm=1e-4)
    loose = _agent(seed=1, learning_starts=1, train_freq=1, grad_clip_norm=1e6)

    before = [p.clone() for p in tight.online.parameters()]
    for agent in (tight, loose):
        agent.update(_obs(2.0), 1, 1000.0, _obs(3.0), False)

    assert tight.last_grad_norm > 1e-4, "gradient was already under the clip; test is vacuous"

    def moved(agent):
        return sum(
            float((p - q).abs().sum().detach())
            for p, q in zip(agent.online.parameters(), before)
        )

    assert moved(tight) < moved(loose), "clipping did not restrain the step"


# ---------------------------------------------------------------------------
# 3. The two required ablations, each at a single backup
# ---------------------------------------------------------------------------


def _one_backup_agent(**overrides) -> DQNAgent:
    """An agent that takes exactly ONE gradient step, on the 200th transition.

    learning_starts gates both the replay and no-replay paths identically, so
    the only difference between the two conditions is which transitions end up
    in the batch — which is the whole point of the comparison.
    """
    return _agent(
        seed=2, learning_starts=200, train_freq=1, batch_size=64,
        target_update_every=10_000, **overrides
    )


def test_no_replay_ablation_trains_on_a_single_transition():
    """With replay off, the batch is the transition just observed, so a buffer
    full of contradicting history has no influence on the step."""
    control = _one_backup_agent(no_replay=False)
    ablated = _one_backup_agent(no_replay=True)

    for agent in (control, ablated):
        for _ in range(199):
            agent.update(_obs(1.0), 0, 0.0, _obs(1.0), False)
        assert agent.gradient_steps == 0
        agent.update(_obs(1.0), 0, 100.0, _obs(1.0), False)
        assert agent.gradient_steps == 1

    # The control's single batch is drawn from 200 transitions, 199 of which
    # carry reward 0; the ablated agent's batch is the one carrying reward 100.
    assert ablated.q_values(_obs(1.0))[0] > control.q_values(_obs(1.0))[0]


def test_no_target_network_ablation_bootstraps_off_the_online_net():
    """Asserted at a SINGLE backup, in the spirit of Phase 2's
    test_update_bootstraps_off_the_max_not_the_behaviour_action: once converged
    the two agree, and no convergence test can tell them apart.

    The control's target network is zeroed so the two conditions predict
    different TD targets, and both targets are computed by hand here.
    """
    obs, next_obs, action, reward = _obs(1.0), _obs(2.0), 1, 5.0
    control = _agent(seed=5, learning_starts=1, train_freq=1, no_replay=True)
    ablated = _agent(seed=5, learning_starts=1, train_freq=1, no_replay=True,
                     no_target_network=True)

    with torch.no_grad():
        for p in control.target.parameters():
            p.zero_()
    assert ablated.target is None, "the ablated agent must not hold a target network"

    q_sa = control.q_values(obs)[action]
    max_online_next = float(np.max(control.q_values(next_obs)))

    control.update(obs, action, reward, next_obs, False)
    ablated.update(obs, action, reward, next_obs, False)

    # A zeroed network outputs 0 for any input, so the control bootstraps off 0.
    expected_control = F.huber_loss(
        torch.tensor([q_sa]), torch.tensor([reward + 0.99 * 0.0])
    )
    expected_ablated = F.huber_loss(
        torch.tensor([q_sa]), torch.tensor([reward + 0.99 * max_online_next])
    )
    assert control.last_loss == pytest.approx(float(expected_control), rel=1e-5)
    assert ablated.last_loss == pytest.approx(float(expected_ablated), rel=1e-5)
    assert control.last_loss != pytest.approx(ablated.last_loss, rel=1e-5)


# ---------------------------------------------------------------------------
# 4. The anchor
# ---------------------------------------------------------------------------


def _one_hot(state: int) -> np.ndarray:
    """tiny_mdp states as inputs a network can take.

    One-hot rather than the raw index: an index would make QUIET=0 and BUSY=1
    numerically adjacent and force the network to learn a linear relationship
    between two states that have none.
    """
    vector = np.zeros(N_TINY_STATES, dtype=np.float64)
    vector[state] = 1.0
    return vector


def _train_on_tiny_mdp(agent: DQNAgent, n_episodes: int, seed: int = 0) -> DQNAgent:
    """Same loop as test_tabular._train_on_tiny_mdp, including its done=False.

    The tiny MDP is *continuing*: the horizon cut is truncation, not
    termination, so the agent must keep bootstrapping through it.
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


def _tiny_agent(seed: int) -> DQNAgent:
    """Deliberately smaller and faster than the shipped network.

    hidden_layers=(32, 32) keeps this test near two seconds. The anchor is
    checking the update rule, not the capacity of the 128x128 net.
    """
    return DQNAgent(
        obs_dim=N_TINY_STATES,
        n_actions=N_TINY_ACTIONS,
        dcfg=_dcfg(hidden_layers=(32, 32), batch_size=32, learning_starts=100,
                   train_freq=2, target_update_every=200, lr=0.001),
        gamma=GAMMA,  # tiny_mdp's own 0.9, NOT the project's 0.99 — see below
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99,
        seed=seed,
        feature_scales=np.ones(N_TINY_STATES),  # one-hot needs no scaling
    )


def _learned_q(agent: DQNAgent) -> np.ndarray:
    return np.array([agent.q_values(_one_hot(s)) for s in range(N_TINY_STATES)])


# Episodes and tolerance are both MEASURED, not chosen (plan Task 3 Step 4).
#
# Max |Q - q*| across seeds 0, 1, 2, varying only the episode count:
#
#     episodes    seed0    seed1    seed2    worst
#           60    0.420    0.418    0.429    0.429
#          120    0.017    0.017    0.013    0.017
#          250    0.006    0.002    0.006    0.006
#          500    0.016    0.002    0.002    0.016
#
# Two things that table establishes, and neither could be assumed:
#
#  1. At 60 episodes every entry was low by a near-identical 0.43 — a uniform
#     offset, which is what an unconverged bootstrap looks like and NOT what a
#     wrong backup looks like. The error then collapses by 25x rather than
#     plateauing at a floor, which is the evidence that the update rule is
#     right and 60 episodes was simply too few.
#  2. The residual does not go to zero. It sits around 0.002-0.017 and wobbles
#     back up at 500 episodes: that is the constant-learning-rate noise floor,
#     the same phenomenon SARSA and MC show on this fixture. A tolerance below
#     ~0.02 would be flaky.
#
# 120 episodes is the cheapest point past the collapse. The tolerance is set
# above the observed spread at that point (worst 0.017), not at a round number
# picked for comfort.
TINY_MDP_EPISODES = 120
TINY_MDP_TOLERANCE = 0.05

# Unlike the tabular learners, this anchor costs seconds rather than
# milliseconds, so both assertions share one training run per seed.
TINY_MDP_SEEDS = (0, 1, 2)


@pytest.fixture(scope="module")
def tiny_mdp_tables() -> list[np.ndarray]:
    return [
        _learned_q(_train_on_tiny_mdp(_tiny_agent(s), TINY_MDP_EPISODES, seed=s))
        for s in TINY_MDP_SEEDS
    ]


def test_converges_on_the_tiny_mdp(tiny_mdp_tables):
    """THE Phase 3 anchor (D-014). Learned Q vs a human's pen-and-paper answer.

    tiny_mdp's hand-derived q_* exists BEFORE this learner, so a disagreement
    is unambiguously the learner's fault. If this fails, fix the agent — never
    the expected values.

    Runs at tiny_mdp's own GAMMA of 0.9, not the project's 0.99. HAND_COMPUTED_Q
    was derived on paper at 0.9 (D-014); grading a network trained at 0.99
    against it would compare the agent to the wrong answer and look exactly like
    a convergence failure.
    """
    errors = [np.abs(q - HAND_COMPUTED_Q).max() for q in tiny_mdp_tables]
    worst = max(errors)
    assert worst < TINY_MDP_TOLERANCE, (
        f"max |Q - q*| per seed = {[f'{e:.3f}' for e in errors]}\n"
        f"learned (seed {TINY_MDP_SEEDS[-1]}):\n{tiny_mdp_tables[-1]}\n"
        f"q*:\n{HAND_COMPUTED_Q}"
    )


def test_recovers_the_optimal_policy_on_the_tiny_mdp(tiny_mdp_tables):
    """A weaker claim than the values, and the one that actually matters: the
    action gap on this fixture is >= 2.3 (tiny_mdp.MIN_ACTION_GAP), so a
    roughly-converged network still ranks the actions correctly."""
    for seed, q in zip(TINY_MDP_SEEDS, tiny_mdp_tables):
        assert np.array_equal(np.argmax(q, axis=1), HAND_COMPUTED_POLICY), (
            f"seed {seed} learned policy from:\n{q}"
        )
