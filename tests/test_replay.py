"""The replay buffer, tested as the ring buffer it is.

Roadmap calls this out as a classic interview question, so it is written by
hand (CONSTRAINTS #7) and every property an interviewer would probe is pinned:
capacity, overwrite order, sampling shape, and reproducibility under a seed.
"""

import numpy as np
import pytest

from soc_triage.agents.replay import ReplayBuffer

OBS_DIM = 4


def _fill(buffer, n, start=0):
    """Push n transitions whose every field encodes its own step number.

    obs is all-i, next_obs is all-(i+0.5), reward is i and action is i % 5, so
    a misaligned batch is detectable from the batch alone.
    """
    for i in range(start, start + n):
        buffer.push(
            np.full(OBS_DIM, float(i)), i % 5, float(i), np.full(OBS_DIM, float(i) + 0.5), False
        )


def test_length_grows_then_stops_at_capacity():
    buffer = ReplayBuffer(capacity=10, obs_dim=OBS_DIM, seed=0)
    assert len(buffer) == 0
    _fill(buffer, 4)
    assert len(buffer) == 4
    _fill(buffer, 20, start=4)
    assert len(buffer) == 10


def test_oldest_entry_is_the_one_overwritten():
    buffer = ReplayBuffer(capacity=3, obs_dim=OBS_DIM, seed=0)
    _fill(buffer, 5)  # pushes rewards 0,1,2,3,4 -> buffer should hold 2,3,4
    obs, action, reward, next_obs, done = buffer.sample(64)
    assert set(np.unique(reward)) == {2.0, 3.0, 4.0}


def test_sample_returns_declared_shapes_and_dtypes():
    buffer = ReplayBuffer(capacity=50, obs_dim=OBS_DIM, seed=0)
    _fill(buffer, 50)
    obs, action, reward, next_obs, done = buffer.sample(8)
    assert obs.shape == (8, OBS_DIM) and obs.dtype == np.float32
    assert action.shape == (8,) and action.dtype == np.int64
    assert reward.shape == (8,) and reward.dtype == np.float32
    assert next_obs.shape == (8, OBS_DIM) and next_obs.dtype == np.float32
    assert done.shape == (8,) and done.dtype == np.float32


def test_transitions_stay_aligned_across_the_five_arrays():
    """A ring buffer bug that shuffles one array relative to the others
    produces a plausible-looking batch of nonsense transitions, and nothing
    downstream would ever error."""
    buffer = ReplayBuffer(capacity=50, obs_dim=OBS_DIM, seed=1)
    _fill(buffer, 50)
    obs, action, reward, next_obs, done = buffer.sample(32)
    for i in range(32):
        step = reward[i]
        assert np.allclose(obs[i], step)
        assert np.allclose(next_obs[i], step + 0.5)
        assert action[i] == int(step) % 5


def test_same_seed_gives_identical_samples():
    a, b = ReplayBuffer(20, OBS_DIM, seed=7), ReplayBuffer(20, OBS_DIM, seed=7)
    _fill(a, 20)
    _fill(b, 20)
    for x, y in zip(a.sample(6), b.sample(6)):
        assert np.array_equal(x, y)


def test_different_seeds_give_different_samples():
    a, b = ReplayBuffer(200, OBS_DIM, seed=1), ReplayBuffer(200, OBS_DIM, seed=2)
    _fill(a, 200)
    _fill(b, 200)
    assert not np.array_equal(a.sample(16)[2], b.sample(16)[2])


def test_done_flag_survives_as_a_float():
    buffer = ReplayBuffer(4, OBS_DIM, seed=0)
    buffer.push(np.zeros(OBS_DIM), 0, 1.0, np.zeros(OBS_DIM), True)
    assert buffer.sample(1)[4][0] == 1.0


def test_sampling_an_empty_buffer_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(4, OBS_DIM, seed=0).sample(1)
