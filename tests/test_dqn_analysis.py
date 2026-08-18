"""The analysis path: the guards that stop a wrong number being reported.

These scripts run once, at the end of an eight-hour sweep, against sixty files
nobody will read individually. That is exactly the situation in which a silent
mistake survives — so the checks that refuse to average incomparable runs are
tested here rather than trusted.

Synthetic runs throughout. Nothing here trains anything; the point is the
arithmetic and the refusals, both of which are independent of whether the agent
learned.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_dqn  # noqa: E402
from aggregate_dqn import across_runs, curve_matrix, load_runs  # noqa: E402
from dqn_ablations import max_drawdown, volatility  # noqa: E402


def _run(repeat: int, *, n_episodes: int = 100, config_hash: str = "abc",
         no_replay: bool = False, no_target_network: bool = False,
         curve: list | None = None, mttd: float | None = 5.0) -> dict:
    """One synthetic repeat, shaped exactly like train_dqn writes them."""
    curve = curve if curve is not None else [[50, 1.0], [100, 2.0]]
    return {
        "tag": "dqn", "repeat": repeat, "n_episodes": n_episodes,
        "eval_every": 50, "seed_base": 1000000 + repeat * n_episodes,
        "config_hash": config_hash, "no_replay": no_replay,
        "no_target_network": no_target_network, "wall_min": 1.0,
        "episode_rewards": [0.0] * n_episodes,
        "episode_losses": [1.0] * n_episodes,
        "curve": curve,
        "summary": {
            "n_episodes": 30,
            "mttd_undefined_episodes": 0,
            "recall_at_deadline": {"mean": 0.5, "std": 0.1},
            "total_reward": {"mean": 10.0 + repeat, "std": 2.0},
            "mttd_min": {"mean": mttd, "std": None if mttd is None else 1.0},
        },
        "eval_seeds": list(range(101, 131)),
        "per_seed_total_reward": [1.0] * 30,
    }


def _write(tmp_path: Path, monkeypatch, runs: list[dict], tag: str = "dqn") -> None:
    directory = tmp_path / tag
    directory.mkdir(parents=True, exist_ok=True)
    for run in runs:
        (directory / f"repeat{run['repeat']}.json").write_text(
            json.dumps(run), encoding="utf-8"
        )
    monkeypatch.setattr(aggregate_dqn, "RUNS_DIR", tmp_path)


# --- the refusals -----------------------------------------------------------


def test_mixed_episode_counts_are_refused(tmp_path, monkeypatch):
    """The exact accident that nearly happened: a 40-episode smoke test left
    files beside a 20000-episode sweep. Averaging them would report a mean over
    a mixture and shrink n, with nothing anywhere to notice."""
    _write(tmp_path, monkeypatch, [_run(0, n_episodes=100), _run(1, n_episodes=40)])
    with pytest.raises(SystemExit, match="n_episodes"):
        load_runs("dqn")


def test_mixed_config_hashes_are_refused(tmp_path, monkeypatch):
    """A changed env_default.yaml mid-sweep means the runs faced different
    environments; the hash is the only thing that reveals it."""
    _write(tmp_path, monkeypatch, [_run(0), _run(1, config_hash="different")])
    with pytest.raises(SystemExit, match="config_hash"):
        load_runs("dqn")


def test_mixed_ablation_flags_are_refused(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, [_run(0), _run(1, no_replay=True)])
    with pytest.raises(SystemExit, match="no_replay"):
        load_runs("dqn")


def test_missing_condition_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_dqn, "RUNS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="no results"):
        load_runs("dqn")


def test_matching_runs_load_in_repeat_order(tmp_path, monkeypatch):
    """Sorted numerically, not lexically: repeat10 must not sort before repeat2."""
    _write(tmp_path, monkeypatch, [_run(i) for i in (0, 2, 10)])
    assert [r["repeat"] for r in load_runs("dqn")] == [0, 2, 10]


# --- the arithmetic ---------------------------------------------------------


def test_curve_matrix_is_runs_by_points(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, [_run(0), _run(1)])
    episodes, values = curve_matrix(load_runs("dqn"))
    assert episodes.tolist() == [50, 100]
    assert values.shape == (2, 2)


def test_undefined_metrics_are_dropped_not_zeroed(tmp_path, monkeypatch):
    """summarise() reports mttd as None when no episode caught an incident.
    Counting that as zero would invent the best possible detection time out of
    the worst possible outcome."""
    _write(tmp_path, monkeypatch, [_run(0, mttd=4.0), _run(1, mttd=None)])
    mean, _, n = across_runs(load_runs("dqn"), "mttd_min")
    assert mean == pytest.approx(4.0)
    assert n == 1, "the contributing-run count must reveal the shrunken sample"


def test_metric_undefined_in_every_run_returns_none(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, [_run(0, mttd=None), _run(1, mttd=None)])
    assert across_runs(load_runs("dqn"), "mttd_min") == (None, None, 0)


def test_across_runs_averages_run_means(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, [_run(0), _run(1), _run(2)])  # 10, 11, 12
    mean, std, n = across_runs(load_runs("dqn"), "total_reward")
    assert mean == pytest.approx(11.0)
    assert std == pytest.approx(np.std([10.0, 11.0, 12.0]))
    assert n == 3


# --- the instability measures ----------------------------------------------


def test_a_flat_curve_has_zero_volatility_and_zero_drawdown():
    flat = np.ones((3, 10))
    assert volatility(flat) == 0.0
    assert max_drawdown(flat) == 0.0


def test_volatility_is_mean_absolute_step():
    """Sawtooth 0,2,0,2: every step moves 2, so volatility is exactly 2."""
    saw = np.array([[0.0, 2.0, 0.0, 2.0]])
    assert volatility(saw) == pytest.approx(2.0)


def test_a_monotonic_rise_has_no_drawdown():
    assert max_drawdown(np.array([[1.0, 2.0, 3.0, 4.0]])) == pytest.approx(0.0)


def test_drawdown_measures_the_fall_from_the_peak():
    """Peak 10 then down to 3 is a drawdown of 7, even though the curve ends
    higher than it started — which is the case a mean would hide."""
    assert max_drawdown(np.array([[1.0, 10.0, 3.0, 6.0]])) == pytest.approx(7.0)


def test_drawdown_averages_across_runs():
    values = np.array([[1.0, 10.0, 3.0], [1.0, 5.0, 4.0]])  # 7 and 1
    assert max_drawdown(values) == pytest.approx(4.0)
