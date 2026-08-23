"""Load and validate config/training_default.yaml into typed, frozen objects.

Split from the environment loader when the single `config.py` passed 500 lines
(CONSTRAINTS #12). The two files load two different YAMLs and share nothing but
the checks in `validation.py`, so the seam is where the split belongs.

Sections are added as phases need them — no building ahead. What every loader
here has in common: a value that is wrong in a way the algorithm cannot detect
(a learner's seed block overlapping another's, a Huber delta below the measured
collapse threshold) is refused at load time rather than discovered in a results
table eight hours later.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from soc_triage.config.validation import ConfigError, _require


@dataclass(frozen=True)
class CommonTrainingConfig:
    gamma: float
    n_episodes: int
    eval_every: int
    log_smoothing_window: int


@dataclass(frozen=True)
class DPConfig:
    n_estimation_episodes: int
    estimation_seed_start: int
    value_iteration_theta: float
    policy_eval_theta: float
    max_sweeps: int


@dataclass(frozen=True)
class EpsilonConfig:
    """The exploration schedule, shared by every epsilon-greedy learner.

    `decay` is applied once per EPISODE, not per step — see
    `QLearningAgent.end_episode`. `min` stays above zero on purpose: queue
    composition shifts during a shift, so a fully greedy agent stops adapting.
    """

    start: float
    min: float
    decay: float


@dataclass(frozen=True)
class QLearningConfig:
    alpha: float
    train_seed_start: int


@dataclass(frozen=True)
class SarsaConfig:
    alpha: float
    train_seed_start: int


@dataclass(frozen=True)
class MonteCarloConfig:
    alpha: float
    first_visit: bool
    train_seed_start: int


@dataclass(frozen=True)
class DQNConfig:
    """Phase 3. `feature_scales` is (name, divisor) pairs rather than a dict so
    the whole config stays hashable and frozen; `state.feature_scale_vector`
    turns it into the ordered array the agent multiplies by."""

    hidden_layers: tuple[int, ...]
    activation: str
    lr: float
    batch_size: int
    replay_capacity: int
    learning_starts: int
    train_freq: int
    target_update_every: int
    grad_clip_norm: float
    loss: str
    huber_delta: float
    feature_scales: tuple[tuple[str, float], ...]
    train_seed_start: int
    ablation_seed_start: int
    no_replay: bool
    no_target_network: bool


@dataclass(frozen=True)
class TrainingConfig:
    common: CommonTrainingConfig
    dp: DPConfig
    epsilon: EpsilonConfig
    q_learning: QLearningConfig
    sarsa: SarsaConfig
    monte_carlo: MonteCarloConfig
    dqn: DQNConfig


def load_training_config(path: str | Path) -> TrainingConfig:
    """Read the training YAML; validate the sections Phase 1 uses."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} did not parse to a mapping")

    common_raw = _require(raw, "common", "training")
    common = CommonTrainingConfig(
        gamma=float(_require(common_raw, "gamma", "common")),
        n_episodes=int(_require(common_raw, "n_episodes", "common")),
        eval_every=int(_require(common_raw, "eval_every", "common")),
        log_smoothing_window=int(_require(common_raw, "log_smoothing_window", "common")),
    )

    dp_raw = _require(raw, "dp", "training")
    dp = DPConfig(
        n_estimation_episodes=int(_require(dp_raw, "n_estimation_episodes", "dp")),
        estimation_seed_start=int(_require(dp_raw, "estimation_seed_start", "dp")),
        value_iteration_theta=float(_require(dp_raw, "value_iteration_theta", "dp")),
        policy_eval_theta=float(_require(dp_raw, "policy_eval_theta", "dp")),
        max_sweeps=int(_require(dp_raw, "max_sweeps", "dp")),
    )

    # Phase 2 sections. Added when Phase 2 started, not before — the loader
    # reads what the current phase needs and nothing more (no building ahead).
    epsilon_raw = _require(raw, "epsilon", "training")
    epsilon = EpsilonConfig(
        start=float(_require(epsilon_raw, "start", "epsilon")),
        min=float(_require(epsilon_raw, "min", "epsilon")),
        decay=float(_require(epsilon_raw, "decay", "epsilon")),
    )

    q_learning_raw = _require(raw, "q_learning", "training")
    q_learning = QLearningConfig(
        alpha=float(_require(q_learning_raw, "alpha", "q_learning")),
        train_seed_start=int(_require(q_learning_raw, "train_seed_start", "q_learning")),
    )

    sarsa_raw = _require(raw, "sarsa", "training")
    sarsa = SarsaConfig(
        alpha=float(_require(sarsa_raw, "alpha", "sarsa")),
        train_seed_start=int(_require(sarsa_raw, "train_seed_start", "sarsa")),
    )

    mc_raw = _require(raw, "monte_carlo", "training")
    monte_carlo = MonteCarloConfig(
        alpha=float(_require(mc_raw, "alpha", "monte_carlo")),
        first_visit=bool(_require(mc_raw, "first_visit", "monte_carlo")),
        train_seed_start=int(_require(mc_raw, "train_seed_start", "monte_carlo")),
    )

    # Phase 3 section, added when Phase 3 started (no building ahead).
    dqn_raw = _require(raw, "dqn", "training")
    scales_raw = _require(dqn_raw, "feature_scales", "dqn")
    ablations_raw = _require(dqn_raw, "ablations", "dqn")
    dqn = DQNConfig(
        hidden_layers=tuple(int(h) for h in _require(dqn_raw, "hidden_layers", "dqn")),
        activation=str(_require(dqn_raw, "activation", "dqn")),
        lr=float(_require(dqn_raw, "lr", "dqn")),
        batch_size=int(_require(dqn_raw, "batch_size", "dqn")),
        replay_capacity=int(_require(dqn_raw, "replay_capacity", "dqn")),
        learning_starts=int(_require(dqn_raw, "learning_starts", "dqn")),
        train_freq=int(_require(dqn_raw, "train_freq", "dqn")),
        target_update_every=int(_require(dqn_raw, "target_update_every", "dqn")),
        grad_clip_norm=float(_require(dqn_raw, "grad_clip_norm", "dqn")),
        loss=str(_require(dqn_raw, "loss", "dqn")),
        huber_delta=float(_require(dqn_raw, "huber_delta", "dqn")),
        feature_scales=tuple((str(k), float(v)) for k, v in scales_raw.items()),
        train_seed_start=int(_require(dqn_raw, "train_seed_start", "dqn")),
        ablation_seed_start=int(_require(dqn_raw, "ablation_seed_start", "dqn")),
        no_replay=bool(_require(ablations_raw, "no_replay", "dqn.ablations")),
        no_target_network=bool(
            _require(ablations_raw, "no_target_network", "dqn.ablations")
        ),
    )

    if not 0.0 < common.gamma <= 1.0:
        raise ConfigError("'common.gamma' must be in (0, 1]")
    for name, alpha in (("sarsa", sarsa.alpha), ("monte_carlo", monte_carlo.alpha)):
        if not 0.0 < alpha <= 1.0:
            raise ConfigError(f"'{name}.alpha' must be in (0, 1]")
    if not monte_carlo.first_visit:
        # Every-visit MC is a real algorithm, but it is not the one this project
        # implements, documents or tests. Flipping the flag would silently make
        # MonteCarloAgent disagree with its own docstring and with S&B §5.1.
        raise ConfigError(
            "'monte_carlo.first_visit' must be true — every-visit MC is not "
            "implemented (see agents/monte_carlo.py and DECISIONS D-017)"
        )

    # These three fail loudly at load time on purpose. An out-of-range alpha
    # produces a diverging Q-table, and an incoherent epsilon schedule produces
    # an agent that never explores — both look like algorithm bugs and cost an
    # afternoon to trace back to a typo in the YAML.
    if not 0.0 < q_learning.alpha <= 1.0:
        raise ConfigError("'q_learning.alpha' must be in (0, 1]")
    if not 0.0 <= epsilon.min <= epsilon.start <= 1.0:
        raise ConfigError(
            "epsilon must satisfy 0 <= min <= start <= 1 "
            f"(got start={epsilon.start}, min={epsilon.min})"
        )
    if not 0.0 < epsilon.decay <= 1.0:
        raise ConfigError("'epsilon.decay' must be in (0, 1]")
    # Every learner trains on its own block of shifts (D-016). Enforced here
    # rather than trusted to the YAML comments, per CONSTRAINTS #2's requirement
    # that seed separation live in code.
    seed_starts = {
        "q_learning": q_learning.train_seed_start,
        "sarsa": sarsa.train_seed_start,
        "monte_carlo": monte_carlo.train_seed_start,
        "dqn": dqn.train_seed_start,
    }
    for name, start in seed_starts.items():
        if start < 100_000:
            raise ConfigError(
                f"'{name}.train_seed_start' must be >= 100000 to stay clear of the "
                "train (1-10), eval (101-105), calibration (1000-3099), and DP "
                "estimation (10000-59999) seed blocks"
            )
    if len(set(seed_starts.values())) != len(seed_starts):
        raise ConfigError(f"learner training seed blocks must be distinct: {seed_starts}")
    # The ablation sweep gets its own block rather than joining seed_starts, so
    # the error names 'dqn.ablation_seed_start' instead of a dict key that does
    # not exist in the YAML. Sharing a block with the main DQN run would make
    # the control condition and the ablations train on identical alert streams,
    # which is the one confound an ablation must not have.
    if dqn.ablation_seed_start < 100_000:
        raise ConfigError(
            "'dqn.ablation_seed_start' must be >= 100000 to stay clear of the "
            "train, eval, calibration and DP estimation seed blocks"
        )
    if dqn.ablation_seed_start in seed_starts.values():
        raise ConfigError(
            f"'dqn.ablation_seed_start' ({dqn.ablation_seed_start}) must differ "
            f"from every learner training block: {seed_starts}"
        )

    # Phase 3 checks. Each of these produces a failure that does not look like a
    # config error at runtime: a replay_capacity below batch_size hangs sample(),
    # a bad activation raises deep inside nn.Sequential, and an unscaled column
    # just yields a worse result with no error at all.
    if not dqn.hidden_layers or any(h <= 0 for h in dqn.hidden_layers):
        raise ConfigError("'dqn.hidden_layers' must be a non-empty list of positive ints")
    if dqn.activation not in ("relu", "tanh"):
        raise ConfigError(f"'dqn.activation' must be relu or tanh, got {dqn.activation!r}")
    if dqn.lr <= 0:
        raise ConfigError("'dqn.lr' must be positive")
    if dqn.batch_size <= 0:
        raise ConfigError("'dqn.batch_size' must be positive")
    if dqn.replay_capacity < dqn.batch_size:
        raise ConfigError(
            f"'dqn.replay_capacity' ({dqn.replay_capacity}) must be at least "
            f"batch_size ({dqn.batch_size}) or sampling can never fill a batch"
        )
    if dqn.learning_starts < dqn.batch_size:
        raise ConfigError(
            f"'dqn.learning_starts' ({dqn.learning_starts}) must be at least "
            f"batch_size ({dqn.batch_size}) or the first sample() has too little data"
        )
    if dqn.train_freq < 1 or dqn.target_update_every < 1:
        raise ConfigError("'dqn.train_freq' and 'dqn.target_update_every' must be >= 1")
    if dqn.grad_clip_norm <= 0:
        raise ConfigError("'dqn.grad_clip_norm' must be positive")
    if dqn.loss != "huber":
        # Same guard as monte_carlo.first_visit: MSE is a real choice but it is
        # not the one implemented, documented or tested, and flipping the key
        # would make the agent disagree with its own docstring.
        raise ConfigError("'dqn.loss' must be 'huber' — no other loss is implemented")
    if dqn.huber_delta <= 0:
        raise ConfigError("'dqn.huber_delta' must be positive")
    if dqn.huber_delta < 50.0:
        # Not a taste bound — a measured one, and the reason Phase 3's first
        # sweep was thrown away. At torch's default delta of 1.0 the -150 and
        # -200 penalties in env_default.yaml produce the same gradient as a
        # routine +-1 error, and all 20 runs collapsed to BULK_CLOSE (E-016).
        # A 5x3 sweep collapsed 3/3 seeds at delta 10 and 1/3 at delta 25, and
        # 0/3 at every value from 50 up. Same shape of guard as the seed-block
        # checks above: cheap here, eight hours of dead compute if it is missed.
        raise ConfigError(
            f"'dqn.huber_delta' is {dqn.huber_delta}, below the measured "
            f"collapse threshold of 50 (E-016). Values at or under 25 make the "
            f"agent collapse to BULK_CLOSE and catch ~1% of incidents."
        )
    for name, divisor in dqn.feature_scales:
        if divisor <= 0:
            raise ConfigError(f"'dqn.feature_scales.{name}' must be positive, got {divisor}")

    if dp.value_iteration_theta <= 0 or dp.policy_eval_theta <= 0:
        raise ConfigError("DP convergence thresholds must be positive")
    if dp.estimation_seed_start < 10_000:
        raise ConfigError(
            "'dp.estimation_seed_start' must be >= 10000 to stay clear of the "
            "train (1-10), eval (101-105), and calibration (1000-3099) seed blocks"
        )
    return TrainingConfig(
        common=common,
        dp=dp,
        epsilon=epsilon,
        q_learning=q_learning,
        sarsa=sarsa,
        monte_carlo=monte_carlo,
        dqn=dqn,
    )
