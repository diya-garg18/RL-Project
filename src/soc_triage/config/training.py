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

from soc_triage.config.training_validation import validate_training_config
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
class FeaturesConfig:
    """Input scaling for every function-approximation agent (D-032).

    `scales` is (name, divisor) pairs rather than a dict so the whole config
    stays hashable and frozen; `state.feature_scale_vector` turns it into the
    ordered array the agent multiplies by.

    This lived under `dqn:` through Phase 3, when the DQN was the only consumer.
    It is shared now because the divisors are domain constants -- the shift is
    480 minutes, severity runs 0-3 -- and Phase 4's REINFORCE and actor-critic
    read the identical 17 columns. Two copies would let the DQN and REINFORCE be
    scaled differently with nothing failing, which would quietly turn the
    sample-efficiency comparison into a comparison of preprocessing.
    """

    scales: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class DQNConfig:
    """Phase 3 hyperparameters. Input scaling is NOT here -- see FeaturesConfig."""

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
    train_seed_start: int
    ablation_seed_start: int
    no_replay: bool
    no_target_network: bool


@dataclass(frozen=True)
class ReinforceConfig:
    """Phase 4. Note what is absent: there is no epsilon schedule.

    Every earlier learner in this project explores by sometimes ignoring its own
    policy. REINFORCE explores by *having* a stochastic policy and sampling from
    it, so exploration decays only as the policy sharpens -- there is nothing to
    schedule and nothing to decay. That absence is the most visible structural
    difference between value-based and policy-gradient methods, and it is why
    this config class does not simply reuse EpsilonConfig.
    """

    hidden_layers: tuple[int, ...]
    activation: str
    lr: float
    use_baseline: bool
    baseline_lr: float
    grad_clip_norm: float
    train_seed_start: int
    ablation_seed_start: int
    clip_experiment_seed_start: int


@dataclass(frozen=True)
class ActorCriticConfig:
    """Phase 4, S&B §13.5. Note what is here that ReinforceConfig does not have.

    `entropy_coef` has no counterpart in REINFORCE and none in the textbook
    either: S&B's boxed one-step actor-critic carries no entropy term. It is this
    project's addition, and setting it to 0.0 recovers the textbook update
    exactly -- which is why the loader accepts zero and rejects only negatives.
    A negative coefficient would reward CERTAINTY and delete exploration while
    still training and still logging a curve.

    There is no `ablation_seed_start` here. ROADMAP's Phase 4 requires an ablation
    of REINFORCE's baseline, not of the entropy bonus, and a seed block for an
    experiment nobody has been asked to run is building ahead.
    """

    hidden_layers: tuple[int, ...]
    activation: str
    actor_lr: float
    critic_lr: float
    entropy_coef: float
    grad_clip_norm: float
    train_seed_start: int
    entropy_experiment_seed_start: int


@dataclass(frozen=True)
class RLHFConfig:
    """Phase 5a preference collection (FEATURE_011).

    `pair_seed_start` is a block of its own, following the D-016 convention,
    and it must not be the eval block. The reward model is fitted to human
    judgements of these episodes and Phase 5c re-trains policies on that reward
    model; if the labelled episodes were eval episodes, human judgement of
    eval-seed outcomes would end up inside the reward the policy maximises.
    That is CONSTRAINTS #2 violated through a person rather than through code,
    and no test downstream would catch it.

    `pair_must_share_seed` is a key rather than a constant so the intent is
    visible in the YAML, but it is validated to True — a pair whose two sides
    ran different alert streams compares luck, not policies.

    `labellers` is ordered, and the order is load-bearing: it fixes which
    labeller gets which of the single-label pairs (D-040). Reordering it after
    labelling has begun would hand every unanswered pair to the other person.

    `ui_host` defaults to loopback and the wildcard is rejected in validation —
    the page serves unlabelled shift data and writes to a database of
    irreplaceable labels, neither of which belongs on a network.
    """

    target_pairs: int
    double_labelled_pairs: int
    pair_must_share_seed: bool
    pair_seed_start: int
    n_pair_seeds: int
    pair_sampling_seed: int
    policies: tuple[str, ...]
    labellers: tuple[str, ...]
    max_seconds_per_pair: int
    ui_host: str
    ui_port: int


@dataclass(frozen=True)
class TrainingConfig:
    common: CommonTrainingConfig
    dp: DPConfig
    epsilon: EpsilonConfig
    q_learning: QLearningConfig
    sarsa: SarsaConfig
    monte_carlo: MonteCarloConfig
    features: FeaturesConfig
    dqn: DQNConfig
    reinforce: ReinforceConfig
    actor_critic: ActorCriticConfig
    rlhf: RLHFConfig


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

    # Shared input scaling (D-032). Read before the agents that consume it.
    features_raw = _require(raw, "features", "training")
    scales_raw = _require(features_raw, "scales", "features")
    features = FeaturesConfig(
        scales=tuple((str(k), float(v)) for k, v in scales_raw.items()),
    )

    # Phase 3 section, added when Phase 3 started (no building ahead).
    dqn_raw = _require(raw, "dqn", "training")
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
        train_seed_start=int(_require(dqn_raw, "train_seed_start", "dqn")),
        ablation_seed_start=int(_require(dqn_raw, "ablation_seed_start", "dqn")),
        no_replay=bool(_require(ablations_raw, "no_replay", "dqn.ablations")),
        no_target_network=bool(
            _require(ablations_raw, "no_target_network", "dqn.ablations")
        ),
    )

    # Phase 4 section, added when Phase 4 started (no building ahead).
    reinforce_raw = _require(raw, "reinforce", "training")
    reinforce = ReinforceConfig(
        hidden_layers=tuple(int(h) for h in _require(reinforce_raw, "hidden_layers", "reinforce")),
        activation=str(_require(reinforce_raw, "activation", "reinforce")),
        lr=float(_require(reinforce_raw, "lr", "reinforce")),
        use_baseline=bool(_require(reinforce_raw, "use_baseline", "reinforce")),
        baseline_lr=float(_require(reinforce_raw, "baseline_lr", "reinforce")),
        grad_clip_norm=float(_require(reinforce_raw, "grad_clip_norm", "reinforce")),
        train_seed_start=int(_require(reinforce_raw, "train_seed_start", "reinforce")),
        ablation_seed_start=int(_require(reinforce_raw, "ablation_seed_start", "reinforce")),
        clip_experiment_seed_start=int(
            _require(reinforce_raw, "clip_experiment_seed_start", "reinforce")
        ),
    )

    actor_critic_raw = _require(raw, "actor_critic", "training")
    actor_critic = ActorCriticConfig(
        hidden_layers=tuple(
            int(h) for h in _require(actor_critic_raw, "hidden_layers", "actor_critic")
        ),
        activation=str(_require(actor_critic_raw, "activation", "actor_critic")),
        actor_lr=float(_require(actor_critic_raw, "actor_lr", "actor_critic")),
        critic_lr=float(_require(actor_critic_raw, "critic_lr", "actor_critic")),
        entropy_coef=float(_require(actor_critic_raw, "entropy_coef", "actor_critic")),
        grad_clip_norm=float(_require(actor_critic_raw, "grad_clip_norm", "actor_critic")),
        train_seed_start=int(_require(actor_critic_raw, "train_seed_start", "actor_critic")),
        entropy_experiment_seed_start=int(
            _require(actor_critic_raw, "entropy_experiment_seed_start", "actor_critic")
        ),
    )

    # Phase 5a section (FEATURE_011). Read now because Phase 5 is the current
    # block; nothing earlier reads it, so an older phase still loads if this
    # section is absent from a hand-edited config -- except that _require makes
    # it mandatory, which is the right trade: a missing rlhf block means someone
    # is running Phase 5 against a stale config.
    rlhf_raw = _require(raw, "rlhf", "training")
    rlhf = RLHFConfig(
        target_pairs=int(_require(rlhf_raw, "target_pairs", "rlhf")),
        double_labelled_pairs=int(_require(rlhf_raw, "double_labelled_pairs", "rlhf")),
        pair_must_share_seed=bool(_require(rlhf_raw, "pair_must_share_seed", "rlhf")),
        pair_seed_start=int(_require(rlhf_raw, "pair_seed_start", "rlhf")),
        n_pair_seeds=int(_require(rlhf_raw, "n_pair_seeds", "rlhf")),
        pair_sampling_seed=int(_require(rlhf_raw, "pair_sampling_seed", "rlhf")),
        policies=tuple(str(p) for p in _require(rlhf_raw, "policies", "rlhf")),
        labellers=tuple(str(l) for l in _require(rlhf_raw, "labellers", "rlhf")),
        max_seconds_per_pair=int(
            _require(rlhf_raw, "max_seconds_per_pair", "rlhf")
        ),
        ui_host=str(_require(rlhf_raw, "ui_host", "rlhf")),
        ui_port=int(_require(rlhf_raw, "ui_port", "rlhf")),
    )

    validate_training_config(
        common=common,
        dp=dp,
        epsilon=epsilon,
        q_learning=q_learning,
        sarsa=sarsa,
        monte_carlo=monte_carlo,
        features=features,
        dqn=dqn,
        reinforce=reinforce,
        actor_critic=actor_critic,
        rlhf=rlhf,
    )

    return TrainingConfig(
        common=common,
        dp=dp,
        epsilon=epsilon,
        q_learning=q_learning,
        sarsa=sarsa,
        monte_carlo=monte_carlo,
        features=features,
        dqn=dqn,
        reinforce=reinforce,
        actor_critic=actor_critic,
        rlhf=rlhf,
    )
