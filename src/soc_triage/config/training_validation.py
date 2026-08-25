"""Validation for the training config, split out of `training.py` (CONSTRAINTS #12).

`training.py` passed 500 lines when Phase 4's actor-critic section and its two
experiment seed blocks landed. The seam is the same one D-031 used for the
original package split: the loader's job is to turn YAML into typed objects, and
this file's job is to refuse objects that are wrong in a way the algorithm cannot
detect. Nothing here reads YAML, and nothing in `training.py` decides what is
valid.

**Every check here was moved verbatim.** None was reworded, reordered or
renamed, which is what makes the existing suite a real test of the move rather
than a test of a rewrite -- the property D-031 relied on, and the reason that
split was safe.

Why this takes the sub-configs rather than an assembled `TrainingConfig`:
keeping the parameter names identical to the loader's local variables is what
allowed the body to move without edits. A `cfg.` prefix on every reference would
have meant touching ~60 lines, and a typo in any one of them would have silently
disabled a guard.
"""

from soc_triage.config.validation import ConfigError


def validate_training_config(
    *,
    common,
    dp,
    epsilon,
    q_learning,
    sarsa,
    monte_carlo,
    features,
    dqn,
    reinforce,
    actor_critic,
) -> None:
    """Raise ConfigError on any setting that would fail as a bad RESULT rather
    than as an error.

    That is the criterion for belonging here. An out-of-range alpha produces a
    diverging Q-table; a Huber delta below the measured collapse threshold
    produces an agent that ignores catastrophes; an entropy coefficient of the
    wrong sign produces a policy that deletes its own exploration. None of them
    raise on their own, and each costs hours to trace back to a line of YAML.
    """
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
        "reinforce": reinforce.train_seed_start,
        "actor_critic": actor_critic.train_seed_start,
    }
    for name, start in seed_starts.items():
        if start < 100_000:
            raise ConfigError(
                f"'{name}.train_seed_start' must be >= 100000 to stay clear of the "
                "train (1-10), eval (101-130), calibration (1000-3099), and DP "
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
    if reinforce.ablation_seed_start < 100_000:
        raise ConfigError(
            "'reinforce.ablation_seed_start' must be >= 100000 to stay clear of the "
            "train, eval, calibration and DP estimation seed blocks"
        )
    if reinforce.ablation_seed_start in seed_starts.values() or (
        reinforce.ablation_seed_start == dqn.ablation_seed_start
    ):
        raise ConfigError(
            f"'reinforce.ablation_seed_start' ({reinforce.ablation_seed_start}) must "
            f"differ from every other seed block, including 'dqn.ablation_seed_start' "
            f"({dqn.ablation_seed_start})"
        )
    # E-019's clip sweep is the third named block under `reinforce:`, and it gets
    # the same treatment as the other two: a tuning run must never share alert
    # streams with a run whose number gets reported.
    if reinforce.clip_experiment_seed_start < 100_000:
        raise ConfigError(
            "'reinforce.clip_experiment_seed_start' must be >= 100000 to stay clear "
            "of the train, eval, calibration and DP estimation seed blocks"
        )
    if reinforce.clip_experiment_seed_start in seed_starts.values() or (
        reinforce.clip_experiment_seed_start
        in (dqn.ablation_seed_start, reinforce.ablation_seed_start)
    ):
        raise ConfigError(
            f"'reinforce.clip_experiment_seed_start' "
            f"({reinforce.clip_experiment_seed_start}) must differ from every other "
            f"seed block, including the two ablation blocks"
        )
    # E-020's entropy sweep, treated exactly as E-019's clip sweep is: a tuning
    # run must never share alert streams with a run whose number gets reported.
    if actor_critic.entropy_experiment_seed_start < 100_000:
        raise ConfigError(
            "'actor_critic.entropy_experiment_seed_start' must be >= 100000 to stay "
            "clear of the train, eval, calibration and DP estimation seed blocks"
        )
    if actor_critic.entropy_experiment_seed_start in seed_starts.values() or (
        actor_critic.entropy_experiment_seed_start
        in (
            dqn.ablation_seed_start,
            reinforce.ablation_seed_start,
            reinforce.clip_experiment_seed_start,
        )
    ):
        raise ConfigError(
            f"'actor_critic.entropy_experiment_seed_start' "
            f"({actor_critic.entropy_experiment_seed_start}) must differ from every "
            f"other seed block, including the ablation and clip-sweep blocks"
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
    # An unscaled or negatively-scaled column does not raise anywhere downstream;
    # it just trains worse. Caught here or not at all.
    for name, divisor in features.scales:
        if divisor <= 0:
            raise ConfigError(f"'features.scales.{name}' must be positive, got {divisor}")

    # Phase 4 checks. Same rule as the Phase 3 block above: each of these fails
    # at runtime as a bad result rather than as an error. A zero learning rate
    # trains a policy that never moves, which reads as "the task is hard".
    if not reinforce.hidden_layers or any(h <= 0 for h in reinforce.hidden_layers):
        raise ConfigError("'reinforce.hidden_layers' must be a non-empty list of positive ints")
    if reinforce.activation not in ("relu", "tanh"):
        raise ConfigError(
            f"'reinforce.activation' must be relu or tanh, got {reinforce.activation!r}"
        )
    if reinforce.lr <= 0:
        raise ConfigError("'reinforce.lr' must be positive")
    if reinforce.baseline_lr <= 0:
        # Checked even when use_baseline is false: an ablation run must be able
        # to switch the baseline back on without the config being wrong.
        raise ConfigError("'reinforce.baseline_lr' must be positive")
    if reinforce.grad_clip_norm <= 0:
        raise ConfigError("'reinforce.grad_clip_norm' must be positive")

    if not actor_critic.hidden_layers or any(h <= 0 for h in actor_critic.hidden_layers):
        raise ConfigError("'actor_critic.hidden_layers' must be a non-empty list of positive ints")
    if actor_critic.activation not in ("relu", "tanh"):
        raise ConfigError(
            f"'actor_critic.activation' must be relu or tanh, got {actor_critic.activation!r}"
        )
    if actor_critic.actor_lr <= 0:
        raise ConfigError("'actor_critic.actor_lr' must be positive")
    if actor_critic.critic_lr <= 0:
        raise ConfigError("'actor_critic.critic_lr' must be positive")
    if actor_critic.grad_clip_norm <= 0:
        raise ConfigError("'actor_critic.grad_clip_norm' must be positive")
    if actor_critic.entropy_coef < 0:
        # Zero is allowed and is S&B §13.5 exactly. Negative is not an aggressive
        # setting of the same knob -- it flips the sign of the term, paying the
        # policy to become deterministic. It trains, it logs a curve, and it
        # does the opposite of what the key is named for.
        raise ConfigError(
            f"'actor_critic.entropy_coef' must be >= 0 (0.0 is the textbook "
            f"algorithm); a negative value rewards certainty and deletes "
            f"exploration, got {actor_critic.entropy_coef}"
        )

    if dp.value_iteration_theta <= 0 or dp.policy_eval_theta <= 0:
        raise ConfigError("DP convergence thresholds must be positive")
    if dp.estimation_seed_start < 10_000:
        raise ConfigError(
            "'dp.estimation_seed_start' must be >= 10000 to stay clear of the "
            "train (1-10), eval (101-130), and calibration (1000-3099) seed blocks"
        )
