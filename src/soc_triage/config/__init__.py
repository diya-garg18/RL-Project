"""Typed, validated configuration for the whole project.

This was one 657-line module until Phase 4 (D-031). It is now three, and the
package re-exports every name the old module exported, so `from soc_triage.config
import EnvConfig, load_training_config` keeps working exactly as before — the
split is a filing change, not an interface change.

Where things live now:

  validation.py   ConfigError and the three shared checks (_require, _check_prior,
                  _check_ascending). Imported by both loaders; imports neither.
  environment.py  config/env_default.yaml — the MDP itself: arrivals, severities,
                  rewards, state buckets, the 5 actions, the seed blocks.
  training.py     config/training_default.yaml — one section per algorithm, plus
                  the cross-learner rules (seed blocks must be disjoint, etc).

Note the two meanings of the word "config" in this repo, which the split makes
easier to confuse rather than harder: `config/` at the repo root holds the YAML
files, and `soc_triage.config` is the code that reads them.
"""

from soc_triage.config.environment import (
    ActionsConfig,
    AlertTypeConfig,
    ArrivalsConfig,
    AssetCriticalityConfig,
    BulkCloseConfig,
    CompositeCostConfig,
    EnvConfig,
    IncidentConfig,
    RewardConfig,
    SeedsConfig,
    SeverityConfig,
    ShiftConfig,
    StateBucketsConfig,
    VerifyCostConfig,
    load_env_config,
)
from soc_triage.config.training import (
    CommonTrainingConfig,
    DPConfig,
    DQNConfig,
    EpsilonConfig,
    FeaturesConfig,
    MonteCarloConfig,
    QLearningConfig,
    ActorCriticConfig,
    ReinforceConfig,
    SarsaConfig,
    TrainingConfig,
    load_training_config,
)
from soc_triage.config.validation import ConfigError

__all__ = [
    "ActionsConfig",
    "AlertTypeConfig",
    "ArrivalsConfig",
    "AssetCriticalityConfig",
    "BulkCloseConfig",
    "CommonTrainingConfig",
    "CompositeCostConfig",
    "ConfigError",
    "DPConfig",
    "DQNConfig",
    "EnvConfig",
    "EpsilonConfig",
    "FeaturesConfig",
    "IncidentConfig",
    "MonteCarloConfig",
    "QLearningConfig",
    "ActorCriticConfig",
    "ReinforceConfig",
    "RewardConfig",
    "SarsaConfig",
    "SeedsConfig",
    "SeverityConfig",
    "ShiftConfig",
    "StateBucketsConfig",
    "TrainingConfig",
    "VerifyCostConfig",
    "load_env_config",
    "load_training_config",
]
