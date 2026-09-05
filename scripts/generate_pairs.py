"""Run the nine policies on the pair seed block (FEATURE_011 §2, §4, §5).

This is the driver at the top of the Phase 5a pipeline, and the only step in it
that a missing artefact can block:

    generate_pairs.py  -->  rlhf/pairs.py  -->  rlhf/store.py
    (agents + torch)        (pure data)         (pure data)

It exists in two modes because Phase 5a owes a decision before it can write
anything. Three of the nine policies were trained five times (`dqn`,
`reinforce`, `actor_critic`), and a repeat is not a cosmetic choice: REINFORCE's
five differ by 0.7763 +- 0.0833 recall on the eval seeds, and three of them
reproduce `severity_sort` exactly there. Shipping one of those into a blinded
pair set would put `severity_sort` on both sides of a pair under two different
names, and a labeller would express a "preference" between a policy and itself.

    --survey   run every repeat on the pair seeds and report which ones are
               behaviourally indistinguishable. Writes nothing.
    (later)    take the chosen repeats and write the records + pair set.

The survey measures on the PAIR seeds rather than reusing the eval-seed numbers
already in `results/`, for the reason FEATURE_011 §4 forbids labelling eval
episodes at all: those are different alert streams, and identity on one block is
not identity on another. Evaluating an already-trained policy is inference, so
running all fifteen repeats costs seconds, not a training run.
"""

from typing import Sequence

# --- the comparison arithmetic ------------------------------------------------
#
# Pure functions over EpisodeRecords: no agent, no torch, no `results/`. Two
# policies are indistinguishable to a labeller exactly when they take the same
# actions on the same shift, so the whole comparison is a property of action
# sequences and nothing else. Keeping it separable is what makes it testable
# (tests/test_generate_pairs.py) without fifteen checkpoints on disk.


def action_trace(record: dict) -> tuple[int, ...]:
    """The actions one episode took, in order — the policy's whole visible behaviour.

    Deliberately not the rewards. Two policies with identical traces earn
    identical rewards by construction, but the converse is false, and it is the
    trace a labeller is shown (`rlhf/summary.py`).
    """
    return tuple(int(step["action"]) for step in record["steps"])


def variant_traces(records: Sequence[dict]) -> dict[int, tuple[int, ...]]:
    """seed -> action trace, for the records of a single policy variant."""
    return {int(record["seed"]): action_trace(record) for record in records}


def seeds_matching(
    a: dict[int, tuple[int, ...]], b: dict[int, tuple[int, ...]]
) -> list[int]:
    """Seeds on which two variants took exactly the same actions.

    Only seeds both sides ran are considered. A seed one side is missing is not
    evidence either way: counting it as a match would inflate the collapse count
    and counting it as a mismatch would hide one.
    """
    return sorted(seed for seed in a.keys() & b.keys() if a[seed] == b[seed])


def identical_groups(
    traces: dict[str, dict[int, tuple[int, ...]]]
) -> list[tuple[str, ...]]:
    """Partition variants into sets that are behaviourally indistinguishable.

    Equality is over the *whole* trace map, not over shared seeds only. That is
    the stricter reading and it is the one that makes this a partition at all:
    "agrees on every seed we both ran" is not transitive when the seed sets
    differ, so grouping on it would produce overlapping groups and a table that
    cannot be read. In the survey every variant runs every pair seed, so the two
    readings coincide anyway.

    Ordered by group size descending, then alphabetically — collapses are what
    the reader is looking for, so they surface at the top, and the ordering is
    fixed so that two runs over the same records print the same table.
    """
    groups: dict[tuple, list[str]] = {}
    for name, by_seed in traces.items():
        signature = tuple(sorted(by_seed.items()))
        groups.setdefault(signature, []).append(name)

    ordered = [tuple(sorted(members)) for members in groups.values()]
    ordered.sort(key=lambda group: (-len(group), group))
    return ordered


# --- the driver ---------------------------------------------------------------
#
# Everything below needs torch and the gitignored `results/` tree. It is kept
# beneath the pure section, and out of `rlhf/pairs.py` entirely, for the reason
# FEATURE_011 §3 gives: pair construction must stay buildable on a clone with no
# checkpoints and no torch. This script is the one place allowed to need them.

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from soc_triage.agents.baselines import make_baselines  # noqa: E402
from soc_triage.agents.dp import DPAgent  # noqa: E402
from soc_triage.config import load_env_config, load_training_config  # noqa: E402
from soc_triage.env import SOCTriageEnv  # noqa: E402
from soc_triage.evaluation.metrics import summarise  # noqa: E402
from soc_triage.runner import config_hash, run_episodes  # noqa: E402

from run_baselines import RANDOM_AGENT_ACTION_SEED  # noqa: E402
from train import build_agent as build_tabular  # noqa: E402
from train_actor_critic import build_agent as build_actor_critic  # noqa: E402
from train_dqn import build_agent as build_dqn  # noqa: E402
from train_reinforce import build_agent as build_reinforce  # noqa: E402

RESULTS = ROOT / "results"

# The three learners trained more than once, and where each repeat's checkpoint
# lives. Five repeats each, so the survey compares 15 learned policies alongside
# the six single-artefact ones.
MULTI_RUN = {
    "dqn": RESULTS / "dqn_runs" / "dqn" / "repeat{i}.pt",
    "reinforce": RESULTS / "reinforce_runs" / "reinforce" / "reinforce_repeat{i}.pt",
    "actor_critic": RESULTS / "actor_critic_runs" / "actor_critic" / "actor_critic_repeat{i}.pt",
}
N_REPEATS = 5

# The variant every other variant is measured against. It is the collapse the
# survey exists to find: severity_sort is a hand-written baseline, so a learned
# policy that reproduces it is not a second opinion, and pairing the two asks a
# labeller to prefer a policy over itself.
REFERENCE = "severity_sort"


def _require(path: Path, how: str) -> None:
    """Refuse early and name the command that would produce the artefact.

    The survey runs 21 variants; discovering a missing checkpoint on the
    nineteenth wastes the eighteen before it.
    """
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(ROOT)} - run `{how}` first")


def build_variants(cfg, tcfg) -> dict:
    """Every policy the pair set may draw from, keyed by variant label.

    Multi-run learners appear once per repeat, labelled `dqn@3`, because the
    whole point of the survey is that those five are not interchangeable.
    `oracle_greedy`, `fifo` and `cheapest_first` are absent by design
    (FEATURE_011 §5).

    Exploration is pinned off wherever it exists. For the tabular learners and
    the DQN that means epsilon 0 - a partly-random agent is not the policy that
    was learned. The policy-gradient agents have no epsilon (their randomness is
    the policy, D-036), so they are left sampling and their draws are pinned by
    `reseed` instead, exactly as `train_reinforce.py` does at evaluation time.
    """
    variants: dict = {}

    baselines = {agent.name: agent for agent in make_baselines(cfg, RANDOM_AGENT_ACTION_SEED)}
    variants["random"] = baselines["random"]
    variants["severity_sort"] = baselines["severity_sort"]

    dp_policy = RESULTS / "dp_policy.npy"
    _require(dp_policy, "python scripts/run_dp.py")
    variants["dp"] = DPAgent(np.load(dp_policy))

    for name in ("q_learning", "sarsa", "monte_carlo"):
        q_path = RESULTS / f"{name}_Q.npy"
        _require(q_path, f"python scripts/train.py --agent {name}")
        agent = build_tabular(tcfg, name, seed=0)
        agent.Q = np.load(q_path)
        agent.epsilon = 0.0
        variants[name] = agent

    for repeat in range(N_REPEATS):
        path = Path(str(MULTI_RUN["dqn"]).format(i=repeat))
        _require(path, "python scripts/train_dqn.py")
        agent = build_dqn(tcfg, seed=repeat, no_replay=False, no_target_network=False)
        agent.load(str(path))
        # AFTER load, not before: DQNAgent.load restores the epsilon the
        # checkpoint was saved with, so pinning it first would be silently undone
        # and the survey would compare partly-random agents.
        agent.epsilon = 0.0
        variants[f"dqn@{repeat}"] = agent

    for repeat in range(N_REPEATS):
        path = Path(str(MULTI_RUN["reinforce"]).format(i=repeat))
        _require(path, "python scripts/train_reinforce.py")
        agent = build_reinforce(tcfg, seed=repeat, no_baseline=False)
        agent.load(str(path))
        agent.reseed(repeat)
        variants[f"reinforce@{repeat}"] = agent

    for repeat in range(N_REPEATS):
        path = Path(str(MULTI_RUN["actor_critic"]).format(i=repeat))
        _require(path, "python scripts/train_actor_critic.py")
        agent = build_actor_critic(tcfg, seed=repeat)
        agent.load(str(path))
        agent.reseed(repeat)
        variants[f"actor_critic@{repeat}"] = agent

    return variants


def survey(cfg, tcfg, cfg_hash: str) -> None:
    """Run every variant on the pair seeds and report what is distinguishable.

    Writes nothing. The output is evidence for one decision - which repeat each
    multi-run learner contributes - and that decision is a human's: CLAUDE.md
    puts the pair set among the things the students must be able to defend.
    """
    seeds = tuple(
        tcfg.rlhf.pair_seed_start + offset for offset in range(tcfg.rlhf.n_pair_seeds)
    )
    variants = build_variants(cfg, tcfg)
    env = SOCTriageEnv(cfg)

    print(f"pair seeds {seeds[0]}..{seeds[-1]} ({len(seeds)})   config {cfg_hash}")
    print(f"{len(variants)} variants x {len(seeds)} seeds = "
          f"{len(variants) * len(seeds)} episodes, inference only\n")

    traces: dict = {}
    metrics: dict = {}
    for label, agent in variants.items():
        records = run_episodes(env, agent, seeds, cfg, cfg_hash, learn=False)
        traces[label] = variant_traces(records)
        metrics[label] = summarise(records, cfg)

    reference = traces[REFERENCE]

    header_match = f"== {REFERENCE}"
    print(f"{'variant':18s} {'recall':>18s} {'total reward':>20s} {header_match:>16s}")
    for label in variants:
        recall = metrics[label]["recall_at_deadline"]
        reward = metrics[label]["total_reward"]
        matched = len(seeds_matching(traces[label], reference))
        flag = "-" if label == REFERENCE else f"{matched}/{len(seeds)}"
        print(f"{label:18s} {recall['mean']:9.4f} +- {recall['std']:6.4f} "
              f"{reward['mean']:10.2f} +- {reward['std']:7.2f} {flag:>16s}")

    print(f"\nBehaviourally indistinguishable groups "
          f"(identical actions on all {len(seeds)} pair seeds):")
    collapsed = [group for group in identical_groups(traces) if len(group) > 1]
    for group in collapsed:
        print(f"  COLLAPSED:  {'  ==  '.join(group)}")
    if not collapsed:
        print(f"  none - all {len(variants)} variants are distinguishable")

    print("\nA pair drawn from a collapsed group would ask a labeller to prefer a")
    print("policy over itself. Choose one variant per multi-run learner from the")
    print("table above; the write mode then builds the records and the pair set.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--survey", action="store_true",
        help="run every repeat on the pair seeds and report which collapse; writes nothing",
    )
    args = parser.parse_args()

    if not args.survey:
        raise SystemExit(
            "only --survey is built. The write mode needs the repeat decision that "
            "--survey exists to inform - run `python scripts/generate_pairs.py --survey`."
        )

    cfg_path = ROOT / "config" / "env_default.yaml"
    survey(
        load_env_config(cfg_path),
        load_training_config(ROOT / "config" / "training_default.yaml"),
        config_hash(cfg_path),
    )


if __name__ == "__main__":
    main()
