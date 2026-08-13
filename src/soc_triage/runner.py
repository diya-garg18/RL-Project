"""Episode runner — the loop that connects env and agent (FLOW.md Flow A/B).

Runs episodes, encodes observations per the agent's declared obs_kind, and
emits EpisodeRecord dicts (ARCHITECTURE.md §4) — the interchange format that
evaluation, the RLHF pair builder, and the dashboard all consume.

The runner computes no metrics (that's evaluation/) and holds no learning
logic (that's agents/).
"""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from soc_triage.agents.base import Agent
from soc_triage.config import EnvConfig
from soc_triage.env import SOCTriageEnv
from soc_triage.state import EnvSnapshot, discretise, featurise


def _encode(snap: EnvSnapshot, agent: Agent, cfg: EnvConfig) -> Any:
    """Give the agent the observation kind it declared. 'snapshot' is oracle-only."""
    if agent.obs_kind == "disc":
        return discretise(snap, cfg)
    if agent.obs_kind == "cont":
        return featurise(snap, cfg)
    if agent.obs_kind == "snapshot":
        return snap
    raise ValueError(f"unknown obs_kind '{agent.obs_kind}' on agent '{agent.name}'")


def _alert_to_dict(alert) -> dict:
    """Alert -> JSON-serialisable dict. Includes ground truth: EpisodeRecords are
    environment-side logs for evaluation and RLHF outcome rendering, never
    agent observations."""
    return dataclasses.asdict(alert)


def config_hash(cfg_path: str | Path) -> str:
    """Short content hash of the config file, recorded in every EpisodeRecord
    so results are traceable to the exact config that produced them."""
    text = Path(cfg_path).read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def run_episode(
    env: SOCTriageEnv,
    agent: Agent,
    seed: int,
    cfg: EnvConfig,
    cfg_hash: str = "unhashed",
    learn: bool = False,
) -> dict:
    """Run one full shift. Returns an EpisodeRecord dict.

    learn=False (evaluation): agent.update() is never called.
    learn=True  (training):   update() is called with the agent's own obs kind.
    """
    snap = env.reset(seed)
    obs = _encode(snap, agent, cfg)

    steps: list[dict] = []
    total_reward = 0.0
    done = False
    while not done:
        action = agent.act(obs)
        next_snap, reward, done, info = env.step(action)
        next_obs = _encode(next_snap, agent, cfg)

        if learn:
            agent.update(obs, action, reward, next_obs, done)

        steps.append({
            "state_disc": discretise(snap, cfg),
            "action": int(action),
            "reward": float(reward),
            "info": {
                "action_name": info["action_name"],
                "alert_investigated": (
                    _alert_to_dict(info["alert_investigated"])
                    if info["alert_investigated"] is not None else None
                ),
                "was_true_incident": info["was_true_incident"],
                "delay_min": info["delay_min"],
                "n_bulk_closed": len(info["bulk_closed"]),
                "bulk_closed_ids": [a.id for a in info["bulk_closed"]],
                "time_consumed": info["time_consumed"],
                "reward_breakdown": info["reward_breakdown"],
            },
        })
        total_reward += reward
        snap, obs = next_snap, next_obs

    outcome = env.episode_outcome()
    outcome["total_reward"] = float(total_reward)

    return {
        "run_id": f"{agent.name}-seed{seed}",
        "agent_name": agent.name,
        "seed": int(seed),
        "config_hash": cfg_hash,
        "steps": steps,
        "outcome": outcome,
    }


def run_episodes(
    env: SOCTriageEnv,
    agent: Agent,
    seeds: tuple[int, ...],
    cfg: EnvConfig,
    cfg_hash: str = "unhashed",
    learn: bool = False,
) -> list[dict]:
    """Run one episode per seed. Explicit loop — no hidden parallelism."""
    records = []
    for seed in seeds:
        records.append(run_episode(env, agent, seed, cfg, cfg_hash, learn))
    return records


def save_records(records: list[dict], directory: str | Path) -> None:
    """Write each EpisodeRecord as JSON under results/runs/<run_id>.json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = directory / f"{record['run_id']}.json"
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
