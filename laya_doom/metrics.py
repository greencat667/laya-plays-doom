"""JSONL logging of every step and every episode, plus summary stats used by
experiments/compare.py.

Two files per run: ``<run_name>.steps.jsonl`` (one row per decision — world
state text, the executed action, the subsequent result, confidence,
latency) and ``<run_name>.episodes.jsonl`` (one row per episode summary).

Note on ``StepRecord.tool_calls``: kept as a field (see controller.py) for
log-schema parity with an earlier tool-calling pipeline, but it's always an
empty list here — Laya has no tool-call concept at all (see laya_agent.py's
module docstring). The real per-decision signal worth inspecting for Laya
is ``reasoning`` (a compact rendering of the real returned probability
distribution) and ``confidence`` (Laya's own calibrated top-label score).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .controller import EpisodeResult, StepRecord


class MetricsLogger:
    def __init__(self, log_dir: str | Path, run_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.log_dir / f"{run_name}.steps.jsonl"
        self.episodes_path = self.log_dir / f"{run_name}.episodes.jsonl"

    def log_step(self, episode_index: int, record: StepRecord) -> None:
        row = {"episode": episode_index, "logged_at": time.time(), **asdict(record)}
        with self.steps_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def log_episode(self, result: EpisodeResult) -> None:
        row = {"logged_at": time.time(), **asdict(result)}
        with self.episodes_path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _override_rate(episode_rows: list[dict]) -> float | None:
    """Share of all executed steps that a controller safety net overrode.
    None for logs written before EpisodeResult.overridden_steps existed."""
    rows = [r for r in episode_rows if "overridden_steps" in r]
    steps = sum(r.get("steps") or 0 for r in rows)
    return sum(r["overridden_steps"] for r in rows) / steps if steps else None


def summarize_episodes(episode_rows: list[dict]) -> dict:
    """Aggregate stats across a list of episode-row dicts (as produced by
    MetricsLogger.log_episode / load_jsonl), used by experiments/compare.py
    to build the controller-vs-controller table."""
    if not episode_rows:
        return {"episodes": 0}

    n = len(episode_rows)

    def avg(key: str) -> float | None:
        vals = [r[key] for r in episode_rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    action_totals: dict[str, int] = {}
    for row in episode_rows:
        for name, count in (row.get("action_counts") or {}).items():
            action_totals[name] = action_totals.get(name, 0) + count

    return {
        "episodes": n,
        "mean_steps": avg("steps"),
        "mean_survival_tics": avg("survival_tics"),
        "mean_distance_travelled": avg("distance_travelled"),
        "mean_kills": avg("kills"),
        "mean_damage_given": avg("damage_given"),
        "mean_damage_taken": avg("damage_taken"),
        "mean_health_remaining": avg("health_remaining"),
        "mean_ammo_used": avg("ammo_used"),
        "mean_items_collected": avg("items_collected"),
        "mean_total_reward": avg("total_reward"),
        "death_rate": sum(1 for r in episode_rows if r.get("died")) / n,
        "completion_rate": sum(1 for r in episode_rows if r.get("completed")) / n,
        "override_rate": _override_rate(episode_rows),
        "mean_confidence": avg("mean_confidence"),
        "mean_latency_ms": avg("mean_latency_ms"),
        "action_distribution": action_totals,
    }
