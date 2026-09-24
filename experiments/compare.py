"""CLI: run several controllers over the same episode budget and print a
side-by-side comparison table.

    python -m experiments.compare --controllers random heuristic laya --episodes 100
"""

from __future__ import annotations

import argparse

from laya_doom.controller import run_episode
from laya_doom.doom_env import DoomEnv, DoomEnvConfig
from laya_doom.metrics import MetricsLogger, load_jsonl, summarize_episodes
from laya_doom.perception import PerceptionConfig
from laya_doom.state_encoder import EncoderConfig, StateEncoder

from .run import _MEMORY_ALIASES, _build_agent, add_laya_args, add_safety_net_args, safety_nets_from_args

_TABLE_METRICS = [
    "episodes",
    "mean_steps",
    "mean_survival_tics",
    "mean_kills",
    "mean_damage_given",
    "mean_damage_taken",
    "mean_health_remaining",
    "mean_items_collected",
    "mean_total_reward",
    "death_rate",
    "completion_rate",
    "override_rate",
    "mean_confidence",
    "mean_latency_ms",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare controllers over the same episode budget.")
    p.add_argument(
        "--controllers", nargs="+", choices=["random", "heuristic", "laya"], default=["random", "heuristic", "laya"]
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--scenario", default="basic", help='"basic", "my_way_home", or "level" for a real Freedoom map')
    p.add_argument("--doom-map", default=None, help='map name for --scenario level (default "MAP01")')
    p.add_argument(
        "--action-set",
        choices=["stage1", "full"],
        default=None,
        help='default: "full" for --scenario level, "stage1" otherwise',
    )
    p.add_argument("--decision-tics", type=int, default=None)
    p.add_argument("--memory", default="prev_state")
    p.add_argument("--max-steps", type=int, default=500)
    add_laya_args(p)
    add_safety_net_args(p)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-dir", default="logs")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    memory_mode = _MEMORY_ALIASES.get(str(args.memory), "prev_state")
    if args.action_set is None:
        args.action_set = "full" if args.scenario == "level" else "stage1"

    summaries: dict[str, dict] = {}
    for controller in args.controllers:
        run_name = f"compare_{controller}_{args.scenario}_{args.action_set}"
        logger = MetricsLogger(args.log_dir, run_name)
        env = DoomEnv(
            DoomEnvConfig(
                scenario=args.scenario,
                action_set=args.action_set,
                window_visible=False,
                uniform_tics=args.decision_tics,
                seed=args.seed,
                doom_map=args.doom_map,
            sectors_info_enabled=args.door_first,
            )
        )
        agent = _build_agent(argparse.Namespace(**{**vars(args), "controller": controller}))
        encoder = StateEncoder(EncoderConfig(memory_mode=memory_mode))
        perception_config = PerceptionConfig()
        safety_nets = safety_nets_from_args(args)

        try:
            for ep in range(args.episodes):
                result, _ = run_episode(
                    env,
                    agent,
                    encoder,
                    perception_config=perception_config,
                    max_steps=args.max_steps,
                    episode_index=ep,
                    **safety_nets,
                )
                logger.log_episode(result)
                print(f"[{controller}] episode {ep}: steps={result.steps} kills={result.kills} died={result.died}")
        finally:
            env.close()

        summaries[controller] = summarize_episodes(load_jsonl(logger.episodes_path))

    _print_table(summaries)
    return 0


def _print_table(summaries: dict[str, dict]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        for controller, summary in summaries.items():
            print(controller, summary)
        return

    console = Console()
    table = Table(title="Controller comparison")
    table.add_column("metric")
    for controller in summaries:
        table.add_column(controller)

    for key in _TABLE_METRICS:
        row = [key]
        for controller in summaries:
            value = summaries[controller].get(key)
            if value is None:
                row.append("n/a")
            elif isinstance(value, float):
                row.append(f"{value:.2f}")
            else:
                row.append(str(value))
        table.add_row(*row)
    console.print(table)


if __name__ == "__main__":
    raise SystemExit(main())
