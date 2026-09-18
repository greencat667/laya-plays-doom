"""CLI: run several controllers over the same episode budget and print a
side-by-side comparison table.

    python -m experiments.compare --controllers random heuristic laya --episodes 100
"""

from __future__ import annotations

import argparse

from laya_doom.controller import (
    DoorUseConfig,
    ExplorationNudgeConfig,
    FrontierExplorationConfig,
    LowHealthRetreatConfig,
    SecretSearchConfig,
    StuckRecoveryConfig,
    ThreatEngagementConfig,
    ThreatResponseConfig,
    TurnLoopRecoveryConfig,
    WallFollowConfig,
    run_episode,
)
from laya_doom.doom_env import DoomEnv, DoomEnvConfig
from laya_doom.metrics import MetricsLogger, load_jsonl, summarize_episodes
from laya_doom.perception import PerceptionConfig
from laya_doom.state_encoder import EncoderConfig, StateEncoder

from .run import _MEMORY_ALIASES, _build_agent

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
    p.add_argument("--confidence-mode", choices=["always_execute", "confidence_threshold"], default="always_execute")
    p.add_argument("--confidence-threshold", type=float, default=0.15)
    p.add_argument("--model-id", default="convaiinnovations/laya")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--no-shoot-gate",
        action="store_true",
        help="see experiments/run.py --no-shoot-gate — applies to the laya controller only",
    )
    p.add_argument("--shoot-gate-threshold", type=float, default=0.45)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument(
        "--no-stuck-recovery",
        action="store_true",
        help="disable the no-progress-move safety net for all controllers (see controller.StuckRecoveryConfig)",
    )
    p.add_argument(
        "--no-threat-response",
        action="store_true",
        help="disable the unseen-attacker safety net for all controllers (see controller.ThreatResponseConfig)",
    )
    p.add_argument(
        "--no-turn-loop-recovery",
        action="store_true",
        help="disable the stuck-turning safety net for all controllers (see controller.TurnLoopRecoveryConfig)",
    )
    p.add_argument(
        "--no-threat-engagement",
        action="store_true",
        help="disable the turn-toward-off-center-threat safety net for all controllers (see controller.ThreatEngagementConfig)",
    )
    p.add_argument(
        "--no-low-health-retreat",
        action="store_true",
        help="disable the low-health retreat safety net for all controllers (see controller.LowHealthRetreatConfig)",
    )
    p.add_argument("--low-health-threshold", type=int, default=20)
    p.add_argument("--emergency-health-threshold", type=int, default=10)
    p.add_argument(
        "--no-exploration-nudge",
        action="store_true",
        help="disable the circling-breaker safety net for all controllers (see controller.ExplorationNudgeConfig)",
    )
    p.add_argument("--exploration-streak-threshold", type=int, default=15)
    p.add_argument(
        "--no-frontier-exploration",
        action="store_true",
        help="disable directed (ANGLE-based) exploration-nudge turns for all controllers, reverting to the old "
        "blind guess (see controller.FrontierExplorationConfig)",
    )
    p.add_argument("--frontier-lookahead-cells", type=float, default=2.0)
    p.add_argument(
        "--no-door-use",
        action="store_true",
        help="disable the try-use-against-a-stalled-wall safety net for all controllers (see controller.DoorUseConfig)",
    )
    p.add_argument("--door-use-stall-threshold", type=int, default=3)
    p.add_argument(
        "--no-secret-search",
        action="store_true",
        help="disable the systematic turn-and-use sequence for all controllers (see controller.SecretSearchConfig)",
    )
    p.add_argument(
        "--no-wall-follow",
        action="store_true",
        help="disable the wall-following maze fallback for all controllers (see controller.WallFollowConfig)",
    )
    p.add_argument("--wall-follow-hand", choices=["left", "right"], default="right")
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
            )
        )
        agent_args = argparse.Namespace(
            controller=controller,
            action_set=args.action_set,
            seed=args.seed,
            model_id=args.model_id,
            device=args.device,
            confidence_mode=args.confidence_mode,
            confidence_threshold=args.confidence_threshold,
            no_shoot_gate=args.no_shoot_gate,
            shoot_gate_threshold=args.shoot_gate_threshold,
        )
        agent = _build_agent(agent_args)
        encoder = StateEncoder(EncoderConfig(memory_mode=memory_mode))
        perception_config = PerceptionConfig()
        stuck_recovery = StuckRecoveryConfig(enabled=not args.no_stuck_recovery)
        threat_response = ThreatResponseConfig(enabled=not args.no_threat_response)
        turn_loop_recovery = TurnLoopRecoveryConfig(enabled=not args.no_turn_loop_recovery)
        threat_engagement = ThreatEngagementConfig(enabled=not args.no_threat_engagement)
        low_health_retreat = LowHealthRetreatConfig(
            enabled=not args.no_low_health_retreat,
            health_threshold=args.low_health_threshold,
            emergency_health_threshold=args.emergency_health_threshold,
        )
        exploration_nudge = ExplorationNudgeConfig(
            enabled=not args.no_exploration_nudge, streak_threshold=args.exploration_streak_threshold
        )
        frontier_exploration = FrontierExplorationConfig(
            enabled=not args.no_frontier_exploration, lookahead_cells=args.frontier_lookahead_cells
        )
        door_use = DoorUseConfig(enabled=not args.no_door_use, stall_threshold=args.door_use_stall_threshold)
        secret_search = SecretSearchConfig(enabled=not args.no_secret_search)
        wall_follow = WallFollowConfig(enabled=not args.no_wall_follow, hand=args.wall_follow_hand)

        try:
            for ep in range(args.episodes):
                result, _ = run_episode(
                    env,
                    agent,
                    encoder,
                    perception_config=perception_config,
                    max_steps=args.max_steps,
                    episode_index=ep,
                    stuck_recovery=stuck_recovery,
                    threat_response=threat_response,
                    turn_loop_recovery=turn_loop_recovery,
                    threat_engagement=threat_engagement,
                    low_health_retreat=low_health_retreat,
                    exploration_nudge=exploration_nudge,
                    door_use=door_use,
                    frontier_exploration=frontier_exploration,
                    secret_search=secret_search,
                    wall_follow=wall_follow,
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
