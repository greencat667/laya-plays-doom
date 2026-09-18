"""CLI: run N episodes of one controller.

    python -m experiments.run --controller laya --episodes 100 \\
        --decision-tics 4 --memory 1 --scenario basic

Logs every step and episode to --log-dir (JSONL) and prints a summary.
"""

from __future__ import annotations

import argparse
import sys

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
from laya_doom.doom_env import WINDOW_SCALE_RESOLUTIONS, DoomEnv, DoomEnvConfig
from laya_doom.metrics import MetricsLogger, load_jsonl, summarize_episodes
from laya_doom.perception import PerceptionConfig
from laya_doom.state_encoder import EncoderConfig, StateEncoder

from .heuristic_agent import HeuristicAgent
from .random_agent import RandomAgent

# --memory accepts the self-documenting names, or the numbered list from the
# brief (1. stateless, 2. previous-state + current-state, 3. tiny rolling
# memory) as a convenience alias.
_MEMORY_ALIASES = {
    "0": "stateless",
    "1": "prev_state",
    "2": "rolling",
    "stateless": "stateless",
    "prev_state": "prev_state",
    "prev-state": "prev_state",
    "rolling": "rolling",
}


def _build_agent(args: argparse.Namespace):
    if args.controller == "random":
        return RandomAgent(action_set=args.action_set, seed=args.seed)
    if args.controller == "heuristic":
        return HeuristicAgent(action_set=args.action_set)
    if args.controller == "laya":
        from laya_doom.laya_agent import LayaAgent

        return LayaAgent(
            action_set=args.action_set,
            model_id=args.model_id,
            device=args.device,
            confidence_mode=args.confidence_mode,
            confidence_threshold=args.confidence_threshold,
            use_shoot_gate=not args.no_shoot_gate,
            shoot_gate_threshold=args.shoot_gate_threshold,
        )
    raise ValueError(f"unknown controller: {args.controller}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run N episodes of one laya-doom controller.")
    p.add_argument("--controller", choices=["random", "heuristic", "laya"], required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--scenario", default="basic", help='"basic", "my_way_home", or "level" for a real Freedoom map')
    p.add_argument("--doom-map", default=None, help='map name for --scenario level (default "MAP01")')
    p.add_argument(
        "--action-set",
        choices=["stage1", "full"],
        default=None,
        help='default: "full" for --scenario level (needs use() for doors), "stage1" otherwise',
    )
    p.add_argument(
        "--decision-tics",
        type=int,
        default=None,
        help="uniform tic count per action, overriding per-action defaults — the control-frequency knob",
    )
    p.add_argument("--memory", default="prev_state", help="stateless|prev_state|rolling (or 0/1/2)")
    p.add_argument("--rolling-window", type=int, default=3)
    p.add_argument("--no-goal-line", action="store_true", help="omit the GOAL line from the encoded state")
    p.add_argument(
        "--confidence-mode", choices=["always_execute", "confidence_threshold"], default="always_execute"
    )
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.15,
        help="see LayaAgent's docstring for why this defaults much lower than a tool-calling model's threshold would",
    )
    p.add_argument(
        "--model-id",
        default="convaiinnovations/laya",
        help='HuggingFace model id or local path for laya.load() — the Laya-side equivalent of the sibling '
        "project's --weights model-ladder lever, except Laya ships one checkpoint, not a depth ladder",
    )
    p.add_argument(
        "--device",
        default=None,
        help='torch device for laya.load() — omit to auto-detect (cuda > mps > cpu); this machine auto-detects "mps"',
    )
    p.add_argument(
        "--no-shoot-gate",
        action="store_true",
        help="use one `choice` call over every label including combat (the original design) instead of the "
        "default two-call design (a separate `noul` should_shoot gate + an AMMO>0 guard, then a movement "
        "choice over the rest) — see laya_agent.py's module-level comment for why the gate is the default: "
        "shoot/attack never won the multi-way choice on its own (0/8 on hand-built states)",
    )
    p.add_argument(
        "--shoot-gate-threshold",
        type=float,
        default=0.45,
        help="P(should_shoot) cutoff for the shoot gate — picked from a real 8-state spread (see "
        "laya_agent.py), not a calibrated cutoff; only applies when the gate is enabled",
    )
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument(
        "--no-stuck-recovery",
        action="store_true",
        help="disable the safety net that forces a turn after repeated no-progress moves (see controller.StuckRecoveryConfig) — use this to measure Laya's raw, unassisted behaviour",
    )
    p.add_argument(
        "--no-threat-response",
        action="store_true",
        help="disable the safety net that forces a turn after taking damage from an unseen attacker (see controller.ThreatResponseConfig)",
    )
    p.add_argument(
        "--no-turn-loop-recovery",
        action="store_true",
        help="disable the safety net that forces a move_forward attempt after too many consecutive turn "
        "actions with no move ever proposed (see controller.TurnLoopRecoveryConfig) — found on a real "
        "--scenario level run that got stuck turning in a corner even after a path opened up",
    )
    p.add_argument(
        "--no-threat-engagement",
        action="store_true",
        help="disable the safety net that turns toward a visible, near-enough, off-center enemy instead of "
        "letting Laya's movement choice stand (see controller.ThreatEngagementConfig) — found on a real "
        "--scenario level run that died to one zombieman it never turned to face",
    )
    p.add_argument(
        "--no-low-health-retreat",
        action="store_true",
        help="disable the safety net that retreats when a visible enemy is present and health is low (see "
        "controller.LowHealthRetreatConfig) — ported from the sibling Needle project",
    )
    p.add_argument(
        "--low-health-threshold",
        type=int,
        default=20,
        help="health at or below which low-health-retreat kicks in with a visible enemy, unless Laya already chose attack/shoot (default 20)",
    )
    p.add_argument(
        "--emergency-health-threshold",
        type=int,
        default=10,
        help="health at or below which low-health-retreat overrides even an attack/shoot decision (default 10)",
    )
    p.add_argument(
        "--no-exploration-nudge",
        action="store_true",
        help="disable the circling-breaker safety net (see controller.ExplorationNudgeConfig) — a first pass "
        "at exit-seeking, not real wayfinding — ported from the sibling Needle project",
    )
    p.add_argument(
        "--exploration-streak-threshold",
        type=int,
        default=15,
        help="consecutive already-visited-cell decisions (with no enemy/pickup) before exploration-nudge forces a heading change (default 15)",
    )
    p.add_argument(
        "--no-frontier-exploration",
        action="store_true",
        help="make exploration-nudge's forced heading change blind again (the old open_left/open_right/alternate "
        "guess) instead of the default directed guess toward the nearest known-unvisited grid cell, computed "
        "from the now-verified ANGLE game variable (see controller.FrontierExplorationConfig / wayfinding.py "
        "and the README's ANGLE-verification section) — for regression/comparison",
    )
    p.add_argument(
        "--frontier-lookahead-cells",
        type=float,
        default=2.0,
        help="how many area_cell_size-unit cells ahead each candidate heading is projected when picking a "
        "directed exploration-nudge turn (default 2.0)",
    )
    p.add_argument(
        "--no-door-use",
        action="store_true",
        help="disable the safety net that tries `use` once after WALL ahead near holds for --door-use-stall-"
        "threshold consecutive steps (see controller.DoorUseConfig) — added after real logged data showed "
        "`use` never wins Laya's crowded movement choice (max probability 0.169 over 2,950 real decisions, "
        "0 times chosen) no matter how close to a wall/door",
    )
    p.add_argument(
        "--door-use-stall-threshold",
        type=int,
        default=3,
        help="consecutive WALL-ahead-near steps before door-use tries `use` once (default 3)",
    )
    p.add_argument(
        "--no-secret-search",
        action="store_true",
        help="disable the systematic turn-and-use sequence that runs once frontier exploration is exhausted "
        "(no reachable unvisited territory nearby) — see controller.SecretSearchConfig. Doom/Freedoom secret "
        "doors look identical to ordinary walls, so this tries `use` against several nearby wall-facing "
        "directions instead of only whichever single wall is currently faced",
    )
    p.add_argument(
        "--no-wall-follow",
        action="store_true",
        help="disable the classic right/left-hand wall-following maze fallback that engages once plain "
        "nudging and a full secret-door sweep have both come up empty (see controller.WallFollowConfig)",
    )
    p.add_argument("--wall-follow-hand", choices=["left", "right"], default="right")
    p.add_argument("--render", action="store_true", help="open a visible Doom window")
    p.add_argument(
        "--window-scale",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Doom window size with --render: 1=320x240, 2=640x480 (exact 2x), 3=1024x768 (closest 4:3 preset "
        "to 3x — ViZDoom has no exact 3x). Purely a display size; perception.py samples by fraction of "
        "width/height, so behavior is unaffected — ported from the sibling Needle project.",
    )
    p.add_argument("--dashboard", action="store_true", help="show the live terminal dashboard")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--run-name", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    memory_mode = _MEMORY_ALIASES.get(str(args.memory))
    if memory_mode is None:
        print(f"invalid --memory value: {args.memory!r} (want stateless|prev_state|rolling|0|1|2)", file=sys.stderr)
        return 2
    if args.action_set is None:
        args.action_set = "full" if args.scenario == "level" else "stage1"

    run_name = args.run_name or f"{args.controller}_{args.scenario}_{args.action_set}"
    logger = MetricsLogger(args.log_dir, run_name)

    env = DoomEnv(
        DoomEnvConfig(
            scenario=args.scenario,
            action_set=args.action_set,
            window_visible=args.render,
            uniform_tics=args.decision_tics,
            seed=args.seed,
            doom_map=args.doom_map,
            screen_resolution=WINDOW_SCALE_RESOLUTIONS[args.window_scale],
        )
    )
    agent = _build_agent(args)
    encoder = StateEncoder(
        EncoderConfig(
            memory_mode=memory_mode,
            rolling_window=args.rolling_window,
            include_goal_line=not args.no_goal_line,
        )
    )
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

    dashboard = None
    if args.dashboard:
        from laya_doom.dashboard import TerminalDashboard

        dashboard = TerminalDashboard().__enter__()

    try:
        for ep in range(args.episodes):
            records_so_far: list = []

            def on_step(perception, decision, record, _records=records_so_far, _ep=ep):
                _records.append(record)
                logger.log_step(_ep, record)
                if dashboard is not None:
                    from laya_doom.dashboard import compute_running_stats

                    dashboard.update(perception, decision, record, compute_running_stats(_ep, _records))

            result, _ = run_episode(
                env,
                agent,
                encoder,
                perception_config=perception_config,
                max_steps=args.max_steps,
                episode_index=ep,
                on_step=on_step,
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
            if dashboard is not None:
                dashboard.episode_done(result)
            else:
                print(
                    f"episode {ep}: steps={result.steps} kills={result.kills} "
                    f"health={result.health_remaining} died={result.died} "
                    f"reward={result.total_reward:.1f} mean_conf={result.mean_confidence} "
                    f"mean_latency_ms={result.mean_latency_ms:.1f}"
                )
    finally:
        if dashboard is not None:
            dashboard.__exit__(None, None, None)
        env.close()

    summary = summarize_episodes(load_jsonl(logger.episodes_path))
    print("\n=== summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nlogs: {logger.steps_path}\n      {logger.episodes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
