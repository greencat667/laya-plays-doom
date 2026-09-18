"""CLI: run N episodes of one controller.

    python -m experiments.run --controller laya --episodes 100 \\
        --decision-tics 4 --memory 1 --scenario basic

Logs every step and episode to --log-dir (JSONL) and prints a summary.
"""

from __future__ import annotations

import argparse
import sys

from laya_doom.controller import StuckRecoveryConfig, ThreatResponseConfig, run_episode
from laya_doom.doom_env import DoomEnv, DoomEnvConfig
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
    p.add_argument("--render", action="store_true", help="open a visible Doom window")
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
