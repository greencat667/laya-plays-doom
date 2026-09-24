"""A Karpathy-style autoresearch loop, adapted honestly to what this
project actually has: this script is the tireless experiment executor,
not the hypothesis generator. Each invocation runs ONE small, named,
falsifiable change against the current defaults, paired on identical
seeds, and appends the real result to autoresearch_log.jsonl -- win or
lose. Picking the *next* hypothesis from what the log just showed is a
human/agent-in-the-loop step, deliberately not automated here: a static
script cannot read `docs/doom-strategy-research.md`, notice a real run's
`override_reason` counts looked wrong, and propose a genuinely new idea --
that's the part of "autoresearch" that still needs a researcher, human or
Claude, reading the log between rounds (``autoresearch log``) and adding
the next hypothesis to HYPOTHESES below.

    python -m experiments.autoresearch list
    python -m experiments.autoresearch run wall_follow_longer --episodes 8
    python -m experiments.autoresearch log

Every round is a PAIRED comparison: the same seeds run once against the
unmodified defaults ("baseline") and once against defaults-plus-one-change
("candidate"), so the delta isn't contaminated by which seeds happened to
land on an easy vs. hard start. The accept bar is a stated heuristic, not
a real significance test -- see run_round's docstring -- consistent with
this project's own "report the real number, not a p-value we didn't
actually compute" standard.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

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
from laya_doom.laya_agent import LayaAgent
from laya_doom.perception import PerceptionConfig
from laya_doom.planner import FrontierPlannerConfig
from laya_doom.state_encoder import EncoderConfig, StateEncoder

LOG_PATH = Path(__file__).resolve().parent / "autoresearch_log.jsonl"


def _json_default(obj):
    """A hypothesis's overrides can legitimately contain a frozenset (see
    threat_engage_wider/threat_engage_narrow's engage_distances) -- json
    only knows how to serialize a list, so convert deterministically
    (sorted) rather than letting json.dumps raise on it."""
    if isinstance(obj, (frozenset, set)):
        return sorted(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# One entry per round, ever. Never delete or edit an old one after it's
# been run -- the log references these by id, and the whole point of the
# loop is an honest, append-only record of what was tried.
HYPOTHESES: dict[str, dict] = {
    "control_noop": {
        "description": (
            "Sanity check on the harness itself, not a real hypothesis: candidate == baseline "
            "(no overrides at all). If this doesn't come back with ~zero delta on every metric, "
            "the paired-seed comparison isn't actually reproducible and nothing else in this log "
            "can be trusted."
        ),
        "overrides": {},
    },
    "wall_follow_longer": {
        "description": (
            "Let the last-resort wall-follow fallback run for longer before giving up "
            "(60 -> 150 steps) -- real Freedoom corridors may need more than 60 steps of "
            "hand-on-wall travel to actually reach a door."
        ),
        "overrides": {"wall_follow": {"max_follow_steps": 150}},
    },
    "wall_follow_left_hand": {
        "description": (
            "Left-hand rule instead of right-hand -- MAP01's layout from the default spawn "
            "may be more left-hand-friendly."
        ),
        "overrides": {"wall_follow": {"hand": "left"}},
    },
    "secret_search_more_headings": {
        "description": (
            "Sweep 8 headings (every 45 degrees) instead of 4 (every 90) once secret-search "
            "escalates -- a secret door that isn't exactly ahead/behind/either-side is currently "
            "never tried."
        ),
        "overrides": {
            "secret_search": {"headings_deg": (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)}
        },
    },
    "tighter_frontier_lookahead": {
        "description": (
            "Project frontier-exploration headings only 1 cell ahead instead of 2 -- a shorter "
            "lookahead is less likely to project across a wall the pipeline can't see."
        ),
        "overrides": {"frontier_exploration": {"lookahead_cells": 1.0}},
    },
    "faster_circling_trigger": {
        "description": (
            "Fire the circling detector after 8 revisited-cell decisions instead of 15, so "
            "wayfinding escalation kicks in sooner on a real run."
        ),
        "overrides": {"exploration_nudge": {"streak_threshold": 8}},
    },
    "wall_follow_disabled": {
        "description": (
            "Two rounds in (wall_follow_longer, wall_follow_left_hand) both showed WORSE new-cell "
            "coverage and kills than baseline, regardless of hand or duration -- raising the sharper "
            "question of whether the mechanism helps at all on these seeds. Disable it entirely "
            "(secret-search's turn-and-use sweep still runs; only the last-resort wall-follow rung "
            "is removed) and compare against the same unmodified baseline."
        ),
        "overrides": {"wall_follow": {"enabled": False}},
    },
    "earlier_wall_follow_escalation": {
        "description": (
            "Skip straight to wall-following after the very first failed secret-search cycle "
            "(escalate_after 2 -> 1 stays the same for secret-search, but wall-follow's own "
            "activate_after_searches goes 1 -> 0-equivalent isn't valid, so this instead makes "
            "secret-search escalate to wall-follow after 1 nudge instead of 2) -- reach the "
            "formally-guaranteed fallback sooner."
        ),
        "overrides": {"secret_search": {"escalate_after": 1}},
    },
    "wall_follow_shorter": {
        "description": (
            "Opposite direction from wall_follow_longer, which made coverage/kills worse at 150 "
            "steps: try LESS time per wall-follow attempt (60 -> 25) instead of more, on the "
            "theory that a shorter, more frequently-re-evaluated attempt loses less time to a "
            "blind rule when it isn't working, and hands back to Laya's own judgment sooner."
        ),
        "overrides": {"wall_follow": {"max_follow_steps": 25}},
    },
    "wall_follow_conservative_activation": {
        "description": (
            "wall_follow_longer and wall_follow_left_hand both showed the mechanism itself may be "
            "net-negative on these seeds (see wall_follow_disabled). Rather than removing it "
            "outright, make it much more reluctant to engage: require 3 failed secret-search "
            "cycles in a row (instead of 1) before handing control to the blind maze rule at all."
        ),
        "overrides": {"wall_follow": {"activate_after_searches": 3}},
    },
    "secret_search_diagonal_headings": {
        "description": (
            "The default secret-search sweep checks the 4 cardinal headings relative to the wall "
            "it triggered against (0/90/180/270 -- ahead, both sides, behind). Try the 4 diagonal "
            "headings instead (45/135/225/315), on the theory that a secret door built into a "
            "corner or an angled corridor junction is currently never faced squarely."
        ),
        "overrides": {"secret_search": {"headings_deg": (45.0, 135.0, 225.0, 315.0)}},
    },
    "secret_search_slower_escalation": {
        "description": (
            "Opposite direction from earlier_wall_follow_escalation: require 4 failed nudge "
            "cycles (instead of 2) before escalating to the active turn-and-use sweep, giving "
            "plain directed nudging more chances to find real new ground on its own first."
        ),
        "overrides": {"secret_search": {"escalate_after": 4}},
    },
    "frontier_lookahead_wider": {
        "description": (
            "Opposite direction from tighter_frontier_lookahead: project candidate headings "
            "4 cells ahead instead of 2, on the theory that a longer lookahead more reliably "
            "distinguishes a heading that leads somewhere new from one that loops back into "
            "already-visited cells nearby."
        ),
        "overrides": {"frontier_exploration": {"lookahead_cells": 4.0}},
    },
    "slower_circling_trigger": {
        "description": (
            "Opposite direction from faster_circling_trigger: require 25 revisited-cell "
            "decisions in a row (instead of 15) before declaring 'stuck circling', in case 15 "
            "is cutting off legitimate exploration of a large, genuinely-connected room too early."
        ),
        "overrides": {"exploration_nudge": {"streak_threshold": 25}},
    },
    "door_use_eager": {
        "description": (
            "Try `use` the moment a wall has been near for just 1 step (instead of 3), on the "
            "theory that the 3-step wait is losing real door/switch opportunities to whatever "
            "other net fires first in the meantime."
        ),
        "overrides": {"door_use": {"stall_threshold": 1}},
    },
    "door_use_patient": {
        "description": (
            "Wait for a wall to be near for 6 consecutive steps (instead of 3) before trying "
            "`use`, on the theory that 3 steps is too eager and fires on walls the agent is "
            "about to turn away from anyway (a real, if harmless, wasted attempt either way)."
        ),
        "overrides": {"door_use": {"stall_threshold": 6}},
    },
    "threat_engage_wider": {
        "description": (
            "Widen ThreatEngagementConfig to also turn toward a 'far' off-center enemy, not just "
            "very-near/near/medium -- on the theory that reacting to a threat earlier, before it "
            "closes distance, gives more time to actually win the engagement."
        ),
        "overrides": {"threat_engagement": {"engage_distances": frozenset({"very-near", "near", "medium", "far"})}},
    },
    "threat_engage_narrow": {
        "description": (
            "Narrow ThreatEngagementConfig to only very-near/near (dropping medium), on the "
            "theory that turning toward a merely medium-distance off-center enemy interrupts "
            "navigation/exploration more often than it needs to for a threat that isn't yet close."
        ),
        "overrides": {"threat_engagement": {"engage_distances": frozenset({"very-near", "near"})}},
    },
    "turn_loop_more_patient": {
        "description": (
            "Allow 10 consecutive executed turns (instead of 6) before forcing a move_forward "
            "test, in case 6 is cutting off a legitimate multi-turn scan (e.g. searching a wide "
            "room) before it finds a real heading to commit to."
        ),
        "overrides": {"turn_loop_recovery": {"max_consecutive_turns": 10}},
    },
    "turn_loop_less_patient": {
        "description": (
            "Force a move_forward test after only 3 consecutive executed turns (instead of 6), "
            "on the theory that 6 lets a genuinely stuck spin waste more steps than necessary "
            "before the recovery net intervenes."
        ),
        "overrides": {"turn_loop_recovery": {"max_consecutive_turns": 3}},
    },
    "stuck_recovery_stricter_progress": {
        "description": (
            "Require 8 game units of real movement (instead of 3) to count as genuine progress "
            "for StuckRecoveryConfig's purposes, on the theory that 3 units is small enough to "
            "let near-stationary shuffling (e.g. sliding along a wall without net displacement) "
            "slip through as 'progress' and delay the recovery turn it should be triggering."
        ),
        "overrides": {"stuck_recovery": {"min_progress_units": 8.0}},
    },
    "low_health_more_cautious": {
        "description": (
            "Retreat earlier and more conservatively: health_threshold 20 -> 35, "
            "emergency_health_threshold 10 -> 20 -- ported motivation directly from ThreatEngagementConfig's "
            "documented death (51 -> 10 -> 4 -> 0 against one zombieman, never disengaging "
            "until truly critical); test whether disengaging even earlier measurably reduces "
            "deaths without costing meaningful kills/coverage."
        ),
        "overrides": {"low_health_retreat": {"health_threshold": 35, "emergency_health_threshold": 20}},
    },
    "door_use_aimed": {
        "description": (
            "Square up to the nearest cardinal heading before door_use presses `use`, then walk through if "
            "the wall clears -- MAP01's only door out of the start area (x=736) was reached on seeds "
            "5000/5001 but pressed at 33 degrees, hitting the frame; a replay aimed at 0 degrees went through"
        ),
        "overrides": {"door_use": {"aim": True}},
    },
    "frontier_planner": {
        "description": (
            "When circling is detected, travel along cell-to-cell moves already made to the nearest untried "
            "edge of explored space, face it squarely, push forward, and press use once if blocked "
            "(laya_doom/planner.py) -- MAP01's 26-cell ceiling is one door the agent reached for 17/1200 steps"
        ),
        "overrides": {"frontier_planner": {"enabled": True}},
    },
    "frontier_planner_door_first": {
        "description": (
            "frontier_planner, but frontiers facing a door-shaped sector (thin, 8-24 units deep, two passable "
            "sides, from ViZDoom's level geometry read once at episode start) are tried before plain walls -- "
            "on seed 5000 the plain planner pressed ~25 plain walls before reaching MAP01's first door"
        ),
        "overrides": {"frontier_planner": {"door_first": True}},
        "env_overrides": {"sectors_info_enabled": True},
    },
    "frontier_planner_door_first_v2": {
        "description": (
            "door_first, fixed after v1 lost: door missions walk to 32 units in front of the doorway's centre "
            "before pressing, press any door at most once per 40 steps (a second press shuts an open door -- "
            "confirmed by replay), and retry a failed door once after the cooldown instead of exhausting it"
        ),
        "overrides": {"frontier_planner": {"door_first": True}},
        "env_overrides": {"sectors_info_enabled": True},
    },
    "laya_distilled_head": {
        "description": (
            "Laya's movement choice answered by a stuntd-trained head on the frozen encoder "
            "(scripts/distill_movement_head.py: logged moves that achieved something plus reactive-net "
            "corrections, mirror-augmented) instead of zero-shot. Needs stuntd + laya>=0.3.4 in the env "
            "running BOTH arms, so the baseline is zero-shot on the same laya version."
        ),
        "overrides": {},
        "agent_overrides": {"movement_head": "models/movement_head"},
    },
    "laya_distilled_head_frontier": {
        "description": (
            "laya_distilled_head with a map-aware teacher: a FRONTIER <direction> <distance> line (toward the "
            "planner's next waypoint) in the state text for BOTH arms, and a head trained on logs that carry it, "
            "including the planner's own moves -- the first head only copied the reactive nets"
        ),
        "overrides": {},
        "agent_overrides": {"movement_head": "models/movement_head_frontier"},
        "encoder_overrides": {"include_frontier_hint": True},
    },
    "door_first_v2_with_head": {
        "description": (
            "door_first v2 (gets through doors, but deaths 0 -> 6/10 zero-shot) plus the map-aware stuntd head "
            "and its FRONTIER line (0/10 deaths in its own A/B), against the untouched default"
        ),
        "overrides": {"frontier_planner": {"door_first": True}},
        "env_overrides": {"sectors_info_enabled": True},
        "agent_overrides": {"movement_head": "models/movement_head_frontier"},
        "candidate_encoder_overrides": {"include_frontier_hint": True},
    },
    "door_first_v2_with_door_head": {
        "description": (
            "door_first_v2_with_head, but the head retrained on logs of door_first v2 play (seed 6200) plus the "
            "plain-planner FRONTIER logs, so its teacher includes door-first's approach-and-walk-through"
        ),
        "overrides": {"frontier_planner": {"door_first": True}},
        "env_overrides": {"sectors_info_enabled": True},
        "agent_overrides": {"movement_head": "models/movement_head_door"},
        "candidate_encoder_overrides": {"include_frontier_hint": True},
    },
    "direction_resolved": {
        "description": (
            "Laya picks only the action TYPE (strafe/turn_small/turn_large/...) and code picks left/right "
            "from perception (enemy side, pickup side, only-open side, else keep last side) -- because the "
            "real model's own left/right choice is pure label-order bias (scripts/probe_direction_bias.py)"
        ),
        "overrides": {},
        "agent_overrides": {"direction_mode": "resolved"},
    },
}

_CONFIG_FACTORIES = {
    "stuck_recovery": StuckRecoveryConfig,
    "turn_loop_recovery": TurnLoopRecoveryConfig,
    "exploration_nudge": ExplorationNudgeConfig,
    "frontier_exploration": FrontierExplorationConfig,
    "secret_search": SecretSearchConfig,
    "wall_follow": WallFollowConfig,
    "door_use": DoorUseConfig,
    "threat_response": ThreatResponseConfig,
    "threat_engagement": ThreatEngagementConfig,
    "low_health_retreat": LowHealthRetreatConfig,
    "frontier_planner": FrontierPlannerConfig,
}


def _build_configs(overrides: dict) -> dict:
    configs = {name: factory() for name, factory in _CONFIG_FACTORIES.items()}
    for name, fields in overrides.items():
        configs[name] = replace(configs[name], **fields)
    return configs


def _run_arm(
    agent: LayaAgent,
    seeds: list[int],
    configs: dict,
    max_steps: int,
    doom_map: str | None,
    env_overrides: dict | None = None,
    encoder_overrides: dict | None = None,
) -> list[dict]:
    """One arm (baseline or candidate) over the given seeds. Fresh DoomEnv
    + StateEncoder per seed (matching experiments/run.py's own setup);
    the same LayaAgent is reused throughout -- run_episode calls its
    reset() once per episode, and laya_agent.py's own docstring is explicit
    that there's no persistent state between predict() calls for that to
    leak across episodes or arms."""
    rows = []
    for seed in seeds:
        env = DoomEnv(
            DoomEnvConfig(
                scenario="level",
                action_set="full",
                window_visible=False,
                seed=seed,
                doom_map=doom_map,
                **(env_overrides or {}),
            )
        )
        encoder = StateEncoder(EncoderConfig(memory_mode="prev_state", **(encoder_overrides or {})))
        try:
            result, records = run_episode(
                env,
                agent,
                encoder,
                perception_config=PerceptionConfig(),
                max_steps=max_steps,
                episode_index=seed,
                **configs,
            )
        finally:
            env.close()
        rows.append(
            {
                "seed": seed,
                "completed": bool(result.completed),
                "died": bool(result.died),
                "steps": result.steps,
                "distance": result.distance_travelled,
                "new_cells": len(encoder.visited_cells),
                "kills": result.kills,
                "health_remaining": result.health_remaining,
                # Share of executed actions that were Laya's own proposal rather
                # than a controller safety-net override.
                "laya_share": sum(not r.overridden for r in records) / max(len(records), 1),
            }
        )
    return rows


def run_round(
    hypothesis_id: str,
    episodes: int,
    max_steps: int,
    seed_base: int,
    doom_map: str | None,
    model_id: str,
    device: str | None,
) -> dict:
    """Run one paired round and log it.

    Accept bar (a stated heuristic, not a computed p-value -- flagged
    honestly as exactly that): candidate wins if it completes strictly
    more real episodes than baseline (the actual goal), OR its new-cell
    coverage improves by more than one paired standard deviation. Raw
    distance is deliberately NOT part of the bar -- this project's own
    README documents wall-following inflating distance without
    proportional new-ground coverage, so rewarding distance here would
    reward exactly the wrong thing.
    """
    if hypothesis_id not in HYPOTHESES:
        raise SystemExit(f"unknown hypothesis {hypothesis_id!r} -- see `autoresearch list`")
    spec = HYPOTHESES[hypothesis_id]
    seeds = list(range(seed_base, seed_base + episodes))

    baseline_configs = _build_configs({})
    candidate_configs = _build_configs(spec["overrides"])

    baseline_agent = LayaAgent(action_set="full", model_id=model_id, device=device)
    # agent_overrides builds a second LayaAgent variant for the candidate arm
    # (e.g. direction_mode) -- config overrides alone can't express those.
    agent_overrides = spec.get("agent_overrides") or {}
    candidate_agent = (
        LayaAgent(action_set="full", model_id=model_id, device=device, **agent_overrides)
        if agent_overrides
        else baseline_agent
    )

    print(f"[{hypothesis_id}] {spec['description']}")
    t0 = time.perf_counter()
    # encoder_overrides change the state text itself, so they apply to BOTH arms
    # (e.g. a FRONTIER line zero-shot Laya should see too, for a fair comparison).
    encoder_overrides = spec.get("encoder_overrides") or {}
    baseline = _run_arm(baseline_agent, seeds, baseline_configs, max_steps, doom_map, None, encoder_overrides)
    # env_overrides (e.g. sectors_info_enabled) apply to the candidate arm only,
    # so the baseline arm stays byte-identical to earlier rounds.
    env_overrides = spec.get("env_overrides") or {}
    # candidate_encoder_overrides change the candidate arm's state text only, for a
    # candidate compared against the untouched default (e.g. a head trained on a line
    # the default doesn't have).
    candidate_encoder = {**encoder_overrides, **(spec.get("candidate_encoder_overrides") or {})}
    candidate = _run_arm(
        candidate_agent, seeds, candidate_configs, max_steps, doom_map, env_overrides, candidate_encoder
    )
    elapsed = time.perf_counter() - t0

    def _mean(key: str, rows: list[dict]) -> float:
        return statistics.fmean(float(r[key]) for r in rows)

    def _paired_deltas(key: str) -> list[float]:
        return [float(c[key]) - float(b[key]) for b, c in zip(baseline, candidate)]

    metrics = {}
    for key in ("distance", "new_cells", "kills", "steps", "laya_share"):
        deltas = _paired_deltas(key)
        mean_delta = statistics.fmean(deltas)
        stdev_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        metrics[key] = {
            "baseline_mean": _mean(key, baseline),
            "candidate_mean": _mean(key, candidate),
            "mean_delta": mean_delta,
            "stdev_delta": stdev_delta,
        }

    baseline_completed = sum(r["completed"] for r in baseline)
    candidate_completed = sum(r["completed"] for r in candidate)
    baseline_died = sum(r["died"] for r in baseline)
    candidate_died = sum(r["died"] for r in candidate)

    nc = metrics["new_cells"]
    accept = (candidate_completed > baseline_completed) or (
        nc["mean_delta"] > 0 and nc["mean_delta"] > nc["stdev_delta"]
    )
    # Byte-identical arms mean the change never engaged on these seeds (e.g.
    # a health threshold that was never crossed) -- untested, not rejected.
    no_op = baseline == candidate
    decision = "no_op" if no_op else ("accept" if accept else "reject")

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypothesis_id": hypothesis_id,
        "description": spec["description"],
        "overrides": spec["overrides"],
        "agent_overrides": agent_overrides,
        "env_overrides": env_overrides,
        "encoder_overrides": encoder_overrides,
        "candidate_encoder_overrides": spec.get("candidate_encoder_overrides") or {},
        "episodes": episodes,
        "seeds": seeds,
        "max_steps": max_steps,
        "elapsed_s": round(elapsed, 1),
        "baseline_completed": baseline_completed,
        "candidate_completed": candidate_completed,
        "baseline_died": baseline_died,
        "candidate_died": candidate_died,
        "metrics": metrics,
        "decision": decision,
        "baseline_rows": baseline,
        "candidate_rows": candidate,
    }

    with LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")

    _print_summary(record)
    return record


def _print_summary(record: dict) -> None:
    print(f"\n=== {record['hypothesis_id']}: {record['decision'].upper()} ===")
    print(
        f"completed: baseline {record['baseline_completed']}/{record['episodes']}  "
        f"candidate {record['candidate_completed']}/{record['episodes']}"
    )
    print(
        f"died:      baseline {record['baseline_died']}/{record['episodes']}  "
        f"candidate {record['candidate_died']}/{record['episodes']}"
    )
    for key, m in record["metrics"].items():
        print(
            f"{key:>10}: baseline {m['baseline_mean']:7.2f}  candidate {m['candidate_mean']:7.2f}  "
            f"delta {m['mean_delta']:+7.2f} (stdev {m['stdev_delta']:.2f})"
        )
    print(f"log: {LOG_PATH}  ({record['elapsed_s']}s)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Paired autoresearch round for Laya's Doom wayfinding.")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run one paired baseline-vs-candidate round")
    run_p.add_argument("hypothesis_id", choices=sorted(HYPOTHESES))
    run_p.add_argument("--episodes", type=int, default=8)
    run_p.add_argument("--max-steps", type=int, default=800)
    run_p.add_argument("--seed-base", type=int, default=5000)
    run_p.add_argument("--doom-map", default=None)
    run_p.add_argument("--model-id", default="convaiinnovations/laya")
    run_p.add_argument("--device", default=None)

    sub.add_parser("list", help="list known hypotheses")
    sub.add_parser("log", help="print the running research log")

    args = p.parse_args(argv)

    if args.cmd == "list":
        for hid, spec in HYPOTHESES.items():
            print(f"{hid}: {spec['description']}")
        return 0

    if args.cmd == "log":
        if not LOG_PATH.exists():
            print("no rounds logged yet")
            return 0
        for line in LOG_PATH.read_text().splitlines():
            record = json.loads(line)
            nc_delta = record["metrics"]["new_cells"]["mean_delta"]
            print(
                f"{record['timestamp']}  {record['hypothesis_id']:<28} {record['decision']:<7} "
                f"completed {record['candidate_completed']}/{record['episodes']} "
                f"(baseline {record['baseline_completed']}) new_cells delta {nc_delta:+.1f}"
            )
        return 0

    run_round(
        args.hypothesis_id, args.episodes, args.max_steps, args.seed_base, args.doom_map, args.model_id, args.device
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
