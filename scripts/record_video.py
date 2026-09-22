"""Record a real Laya episode as an mp4: the actual Doom screen buffer
(not an OS screen capture — see the "why not screencapture" note below)
side by side with the same terminal-dashboard panel experiments/run.py
--dashboard shows, one frame per decision step held for that step's real
tic duration, so playback speed matches real gameplay.

    python -m scripts.record_video --scenario level --max-steps 400

Why not a real screen recording: macOS's Screen Recording TCC permission
blocks `screencapture`/`CGWindowListCreateImage` for this session with no
override available (confirmed separately, not assumed) — see the README's
recording-tooling section. This sidesteps that entirely by reading
ViZDoom's own internal screen_buffer array (works headless, no window or
OS capture permission needed at all) instead of capturing the display, and
synthesizes the "terminal output" half by rendering the exact same
rich Panels dashboard.py's live TerminalDashboard.update() shows, into a
Console(record=True) buffer instead of a live terminal — genuine text,
not fabricated, just captured a different way. No changes to
controller.py/doom_env.py/dashboard.py: this only uses run_episode's
existing on_step hook plus each env's public .get_state().
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from laya_doom import actions as actions_mod
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
from laya_doom.dashboard import _ACTION_GLYPHS, compute_running_stats
from laya_doom.doom_env import WINDOW_SCALE_RESOLUTIONS, DoomEnv, DoomEnvConfig
from laya_doom.laya_agent import LayaAgent
from laya_doom.perception import PerceptionConfig
from laya_doom.state_encoder import EncoderConfig, StateEncoder

DOOM_TICS_PER_SECOND = 35
PANEL_WIDTH_PX = 480
PANEL_TEXT_COLUMNS = 56

_FONT_CANDIDATES = (
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel_text(perception, decision, record, stats, history: list[str]) -> str:
    """Same panels/fields as dashboard.TerminalDashboard.update(), rendered
    into an in-memory Console instead of a live terminal, plus the
    override info StepRecord carries that the live dashboard doesn't show."""
    state_table = Table.grid(padding=(0, 1))
    state_table.add_row("Health:", str(perception.health))
    state_table.add_row("Armor:", str(perception.armor))
    state_table.add_row("Ammo:", str(perception.ammo))
    if perception.enemies:
        e = perception.enemies[0]
        state_table.add_row("Enemy:", f"{e.kind} / {e.bearing} / {e.distance}")
    else:
        state_table.add_row("Enemy:", "none visible")
    state_table.add_row("Kills so far:", str(perception.killcount))

    laya_table = Table.grid(padding=(0, 1))
    laya_table.add_row("Decision:", decision.action.upper())
    conf = f"{decision.confidence:.2f}" if decision.confidence is not None else "n/a"
    laya_table.add_row("Confidence:", conf)
    laya_table.add_row("Latency:", f"{decision.latency_ms:.0f} ms")
    if record.overridden:
        laya_table.add_row("Override:", f"{record.override_reason} (proposed {record.proposed_action})")
    if decision.reasoning:
        laya_table.add_row("Reasoning:", decision.reasoning[:200])

    stats_table = Table.grid(padding=(0, 1))
    stats_table.add_row("Step:", str(record.step))
    stats_table.add_row("Distance:", f"{stats.get('distance', 0.0):.0f}")
    avg_conf = stats.get("avg_confidence")
    stats_table.add_row("Avg confidence:", f"{avg_conf:.2f}" if avg_conf is not None else "n/a")
    stats_table.add_row("Avg latency:", f"{stats.get('avg_latency_ms', 0.0):.0f} ms")

    body = Group(
        Panel(state_table, title="CURRENT STATE"),
        Panel(laya_table, title="LAYA"),
        Panel(Text("  ".join(history) or "-"), title="LAST ACTIONS"),
        Panel(stats_table, title=f"EPISODE {stats.get('episode', 0)}"),
    )
    buf = io.StringIO()
    console = Console(record=True, width=PANEL_TEXT_COLUMNS, file=buf)
    console.print(body)
    return console.export_text(styles=False)


def _compose_frame(screen_buffer, panel_text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    game_img = Image.fromarray(screen_buffer)
    panel_img = Image.new("RGB", (PANEL_WIDTH_PX, game_img.height), (14, 14, 18))
    draw = ImageDraw.Draw(panel_img)
    y = 6
    for line in panel_text.splitlines():
        draw.text((8, y), line, font=font, fill=(215, 215, 215))
        y += 14
        if y > game_img.height - 14:
            break
    combined = Image.new("RGB", (game_img.width + PANEL_WIDTH_PX, game_img.height), (0, 0, 0))
    combined.paste(game_img, (0, 0))
    combined.paste(panel_img, (game_img.width, 0))
    return combined


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record a real Laya episode as an mp4 (frame + dashboard panel).")
    p.add_argument("--scenario", default="level", help='"basic", "my_way_home", or "level" (default)')
    p.add_argument("--doom-map", default=None)
    p.add_argument("--action-set", choices=["stage1", "full"], default=None)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--model-id", default="convaiinnovations/laya")
    p.add_argument("--device", default=None)
    p.add_argument("--confidence-mode", choices=["always_execute", "confidence_threshold"], default="always_execute")
    p.add_argument("--confidence-threshold", type=float, default=0.15)
    p.add_argument("--no-shoot-gate", action="store_true")
    p.add_argument("--shoot-gate-threshold", type=float, default=0.45)
    p.add_argument("--window-scale", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--wall-follow-hand", choices=["left", "right"], default="right")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="logs/laya_playthrough.mp4")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH — required to encode the recorded frames.")
        return 2
    action_set = args.action_set or ("full" if args.scenario == "level" else "stage1")

    env = DoomEnv(
        DoomEnvConfig(
            scenario=args.scenario,
            action_set=action_set,
            window_visible=False,
            doom_map=args.doom_map,
            seed=args.seed,
            screen_resolution=WINDOW_SCALE_RESOLUTIONS[args.window_scale],
        )
    )
    agent = LayaAgent(
        action_set=action_set,
        model_id=args.model_id,
        device=args.device,
        confidence_mode=args.confidence_mode,
        confidence_threshold=args.confidence_threshold,
        use_shoot_gate=not args.no_shoot_gate,
        shoot_gate_threshold=args.shoot_gate_threshold,
    )
    encoder = StateEncoder(EncoderConfig(memory_mode="prev_state"))
    perception_config = PerceptionConfig()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font = _load_font(11)

    frame_dir = Path(tempfile.mkdtemp(prefix="laya_video_"))
    frame_paths: list[Path] = []
    frame_durations: list[float] = []
    history: list[str] = []
    records_so_far: list = []

    def on_step(perception, decision, record):
        history.append(_ACTION_GLYPHS.get(decision.action, decision.action))
        if len(history) > 12:
            del history[0]
        records_so_far.append(record)
        stats = compute_running_stats(0, records_so_far)
        stats["episode"] = 0
        panel_text = _panel_text(perception, decision, record, stats, history)

        state = env.get_state()
        if state is not None and state.screen_buffer is not None:
            frame = _compose_frame(state.screen_buffer, panel_text, font)
            frame_path = frame_dir / f"frame_{record.step:05d}.png"
            frame.save(frame_path)
            frame_paths.append(frame_path)
            tics = actions_mod.DEFAULT_TICS.get(record.action, 4)
            frame_durations.append(tics / DOOM_TICS_PER_SECOND)

        print(
            f"step {record.step}: action={record.action} "
            f"(proposed={record.proposed_action} overridden={record.overridden} "
            f"reason={record.override_reason!r}) health={perception.health} "
            f"ammo={perception.ammo} kills={perception.killcount} "
            f"conf={decision.confidence} reward={record.reward:.2f}"
        )

    try:
        result, _ = run_episode(
            env,
            agent,
            encoder,
            perception_config=perception_config,
            max_steps=args.max_steps,
            episode_index=0,
            on_step=on_step,
            stuck_recovery=StuckRecoveryConfig(),
            threat_response=ThreatResponseConfig(),
            turn_loop_recovery=TurnLoopRecoveryConfig(),
            threat_engagement=ThreatEngagementConfig(),
            low_health_retreat=LowHealthRetreatConfig(),
            exploration_nudge=ExplorationNudgeConfig(),
            door_use=DoorUseConfig(),
            frontier_exploration=FrontierExplorationConfig(),
            secret_search=SecretSearchConfig(),
            wall_follow=WallFollowConfig(hand=args.wall_follow_hand),
        )
    finally:
        env.close()

    print(
        f"\n=== episode done: steps={result.steps} kills={result.kills} "
        f"health={result.health_remaining} died={result.died} "
        f"completed={result.completed} reward={result.total_reward:.1f} ==="
    )

    if not frame_paths:
        print("no frames captured — nothing to encode")
        shutil.rmtree(frame_dir, ignore_errors=True)
        return 1

    concat_path = frame_dir / "concat.txt"
    with concat_path.open("w") as fh:
        for path, duration in zip(frame_paths, frame_durations):
            fh.write(f"file '{path.name}'\nduration {duration:.4f}\n")
        # concat demuxer quirk: the final image's duration line is ignored
        # unless the last file is also listed once more with no duration.
        fh.write(f"file '{frame_paths[-1].name}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_path.resolve()),
    ]
    proc = subprocess.run(cmd, cwd=frame_dir, capture_output=True, text=True)
    shutil.rmtree(frame_dir, ignore_errors=True)
    if proc.returncode != 0:
        print("ffmpeg failed:")
        print(proc.stderr[-4000:])
        return proc.returncode

    print(f"\nwrote {out_path.resolve()} ({len(frame_paths)} decision frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
