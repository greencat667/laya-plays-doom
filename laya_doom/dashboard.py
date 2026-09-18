"""A terminal dashboard (rich) showing the Doom state, Laya's decision,
confidence and latency, a short action history, and running episode stats —
the "watch Laya attempting to play" view. Intentionally terminal-only for
now, per the brief ("A terminal UI is acceptable initially"); a browser
dashboard is a natural follow-up once the core loop is proven out.
"""

from __future__ import annotations

from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .controller import EpisodeResult, StepRecord
from .laya_agent import Decision
from .perception import Perception

_ACTION_GLYPHS = {
    "move_forward": "↑ forward",
    "move_backward": "↓ backward",
    "strafe_left": "← strafe",
    "strafe_right": "→ strafe",
    "turn_left": "↺ turn",
    "turn_right": "↻ turn",
    "turn_left_small": "↺ turn",
    "turn_right_small": "↻ turn",
    "turn_left_large": "↺↺ turn",
    "turn_right_large": "↻↻ turn",
    "attack": "🔥 shoot",
    "shoot": "🔥 shoot",
    "use": "▷ use",
    "wait": "… wait",
}


def compute_running_stats(episode: int, records: list[StepRecord]) -> dict:
    if not records:
        return {"episode": episode, "kills": 0, "distance": 0.0, "avg_confidence": None, "avg_latency_ms": 0.0}
    distance = 0.0
    for prev, cur in zip(records, records[1:]):
        distance += ((cur.x - prev.x) ** 2 + (cur.y - prev.y) ** 2) ** 0.5
    confidences = [r.confidence for r in records if r.confidence is not None]
    latencies = [r.latency_ms for r in records]
    return {
        "episode": episode,
        "distance": distance,
        "avg_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


class TerminalDashboard:
    def __init__(self, history_len: int = 12):
        self.console = Console()
        self._live = Live(console=self.console, refresh_per_second=8, screen=False)
        self._history: deque[str] = deque(maxlen=history_len)

    def __enter__(self) -> "TerminalDashboard":
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._live.__exit__(*exc)

    def update(self, perception: Perception, decision: Decision, record: StepRecord, stats: dict) -> None:
        self._history.append(_ACTION_GLYPHS.get(decision.action, decision.action))

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
        if decision.reasoning:
            laya_table.add_row("Reasoning:", decision.reasoning)

        stats_table = Table.grid(padding=(0, 1))
        stats_table.add_row("Distance:", f"{stats.get('distance', 0.0):.0f}")
        avg_conf = stats.get("avg_confidence")
        stats_table.add_row("Avg confidence:", f"{avg_conf:.2f}" if avg_conf is not None else "n/a")
        stats_table.add_row("Avg latency:", f"{stats.get('avg_latency_ms', 0.0):.0f} ms")

        body = Group(
            Panel(state_table, title="CURRENT STATE"),
            Panel(laya_table, title="LAYA"),
            Panel(Text("  ".join(self._history) or "—"), title="LAST ACTIONS"),
            Panel(stats_table, title=f"EPISODE {stats.get('episode', 0)}"),
        )
        self._live.update(body)

    def episode_done(self, result: EpisodeResult) -> None:
        self.console.print(
            f"[bold]Episode {result.episode} done[/bold] — steps={result.steps} "
            f"kills={result.kills} health={result.health_remaining} died={result.died} "
            f"reward={result.total_reward:.1f}"
        )
