"""Perception -> compact text block that Laya actually reads.

Three memory modes (configurable, per the brief's "experiment with" list):

- ``stateless``: just the current tick's facts, no LAST_ACTION/LAST_RESULT,
  no deltas. Tests whether a reactive policy needs history at all.
- ``prev_state``: adds LAST_ACTION/LAST_RESULT plus deltas computed against
  the previous tick (HEALTH_CHANGE, AMMO_CHANGE, VISIBLE_ENEMIES_CHANGE,
  MOVED) — one step of consequence.
- ``rolling``: adds a short HISTORY line of the last N action->result pairs,
  in addition to everything ``prev_state`` gives.

Kept deliberately format-stable (fixed line order, fixed tokens) so the
compact text is easy to diff/inspect in logs and in the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .perception import Perception

MemoryMode = Literal["stateless", "prev_state", "rolling"]

# Prepended to the encoded world-state text when EncoderConfig.include_goal_line
# is True. Kept to one short line, no chain-of-thought, per the brief. See
# laya_agent.py's module docstring for why this lives in the world-state
# text rather than any system/instructions parameter — Laya's `predict()`
# takes only `state` and typed `questions`, no free-text system prompt at all.
GOAL_LINE = "GOAL stay_alive kill_threats explore collect find_exit"


@dataclass
class EncoderConfig:
    memory_mode: MemoryMode = "prev_state"
    rolling_window: int = 3
    move_delta_threshold: float = 4.0  # game units; below this, MOVED is "no"
    # See laya_agent.py's module docstring: laya.Agent.predict() has no
    # system/instructions parameter at all (state + typed questions only),
    # so the brief's goal/priority line is injected here instead, as part
    # of the state text Laya actually reads. Toggle to measure whether it
    # does anything at all.
    include_goal_line: bool = True
    # Stage 2 (navigation): a coarse "have I been here before" signal, per
    # the brief's "visited_direction estimate" — simplified to the current
    # position's grid cell rather than full per-direction estimation, since
    # that would need trusting ViZDoom's player-angle sign convention, which
    # perception.py deliberately avoids elsewhere (see its module docstring)
    # by using screen-space bearing instead of trig. Position (POSITION_X/Y)
    # is bucketed into area_cell_size-unit square cells and the episode
    # remembers which cells it's already visited.
    include_area_hint: bool = True
    area_cell_size: float = 128.0
    # Real levels (scenario="level") have keys/doors; surfaced as a KEYS
    # line — see controller.run_episode, which infers which key was picked
    # up (ViZDoom doesn't expose a key-inventory game variable) and passes
    # it into encode() each step.
    include_keys_hint: bool = True
    # A hit with no ENEMY line means the attacker is outside perception.py's
    # visible-FOV labels — surfaced as "THREAT unseen behind" (see encode()
    # for why "behind" specifically). Only meaningful in prev_state/rolling
    # (needs a previous HEALTH to compare against).
    include_threat_hint: bool = True
    # A "FRONTIER <direction> <distance>" line pointing toward the frontier
    # planner's next waypoint (planner.FrontierPlanner.hint) -- map knowledge
    # the rest of the text lacks, so a trained head (scripts/
    # distill_movement_head.py) can learn the planner's choices. Off by
    # default: it changes what zero-shot Laya reads.
    include_frontier_hint: bool = False


@dataclass(frozen=True)
class HistoryEntry:
    action: str
    result: str


def summarize_result(before: Perception | None, after: Perception | None) -> str:
    """Compact LAST_RESULT token summarizing what an action accomplished,
    e.g. "enemy_closer", "took_damage", "killed_enemy", "no_change"."""
    if after is None:
        return "episode_ended"
    if not after.alive:
        return "died"
    if before is None:
        return "spawned"

    if after.damage_taken > before.damage_taken:
        return "took_damage"
    if after.killcount > before.killcount:
        return "killed_enemy"
    if after.itemcount > before.itemcount:
        return "picked_up_item"

    if before.enemies and after.enemies:
        nearest_before = min(e.raw_distance_units for e in before.enemies)
        nearest_after = min(e.raw_distance_units for e in after.enemies)
        if nearest_after < nearest_before - 1e-6:
            return "enemy_closer"
        if nearest_after > nearest_before + 1e-6:
            return "enemy_farther"
    if before.enemies and not after.enemies:
        return "enemy_gone"
    if after.enemies and not before.enemies:
        return "enemy_appeared"

    dx = after.x - before.x
    dy = after.y - before.y
    if (dx * dx + dy * dy) ** 0.5 > 4.0:
        return "moved"

    return "no_change"


class StateEncoder:
    def __init__(self, config: EncoderConfig | None = None):
        self.config = config or EncoderConfig()
        self._history: list[HistoryEntry] = []
        self._visited_cells: set[tuple[int, int]] = set()
        # Set by encode() whenever include_area_hint is on — exposed so
        # controller.run_episode can read the real AREA new/revisited
        # signal directly (for ExplorationNudgeConfig's circling detector)
        # without re-deriving it by parsing the encoded text.
        self.last_area_new: bool | None = None

    def reset(self) -> None:
        self._history.clear()
        self._visited_cells.clear()
        self.last_area_new = None

    def _area_cell(self, perception: Perception) -> tuple[int, int]:
        size = self.config.area_cell_size
        return (int(perception.x // size), int(perception.y // size))

    @property
    def visited_cells(self) -> frozenset[tuple[int, int]]:
        """Read-only view of the AREA-hint visited-cells grid, exposed for
        controller.py's FrontierExplorationConfig (see wayfinding.py) to
        compute a directed exploration heading from, the same way
        ``last_area_new`` is exposed for the circling detector above."""
        return frozenset(self._visited_cells)

    def record(self, action: str, result: str) -> None:
        if self.config.memory_mode != "rolling":
            return
        self._history.append(HistoryEntry(action=action, result=result))
        if len(self._history) > self.config.rolling_window:
            self._history = self._history[-self.config.rolling_window :]

    def encode(
        self,
        perception: Perception,
        prev_perception: Perception | None,
        last_action: str | None,
        last_result: str | None,
        keys_held: frozenset[str] = frozenset(),
        frontier_hint: str | None = None,
    ) -> str:
        lines: list[str] = []
        if self.config.include_goal_line:
            lines.append(GOAL_LINE)
        lines.extend(
            [
                f"HEALTH {perception.health}",
                f"ARMOR {perception.armor}",
                f"AMMO {perception.ammo}",
            ]
        )
        if self.config.include_keys_hint and keys_held:
            lines.append(f"KEYS {' '.join(sorted(keys_held))}")

        for enemy in perception.enemies:
            lines.append(f"ENEMY {enemy.kind} {enemy.bearing} {enemy.distance}")
        for pickup in perception.pickups:
            lines.append(f"PICKUP {pickup.kind} {pickup.bearing} {pickup.distance}")

        if perception.wall_near:
            lines.append("WALL ahead near")
        elif perception.wall_ahead:
            lines.append("WALL ahead far")
        lines.append(f"PATH left {'open' if perception.open_left else 'blocked'}")
        lines.append(f"PATH right {'open' if perception.open_right else 'blocked'}")

        if self.config.include_area_hint:
            cell = self._area_cell(perception)
            self.last_area_new = cell not in self._visited_cells
            lines.append(f"AREA {'new' if self.last_area_new else 'revisited'}")
            self._visited_cells.add(cell)

        if self.config.include_frontier_hint and frontier_hint:
            lines.append(frontier_hint)

        if self.config.memory_mode in ("prev_state", "rolling") and last_action is not None:
            lines.append(f"LAST_ACTION {last_action}")
            if last_result is not None:
                lines.append(f"LAST_RESULT {last_result}")

        if self.config.memory_mode in ("prev_state", "rolling") and prev_perception is not None:
            health_change = perception.health - prev_perception.health
            lines.append(f"HEALTH_CHANGE {health_change:+d}")
            if self.config.include_threat_hint and health_change < 0 and not perception.enemies:
                # Took damage with nothing in the visible-labels FOV to
                # blame it on — perception.py only reports what's currently
                # on screen, so a hit with no ENEMY line means the attacker
                # is outside that view. "behind" is a reasonable default
                # guess, not a measured direction: ViZDoom has no damage-
                # source-direction game variable, so there's no way to
                # actually tell front/back/side apart here.
                lines.append("THREAT unseen behind")
            lines.append(f"AMMO_CHANGE {perception.ammo - prev_perception.ammo:+d}")
            enemy_change = len(perception.enemies) - len(prev_perception.enemies)
            lines.append(f"VISIBLE_ENEMIES_CHANGE {enemy_change:+d}")
            dx = perception.x - prev_perception.x
            dy = perception.y - prev_perception.y
            moved = (dx * dx + dy * dy) ** 0.5 > self.config.move_delta_threshold
            lines.append(f"MOVED {'yes' if moved else 'no'}")

        if self.config.memory_mode == "rolling" and self._history:
            history_str = " ".join(f"{h.action}->{h.result}" for h in self._history)
            lines.append(f"HISTORY {history_str}")

        return "\n".join(lines)
