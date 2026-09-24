"""Frontier planner: goes to the edge of explored space instead of hoping to
stumble on it.

Motivation (MAP01, see DoorUseConfig's docstring): the agent's first
exploration ceiling is one door out of the start area, and the agent only
reached it for 17 of 1,200 steps on seed 5000 -- the frontier heading in
wayfinding.py ignores walls, so it pulls toward walls, not doorways.

The map is built purely from experience, no level geometry:

- nodes: grid cells the player has stood in (same cells as StateEncoder's
  AREA hint), each with the real position closest to the cell centre;
- edges: cell-to-cell moves the player actually made, so a path along
  them is known to be walkable (modulo doors that have since shut);
- frontier: every (visited cell, cardinal heading) whose neighbour cell in
  that direction has never been visited.

A mission travels along known edges to the nearest untried frontier, faces
its heading squarely (0/90/180/270 degrees, so an aimed ``use`` lands on an
axis-aligned door, which is what opened MAP01's), and pushes forward. If
the wall ahead blocks it, it presses ``use`` once; if that doesn't clear
the way, the frontier is marked exhausted. Either way each frontier is
tried at most once, so aimed ``use`` is only ever spent at the edge of
explored space -- not at every wall, which is what sank
``DoorUseConfig(aim=True)``.

``door_first`` optionally ranks frontiers that face a door-shaped sector
ahead of plain walls, using ViZDoom's level geometry (read once at episode
start): a Doom door is a thin sector, 8-24 units deep, spanning a doorway
between two passable lines. On MAP01 this finds the start-area door as
sectors 26-28 (16 x 128 units each) plus ~17 other doorway-shaped spots.
Sector floor/ceiling heights are NOT used: ViZDoom reports them wrongly
(the central hall as floor = ceiling = 64, the corridor as ceiling below
floor), while line positions matched the real game exactly. It's level
data the rest of the pipeline never sees, used only to *order* frontiers,
which are by definition next to space the agent has already explored.

door_first is off by default: it lost a paired 10-episode, 2,400-step run
against the plain planner (autoresearch run frontier_planner_door_first):
new cells 58.1 -> 46.5, deaths 0 -> 2, and seeds 5000/5001 fell from 68/71
to 27/19 cells. The ranking worked -- on seed 5000 it reached MAP01's door
by mission 4 -- but the door frontiers were then exhausted in two quick
presses: one from (672, -96), the corridor's top edge, where the centre
depth ray points at the door frame, and one ten steps later from
(673, -195). Likely (not yet confirmed): the first press opened the door
without the centre ray seeing it, and the second, on an open door, shut
it again (Doom doors toggle). A door frontier probably needs lining up
with the doorway's centre and a delayed retry, not one-strike exhaustion.

Confirmed afterwards by replaying seed 5000 to the door: a second ``use``
on the open door slammed it shut within 4 tics (centre depth 871 -> 7).
So door missions now (a) walk to ``door_standoff`` units in front of the
doorway's centre before pressing, measured from the farthest of the door's
thin sectors (MAP01's door is three 16-unit strips), (b) press any one
door at most once per ``door_cooldown_steps``, keyed by the door rather
than the frontier (two frontiers face MAP01's door), and (c) get
``door_attempts`` tries, spaced by the cooldown, before being exhausted.

That fixed the door (seeds 5000/5001 now line up, press once and walk
through), but v2 still lost its paired run (autoresearch run
frontier_planner_door_first_v2): new cells 58.1 -> 55.5 (-2.6 +/- 24.1),
kills 5.9 -> 7.0, and deaths 0/10 -> 6/10. Seed 5007 set the campaign
best (81 cells), but getting through doors sooner puts the agent into
monster-filled rooms sooner, and the combat nets don't cope -- survival,
not doors, is now what limits door_first. Still off by default.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from . import wayfinding
from .perception import Perception

Cell = wayfinding.Cell
Frontier = tuple[Cell, float]  # (visited cell, cardinal heading toward the unvisited neighbour)
Box = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)

_CARDINAL_STEPS: dict[float, tuple[int, int]] = {0.0: (1, 0), 90.0: (0, 1), 180.0: (-1, 0), 270.0: (0, -1)}


@dataclass
class FrontierPlannerConfig:
    # On by default since a paired 10-episode, 2,400-step MAP01 run
    # (autoresearch run frontier_planner): new cells 34.8 -> 58.1, better on
    # 8/10 seeds, kills 3.5 -> 5.9, deaths 1 -> 0; completion still 0/10.
    enabled: bool = True
    waypoint_radius: float = 24.0  # game units: "arrived" at a cell's anchor
    face_tolerance_deg: float = 25.0  # travel: move forward once roughly facing the waypoint
    aim_tolerance_deg: float = 4.0  # probe: square up this precisely before pushing / using
    max_aim_steps: int = 12
    stall_steps: int = 8  # travel: consecutive no-progress steps before a mission is abandoned
    push_stall_steps: int = 3  # push: same, once squared up at the frontier (most are plain walls)
    max_probe_steps: int = 14
    min_progress_units: float = 3.0
    door_first: bool = False  # needs DoomEnvConfig(sectors_info_enabled=True)
    door_max_depth: float = 24.0  # door-shaped sector: thinnest side at most this...
    door_min_width: float = 48.0  # ...and longest side at least this (game units)
    door_standoff: float = 32.0  # stand this far in front of the door's centre to press (use reach is 64)
    door_cooldown_steps: int = 40  # never re-press a door sooner: a second press shuts an open door
    door_attempts: int = 2  # tries per door frontier, spaced by the cooldown, before it's exhausted
    max_approach_steps: int = 20


@dataclass
class _Mission:
    target: Frontier
    phase: str = "travel"  # "travel" -> "aim" -> "push" -> ("wait_open" -> "push")
    counter: int = 0
    stalled: int = 0
    used: bool = False
    waited: bool = False
    door: Box | None = None  # set when the target frontier faces a known door-shaped sector


@dataclass
class FrontierPlanner:
    config: FrontierPlannerConfig
    cell_size: float = 128.0
    anchors: dict[Cell, tuple[float, float]] = field(default_factory=dict)
    edges: dict[Cell, set[Cell]] = field(default_factory=dict)
    exhausted: set[Frontier] = field(default_factory=set)
    doors: list[Box] = field(default_factory=list)  # door-shaped sector bounding boxes, see door_first
    door_pressed_at: dict[Box, int] = field(default_factory=dict)
    door_failures: dict[Frontier, int] = field(default_factory=dict)
    retry_at: dict[Frontier, int] = field(default_factory=dict)
    steps: int = 0
    mission: _Mission | None = None
    _last_cell: Cell | None = None
    _last_pos: tuple[float, float] | None = None

    def reset(self) -> None:
        self.anchors.clear()
        self.edges.clear()
        self.exhausted.clear()
        self.doors.clear()
        self.door_pressed_at.clear()
        self.door_failures.clear()
        self.retry_at.clear()
        self.steps = 0
        self.mission = None
        self._last_cell = None
        self._last_pos = None

    # --- map building -------------------------------------------------

    def observe(self, perception: Perception) -> None:
        """Record the player's current position: once per step, before step()."""
        self.steps += 1
        cell = wayfinding.cell_of(perception.x, perception.y, self.cell_size)
        centre = ((cell[0] + 0.5) * self.cell_size, (cell[1] + 0.5) * self.cell_size)
        best = self.anchors.get(cell)
        if best is None or math.dist((perception.x, perception.y), centre) < math.dist(best, centre):
            self.anchors[cell] = (perception.x, perception.y)
        self.edges.setdefault(cell, set())
        if self._last_cell is not None and self._last_cell != cell:
            self.edges[self._last_cell].add(cell)
            self.edges[cell].add(self._last_cell)
        self._last_cell = cell

    def frontiers(self) -> list[Frontier]:
        found = []
        for cell in self.anchors:
            for heading, (dx, dy) in _CARDINAL_STEPS.items():
                neighbour = (cell[0] + dx, cell[1] + dy)
                frontier = (cell, heading)
                if (
                    neighbour not in self.anchors
                    and frontier not in self.exhausted
                    and self.retry_at.get(frontier, 0) <= self.steps
                ):
                    found.append(frontier)
        return found

    def faces_door(self, frontier: Frontier) -> bool:
        return self.door_for(frontier) is not None

    def door_for(self, frontier: Frontier) -> Box | None:
        """The door-shaped sector a frontier faces, if any: thin along the
        frontier's heading, overlapping the frontier cell's width, and starting
        between just behind the cell's anchor and 1.5 cells ahead of it
        (measured from where the player actually stood, since a door inside
        the cell could be on either side). Of several (a door and its track
        strips), the farthest along the heading, so standing off from it keeps
        every one of them within use reach."""
        (cx, cy), heading = frontier
        size = self.cell_size
        along_x = heading in (0.0, 180.0)
        lo, hi = (cy * size, (cy + 1) * size) if along_x else (cx * size, (cx + 1) * size)
        anchor = self.anchors[(cx, cy)][0 if along_x else 1]
        sign = 1 if heading in (0.0, 90.0) else -1
        best: tuple[float, Box] | None = None
        for box in self.doors:
            x0, y0, x1, y1 = box
            near, far, side0, side1 = (x0, x1, y0, y1) if along_x else (y0, y1, x0, x1)
            thin_along_heading = (far - near) < (side1 - side0)
            overlaps = side0 < hi and side1 > lo
            ahead = (near - anchor) if sign > 0 else (anchor - far)
            if thin_along_heading and overlaps and -16.0 <= ahead <= 1.5 * size and (best is None or ahead > best[0]):
                best = (ahead, box)
        return None if best is None else best[1]

    def approach_point(self, door: Box, heading: float) -> tuple[float, float]:
        """door_standoff units in front of the door's centre, facing it along heading."""
        x0, y0, x1, y1 = door
        cx, cy, off = (x0 + x1) / 2, (y0 + y1) / 2, self.config.door_standoff
        return {0.0: (x0 - off, cy), 180.0: (x1 + off, cy), 90.0: (cx, y0 - off), 270.0: (cx, y1 + off)}[heading]

    def _bfs(self, start: Cell) -> dict[Cell, Cell | None]:
        parents: dict[Cell, Cell | None] = {start: None}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for nxt in sorted(self.edges.get(cell, ())):
                if nxt not in parents:
                    parents[nxt] = cell
                    queue.append(nxt)
        return parents

    def nearest_frontier(self, perception: Perception) -> Frontier | None:
        """The untried frontier whose cell is fewest known moves away (ties:
        the heading needing the smallest turn). With door_first, frontiers
        facing a door-shaped sector come before all others."""
        start = wayfinding.cell_of(perception.x, perception.y, self.cell_size)
        parents = self._bfs(start)
        depth = {}
        for cell in parents:
            d, node = 0, cell
            while parents[node] is not None:
                node, d = parents[node], d + 1
            depth[cell] = d
        reachable = [f for f in self.frontiers() if f[0] in depth]
        if not reachable:
            return None
        door_first = self.config.door_first and bool(self.doors)
        return min(
            reachable,
            key=lambda f: (
                door_first and not self.faces_door(f),
                depth[f[0]],
                abs(wayfinding.angular_diff(perception.angle, f[1])),
                f,
            ),
        )

    def hint(self, perception: Perception) -> str | None:
        """``FRONTIER <direction> <distance>`` toward where the planner would go
        next (its mission's target, else the nearest untried frontier): the
        next waypoint along known moves, or the frontier's own heading once
        standing in its cell. Direction is relative to the current facing
        (ahead / front-left / left / behind / right / front-right), distance
        is to the frontier cell's anchor (near < 1 cell < medium < 3 cells <
        far). None when there's no frontier to point at."""
        frontier = self.mission.target if self.mission is not None else self.nearest_frontier(perception)
        if frontier is None:
            return None
        cell, heading = frontier
        pos = (perception.x, perception.y)
        waypoint = self._next_waypoint(perception, cell)
        if waypoint is None:
            return None
        if wayfinding.cell_of(*pos, self.cell_size) == cell and math.dist(pos, waypoint) <= self.config.waypoint_radius:
            bearing = heading
        else:
            bearing = math.degrees(math.atan2(waypoint[1] - pos[1], waypoint[0] - pos[0])) % 360.0
        rel = wayfinding.angular_diff(perception.angle, bearing)
        side = "left" if rel > 0 else "right"
        if abs(rel) <= 20:
            direction = "ahead"
        elif abs(rel) <= 70:
            direction = f"front-{side}"
        elif abs(rel) <= 135:
            direction = side
        else:
            direction = "behind"
        span = math.dist(pos, self.anchors[cell])
        distance = "near" if span < self.cell_size else "medium" if span < 3 * self.cell_size else "far"
        return f"FRONTIER {direction} {distance}"

    def _next_waypoint(self, perception: Perception, target: Cell) -> tuple[float, float] | None:
        start = wayfinding.cell_of(perception.x, perception.y, self.cell_size)
        if start == target:
            return self.anchors[target]
        parents = self._bfs(start)
        if target not in parents:
            return None
        node = target
        while parents[node] != start:
            node = parents[node]
        return self.anchors[node]

    # --- missions ------------------------------------------------------

    def start_mission(self, perception: Perception) -> bool:
        target = self.nearest_frontier(perception)
        if target is None:
            return False
        self.mission = _Mission(target, door=self.door_for(target) if self.doors else None)
        self._last_pos = None
        return True

    def _finish(self, exhaust: bool = True) -> None:
        if exhaust and self.mission is not None:
            self.exhausted.add(self.mission.target)
        self.mission = None

    def _door_failed(self) -> None:
        """A door mission that didn't get through: retry after the cooldown
        until door_attempts are used up, then exhaust the frontier."""
        m = self.mission
        failures = self.door_failures.get(m.target, 0) + 1
        self.door_failures[m.target] = failures
        if failures < self.config.door_attempts:
            self.retry_at[m.target] = self.steps + self.config.door_cooldown_steps
            self._finish(exhaust=False)
        else:
            self._finish()

    def step(self, perception: Perception, action_set: str) -> str | None:
        """The mission's action for this step, or None once it's over (the
        caller falls through to its other nets). Call observe() first."""
        m = self.mission
        if m is None:
            return None
        cell, heading = m.target
        pos = (perception.x, perception.y)
        progressed = self._last_pos is not None and math.dist(pos, self._last_pos) >= self.config.min_progress_units
        self._last_pos = pos
        m.stalled = 0 if progressed else m.stalled + 1

        dx, dy = _CARDINAL_STEPS[heading]
        if (cell[0] + dx, cell[1] + dy) in self.anchors:
            self._finish()  # reached the new cell (or it got visited some other way)
            return None

        if m.phase == "travel":
            if m.stalled >= self.config.stall_steps:
                self._finish()
                return None
            waypoint = self._next_waypoint(perception, cell)
            if waypoint is None:
                self._finish()
                return None
            if wayfinding.cell_of(*pos, self.cell_size) == cell and (
                math.dist(pos, waypoint) <= self.config.waypoint_radius or not perception.open_forward
            ):
                m.phase, m.counter, m.stalled = ("approach" if m.door else "aim"), 0, 0
            else:
                target_heading = math.degrees(math.atan2(waypoint[1] - pos[1], waypoint[0] - pos[0])) % 360.0
                if abs(wayfinding.angular_diff(perception.angle, target_heading)) <= self.config.face_tolerance_deg:
                    return "move_forward"
                return wayfinding.fine_turn_action(perception.angle, target_heading, action_set)

        if m.phase == "approach":
            point = self.approach_point(m.door, heading)
            m.counter += 1
            if (
                math.dist(pos, point) <= 12.0
                or m.stalled >= self.config.push_stall_steps
                or m.counter > self.config.max_approach_steps
            ):
                m.phase, m.counter, m.stalled = "aim", 0, 0
            else:
                target_heading = math.degrees(math.atan2(point[1] - pos[1], point[0] - pos[0])) % 360.0
                if abs(wayfinding.angular_diff(perception.angle, target_heading)) <= 20.0:
                    return "move_forward"
                return wayfinding.fine_turn_action(perception.angle, target_heading, action_set)

        if m.phase == "aim":
            aimed = abs(wayfinding.angular_diff(perception.angle, heading)) <= self.config.aim_tolerance_deg
            if not aimed and m.counter < self.config.max_aim_steps:
                m.counter += 1
                return wayfinding.fine_turn_action(perception.angle, heading, action_set)
            m.phase, m.counter, m.stalled = "push", 0, 0

        if m.phase == "wait_open":
            if not m.waited:
                m.waited = True  # a real door clears ~4 tics after the press; use itself is 2
                return "wait"
            if perception.wall_near:
                if m.door is not None:
                    self._door_failed()
                else:
                    self._finish()  # pressed use, nothing opened
                return None
            m.phase, m.stalled = "push", 0

        # push: walk the faced heading into the unvisited neighbour
        m.counter += 1
        if m.counter > self.config.max_probe_steps:
            self._finish()
            return None
        if perception.wall_near and m.stalled >= 1:
            if m.used:
                if m.door is not None:
                    self._door_failed()
                else:
                    self._finish()
                return None
            if m.door is not None:
                last = self.door_pressed_at.get(m.door)
                if last is not None and self.steps - last < self.config.door_cooldown_steps:
                    # pressed moments ago (maybe from the door's other frontier):
                    # another press would shut it if it's open -- come back later
                    self.retry_at[m.target] = last + self.config.door_cooldown_steps
                    self._finish(exhaust=False)
                    return None
                self.door_pressed_at[m.door] = self.steps
            m.used, m.phase = True, "wait_open"
            return "use"
        if m.stalled >= self.config.push_stall_steps:
            self._finish()
            return None
        return "move_forward"


def door_like_boxes(sectors, max_depth: float = 24.0, min_width: float = 48.0) -> list[Box]:
    """Bounding boxes of door-shaped sectors: four lines, thinnest side at most
    ``max_depth``, longest at least ``min_width``, and at least two
    passable (non-blocking) sides to walk through. ``sectors`` is ViZDoom's
    ``state.sectors`` or any stand-in with ``lines`` of ``x1, y1, x2, y2,
    is_blocking``. Heights are deliberately ignored (see module docstring)."""
    boxes = []
    for sector in sectors or ():
        lines = sector.lines
        if len(lines) != 4 or sum(not line.is_blocking for line in lines) < 2:
            continue
        xs = [v for line in lines for v in (line.x1, line.x2)]
        ys = [v for line in lines for v in (line.y1, line.y2)]
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        if min(width, height) <= max_depth and max(width, height) >= min_width:
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes
