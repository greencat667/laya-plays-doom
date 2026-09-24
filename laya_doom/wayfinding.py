"""Frontier-directed exploration helpers — the one deliberate exception to
perception.py's "never trust ViZDoom's ANGLE sign convention" rule, and
built only after verifying that convention empirically, not from the
ViZDoom docs alone.

Real, printed evidence (reproduce with scripts/verify_angle.py and
scripts/verify_angle_movement.py):

- A fresh episode starts at ANGLE 0.0.
- Five real ``turn_left`` actions (env.execute(...), 3 tics each) against a
  real DoomEnv on the ``basic`` scenario increased ANGLE monotonically:
  +5.27, +7.03, +10.55, +10.55, +10.55 degrees per action (settling at
  ~10.5 deg/action once turning speed ramps up).
- Five real ``turn_right`` actions from there decreased it by the same
  ~10.55 degrees per action, wrapping from 1.76 down to 351.21 (i.e.
  -10.55 mod 360) rather than going negative.
- A single raw 1-tic TURN_LEFT/TURN_RIGHT button press moved ANGLE by
  +3.52 / -3.52 degrees respectively (3 * 3.52 = 10.56, consistent with
  the 3-tic action deltas above).
- Real POSITION_X/Y deltas after ``move_forward`` matched
  ``(cos(angle_deg), sin(angle_deg))`` as the movement direction, to
  within measurement noise (e.g. angle=86.13 -> actual direction
  (+0.067, +0.998) vs predicted (+0.067, +0.998); angle=172.27 -> actual
  (-0.976, +0.216) vs predicted (-0.991, +0.135), same sign, same rough
  magnitude).

So: ANGLE is degrees, wraps 0-360, ``turn_left`` increases it (CCW in the
standard math sense), ``turn_right`` decreases it (CW), and the forward
movement unit vector is ``(cos(angle_deg), sin(angle_deg))`` in
POSITION_X/Y space. Everything below is built directly on that verified
convention — used ONLY for FrontierExplorationConfig's exploration-nudge
override in controller.py, never for bearing/aiming, which stays
screen-space as perception.py's own module docstring explains.
"""

from __future__ import annotations

import math

Cell = tuple[int, int]

# Ordered smallest absolute turn first: straight ahead, then the closer of
# each +/- pair, out to a full about-face. Deliberately favours the
# smallest heading change that reaches unexplored territory (less
# disruptive to whatever heading Laya/the other safety nets already had it
# facing) over a larger one, even when both would work.
_DEFAULT_CANDIDATE_OFFSETS_DEG: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)

# Below this angular difference (degrees) between current heading and the
# chosen target heading, move_forward is issued directly instead of another
# turn — no point turning a few degrees when "close enough" already faces
# roughly the right way. Public (not underscore-prefixed): controller.py's
# secret-search mechanism also uses it directly to decide "am I facing this
# wall segment yet, or do I still need to turn".
FACING_TOLERANCE_DEG = 20.0


def heading_unit_vector(angle_deg: float) -> tuple[float, float]:
    """(cos, sin) of angle_deg — the verified real-game forward-movement
    direction for a given ANGLE reading, see this module's docstring."""
    rad = math.radians(angle_deg)
    return math.cos(rad), math.sin(rad)


def cell_of(x: float, y: float, cell_size: float) -> Cell:
    """Same bucketing StateEncoder._area_cell uses (floor division into
    cell_size-unit square cells) — kept as a free function here so it can
    be applied to a *projected* point ahead of the player, not just the
    player's own current position."""
    return (int(x // cell_size), int(y // cell_size))


def project_cell(x: float, y: float, heading_deg: float, distance: float, cell_size: float) -> Cell:
    """The grid cell that lies `distance` game units ahead of (x, y) along
    `heading_deg`, using the verified (cos, sin) convention."""
    dx, dy = heading_unit_vector(heading_deg)
    return cell_of(x + dx * distance, y + dy * distance, cell_size)


def nearest_unvisited_cell(
    current_cell: Cell, visited_cells: frozenset[Cell] | set[Cell], max_radius: int = 20
) -> Cell | None:
    """Ring search (Chebyshev rings, so it checks the full square ring at
    each radius) outward from current_cell for the nearest grid cell NOT in
    visited_cells. Returns current_cell itself if that's already unvisited
    (degenerate case), or None if every cell within max_radius rings is
    visited — "fully explored locally", the same honest fallback every
    other mechanism in this project uses rather than guessing. Ties within
    a ring are broken by scan order (row-major), which is deterministic,
    not randomized, but not meaningfully "smarter" than any other tie-break
    within the same ring.
    """
    if current_cell not in visited_cells:
        return current_cell
    cx, cy = current_cell
    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                cell = (cx + dx, cy + dy)
                if cell not in visited_cells:
                    return cell
    return None


def best_exploration_heading(
    x: float,
    y: float,
    current_heading_deg: float,
    visited_cells: frozenset[Cell] | set[Cell],
    cell_size: float,
    lookahead_distance: float | None = None,
    candidate_offsets_deg: tuple[float, ...] = _DEFAULT_CANDIDATE_OFFSETS_DEG,
) -> float | None:
    """Real frontier-directed heading choice, replacing the old blind
    "just pick some heading" nudge. For each candidate heading (current
    heading plus each offset, smallest turn first), projects a point
    `lookahead_distance` game units ahead and checks whether that lands in
    a cell NOT already in `visited_cells`. Returns the first (smallest-turn)
    candidate heading whose projected cell is unvisited.

    If every candidate is already visited (the immediate surroundings are
    "locally exhausted" — plausible after enough circling), falls back to
    aiming directly at the nearest unvisited cell anywhere on the known
    grid (`nearest_unvisited_cell` above). Returns None only when there is
    no visited-cell information to use at all (`visited_cells` empty — the
    encoder's area hint was disabled, or this is called before any cell has
    ever been recorded) or the nearest-unvisited fallback also comes back
    empty (target coincides exactly with the player's own position, or
    every cell within its search radius is visited) — the caller is
    expected to fall back to a different, non-directed mechanism in that
    case, same as every other "nudge, not a guarantee" net in this project.
    """
    if not visited_cells:
        return None
    if lookahead_distance is None:
        lookahead_distance = cell_size * 2.0

    for offset in candidate_offsets_deg:
        heading = (current_heading_deg + offset) % 360.0
        cell = project_cell(x, y, heading, lookahead_distance, cell_size)
        if cell not in visited_cells:
            return heading

    current_cell = cell_of(x, y, cell_size)
    target_cell = nearest_unvisited_cell(current_cell, visited_cells)
    if target_cell is None:
        return None
    target_x = (target_cell[0] + 0.5) * cell_size
    target_y = (target_cell[1] + 0.5) * cell_size
    dx, dy = target_x - x, target_y - y
    if dx == 0.0 and dy == 0.0:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def angular_diff(current_heading_deg: float, target_heading_deg: float) -> float:
    """Signed shortest angular distance from current to target, wrapped to
    (-180, 180]. Positive means the target is reached by INCREASING ANGLE
    — i.e. turning left, per the verified convention in this module's
    docstring (turn_left increases ANGLE / CCW, turn_right decreases it /
    CW). Negative means turning right."""
    return (target_heading_deg - current_heading_deg + 180.0) % 360.0 - 180.0


def fine_turn_action(current_heading_deg: float, target_heading_deg: float, action_set: str) -> str:
    """A turn toward the target sized to the remaining error (large > 35
    degrees > normal > 12 degrees > small), so precise aiming converges
    instead of overshooting the way repeated fixed large turns would."""
    diff = angular_diff(current_heading_deg, target_heading_deg)
    side = "left" if diff > 0 else "right"
    if action_set == "stage1":
        return f"turn_{side}"
    if abs(diff) > 35:
        return f"turn_{side}_large"
    if abs(diff) > 12:
        return f"turn_{side}"
    return f"turn_{side}_small"


def turn_action_for_heading(current_heading_deg: float, target_heading_deg: float, action_set: str) -> str:
    """Deterministic turn_left / turn_right / move_forward choice toward
    target_heading_deg, using ``angular_diff`` above. Within
    ``FACING_TOLERANCE_DEG`` of already facing the target, returns
    move_forward instead of another turn.
    """
    diff = angular_diff(current_heading_deg, target_heading_deg)
    if abs(diff) <= FACING_TOLERANCE_DEG:
        return "move_forward"
    direction = "left" if diff > 0 else "right"
    return f"turn_{direction}" if action_set == "stage1" else f"turn_{direction}_large"
