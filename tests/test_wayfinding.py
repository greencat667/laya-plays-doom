import math

from laya_doom import wayfinding


def test_heading_unit_vector_matches_verified_axes():
    # Real, verified convention (see wayfinding.py's module docstring and
    # the README's ANGLE-verification section): angle=0 -> +x, angle=90 ->
    # +y, matching real POSITION_X/Y deltas measured against a real DoomEnv.
    dx, dy = wayfinding.heading_unit_vector(0.0)
    assert dx == 1.0 and abs(dy) < 1e-9

    dx, dy = wayfinding.heading_unit_vector(90.0)
    assert abs(dx) < 1e-9 and dy == 1.0


def test_cell_of_matches_state_encoder_bucketing():
    # Same floor-division bucketing as StateEncoder._area_cell.
    assert wayfinding.cell_of(10.0, 10.0, 100.0) == (0, 0)
    assert wayfinding.cell_of(150.0, -10.0, 100.0) == (1, -1)
    assert wayfinding.cell_of(-384.0, 32.0, 128.0) == (-3, 0)


def test_project_cell_ahead_along_heading():
    # Facing east (angle=0), 200 units ahead at cell_size=100 should land
    # two cells over on the x axis, same row.
    assert wayfinding.project_cell(0.0, 0.0, 0.0, 200.0, 100.0) == (2, 0)
    # Facing north (angle=90).
    assert wayfinding.project_cell(0.0, 0.0, 90.0, 200.0, 100.0) == (0, 2)


def test_nearest_unvisited_cell_returns_current_if_unvisited():
    assert wayfinding.nearest_unvisited_cell((0, 0), frozenset()) == (0, 0)


def test_nearest_unvisited_cell_finds_adjacent_gap():
    visited = frozenset({(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)})
    result = wayfinding.nearest_unvisited_cell((0, 0), visited)
    assert result is not None
    assert result not in visited
    # Nearest ring is radius 1 — must not have to search further out.
    assert max(abs(result[0]), abs(result[1])) == 1


def test_nearest_unvisited_cell_none_when_fully_explored_within_radius():
    visited = {(x, y) for x in range(-2, 3) for y in range(-2, 3)}
    assert wayfinding.nearest_unvisited_cell((0, 0), visited, max_radius=2) is None


def test_best_exploration_heading_none_without_visited_cells():
    assert wayfinding.best_exploration_heading(0.0, 0.0, 0.0, frozenset(), 100.0) is None


def test_best_exploration_heading_prefers_unvisited_straight_ahead():
    # Only the current cell is visited -- straight ahead (offset 0) should
    # already land somewhere unvisited, so it should win over any turn.
    visited = frozenset({(0, 0)})
    heading = wayfinding.best_exploration_heading(
        10.0, 10.0, 0.0, visited, cell_size=100.0, lookahead_distance=200.0
    )
    assert heading == 0.0


def test_best_exploration_heading_turns_when_straight_ahead_is_visited():
    # Build a "corridor" of visited cells straight ahead (east) but leave
    # north (90 degrees) open -- the smallest-turn unvisited candidate.
    visited = {(0, 0), (1, 0), (2, 0), (3, 0)}
    heading = wayfinding.best_exploration_heading(
        10.0, 10.0, 0.0, frozenset(visited), cell_size=100.0, lookahead_distance=150.0
    )
    # 45 offset (heading 45) projects into (1,1)-ish, unvisited; should be
    # preferred over the fully-visited straight-ahead direction. Whatever
    # it picks must not be the visited-straight-ahead heading (0.0).
    assert heading != 0.0


def test_best_exploration_heading_falls_back_to_nearest_unvisited_when_all_candidates_visited():
    # Visit current cell plus every cell reachable by every candidate
    # offset's projection -- forces the nearest-unvisited-cell fallback.
    x, y, cell_size, lookahead = 50.0, 50.0, 100.0, 200.0
    current_cell = wayfinding.cell_of(x, y, cell_size)
    visited = {current_cell}
    for offset in wayfinding._DEFAULT_CANDIDATE_OFFSETS_DEG:
        visited.add(wayfinding.project_cell(x, y, offset, lookahead, cell_size))
    heading = wayfinding.best_exploration_heading(
        x, y, 0.0, frozenset(visited), cell_size=cell_size, lookahead_distance=lookahead
    )
    assert heading is not None
    # The resulting heading, projected one cell size ahead, should land on
    # an unvisited cell (or at least not literally the current cell).
    target_cell = wayfinding.cell_of(
        x + math.cos(math.radians(heading)) * cell_size * 3,
        y + math.sin(math.radians(heading)) * cell_size * 3,
        cell_size,
    )
    assert target_cell != current_cell


def test_turn_action_for_heading_move_forward_when_already_facing():
    assert wayfinding.turn_action_for_heading(10.0, 15.0, "stage1") == "move_forward"


def test_turn_action_for_heading_turns_left_for_positive_diff():
    # Target is CCW of current (increasing angle) -> turn_left, per the
    # verified convention (turn_left increases ANGLE).
    assert wayfinding.turn_action_for_heading(0.0, 90.0, "stage1") == "turn_left"
    assert wayfinding.turn_action_for_heading(0.0, 90.0, "full") == "turn_left_large"


def test_turn_action_for_heading_turns_right_for_negative_diff():
    assert wayfinding.turn_action_for_heading(90.0, 0.0, "stage1") == "turn_right"


def test_turn_action_for_heading_handles_wraparound():
    # current=350, target=20 -> shortest path is +30 (through 360/0), i.e.
    # turn_left, not the naive -330.
    assert wayfinding.turn_action_for_heading(350.0, 20.0, "stage1") == "turn_left"
    # current=10, target=340 -> shortest path is -30, i.e. turn_right.
    assert wayfinding.turn_action_for_heading(10.0, 340.0, "stage1") == "turn_right"
