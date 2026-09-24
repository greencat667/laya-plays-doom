"""FrontierPlanner (planner.py): experience map, frontier choice, and the
travel -> aim -> push -> use mission, all against hand-built perceptions."""

from laya_doom.planner import FrontierPlanner, FrontierPlannerConfig, door_like_boxes
from tests.test_state_encoder import make_perception

CELL = 128.0


def _planner(**config) -> FrontierPlanner:
    return FrontierPlanner(FrontierPlannerConfig(enabled=True, **config), cell_size=CELL)


def _walk(planner: FrontierPlanner, points) -> None:
    for x, y in points:
        planner.observe(make_perception(x=x, y=y))


def test_observe_builds_edges_only_from_moves_actually_made():
    planner = _planner()
    _walk(planner, [(64, 64), (192, 64), (192, 192)])
    assert planner.edges[(0, 0)] == {(1, 0)}
    assert planner.edges[(1, 0)] == {(0, 0), (1, 1)}
    assert (0, 1) not in planner.edges[(0, 0)]  # adjacent but never walked between


def test_anchor_is_the_position_closest_to_the_cell_centre():
    planner = _planner()
    _walk(planner, [(5, 5), (60, 70), (120, 120)])
    assert planner.anchors[(0, 0)] == (60, 70)


def test_frontiers_are_cardinal_edges_into_unvisited_cells():
    planner = _planner()
    _walk(planner, [(64, 64), (192, 64)])
    frontiers = set(planner.frontiers())
    assert ((1, 0), 0.0) in frontiers  # east of the eastern cell
    assert ((0, 0), 0.0) not in frontiers  # east of (0,0) is visited
    assert len(frontiers) == 6


def test_nearest_frontier_prefers_fewest_moves_then_smallest_turn():
    planner = _planner()
    _walk(planner, [(64, 64), (192, 64), (320, 64)])
    here = make_perception(x=64, y=64, angle=0.0)
    assert planner.nearest_frontier(here) == ((0, 0), 90.0)  # (0,0) itself; north beats west/south on turn size
    planner.exhausted |= {((0, 0), h) for h in (90.0, 180.0, 270.0)}
    assert planner.nearest_frontier(here)[0] == (1, 0)


def test_mission_travels_to_the_frontier_cell_then_aims_and_pushes():
    planner = _planner()
    _walk(planner, [(64, 64), (192, 64)])
    planner.exhausted |= {f for f in planner.frontiers() if f != ((1, 0), 0.0)}
    start = make_perception(x=64, y=64, angle=0.0)
    planner.observe(start)
    assert planner.start_mission(start)
    assert planner.step(start, "full") == "move_forward"  # facing the (1,0) anchor already
    there = make_perception(x=192, y=64, angle=30.0)
    planner.observe(there)
    assert planner.step(there, "full") == "turn_right"  # arrived: square up to 0 degrees
    squared = make_perception(x=192, y=64, angle=1.0)
    planner.observe(squared)
    assert planner.step(squared, "full") == "move_forward"


def test_blocked_push_presses_use_once_then_gives_up_if_nothing_opens():
    planner = _planner()
    _walk(planner, [(192, 64)])
    planner.exhausted |= {f for f in planner.frontiers() if f != ((1, 0), 0.0)}
    wall = make_perception(x=192, y=64, angle=0.0, wall_near=True)
    planner.observe(wall)
    assert planner.start_mission(wall)
    assert planner.step(wall, "full") == "move_forward"
    assert planner.step(wall, "full") == "use"  # no progress into a wall
    assert planner.step(wall, "full") == "wait"  # a real door clears a step later
    assert planner.step(wall, "full") is None  # still blocked: exhausted
    assert ((1, 0), 0.0) in planner.exhausted and planner.mission is None


def test_mission_ends_once_the_new_cell_is_entered():
    planner = _planner()
    _walk(planner, [(192, 64)])
    planner.exhausted |= {f for f in planner.frontiers() if f != ((1, 0), 0.0)}
    here = make_perception(x=192, y=64, angle=0.0)
    planner.observe(here)
    planner.start_mission(here)
    planner.step(here, "full")
    through = make_perception(x=260, y=64, angle=0.0)
    planner.observe(through)
    assert planner.step(through, "full") is None
    assert planner.mission is None


def test_travel_abandons_after_stalling():
    planner = _planner(stall_steps=3)
    _walk(planner, [(64, 64), (192, 64)])
    planner.exhausted |= {f for f in planner.frontiers() if f != ((1, 0), 0.0)}
    stuck = make_perception(x=64, y=64, angle=0.0)
    planner.observe(stuck)
    planner.start_mission(stuck)
    actions = [planner.step(stuck, "full") for _ in range(4)]
    assert actions[-1] is None and ((1, 0), 0.0) in planner.exhausted


def _sector(*segments):
    """segments: (x1, y1, x2, y2, is_blocking)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        lines=[SimpleNamespace(x1=a, y1=b, x2=c, y2=d, is_blocking=blk) for a, b, c, d, blk in segments]
    )


def test_door_like_boxes_finds_thin_sectors_spanning_a_doorway():
    # MAP01's start-area door (sector 27): 16 deep, 128 wide, passable on both long sides.
    door = _sector(
        (704, -224, 704, -96, False), (720, -224, 720, -96, False),
        (704, -96, 720, -96, True), (704, -224, 720, -224, True),
    )
    room = _sector((0, 0, 256, 0, True), (256, 0, 256, 256, True), (256, 256, 0, 256, True), (0, 256, 0, 0, False))
    ledge = _sector((0, 0, 16, 0, True), (16, 0, 16, 128, True), (16, 128, 0, 128, True), (0, 128, 0, 0, False))
    assert door_like_boxes([door, room, ledge]) == [(704, -224, 720, -96)]  # ledge has only one passable side
    assert door_like_boxes(None) == []


def test_door_first_ranks_the_frontier_facing_a_door_ahead_of_nearer_walls():
    planner = _planner(door_first=True)
    _walk(planner, [(64, 64), (192, 64), (320, 64)])
    here = make_perception(x=64, y=64, angle=90.0)
    assert planner.nearest_frontier(here) == ((0, 0), 90.0)  # no doors known: nearest wins
    planner.doors = [(400.0, 0.0, 416.0, 128.0)]  # a 16-deep door inside the far cell (2,0), facing east
    assert planner.nearest_frontier(here) == ((2, 0), 0.0)


def test_faces_door_matches_map01s_start_door_from_the_corridor_end():
    planner = _planner(door_first=True)
    _walk(planner, [(688, -113), (688, -180)])  # cells (5,-1) and (5,-2), where the agent stood
    planner.doors = [(704, -224, 720, -96)]  # the real door, sector 27
    assert planner.faces_door(((5, -1), 0.0))
    assert planner.faces_door(((5, -2), 0.0))
    assert not planner.faces_door(((5, -1), 90.0))  # door is thin along x, not y
    assert not planner.faces_door(((5, -1), 180.0))  # it's east, not west
    planner.doors = [(1816, -224, 1832, -96)]  # the mirrored door across the hall: too far
    assert not planner.faces_door(((5, -1), 0.0))


# --- door missions: line up, one press per cooldown, retry -----------------

MAP01_DOOR_STRIPS = [(688, -224, 704, -96), (704, -224, 720, -96), (720, -224, 736, -96)]


def _door_planner(**config) -> FrontierPlanner:
    planner = _planner(door_first=True, **config)
    _walk(planner, [(672, -100), (672, -200)])  # cells (5,-1) and (5,-2), like seed 5000's corridor end
    planner.doors = list(MAP01_DOOR_STRIPS)
    planner.exhausted |= {f for f in planner.frontiers() if f[1] != 0.0}
    return planner


def test_door_for_picks_the_farthest_strip_and_the_approach_point_centres_on_it():
    planner = _door_planner()
    door = planner.door_for(((5, -1), 0.0))
    assert door == (720, -224, 736, -96)
    assert planner.approach_point(door, 0.0) == (688.0, -160.0)  # 32 in front, on the doorway's centre line


def test_door_mission_walks_to_the_doorway_centre_before_aiming():
    planner = _door_planner()
    here = make_perception(x=672, y=-100, angle=0.0)
    planner.observe(here)
    assert planner.start_mission(here) and planner.mission.door is not None
    planner.step(here, "full")  # arrives in the frontier cell -> approach
    action = planner.step(here, "full")
    assert planner.mission.phase == "approach"
    assert action == "turn_right_large"  # (688, -160) is down-right of (672, -100)


def test_a_door_is_never_pressed_twice_within_the_cooldown():
    planner = _door_planner()
    wall = make_perception(x=688, y=-160, angle=0.0, wall_near=True)
    planner.observe(wall)
    planner.door_pressed_at[(720, -224, 736, -96)] = planner.steps - 5  # pressed 5 steps ago from the other frontier
    planner.mission = None
    assert planner.start_mission(wall)
    planner.mission.phase = "push"
    planner.step(wall, "full")  # first push: move_forward
    assert planner.step(wall, "full") is None  # would press: cooldown says come back later
    target = ((5, -1), 0.0) if ((5, -1), 0.0) in planner.retry_at else ((5, -2), 0.0)
    assert target not in planner.exhausted and planner.retry_at[target] > planner.steps


def test_a_failed_door_is_retried_once_after_the_cooldown_then_exhausted():
    planner = _door_planner(door_attempts=2, door_cooldown_steps=40)
    wall = make_perception(x=688, y=-160, angle=0.0, wall_near=True)
    planner.observe(wall)
    planner.start_mission(wall)
    target = planner.mission.target
    planner.mission.phase, planner.mission.used, planner.mission.waited = "wait_open", True, True
    assert planner.step(wall, "full") is None  # still shut after the press
    assert target not in planner.exhausted and planner.retry_at[target] == planner.steps + 40
    assert target not in planner.frontiers()  # cooling down
    planner.steps += 40
    assert target in planner.frontiers()
    planner.start_mission(wall)
    planner.mission.target = target
    planner.mission.phase, planner.mission.used, planner.mission.waited = "wait_open", True, True
    planner.step(wall, "full")
    assert target in planner.exhausted  # second failure: done


# --- FRONTIER hint -----------------------------------------------------------


def test_hint_points_at_the_next_waypoint_relative_to_facing():
    planner = _planner()
    _walk(planner, [(64, 64), (192, 64), (320, 64)])
    planner.exhausted |= {f for f in planner.frontiers() if f != ((2, 0), 0.0)}
    assert planner.hint(make_perception(x=64, y=64, angle=0.0)) == "FRONTIER ahead medium"
    assert planner.hint(make_perception(x=64, y=64, angle=90.0)) == "FRONTIER right medium"
    assert planner.hint(make_perception(x=64, y=64, angle=180.0)) == "FRONTIER behind medium"


def test_hint_uses_the_frontier_heading_once_standing_in_its_cell():
    planner = _planner()
    _walk(planner, [(320, 64)])
    planner.exhausted |= {f for f in planner.frontiers() if f != ((2, 0), 90.0)}
    assert planner.hint(make_perception(x=320, y=64, angle=0.0)) == "FRONTIER left near"


def test_hint_is_none_without_frontiers():
    planner = _planner()
    _walk(planner, [(64, 64)])
    planner.exhausted |= set(planner.frontiers())
    assert planner.hint(make_perception(x=64, y=64)) is None
