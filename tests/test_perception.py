from types import SimpleNamespace

import numpy as np
import pytest

from laya_doom import perception as p


def make_game_variables(**overrides) -> list:
    return [overrides.get(var.name, 0) for var in p._TRACKED_VARS]


def make_label(object_name, x, y, width, height, object_category=None):
    return SimpleNamespace(
        object_name=object_name, x=x, y=y, width=width, height=height, object_category=object_category
    )


def make_depth_buffer(height=240, width=320, fill=200):
    return np.full((height, width), fill, dtype=np.uint8)


def make_state(game_variables=None, labels=None, depth_buffer=None, tic=100):
    return SimpleNamespace(
        game_variables=game_variables if game_variables is not None else make_game_variables(),
        labels=labels if labels is not None else [],
        depth_buffer=depth_buffer if depth_buffer is not None else make_depth_buffer(),
        tic=tic,
    )


def test_kind_of_known_monster():
    assert p.kind_of("Imp", None) == ("enemy", "imp")
    assert p.kind_of("ShotgunGuy", "Monsters") == ("enemy", "shotgun_guy")


def test_kind_of_known_pickup():
    assert p.kind_of("Medikit", None) == ("pickup", "health_pack")
    assert p.kind_of("ClipBox", None) == ("pickup", "ammo")


def test_kind_of_category_fallback():
    assert p.kind_of("SomeNewMonster", "Monsters") == ("enemy", "some_new_monster")
    assert p.kind_of("SomeNewItem", "Items") == ("pickup", "some_new_item")


def test_kind_of_ignored():
    assert p.kind_of("DoomPlayer", None) is None
    assert p.kind_of("TechPillar", "Obstacles") is None


@pytest.mark.parametrize(
    "center_x,width,expected",
    [
        (0, 320, "far-left"),
        (159, 320, "front"),
        (160, 320, "front"),
        (319, 320, "far-right"),
    ],
)
def test_bearing_bucket(center_x, width, expected):
    assert p.bearing_bucket(center_x, width) == expected


def test_bearing_bucket_seven_bins_cover_full_width():
    seen = {p.bearing_bucket(x, 700) for x in range(0, 700, 10)}
    assert seen == set(p.BEARING_BUCKETS)


def test_distance_bucket_boundaries():
    config = p.PerceptionConfig()
    # very-near threshold is 64 game units; GAME_UNITS_PER_DEPTH_UNIT ~= 7.14
    very_near_raw = int(60 / p.GAME_UNITS_PER_DEPTH_UNIT)
    bucket, _ = config.distance_bucket(very_near_raw)
    assert bucket == "very-near"

    far_raw = int(600 / p.GAME_UNITS_PER_DEPTH_UNIT)
    bucket, _ = config.distance_bucket(far_raw)
    assert bucket == "far"


def test_perceive_returns_none_for_finished_episode():
    assert p.perceive(None) is None


def test_perceive_reads_game_variables():
    state = make_state(game_variables=make_game_variables(HEALTH=74, ARMOR=20, SELECTED_WEAPON_AMMO=13))
    perc = p.perceive(state)
    assert perc.health == 74
    assert perc.armor == 20
    assert perc.ammo == 13
    assert perc.alive is True


def test_perceive_dead_flag():
    state = make_state(game_variables=make_game_variables(DEAD=1))
    perc = p.perceive(state)
    assert perc.alive is False


def test_perceive_classifies_and_sorts_enemies_by_distance():
    depth = make_depth_buffer(fill=200)
    # near label bottom-left region, far label center
    near_label = make_label("Imp", x=10, y=100, width=20, height=40)
    far_label = make_label("ShotgunGuy", x=150, y=100, width=20, height=40)
    depth[:, :] = 200  # far everywhere by default
    depth[100:140, 10:30] = 5  # near for the imp's bbox

    state = make_state(labels=[far_label, near_label], depth_buffer=depth)
    perc = p.perceive(state, p.PerceptionConfig(max_enemies=3))

    assert [e.kind for e in perc.enemies] == ["imp", "shotgun_guy"]
    assert perc.enemies[0].distance == "very-near"


def test_perceive_respects_max_enemies():
    labels = [make_label("Imp", x=i * 10, y=0, width=5, height=5) for i in range(5)]
    state = make_state(labels=labels)
    perc = p.perceive(state, p.PerceptionConfig(max_enemies=2))
    assert len(perc.enemies) == 2


def test_max_enemies_keeps_the_nearest_not_the_first_listed():
    depth = make_depth_buffer(fill=200)
    labels = [make_label("Imp", x=i * 40, y=100, width=20, height=40) for i in range(5)]
    depth[100:140, 160:180] = 5  # only the LAST-listed imp (x=160) is close
    state = make_state(labels=labels, depth_buffer=depth)
    perc = p.perceive(state, p.PerceptionConfig(max_enemies=2))
    assert len(perc.enemies) == 2
    assert perc.enemies[0].distance == "very-near"
    assert perc.enemies[0].bearing == "front"


def test_wall_scan_detects_close_wall_ahead():
    depth = make_depth_buffer(width=320, height=240, fill=5)  # everything very close
    state = make_state(depth_buffer=depth)
    perc = p.perceive(state)
    assert perc.wall_ahead is True
    assert perc.wall_near is True
    assert perc.open_forward is False


def test_wall_scan_detects_open_space():
    depth = make_depth_buffer(width=320, height=240, fill=250)  # everything far
    state = make_state(depth_buffer=depth)
    perc = p.perceive(state)
    assert perc.wall_ahead is False
    assert perc.open_forward is True
    assert perc.open_left is True
    assert perc.open_right is True
