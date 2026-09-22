from laya_doom.perception import EnemyPercept, Perception, PickupPercept
from laya_doom.state_encoder import EncoderConfig, GOAL_LINE, StateEncoder, summarize_result


def make_perception(**overrides) -> Perception:
    defaults = dict(
        tic=0,
        alive=True,
        health=100,
        armor=0,
        ammo=50,
        weapon=2,
        killcount=0,
        itemcount=0,
        damagecount=0,
        damage_taken=0,
        hitcount=0,
        x=0.0,
        y=0.0,
        angle=0.0,
        enemies=(),
        pickups=(),
        wall_ahead=False,
        wall_near=False,
        open_forward=True,
        open_left=True,
        open_right=True,
    )
    defaults.update(overrides)
    return Perception(**defaults)


def test_encode_includes_core_facts():
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))
    perc = make_perception(health=74, armor=20, ammo=13)
    text = encoder.encode(perc, None, None, None)
    assert "HEALTH 74" in text
    assert "ARMOR 20" in text
    assert "AMMO 13" in text


def test_encode_lists_enemies_and_pickups():
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))
    perc = make_perception(
        enemies=(EnemyPercept(kind="imp", bearing="front-right", distance="near", raw_distance_units=100.0),),
        pickups=(PickupPercept(kind="health_pack", bearing="front", distance="medium", raw_distance_units=300.0),),
    )
    text = encoder.encode(perc, None, None, None)
    assert "ENEMY imp front-right near" in text
    assert "PICKUP health_pack front medium" in text


def test_stateless_mode_omits_history_and_deltas():
    encoder = StateEncoder(EncoderConfig(memory_mode="stateless", include_goal_line=False))
    prev = make_perception(health=100)
    cur = make_perception(health=80)
    text = encoder.encode(cur, prev, "attack", "took_damage")
    assert "LAST_ACTION" not in text
    assert "HEALTH_CHANGE" not in text


def test_prev_state_mode_includes_deltas_and_last_action():
    encoder = StateEncoder(EncoderConfig(memory_mode="prev_state", include_goal_line=False))
    prev = make_perception(health=100, ammo=10, x=0.0, y=0.0)
    cur = make_perception(health=80, ammo=8, x=50.0, y=0.0)
    text = encoder.encode(cur, prev, "attack", "took_damage")
    assert "LAST_ACTION attack" in text
    assert "LAST_RESULT took_damage" in text
    assert "HEALTH_CHANGE -20" in text
    assert "AMMO_CHANGE -2" in text
    assert "MOVED yes" in text


def test_rolling_mode_accumulates_history():
    encoder = StateEncoder(EncoderConfig(memory_mode="rolling", rolling_window=2, include_goal_line=False))
    perc = make_perception()
    encoder.record("move_forward", "moved")
    encoder.record("attack", "killed_enemy")
    encoder.record("turn_left", "no_change")  # should evict move_forward
    text = encoder.encode(perc, perc, "turn_left", "no_change")
    assert "HISTORY attack->killed_enemy turn_left->no_change" in text
    assert "move_forward" not in text


def test_reset_clears_history():
    encoder = StateEncoder(EncoderConfig(memory_mode="rolling", include_goal_line=False))
    encoder.record("attack", "killed_enemy")
    encoder.reset()
    perc = make_perception()
    text = encoder.encode(perc, None, None, None)
    assert "HISTORY" not in text


def test_goal_line_toggle():
    perc = make_perception()
    with_goal = StateEncoder(EncoderConfig(include_goal_line=True)).encode(perc, None, None, None)
    without_goal = StateEncoder(EncoderConfig(include_goal_line=False)).encode(perc, None, None, None)
    assert with_goal.splitlines()[0] == GOAL_LINE
    assert GOAL_LINE not in without_goal


def test_area_hint_new_then_revisited():
    encoder = StateEncoder(EncoderConfig(include_goal_line=False, area_cell_size=100.0))
    here = make_perception(x=10.0, y=10.0)
    elsewhere = make_perception(x=500.0, y=500.0)

    first = encoder.encode(here, None, None, None)
    assert "AREA new" in first
    assert encoder.last_area_new is True

    second = encoder.encode(here, None, None, None)
    assert "AREA revisited" in second
    assert encoder.last_area_new is False

    third = encoder.encode(elsewhere, None, None, None)
    assert "AREA new" in third
    assert encoder.last_area_new is True


def test_area_hint_toggle_and_reset():
    encoder = StateEncoder(EncoderConfig(include_goal_line=False, include_area_hint=False))
    text = encoder.encode(make_perception(), None, None, None)
    assert "AREA" not in text

    encoder = StateEncoder(EncoderConfig(include_goal_line=False))
    perc = make_perception()
    encoder.encode(perc, None, None, None)
    encoder.reset()
    assert encoder.last_area_new is None
    assert "AREA revisited" not in encoder.encode(perc, None, None, None)


def test_last_area_new_none_before_first_encode_or_when_disabled():
    # Exposed for controller.py's ExplorationNudgeConfig (a circling
    # detector) to read directly rather than re-deriving it by parsing
    # the encoded text.
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))
    assert encoder.last_area_new is None

    disabled = StateEncoder(EncoderConfig(include_goal_line=False, include_area_hint=False))
    disabled.encode(make_perception(), None, None, None)
    assert disabled.last_area_new is None


def test_keys_hint_omitted_when_empty():
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))
    text = encoder.encode(make_perception(), None, None, None)
    assert "KEYS" not in text


def test_keys_hint_lists_held_keys():
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))
    text = encoder.encode(make_perception(), None, None, None, keys_held=frozenset({"blue_key", "red_key"}))
    assert "KEYS blue_key red_key" in text


def test_keys_hint_toggle():
    encoder = StateEncoder(EncoderConfig(include_goal_line=False, include_keys_hint=False))
    text = encoder.encode(make_perception(), None, None, None, keys_held=frozenset({"blue_key"}))
    assert "KEYS" not in text


def test_threat_hint_fires_on_unseen_damage():
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))
    prev = make_perception(health=100, enemies=())
    cur = make_perception(health=85, enemies=())
    text = encoder.encode(cur, prev, "move_forward", "took_damage")
    assert "THREAT unseen behind" in text


def test_threat_hint_omitted_when_enemy_visible():
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))
    prev = make_perception(health=100, enemies=())
    cur = make_perception(
        health=85,
        enemies=(EnemyPercept(kind="imp", bearing="front", distance="near", raw_distance_units=100.0),),
    )
    text = encoder.encode(cur, prev, "move_forward", "took_damage")
    assert "THREAT" not in text


def test_threat_hint_omitted_when_health_unchanged_or_up():
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))
    prev = make_perception(health=100, enemies=())
    same = make_perception(health=100, enemies=())
    healed = make_perception(health=100, enemies=())
    assert "THREAT" not in encoder.encode(same, prev, "wait", "no_change")
    assert "THREAT" not in encoder.encode(healed, make_perception(health=90, enemies=()), "wait", "picked_up_item")


def test_threat_hint_toggle():
    encoder = StateEncoder(EncoderConfig(include_goal_line=False, include_threat_hint=False))
    prev = make_perception(health=100, enemies=())
    cur = make_perception(health=85, enemies=())
    text = encoder.encode(cur, prev, "move_forward", "took_damage")
    assert "THREAT" not in text


def test_summarize_result_priority_order():
    before = make_perception(health=100, damage_taken=0, killcount=0, itemcount=0, x=0, y=0)
    took_damage = make_perception(health=80, damage_taken=20, killcount=0, itemcount=0, x=0, y=0)
    assert summarize_result(before, took_damage) == "took_damage"

    killed = make_perception(health=100, damage_taken=0, killcount=1, itemcount=0, x=0, y=0)
    assert summarize_result(before, killed) == "killed_enemy"

    picked_up = make_perception(health=100, damage_taken=0, killcount=0, itemcount=1, x=0, y=0)
    assert summarize_result(before, picked_up) == "picked_up_item"

    moved = make_perception(health=100, damage_taken=0, killcount=0, itemcount=0, x=50, y=0)
    assert summarize_result(before, moved) == "moved"

    unchanged = make_perception(health=100, damage_taken=0, killcount=0, itemcount=0, x=0, y=0)
    assert summarize_result(before, unchanged) == "no_change"


def test_summarize_result_enemy_distance_change():
    before = make_perception(
        enemies=(EnemyPercept(kind="imp", bearing="front", distance="medium", raw_distance_units=300.0),)
    )
    closer = make_perception(
        enemies=(EnemyPercept(kind="imp", bearing="front", distance="near", raw_distance_units=100.0),)
    )
    assert summarize_result(before, closer) == "enemy_closer"


def test_summarize_result_death_and_none():
    dead = make_perception(alive=False)
    assert summarize_result(make_perception(), dead) == "died"
    assert summarize_result(make_perception(), None) == "episode_ended"
