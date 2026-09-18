from laya_doom.controller import ThreatResponseConfig, _recovery_turn, _should_override_for_threat
from tests.test_state_encoder import make_perception


def test_recovery_turn_prefers_open_side_left():
    perc = make_perception(open_left=True, open_right=False)
    assert _recovery_turn(perc, "stage1", step=0) == "turn_left"


def test_recovery_turn_prefers_open_side_right():
    perc = make_perception(open_left=False, open_right=True)
    assert _recovery_turn(perc, "stage1", step=0) == "turn_right"


def test_recovery_turn_alternates_when_tied():
    perc = make_perception(open_left=True, open_right=True)
    assert _recovery_turn(perc, "stage1", step=0) == "turn_left"
    assert _recovery_turn(perc, "stage1", step=1) == "turn_right"


def test_recovery_turn_uses_full_action_set_names():
    perc = make_perception(open_left=True, open_right=False)
    assert _recovery_turn(perc, "full", step=0) == "turn_left_small"


def test_threat_override_fires_on_unseen_damage():
    config = ThreatResponseConfig()
    perc = make_perception(enemies=())
    assert _should_override_for_threat(config, "took_damage", perc, "move_forward") is True


def test_threat_override_not_fired_when_enemy_visible():
    from laya_doom.perception import EnemyPercept

    config = ThreatResponseConfig()
    perc = make_perception(enemies=(EnemyPercept(kind="imp", bearing="front", distance="near", raw_distance_units=100.0),))
    assert _should_override_for_threat(config, "took_damage", perc, "move_forward") is False


def test_threat_override_not_fired_without_recent_damage():
    config = ThreatResponseConfig()
    perc = make_perception(enemies=())
    assert _should_override_for_threat(config, "no_change", perc, "move_forward") is False


def test_threat_override_not_fired_when_already_turning():
    config = ThreatResponseConfig()
    perc = make_perception(enemies=())
    assert _should_override_for_threat(config, "took_damage", perc, "turn_left_small") is False


def test_threat_override_disabled_via_config():
    config = ThreatResponseConfig(enabled=False)
    perc = make_perception(enemies=())
    assert _should_override_for_threat(config, "took_damage", perc, "move_forward") is False
