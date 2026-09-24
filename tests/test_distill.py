"""Teacher filter for scripts/distill_movement_head.py."""

from scripts.distill_movement_head import build_examples, mirror, teacher_label


def _row(action, result="moved", overridden=False, reason="", state="S"):
    return {"action": action, "result": result, "overridden": overridden, "override_reason": reason, "encoded_state": state}


def test_moves_are_copied_only_when_they_achieved_something():
    assert teacher_label(_row("move_forward")) == "move_forward"
    assert teacher_label(_row("strafe_left", result="no_change")) is None
    assert teacher_label(_row("move_backward", result="took_damage", overridden=True, reason="low_health_retreat")) == "move_backward"


def test_turns_are_copied_only_from_reactive_nets():
    assert teacher_label(_row("turn_left_small", result="no_change")) is None  # Laya's own: label-order noise
    assert teacher_label(_row("turn_right_small", result="no_change", overridden=True, reason="threat_engagement")) == "turn_right_small"
    assert teacher_label(_row("turn_left_large", overridden=True, reason="stuck_recovery")) == "turn_left_large"


def test_map_dependent_overrides_wait_use_and_combat_are_skipped():
    assert teacher_label(_row("move_forward", overridden=True, reason="frontier_planner")) is None
    assert teacher_label(_row("turn_left_large", overridden=True, reason="wall_follow")) is None
    assert teacher_label(_row("wait")) is None
    assert teacher_label(_row("use", overridden=True, reason="door_use")) is None
    assert teacher_label(_row("attack")) is None


def test_identical_states_take_the_majority_label(tmp_path):
    import json

    path = tmp_path / "x.steps.jsonl"
    rows = [_row("move_forward"), _row("move_forward"), _row("strafe_left"), _row("strafe_right", state="T")]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert sorted(build_examples([str(path)], mirrored=False)) == [("S", "move_forward"), ("T", "strafe_right")]


def test_mirror_swaps_left_and_right_in_state_and_label():
    assert mirror("ENEMY imp front-left near\nPATH left open\nPATH right blocked") == (
        "ENEMY imp front-right near\nPATH right open\nPATH left blocked"
    )
    assert mirror("turn_right_small") == "turn_left_small"
    assert mirror("move_forward") == "move_forward"


def test_planner_moves_are_copied_only_when_the_frontier_line_explains_them():
    blind = _row("turn_left_large", result="no_change", overridden=True, reason="frontier_planner", state="HEALTH 100")
    explained = dict(blind, encoded_state="HEALTH 100\nFRONTIER left medium")
    assert teacher_label(blind) is None
    assert teacher_label(explained) == "turn_left_large"
    assert teacher_label(dict(explained, action="move_forward", result="moved")) == "move_forward"
    assert teacher_label(dict(explained, override_reason="wall_follow")) is None


def test_steps_leading_up_to_a_death_are_dropped(tmp_path):
    import json

    rows = [dict(_row("move_forward", state=f"S{i}"), episode=0, step=i) for i in range(10)]
    rows[-1]["result"] = "episode_ended"  # what a real death step logs
    path = tmp_path / "d.steps.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    kept = {text for text, _ in build_examples([str(path)], mirrored=False, drop_before_death=3)}
    assert kept == {f"S{i}" for i in range(7)}  # steps 7-9 (the last 3, incl. the death) dropped
