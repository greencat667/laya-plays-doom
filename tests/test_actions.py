from laya_doom import actions


def test_default_tics_are_positive():
    for name, tics in actions.DEFAULT_TICS.items():
        assert tics >= 1, name


def test_build_action_table_covers_all_names():
    table = actions.build_action_table()
    assert set(table) == set(actions.DEFAULT_TICS)
    for name, spec in table.items():
        assert spec.tics == actions.DEFAULT_TICS[name]


def test_build_action_table_uniform_tics_override():
    table = actions.build_action_table(uniform_tics=7)
    for spec in table.values():
        assert spec.tics == 7


def test_button_vector_matches_pressed_buttons():
    table = actions.build_action_table()
    spec = table["move_forward"]
    vector = spec.button_vector()
    assert len(vector) == len(actions.ALL_BUTTONS)
    forward_index = actions.ALL_BUTTONS.index(__import__("vizdoom").Button.MOVE_FORWARD)
    assert vector[forward_index] == 1
    assert sum(vector) == 1


def test_wait_presses_no_buttons():
    table = actions.build_action_table()
    assert sum(table["wait"].button_vector()) == 0


def test_get_action_names():
    assert actions.get_action_names("stage1") == actions.STAGE1_ACTIONS
    assert actions.get_action_names("full") == actions.FULL_ACTIONS


# No resolve_call_to_action here (unlike the sibling Needle project):
# Laya's choice labels ARE the canonical action names directly (see
# laya_agent.build_criteria), so there's no tool-name/arguments indirection
# to resolve — covered instead by tests/test_laya_agent.py.
