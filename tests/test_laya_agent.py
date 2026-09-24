"""Unit tests for LayaAgent's own logic (laya_agent.py) — label->action
mapping, the shoot-gate design (a `noul` question gating the combat
action, with a deterministic AMMO>0 guard, instead of one `choice` call
over every label — see laya_agent.py's module-level comment block for the
real measured 0/8 result that motivated this), confidence-threshold
gating on the movement fallback, and Decision field population.

Doesn't load the real model: laya.load() downloads ~800MB of weights and
runs on real torch/MPS, so it's faked out here — this tests LayaAgent's
own bookkeeping in isolation, not Laya's actual Doom-playing behaviour
(that's what experiments/compare.py, scripts/probe_criteria.py and
scripts/probe_direction_bias.py are for).
"""

from __future__ import annotations

from unittest.mock import patch

from laya_doom.laya_agent import (
    QUESTION_ID,
    LayaAgent,
    build_criteria,
    build_movement_criteria,
    collapse_directions,
    resolve_direction,
)
from laya_doom.perception import EnemyPercept, Perception, PickupPercept

DEFAULT_MOVE_RESPONSE = {
    "answers": {
        QUESTION_ID: {
            "type": "choice",
            "choice": "move_forward",
            "probabilities": {"move_forward": 0.6, "turn_left": 0.25, "turn_right": 0.15},
            "confidence": 0.6,
        }
    }
}

# A gate response with no "should_shoot" answer at all — the shoot gate
# then sees shoot_prob=None, want_shoot=False, and falls straight through
# to the movement call, same as if the model had nothing to say.
NO_SHOOT_GATE_RESPONSE = {"answers": {}}


class _FakeLayaAgent:
    """Stands in for laya.Agent. predict() returns responses from a small
    queue, consumed in call order (last one repeats) — the shoot-gate
    design makes up to two real predict() calls per decide() (gate, then
    movement), so tests that care about both need to supply both."""

    def __init__(self, responses=None):
        self.predict_calls = []
        self._responses = list(responses) if responses is not None else [NO_SHOOT_GATE_RESPONSE, DEFAULT_MOVE_RESPONSE]

    def predict(self, state, questions):
        self.predict_calls.append((state, questions))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _perception(ammo: int = 50, **overrides) -> Perception:
    """A full, real Perception — the shoot gate now builds its own
    stateless encoding straight from this object (see laya_agent.py's
    ``_gate_encoder``), so a bare stand-in with only ``.ammo`` is no
    longer enough."""
    base = dict(
        tic=0,
        alive=True,
        health=100,
        armor=0,
        ammo=ammo,
        weapon=3,
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
    base.update(overrides)
    return Perception(**base)


def _make_agent(responses=None, **kwargs) -> tuple[LayaAgent, _FakeLayaAgent]:
    fake = _FakeLayaAgent(responses=responses)
    with patch("laya_doom.laya_agent.laya.load", return_value=fake):
        agent = LayaAgent(action_set="stage1", **kwargs)
    return agent, fake


def _gate_response(p: float) -> dict:
    return {"answers": {"should_shoot": {"type": "noul", "noul": p}}}


def test_build_criteria_covers_exactly_the_action_set():
    assert set(build_criteria("stage1")) == {"move_forward", "turn_left", "turn_right", "shoot"}
    assert set(build_criteria("full")) == {
        "move_forward",
        "move_backward",
        "strafe_left",
        "strafe_right",
        "turn_left_small",
        "turn_right_small",
        "turn_left_large",
        "turn_right_large",
        "attack",
        "use",
        "wait",
    }


def test_movement_criteria_excludes_the_combat_action():
    assert "shoot" not in build_movement_criteria("stage1")
    assert set(build_movement_criteria("stage1")) == {"move_forward", "turn_left", "turn_right"}
    assert "attack" not in build_movement_criteria("full")


def test_load_is_called_once_at_construction():
    agent, fake = _make_agent()
    assert agent.agent is fake


def test_decide_queries_the_gate_first_then_falls_back_to_movement_choice():
    agent, fake = _make_agent(responses=[NO_SHOOT_GATE_RESPONSE, DEFAULT_MOVE_RESPONSE])
    decision = agent.decide(_perception(), "HEALTH 100")
    assert decision.action == "move_forward"
    assert len(fake.predict_calls) == 2
    gate_state, gate_questions = fake.predict_calls[0]
    # The gate builds its OWN stateless encoding straight from the
    # Perception object (see laya_agent.py's _gate_encoder and the real
    # episode data motivating that) — it does not reuse the caller's
    # encoded_state string, which may carry a different --memory mode.
    assert "HEALTH 100" in gate_state
    assert "should_shoot" in gate_questions
    assert gate_questions["should_shoot"]["type"] == "noul"
    move_state, move_questions = fake.predict_calls[1]
    assert move_state == "HEALTH 100"  # movement still gets the caller's encoded_state verbatim
    assert QUESTION_ID in move_questions
    assert "shoot" not in move_questions[QUESTION_ID]["criteria"]


def test_shoot_gate_fires_the_combat_action_above_threshold_with_ammo():
    agent, fake = _make_agent(responses=[_gate_response(0.8)], shoot_gate_threshold=0.45)
    enemy_ahead = _perception(ammo=50, enemies=(EnemyPercept("imp", "front", "near", 150.0),))
    decision = agent.decide(enemy_ahead, "HEALTH 100\nENEMY imp front near")
    assert decision.action == "shoot"
    assert decision.confidence == 0.8
    assert len(fake.predict_calls) == 1  # never needed the movement call
    assert "0.800" in decision.reasoning


def test_shoot_gate_below_threshold_falls_back_to_movement():
    agent, fake = _make_agent(
        responses=[_gate_response(0.20), DEFAULT_MOVE_RESPONSE], shoot_gate_threshold=0.45
    )
    decision = agent.decide(_perception(ammo=50), "HEALTH 100")
    assert decision.action == "move_forward"
    assert len(fake.predict_calls) == 2


def test_ammo_guard_blocks_shoot_even_when_gate_says_yes():
    # Real measured behaviour this guards against: the gate alone scored
    # a no-ammo state at P=0.512 (above threshold) despite its own
    # instructions requiring AMMO>0 — see laya_agent.py's comment block.
    agent, fake = _make_agent(
        responses=[_gate_response(0.9), DEFAULT_MOVE_RESPONSE], shoot_gate_threshold=0.45
    )
    no_ammo_enemy_ahead = _perception(ammo=0, enemies=(EnemyPercept("imp", "front", "near", 150.0),))
    decision = agent.decide(no_ammo_enemy_ahead, "HEALTH 100\nENEMY imp front near")
    assert decision.action != "shoot"
    assert decision.action == "move_forward"
    assert "AMMO<=0" in decision.reasoning
    assert len(fake.predict_calls) == 2  # ammo guard forced the movement fallback call


def test_bearing_guard_blocks_shoot_when_no_enemy_is_exactly_ahead():
    # Real measured behaviour this guards against: a live episode logged
    # 120 real shoot decisions, 100% at bearing front-left/front-right and
    # 0% at exact front — 0 damage given. Bearing is a deterministic fact
    # in Perception, so it's checked in code rather than trusted from the
    # gate's own probability (which fired anyway — see laya_agent.py).
    agent, fake = _make_agent(
        responses=[_gate_response(0.9), DEFAULT_MOVE_RESPONSE], shoot_gate_threshold=0.45
    )
    enemy_off_center = _perception(ammo=50, enemies=(EnemyPercept("imp", "front-left", "near", 150.0),))
    decision = agent.decide(enemy_off_center, "HEALTH 100\nENEMY imp front-left near")
    assert decision.action != "shoot"
    assert decision.action == "move_forward"
    assert "no enemy at bearing==front" in decision.reasoning
    assert len(fake.predict_calls) == 2


def test_shoot_gate_disabled_uses_a_single_choice_call_over_every_label():
    agent, fake = _make_agent(
        responses=[
            {
                "answers": {
                    QUESTION_ID: {
                        "choice": "shoot",
                        "probabilities": {"shoot": 0.5, "move_forward": 0.3, "turn_left": 0.2},
                        "confidence": 0.5,
                    }
                }
            }
        ],
        use_shoot_gate=False,
    )
    decision = agent.decide(_perception(), "HEALTH 100")
    assert decision.action == "shoot"
    assert len(fake.predict_calls) == 1
    _, questions = fake.predict_calls[0]
    assert "shoot" in questions[QUESTION_ID]["criteria"]


def test_decide_falls_back_to_wait_on_unrecognized_choice():
    agent, fake = _make_agent(
        responses=[
            NO_SHOOT_GATE_RESPONSE,
            {"answers": {QUESTION_ID: {"choice": "nonsense_label", "probabilities": {}, "confidence": 0.9}}},
        ]
    )
    decision = agent.decide(_perception(), "HEALTH 100")
    assert decision.action == "wait"


def test_decide_falls_back_to_wait_on_missing_answer():
    agent, fake = _make_agent(responses=[NO_SHOOT_GATE_RESPONSE, {"answers": {}}])
    decision = agent.decide(_perception(), "HEALTH 100")
    assert decision.action == "wait"


def test_confidence_threshold_gating_substitutes_wait_below_threshold():
    agent, fake = _make_agent(
        responses=[
            NO_SHOOT_GATE_RESPONSE,
            {"answers": {QUESTION_ID: {"choice": "move_forward", "probabilities": {"move_forward": 0.3}, "confidence": 0.3}}},
        ],
        confidence_mode="confidence_threshold",
        confidence_threshold=0.5,
    )
    decision = agent.decide(_perception(), "HEALTH 100")
    assert decision.action == "wait"
    assert decision.confidence == 0.3  # gating substitutes the action, not the reported confidence


def test_confidence_threshold_gating_executes_at_or_above_threshold():
    agent, fake = _make_agent(
        responses=[
            NO_SHOOT_GATE_RESPONSE,
            {"answers": {QUESTION_ID: {"choice": "move_forward", "probabilities": {"move_forward": 0.5}, "confidence": 0.5}}},
        ],
        confidence_mode="confidence_threshold",
        confidence_threshold=0.5,
    )
    decision = agent.decide(_perception(), "HEALTH 100")
    assert decision.action == "move_forward"


def test_always_execute_ignores_low_confidence():
    agent, fake = _make_agent(
        responses=[
            NO_SHOOT_GATE_RESPONSE,
            {"answers": {QUESTION_ID: {"choice": "move_forward", "probabilities": {"move_forward": 0.05}, "confidence": 0.05}}},
        ],
        confidence_mode="always_execute",
    )
    decision = agent.decide(_perception(), "HEALTH 100")
    assert decision.action == "move_forward"


def test_reasoning_includes_a_real_rendering_of_the_movement_probabilities():
    agent, fake = _make_agent(
        responses=[
            NO_SHOOT_GATE_RESPONSE,
            {
                "answers": {
                    QUESTION_ID: {
                        "choice": "move_forward",
                        "probabilities": {"turn_left": 0.1, "move_forward": 0.7, "turn_right": 0.2},
                        "confidence": 0.7,
                    }
                }
            },
        ]
    )
    decision = agent.decide(_perception(), "HEALTH 100")
    assert "move_forward:0.700 turn_right:0.200 turn_left:0.100 (confidence=0.700)" in decision.reasoning
    assert decision.reasoning.startswith("should_shoot=")


def test_raw_response_includes_both_gate_and_movement_calls():
    gate_resp = NO_SHOOT_GATE_RESPONSE
    move_resp = DEFAULT_MOVE_RESPONSE
    agent, fake = _make_agent(responses=[gate_resp, move_resp])
    decision = agent.decide(_perception(), "HEALTH 100")
    assert decision.raw == {"gate": gate_resp, "movement": move_resp}


def test_reset_is_a_no_op_and_does_not_raise():
    agent, fake = _make_agent(responses=[NO_SHOOT_GATE_RESPONSE, DEFAULT_MOVE_RESPONSE])
    assert agent.reset() is None
    # Calling decide() again afterward still works — no state to have broken.
    decision = agent.decide(_perception(), "HEALTH 100")
    assert decision.action == "move_forward"


# --- direction_mode="resolved" -------------------------------------------


def _collapsed_move_response(label: str) -> dict:
    return {"answers": {QUESTION_ID: {"choice": label, "probabilities": {label: 0.5}, "confidence": 0.5}}}


def test_collapse_directions_merges_each_left_right_pair_in_order():
    collapsed = collapse_directions(build_movement_criteria("full"))
    assert list(collapsed) == ["move_forward", "move_backward", "strafe", "turn_small", "turn_large", "use", "wait"]
    assert list(collapse_directions(build_movement_criteria("stage1"))) == ["move_forward", "turn"]


def test_resolved_mode_offers_only_direction_free_labels():
    agent, fake = _make_agent(responses=[NO_SHOOT_GATE_RESPONSE, _collapsed_move_response("turn")], direction_mode="resolved")
    agent.decide(_perception(), "HEALTH 100")
    _, move_questions = fake.predict_calls[1]
    assert set(move_questions[QUESTION_ID]["criteria"]) == {"move_forward", "turn"}


def test_resolved_mode_turns_toward_an_off_center_enemy():
    agent, _ = _make_agent(responses=[_gate_response(0.1), _collapsed_move_response("turn")], direction_mode="resolved")
    enemy_right = _perception(enemies=(EnemyPercept("imp", "right", "near", 150.0),))
    decision = agent.decide(enemy_right, "HEALTH 100")
    assert decision.action == "turn_right"
    assert "turn -> turn_right (enemy)" in decision.reasoning


def test_resolved_mode_prefers_the_only_open_side_without_a_target():
    agent, _ = _make_agent(responses=[NO_SHOOT_GATE_RESPONSE, _collapsed_move_response("turn")], direction_mode="resolved")
    decision = agent.decide(_perception(open_left=True, open_right=False), "HEALTH 100")
    assert decision.action == "turn_left"


def test_resolved_mode_keeps_the_last_side_on_a_tie():
    agent, _ = _make_agent(responses=[NO_SHOOT_GATE_RESPONSE, _collapsed_move_response("turn")], direction_mode="resolved")
    agent.decide(_perception(open_left=True, open_right=False), "HEALTH 100")  # -> left
    decision = agent.decide(_perception(open_left=True, open_right=True), "HEALTH 100")
    assert decision.action == "turn_left"
    assert "(sticky)" in decision.reasoning


def test_resolved_mode_leaves_non_directional_labels_alone():
    agent, _ = _make_agent(responses=[NO_SHOOT_GATE_RESPONSE, DEFAULT_MOVE_RESPONSE], direction_mode="resolved")
    assert agent.decide(_perception(), "HEALTH 100").action == "move_forward"


def test_resolve_direction_enemy_outranks_pickup():
    perc = _perception(
        enemies=(EnemyPercept("imp", "far-left", "far", 900.0),),
        pickups=(PickupPercept("ammo", "right", "near", 100.0),),
    )
    assert resolve_direction(perc, "right") == ("left", "enemy")



# --- movement_head (stuntd-trained head) ------------------------------------


def test_movement_head_answers_the_movement_question_and_the_gate_stays_zero_shot():
    from types import SimpleNamespace

    agent, fake = _make_agent(responses=[NO_SHOOT_GATE_RESPONSE])
    labels = ("move_forward", "turn_left", "turn_right")
    calls = []

    class _FakeDecider:
        def decide(self, model, head_path, text):
            calls.append(text)
            return SimpleNamespace(label=2, confidence=0.9, probabilities=(0.05, 0.05, 0.9))

    agent._head_decider = _FakeDecider()
    agent._head_model = SimpleNamespace(field="action", labels=labels, temperature=1.0)
    agent._head_path = "head.safetensors"
    decision = agent.decide(_perception(), "HEALTH 100")
    assert decision.action == "turn_right" and decision.confidence == 0.9
    assert calls == ["HEALTH 100"]
    assert len(fake.predict_calls) == 1  # only the zero-shot gate went to the base model
    assert "turn_right:0.900" in decision.reasoning
