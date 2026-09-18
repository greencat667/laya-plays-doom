"""Unit tests for LayaAgent's own logic (laya_agent.py) — label->action
mapping, confidence-threshold gating, and Decision field population.

Doesn't load the real model: laya.load() downloads ~800MB of weights and
runs on real torch/MPS, so it's faked out here exactly the way the sibling
Needle project fakes out needle.Needle in test_needle_agent.py — this
tests LayaAgent's own bookkeeping in isolation, not Laya's actual
Doom-playing behaviour (that's what experiments/compare.py and the
README's "Verified behaviour" section are for).
"""

from __future__ import annotations

from unittest.mock import patch

from laya_doom.laya_agent import LayaAgent, QUESTION_ID, build_criteria


class _FakeLayaAgent:
    """Stands in for laya.Agent. predict() returns a fixed, realistic
    response shape (as seen from the real installed package: {"answers":
    {qid: {"type": "choice", "choice": ..., "probabilities": {...},
    "confidence": ...}}}) so LayaAgent's own parsing/gating logic can be
    tested without touching torch or the network."""

    def __init__(self, response=None):
        self.predict_calls = []
        self._response = response or {
            "answers": {
                QUESTION_ID: {
                    "type": "choice",
                    "choice": "move_forward",
                    "probabilities": {"move_forward": 0.6, "turn_left": 0.25, "turn_right": 0.15},
                    "confidence": 0.6,
                }
            }
        }

    def predict(self, state, questions):
        self.predict_calls.append((state, questions))
        return self._response


def _make_agent(response=None, **kwargs) -> tuple[LayaAgent, _FakeLayaAgent]:
    fake = _FakeLayaAgent(response=response)
    with patch("laya_doom.laya_agent.laya.load", return_value=fake):
        agent = LayaAgent(action_set="stage1", **kwargs)
    return agent, fake


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


def test_load_is_called_once_at_construction():
    agent, fake = _make_agent()
    assert agent.agent is fake


def test_decide_maps_top_choice_straight_to_canonical_action():
    agent, fake = _make_agent(
        response={
            "answers": {
                QUESTION_ID: {
                    "choice": "turn_right",
                    "probabilities": {"turn_right": 0.5, "move_forward": 0.3, "turn_left": 0.2},
                    "confidence": 0.5,
                }
            }
        }
    )
    decision = agent.decide(None, "HEALTH 100")
    assert decision.action == "turn_right"
    assert decision.confidence == 0.5


def test_decide_passes_the_encoded_state_straight_through_as_the_state_arg():
    agent, fake = _make_agent()
    agent.decide(None, "HEALTH 74\nENEMY imp front near")
    state, questions = fake.predict_calls[0]
    assert state == "HEALTH 74\nENEMY imp front near"
    assert QUESTION_ID in questions
    assert questions[QUESTION_ID]["type"] == "choice"


def test_decide_falls_back_to_wait_on_unrecognized_choice():
    agent, fake = _make_agent(
        response={"answers": {QUESTION_ID: {"choice": "nonsense_label", "probabilities": {}, "confidence": 0.9}}}
    )
    decision = agent.decide(None, "HEALTH 100")
    assert decision.action == "wait"


def test_decide_falls_back_to_wait_on_missing_answer():
    agent, fake = _make_agent(response={"answers": {}})
    decision = agent.decide(None, "HEALTH 100")
    assert decision.action == "wait"


def test_confidence_threshold_gating_substitutes_wait_below_threshold():
    agent, fake = _make_agent(
        confidence_mode="confidence_threshold",
        confidence_threshold=0.5,
        response={
            "answers": {
                QUESTION_ID: {
                    "choice": "move_forward",
                    "probabilities": {"move_forward": 0.3},
                    "confidence": 0.3,
                }
            }
        },
    )
    decision = agent.decide(None, "HEALTH 100")
    assert decision.action == "wait"
    assert decision.confidence == 0.3  # gating substitutes the action, not the reported confidence


def test_confidence_threshold_gating_executes_at_or_above_threshold():
    agent, fake = _make_agent(
        confidence_mode="confidence_threshold",
        confidence_threshold=0.5,
        response={
            "answers": {
                QUESTION_ID: {
                    "choice": "move_forward",
                    "probabilities": {"move_forward": 0.5},
                    "confidence": 0.5,
                }
            }
        },
    )
    decision = agent.decide(None, "HEALTH 100")
    assert decision.action == "move_forward"


def test_always_execute_ignores_low_confidence():
    agent, fake = _make_agent(
        confidence_mode="always_execute",
        response={
            "answers": {
                QUESTION_ID: {
                    "choice": "move_forward",
                    "probabilities": {"move_forward": 0.05},
                    "confidence": 0.05,
                }
            }
        },
    )
    decision = agent.decide(None, "HEALTH 100")
    assert decision.action == "move_forward"


def test_reasoning_is_a_real_rendering_of_the_returned_probabilities_sorted_descending():
    agent, fake = _make_agent(
        response={
            "answers": {
                QUESTION_ID: {
                    "choice": "move_forward",
                    "probabilities": {"turn_left": 0.1, "move_forward": 0.7, "turn_right": 0.2},
                    "confidence": 0.7,
                }
            }
        }
    )
    decision = agent.decide(None, "HEALTH 100")
    assert decision.reasoning == "move_forward:0.700 turn_right:0.200 turn_left:0.100 (confidence=0.700)"


def test_reasoning_is_none_when_no_probabilities_returned():
    agent, fake = _make_agent(response={"answers": {QUESTION_ID: {"choice": "wait", "probabilities": {}, "confidence": None}}})
    decision = agent.decide(None, "HEALTH 100")
    assert decision.reasoning is None


def test_raw_response_is_preserved_on_the_decision():
    response = {
        "answers": {QUESTION_ID: {"choice": "move_forward", "probabilities": {"move_forward": 1.0}, "confidence": 1.0}}
    }
    agent, fake = _make_agent(response=response)
    decision = agent.decide(None, "HEALTH 100")
    assert decision.raw == response


def test_reset_is_a_no_op_and_does_not_raise():
    agent, fake = _make_agent()
    assert agent.reset() is None
    # Calling decide() again afterward still works — no state to have broken.
    decision = agent.decide(None, "HEALTH 100")
    assert decision.action == "move_forward"
