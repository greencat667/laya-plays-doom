"""Wraps a laya.Agent bound to one action set (see actions.py).

Architectural note — the biggest real difference from the sibling Needle
project, not an assumption: Laya is not an autoregressive tool-calling
model. ``laya.Agent.predict(state, questions)`` runs ONE non-autoregressive
forward pass over a `choice` question (a fixed label set with a short
"criteria" description per label) and returns per-label probabilities plus
a calibrated confidence for the top label — there is no decode loop, no
free-text generation, and no ``@needle.tool``/trigger/schema-compilation
mechanism to design around. Laya's own top-level API
(``dir(laya)`` on the installed 0.1.6 package) has no ``reset`` at all:
``laya.Agent`` only exposes ``predict`` and ``system_one``. That's not an
oversight this wrapper works around — the sibling Needle project needed
``NeedleAgent.reset()``/``reset_every`` specifically because a long streak
of ``complete()`` calls without resetting could make a *later* autoregressive
decode pathologically slow on repetitive input (confirmed there: 38 calls in,
57s+ and climbing). Laya has no persistent internal state between
``predict()`` calls for that staleness to accumulate in — each call is an
independent forward pass. This was verified empirically on this machine,
not just inferred from the API shape: 200 consecutive ``predict()`` calls
with the byte-identical repetitive input showed no latency trend at all
(first-half mean 21.58ms, second-half mean 21.54ms; max single call 25.56ms,
no outlier blowup) — see the project README's "Laya vs Needle" section for
the full numbers. ``reset()`` is therefore a documented no-op below, not a
missing feature.

No GOAL/system-prompt mechanism either: ``predict()`` takes only ``state``
(str or dict) and ``questions`` (typed dict) — there's nothing resembling
Needle's facts-only ``system=`` parameter to even consider using. The
brief's priority line is injected into the state text itself, exactly as
in the sibling project (see state_encoder.GOAL_LINE), since that is the
only text Laya actually reads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import laya

from .actions import ActionSet, get_action_names

ConfidenceMode = Literal["always_execute", "confidence_threshold"]

QUESTION_ID = "action"

# Criteria text per canonical action name (see actions.get_action_names),
# adapted from the docstrings the sibling Needle project wrote for its
# @needle.tool functions — kept short, one clause per condition, since
# that's the vocabulary style Laya's `criteria` dict expects (label ->
# short description), not a docstring for a compiled function signature.
#
# This is CRITERIA_V1 from the project's own before/after test (see
# README "Laya vs Needle" -> base-checkpoint behaviour). A second,
# far-more-literal wording pass (CRITERIA_V2 in that test) was tried and
# did NOT clearly improve behaviour on the untuned base checkpoint — it
# traded one bias (almost never shooting) for a different one (shooting
# even when the explicit AMMO>0 precondition it was told to check for
# wasn't met) while confidence stayed just as low. V1's wording is kept as
# the shipped default because there was no honest basis to prefer V2's
# result over V1's; see the README for the full before/after data rather
# than re-tuning further to chase a better-looking number.
_ACTION_CRITERIA: dict[str, str] = {
    "move_forward": (
        "the path ahead isn't blocked and no enemy is lined up to shoot — "
        "the default action for making progress and exploring"
    ),
    "move_backward": "retreat — move backward without turning, e.g. when overwhelmed at close range",
    "strafe_left": "sidestep left without turning, to dodge or reposition without losing aim",
    "strafe_right": "sidestep right without turning, to dodge or reposition without losing aim",
    "turn_left": "the path ahead is blocked or a wall is near, or to face an enemy that isn't directly ahead, turning left",
    "turn_right": "the path ahead is blocked or a wall is near, or to face an enemy that isn't directly ahead, turning right",
    "turn_left_small": "a small turn left to fine-aim at an enemy that isn't directly ahead",
    "turn_right_small": "a small turn right to fine-aim at an enemy that isn't directly ahead",
    "turn_left_large": "a large turn left, e.g. when the path ahead is blocked or a wall is near, to scan for a new direction",
    "turn_right_large": "a large turn right, e.g. when the path ahead is blocked or a wall is near, to scan for a new direction",
    "attack": "an ENEMY is reported with bearing front — lined up to hit",
    "shoot": "an ENEMY is reported with bearing front — lined up to hit",
    "use": "interact with a door or switch directly ahead",
    "wait": "nothing else applies — hold position",
}


def build_criteria(action_set: ActionSet) -> dict[str, str]:
    """Build the label -> criteria-description dict for a ``choice``
    question, covering exactly the canonical action names for this action
    set (see actions.get_action_names)."""
    return {name: _ACTION_CRITERIA[name] for name in get_action_names(action_set)}


@dataclass
class Decision:
    action: str
    confidence: float | None
    latency_ms: float
    reasoning: str | None
    raw: dict


class LayaAgent:
    def __init__(
        self,
        action_set: ActionSet = "stage1",
        model_id: str = "convaiinnovations/laya",
        device: str | None = None,
        confidence_mode: ConfidenceMode = "always_execute",
        # Laya's own confidence values, measured on this out-of-domain base
        # checkpoint (fine-tuned for routing/moderation/triage, not Doom —
        # see README), ran far lower than a tool-calling model's execute-
        # band intuition would suggest: 0.02-0.19 across a spread of
        # characteristic Doom states, both before and after the criteria
        # wording pass. A default of 0.6 (the sibling project's Needle
        # default) would gate almost every decision to "wait" here. Kept
        # low by default so --confidence-mode confidence_threshold is at
        # least usable out of the box; see the README before assuming this
        # threshold means the same thing it does for a calibrated
        # tool-calling model.
        confidence_threshold: float = 0.15,
    ):
        self.action_set = action_set
        self.confidence_mode = confidence_mode
        self.confidence_threshold = confidence_threshold
        self._criteria = build_criteria(action_set)
        self._questions = {
            QUESTION_ID: {
                "type": "choice",
                "instructions": (
                    "Given the current Doom game world-state text below, "
                    "which action should the player take right now?"
                ),
                "criteria": self._criteria,
            }
        }
        self.agent = laya.load(model_id, device=device)

    def reset(self) -> None:
        """No-op — see this module's docstring. laya.Agent (0.1.6) has no
        reset method at all, and there's no persistent conversation state
        between predict() calls for one to clear: each call is an
        independent single forward pass. Kept as a real method (not
        omitted) purely to satisfy the same Agent protocol run_episode()
        uses for every controller (see controller.py), so LayaAgent is a
        drop-in the same way RandomAgent/HeuristicAgent/NeedleAgent are."""
        return None

    def decide(self, perception, encoded_state: str) -> Decision:
        t0 = time.perf_counter()
        response = self.agent.predict(encoded_state, self._questions)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = (response.get("answers") or {}).get(QUESTION_ID) or {}
        action = answer.get("choice")
        confidence = answer.get("confidence")
        probabilities = answer.get("probabilities") or {}

        if action is None or action not in self._criteria:
            action = "wait"

        if self.confidence_mode == "confidence_threshold":
            if confidence is None or confidence < self.confidence_threshold:
                action = "wait"

        # Laya doesn't generate free-text reasoning (no decode loop at
        # all — see this module's docstring), so this is never invented
        # text: it's a compact, honest rendering of the REAL returned
        # probability distribution, sorted highest first, plus the
        # calibrated confidence Laya itself reported for the top label.
        reasoning = None
        if probabilities:
            ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
            pairs = " ".join(f"{name}:{prob:.3f}" for name, prob in ranked)
            conf_str = f"{confidence:.3f}" if confidence is not None else "n/a"
            reasoning = f"{pairs} (confidence={conf_str})"

        return Decision(
            action=action,
            confidence=confidence,
            latency_ms=latency_ms,
            reasoning=reasoning,
            raw=response,
        )
