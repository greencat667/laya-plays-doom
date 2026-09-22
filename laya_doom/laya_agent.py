"""Wraps a laya.Agent bound to one action set (see actions.py).

Architectural note: Laya is not an autoregressive tool-calling model.
``laya.Agent.predict(state, questions)`` runs ONE non-autoregressive
forward pass over a `choice` question (a fixed label set with a short
"criteria" description per label) and returns per-label probabilities plus
a calibrated confidence for the top label — there is no decode loop, no
free-text generation, and no trigger/schema-compilation mechanism to
design around. Laya's own top-level API (``dir(laya)`` on the installed
0.1.6 package) has no ``reset`` at all: ``laya.Agent`` only exposes
``predict`` and ``system_one``. That's not an oversight this wrapper
works around — Laya has no persistent internal state between
``predict()`` calls, so there's nothing that could go stale or need
periodic clearing: each call is an independent forward pass. This was
verified empirically on this machine, not just inferred from the API
shape: 200 consecutive ``predict()`` calls with the byte-identical
repetitive input showed no latency trend at all (first-half mean
21.58ms, second-half mean 21.54ms; max single call 25.56ms, no outlier
blowup). ``reset()`` is therefore a documented no-op below, not a
missing feature.

No GOAL/system-prompt mechanism either: ``predict()`` takes only ``state``
(str or dict) and ``questions`` (typed dict) — there's no separate
facts-only ``system=`` parameter to consider using. The brief's priority
line is injected into the state text itself (see state_encoder.GOAL_LINE),
since that is the only text Laya actually reads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import laya

from .actions import ActionSet, get_action_names
from .state_encoder import EncoderConfig, StateEncoder

ConfidenceMode = Literal["always_execute", "confidence_threshold"]

QUESTION_ID = "action"

# Criteria text per canonical action name (see actions.get_action_names) —
# kept short, one clause per condition, matching the vocabulary style
# Laya's `criteria` dict expects (label -> short description).
#
# This is CRITERIA_V1 from the project's own before/after test (see
# README "How the decision engine works" -> base-checkpoint behaviour). A second,
# far-more-literal wording pass (CRITERIA_V2 in that test) was tried and
# did NOT clearly improve behaviour on the untuned base checkpoint — it
# traded one bias (almost never shooting) for a different one (shooting
# even when the explicit AMMO>0 precondition it was told to check for
# wasn't met) while confidence stayed just as low. A third pass
# (CRITERIA_V4 in scripts/probe_criteria.py, terse/symmetric one-clause
# wording) looked dramatic in isolation — shoot won 8/8 hand-built states,
# confidence 0.37-0.54 — but was worse than useless: it fired on
# open_path_no_enemy (no enemy at all) and wall_ahead_near too. That's not
# discrimination, it's a keyword-bait bias toward the most concrete-sounding
# label regardless of state. None of these wording passes are shipped —
# see build_criteria/the shoot-gate design below for what actually is.
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

# The canonical "combat" action per action set — what the shoot gate below
# controls. stage1 calls it "shoot", full calls the same button "attack";
# both map to the same ATTACK button in actions.py.
_COMBAT_ACTION: dict[ActionSet, str] = {"stage1": "shoot", "full": "attack"}

# Laya's own separate documented question primitive for exactly this kind
# of independent binary judgment (a calibrated P(true), not a competing
# label in a multi-way softmax) — reaching for the vendor's own documented
# tool for the specific failure, not an invented workaround.
#
# Real measured motivation (scripts/probe_criteria.py, 8 hand-built
# states, real model): with `shoot`/`attack` competing inside the normal
# `choice` question alongside move/turn, it never won even once (0/8,
# confidence 0.03-0.18 — see README). Pulled out into its own `noul`
# question ("should_shoot"), the SAME model produced a real, monotonic-ish
# signal instead of noise:
#
#   enemy_front_near   -> P=0.682   (correct: shoot)
#   enemy_front_veryn  -> P=0.482   (correct: shoot)
#   enemy_front_left   -> P=0.460   (imperfect: not exactly bearing front,
#                                     but still fires above threshold)
#   enemy_right        -> P=0.427   (correct: below threshold, no shoot)
#   low_health_enemy_far -> P=0.422 (correct: enemy too far, no shoot)
#   open_path_no_enemy -> P=0.378   (correct: no enemy, no shoot)
#   wall_ahead_near    -> P=0.366   (correct: no enemy, no shoot)
#   no_ammo_enemy_front -> P=0.512  (WRONG on the gate alone — the model
#                                     does not reliably respect the AMMO>0
#                                     precondition in its own instructions;
#                                     caught by the deterministic ammo
#                                     guard in decide() below, not by the
#                                     model — logged, not hidden)
#
# _SHOOT_GATE_THRESHOLD=0.45 was picked directly from that real spread (it
# separates the three enemy-ahead states from the four non-combat states),
# not tuned against a held-out set — treat it as a first cut backed by 8
# data points, not a calibrated cutoff, and re-check it if the action
# distribution in a real episode run looks wrong (see README).
#
# Real finding from wiring this into an actual episode (--scenario basic,
# default --memory prev_state): the gate UNDER-fired on the exact case
# that matters (an enemy at bearing "front", medium range — real logged
# P=0.305, below threshold) while OVER-firing on off-center enemies
# (front-left/front-right — 120/120 real shots in a 3-episode run
# coincided with an ENEMY line, but 100% of them were at bearing
# front-left/right, 0% at exact "front", netting 0 damage/kills — worse
# than useless tactically, even though "does it ever shoot at all" looks
# fixed). Root cause, isolated by comparing against the exact same episode
# with --memory stateless (which produced real kills: mean_kills 0.67,
# mean_damage_given 5.0 over 3 episodes — see README): prev_state mode
# adds LAST_ACTION/LAST_RESULT/HEALTH_CHANGE/AMMO_CHANGE/
# VISIBLE_ENEMIES_CHANGE/MOVED lines the 8-state calibration text never
# had, and those extra tokens measurably shift the gate's probability
# away from the calibrated threshold. Fix: the gate always encodes its
# OWN stateless-shaped view of the current perception (see
# ``self._gate_encoder`` below), independent of whatever --memory mode
# the caller configured for the movement choice/logging — decoupling
# "what the gate was calibrated against" from "what the experimenter is
# varying" rather than quietly overfitting the threshold to one memory
# mode.
_SHOOT_GATE_INSTRUCTIONS = (
    "ENEMY reported at bearing front and AMMO greater than 0: should the player shoot?"
)
_SHOOT_GATE_THRESHOLD = 0.45


def build_criteria(action_set: ActionSet) -> dict[str, str]:
    """Build the label -> criteria-description dict for a ``choice``
    question, covering exactly the canonical action names for this action
    set (see actions.get_action_names)."""
    return {name: _ACTION_CRITERIA[name] for name in get_action_names(action_set)}


def build_movement_criteria(action_set: ActionSet) -> dict[str, str]:
    """Same as build_criteria, minus the combat action — the shoot gate
    decides that one separately (see _SHOOT_GATE_INSTRUCTIONS above), so
    it's excluded from the movement `choice` question's competing labels
    rather than left in to compete and lose again."""
    combat = _COMBAT_ACTION[action_set]
    return {name: desc for name, desc in build_criteria(action_set).items() if name != combat}


def _render_reasoning(probabilities: dict, confidence: float | None) -> str | None:
    """Laya doesn't generate free-text reasoning (no decode loop at all —
    see this module's docstring), so this is never invented text: it's a
    compact, honest rendering of the REAL returned probability
    distribution, sorted highest first, plus the calibrated confidence
    Laya itself reported for the top label."""
    if not probabilities:
        return None
    ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
    pairs = " ".join(f"{name}:{prob:.3f}" for name, prob in ranked)
    conf_str = f"{confidence:.3f}" if confidence is not None else "n/a"
    return f"{pairs} (confidence={conf_str})"


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
        # wording pass. A higher default (e.g. 0.6, a typical execute-band
        # threshold for a calibrated tool-calling model) would gate almost
        # every decision to "wait" here. Kept
        # low by default so --confidence-mode confidence_threshold is at
        # least usable out of the box; see the README before assuming this
        # threshold means the same thing it does for a calibrated
        # tool-calling model.
        confidence_threshold: float = 0.15,
        use_shoot_gate: bool = True,
        shoot_gate_threshold: float = _SHOOT_GATE_THRESHOLD,
    ):
        self.action_set = action_set
        self.confidence_mode = confidence_mode
        self.confidence_threshold = confidence_threshold
        # See the shoot-gate comment block above build_criteria(): shoot/
        # attack never won a fair fight inside the normal multi-way
        # `choice` question (0/8 on hand-built states), so by default it's
        # pulled out into its own `noul` "should_shoot" question and the
        # movement `choice` question no longer offers it as a competing
        # label at all. use_shoot_gate=False restores the original
        # single-choice design (all labels, including combat, in one
        # `choice` call) for comparison/regression checks.
        self.use_shoot_gate = use_shoot_gate
        self.shoot_gate_threshold = shoot_gate_threshold
        self._combat_action = _COMBAT_ACTION[action_set]
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
        self._movement_criteria = build_movement_criteria(action_set)
        self._movement_questions = {
            QUESTION_ID: {
                "type": "choice",
                "instructions": (
                    "Given the current Doom game world-state text below, "
                    "which action should the player take right now?"
                ),
                "criteria": self._movement_criteria,
            }
        }
        self._gate_questions = {"should_shoot": {"type": "noul", "instructions": _SHOOT_GATE_INSTRUCTIONS}}
        # Deliberately its own StateEncoder, always in "stateless" mode —
        # see the long comment above _SHOOT_GATE_THRESHOLD for the real
        # episode data that motivated this: the gate's calibration doesn't
        # transfer to text that also carries LAST_ACTION/LAST_RESULT/*_CHANGE
        # lines, so it never sees that text, regardless of what --memory
        # mode the caller is experimenting with for movement/logging.
        self._gate_encoder = StateEncoder(EncoderConfig(memory_mode="stateless"))
        self.agent = laya.load(model_id, device=device)

    def reset(self) -> None:
        """Mostly a no-op — see this module's docstring for why laya.Agent
        itself (0.1.6) has no reset method and no persistent state to
        clear between predict() calls. The one real piece of state this
        resets is ``self._gate_encoder``'s AREA visited-cells tracking, so
        the gate's per-episode "have I been here before" bookkeeping
        doesn't leak across episodes — run_episode() calls agent.reset()
        once per episode (see controller.py), same as every other
        controller here."""
        self._gate_encoder.reset()

    def decide(self, perception, encoded_state: str) -> Decision:
        if not self.use_shoot_gate:
            return self._decide_single_choice(encoded_state)
        return self._decide_with_shoot_gate(perception, encoded_state)

    def _decide_single_choice(self, encoded_state: str) -> Decision:
        """The original design: one `choice` call over every label,
        including combat. Kept for --no-shoot-gate comparison/regression
        runs — see README for why this is not the default."""
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

        reasoning = _render_reasoning(probabilities, confidence)
        return Decision(action=action, confidence=confidence, latency_ms=latency_ms, reasoning=reasoning, raw=response)

    def _decide_with_shoot_gate(self, perception, encoded_state: str) -> Decision:
        t0 = time.perf_counter()
        gate_state = self._gate_encoder.encode(perception, None, None, None)
        gate_response = self.agent.predict(gate_state, self._gate_questions)
        gate_answer = (gate_response.get("answers") or {}).get("should_shoot") or {}
        shoot_prob = gate_answer.get("noul")

        want_shoot = shoot_prob is not None and shoot_prob >= self.shoot_gate_threshold
        # Two deterministic guards, layered on top of the model's own
        # judgment rather than trusting it — the same pattern as this
        # project's other safety nets (StuckRecoveryConfig/
        # ThreatResponseConfig: a transparent, always-logged override,
        # not a hidden one).
        #
        # AMMO>0: the gate alone scored a no-ammo state at P=0.512 (above
        # threshold) despite its own instructions explicitly requiring
        # AMMO>0 — the model doesn't reliably enforce that precondition,
        # so code does.
        #
        # bearing==front: found by watching a real episode, not an
        # isolated test. 120 real shoot decisions in one run all
        # coincided with a real ENEMY line (never firing blind), but 100%
        # of them were at bearing front-left/front-right and 0% at exact
        # front — 0 damage given across the run. Bearing is already a
        # deterministic fact in Perception (perception.py's depth-scan
        # classification), not something that needs a probabilistic
        # judgment at all, so it's checked directly instead of trusting
        # the gate to have implicitly learned "front" from the instructions
        # text (it clearly hasn't — see the README's shoot-gate section).
        ammo_ok = perception.ammo > 0
        enemy_dead_ahead = any(e.bearing == "front" for e in perception.enemies)
        guard_blocked = want_shoot and not (ammo_ok and enemy_dead_ahead)

        shoot_prob_str = f"{shoot_prob:.3f}" if shoot_prob is not None else "n/a"

        if want_shoot and not guard_blocked:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            action = self._combat_action
            confidence = shoot_prob
            reasoning = f"should_shoot={shoot_prob_str} (>= {self.shoot_gate_threshold}) -> {action}"
            raw = {"gate": gate_response}
        else:
            move_response = self.agent.predict(encoded_state, self._movement_questions)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            move_answer = (move_response.get("answers") or {}).get(QUESTION_ID) or {}
            action = move_answer.get("choice")
            confidence = move_answer.get("confidence")
            probabilities = move_answer.get("probabilities") or {}

            if action is None or action not in self._movement_criteria:
                action = "wait"

            if self.confidence_mode == "confidence_threshold":
                if confidence is None or confidence < self.confidence_threshold:
                    action = "wait"

            reasoning = f"should_shoot={shoot_prob_str}"
            if guard_blocked:
                blocked_by = []
                if not ammo_ok:
                    blocked_by.append("AMMO<=0")
                if not enemy_dead_ahead:
                    blocked_by.append("no enemy at bearing==front")
                reasoning += f"  [guard blocked shoot: {', '.join(blocked_by)}]"
            move_reasoning = _render_reasoning(probabilities, confidence)
            if move_reasoning:
                reasoning += "  " + move_reasoning
            raw = {"gate": gate_response, "movement": move_response}

        return Decision(action=action, confidence=confidence, latency_ms=latency_ms, reasoning=reasoning, raw=raw)
