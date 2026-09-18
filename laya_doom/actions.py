"""Semantic actions: the vocabulary Laya (and the baseline agents) choose
from, and the translation from a chosen name into ViZDoom button presses.

Two action sets are provided:

- ``stage1``: the four-action proof-of-concept from the project brief
  (move_forward, turn left/right, shoot).
- ``full``: the complete action list (forward/backward, strafing, small/large
  turns, attack, use) plus the controller-level ``wait`` fallback.

Unlike the sibling Needle project, there is no tool-schema/trigger
machinery here at all: Laya's ``predict()`` picks among a fixed set of
``choice`` labels in one forward pass (see laya_agent.py) rather than
compiling Python function signatures into a decode grammar, so this module
is intentionally smaller than needle_doom's actions.py — it keeps only the
canonical action-name vocabulary, the ActionSpec/button/tic tables, and
``get_action_names``. The mapping from Laya's chosen label straight to a
canonical action name (no argument-based resolution, since every action is
already its own label) lives in laya_agent.py, not here.

Every semantic action maps to a fixed, configurable number of Doom tics
(``DEFAULT_TICS``), overridable uniformly for the decision-rate experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import vizdoom as vzd

ActionSet = Literal["stage1", "full"]

# Fixed order used to build the button vector passed to DoomGame.make_action().
ALL_BUTTONS: tuple[vzd.Button, ...] = (
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK,
    vzd.Button.USE,
)


@dataclass(frozen=True)
class ActionSpec:
    name: str
    pressed: frozenset[vzd.Button]
    tics: int

    def button_vector(self) -> list[int]:
        return [1 if button in self.pressed else 0 for button in ALL_BUTTONS]


# Canonical semantic actions. "shoot" is a stage1-vocabulary alias for
# "attack" with the same buttons/tics; kept as a separate entry so
# LAST_ACTION in the encoded state always shows the name Laya actually
# chose rather than a silently-remapped one.
DEFAULT_TICS: dict[str, int] = {
    "move_forward": 4,
    "move_backward": 4,
    "strafe_left": 4,
    "strafe_right": 4,
    "turn_left": 3,
    "turn_right": 3,
    "turn_left_small": 2,
    "turn_right_small": 2,
    "turn_left_large": 6,
    "turn_right_large": 6,
    "attack": 1,
    "shoot": 1,
    "use": 2,
    "wait": 4,
}

_PRESSED: dict[str, frozenset[vzd.Button]] = {
    "move_forward": frozenset({vzd.Button.MOVE_FORWARD}),
    "move_backward": frozenset({vzd.Button.MOVE_BACKWARD}),
    "strafe_left": frozenset({vzd.Button.MOVE_LEFT}),
    "strafe_right": frozenset({vzd.Button.MOVE_RIGHT}),
    "turn_left": frozenset({vzd.Button.TURN_LEFT}),
    "turn_right": frozenset({vzd.Button.TURN_RIGHT}),
    "turn_left_small": frozenset({vzd.Button.TURN_LEFT}),
    "turn_right_small": frozenset({vzd.Button.TURN_RIGHT}),
    "turn_left_large": frozenset({vzd.Button.TURN_LEFT}),
    "turn_right_large": frozenset({vzd.Button.TURN_RIGHT}),
    "attack": frozenset({vzd.Button.ATTACK}),
    "shoot": frozenset({vzd.Button.ATTACK}),
    "use": frozenset({vzd.Button.USE}),
    "wait": frozenset(),
}

STAGE1_ACTIONS: tuple[str, ...] = ("move_forward", "turn_left", "turn_right", "shoot")
FULL_ACTIONS: tuple[str, ...] = (
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
)

# "wait" is always a legal fallback action (e.g. for confidence gating),
# independent of whether the active action set exposes it as its own label.
_ALL_CANONICAL_NAMES = set(DEFAULT_TICS) | {"wait"}


def build_action_table(uniform_tics: int | None = None) -> dict[str, ActionSpec]:
    """Build the canonical-name -> ActionSpec table.

    ``uniform_tics``, when given, overrides every action's tic count with a
    single value — used by the decision-rate experiments (1 decision every
    2/4/8/16 tics) so the control frequency can be swept independently of
    the per-action defaults tuned for normal play.
    """
    table = {}
    for name, pressed in _PRESSED.items():
        tics = uniform_tics if uniform_tics is not None else DEFAULT_TICS[name]
        table[name] = ActionSpec(name=name, pressed=pressed, tics=max(1, tics))
    return table


def get_action_names(action_set: ActionSet) -> tuple[str, ...]:
    if action_set == "stage1":
        return STAGE1_ACTIONS
    if action_set == "full":
        return FULL_ACTIONS
    raise ValueError(f"unknown action set: {action_set!r}")
