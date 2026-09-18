"""Baseline controller: a small hand-written rule, using the same
Perception object Laya's world-state text is derived from —

    if enemy front and ammo: shoot
    elif enemy right: turn right
    elif enemy left: turn left
    elif pickup visible: turn/move toward it
    else: move forward

(plus a minimal wall-avoidance fallback when nothing else applies, since
without it this baseline just walks into walls forever in any scenario
bigger than a single room). This is the bar LayaAgent has to clear to be
worth its latency/compute cost.

The pickup-seeking branch is grounded in the ViZDoom AI Competition
literature, not a guess: per "ViZDoom Competitions: Playing Doom from
Pixels" (Wydmuch, Kempka, Jaśkowski — https://ar5iv.labs.arxiv.org/html/1809.03470),
the actual Track 1 winner (Marvin) won primarily by prioritizing resource
gathering (medkits/armor) over pure combat aggression — this baseline had
no equivalent behavior at all before (it only ever reacted to enemies).
That paper is also worth reading for what it says top agents from
Facebook/Intel/CMU-era research *couldn't* do even with full RL training:
they "circled the same location," didn't chase targets, and made "zero
sophisticated evasion" attempts against incoming attacks — several of the
same failure modes seen with small models in the sibling Needle project
aren't unique to a tool-calling model either, they're known-hard in this
exact environment.
"""

from __future__ import annotations

import time

from laya_doom.actions import ActionSet
from laya_doom.laya_agent import Decision
from laya_doom.perception import Perception


class HeuristicAgent:
    def __init__(self, action_set: ActionSet = "stage1"):
        self.action_set = action_set

    def reset(self) -> None:
        pass

    def decide(self, perception: Perception, encoded_state: str) -> Decision:
        t0 = time.perf_counter()
        action = self._choose(perception)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return Decision(action=action, confidence=1.0, latency_ms=latency_ms, reasoning="heuristic rule", raw={})

    def _choose(self, perception: Perception) -> str:
        shoot = "shoot" if self.action_set == "stage1" else "attack"
        turn_left = "turn_left" if self.action_set == "stage1" else "turn_left_small"
        turn_right = "turn_right" if self.action_set == "stage1" else "turn_right_small"

        nearest_enemy = perception.enemies[0] if perception.enemies else None
        if nearest_enemy is not None:
            if nearest_enemy.bearing == "front" and perception.ammo > 0:
                return shoot
            if nearest_enemy.bearing in ("front-right", "right", "far-right"):
                return turn_right
            if nearest_enemy.bearing in ("front-left", "left", "far-left"):
                return turn_left

        nearest_pickup = perception.pickups[0] if perception.pickups else None
        if nearest_pickup is not None:
            if nearest_pickup.bearing == "front":
                return "move_forward"
            if nearest_pickup.bearing in ("front-right", "right", "far-right"):
                return turn_right
            if nearest_pickup.bearing in ("front-left", "left", "far-left"):
                return turn_left

        if perception.wall_near:
            return turn_right if perception.open_right else turn_left

        return "move_forward"
