"""Baseline controller: picks uniformly among the legal actions for the
active action set. Runs through the exact same perception/encoding
pipeline as LayaAgent and HeuristicAgent (see controller.run_episode) so
the three are directly comparable."""

from __future__ import annotations

import random
import time

from laya_doom.actions import ActionSet, get_action_names
from laya_doom.laya_agent import Decision
from laya_doom.perception import Perception


class RandomAgent:
    def __init__(self, action_set: ActionSet = "stage1", seed: int | None = None):
        self.action_set = action_set
        self._rng = random.Random(seed)
        self._actions = get_action_names(action_set)

    def reset(self) -> None:
        pass

    def decide(self, perception: Perception, encoded_state: str) -> Decision:
        t0 = time.perf_counter()
        action = self._rng.choice(self._actions)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return Decision(action=action, confidence=None, latency_ms=latency_ms, reasoning=None, raw={})
