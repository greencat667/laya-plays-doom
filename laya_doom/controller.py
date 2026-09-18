"""The explicit per-control-cycle loop:

    while not game.is_episode_finished():
        state = perceive_world(game)
        encoded = encode_state(state)
        decision = agent.decide(state, encoded)
        env.execute(decision.action)
        record_metrics()

``agent`` is anything with a ``decide(perception, encoded_state) -> Decision``
and a ``reset()`` method — LayaAgent (laya_agent.py) and the baseline
controllers in experiments/ all satisfy this, so all three run through
exactly this same function with exactly the same perception/encoding
pipeline, which is what makes them comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

from .doom_env import DoomEnv
from .laya_agent import Decision
from .perception import Perception, PerceptionConfig, perceive
from .state_encoder import StateEncoder, summarize_result


class Agent(Protocol):
    def decide(self, perception: Perception, encoded_state: str) -> Decision: ...

    def reset(self) -> None: ...


_MOVE_ACTION_NAMES = frozenset({"move_forward", "move_backward", "strafe_left", "strafe_right"})
_TURN_ACTION_NAMES = frozenset(
    {"turn_left", "turn_right", "turn_left_small", "turn_right_small", "turn_left_large", "turn_right_large"}
)


@dataclass
class StuckRecoveryConfig:
    """If the agent tries to *move* but doesn't actually make progress for
    ``no_progress_threshold`` consecutive steps (walking into a wall), force
    a turn toward whichever side has more open room instead of letting it
    push forever. Never overrides a decision that wasn't already trying to
    move (turning/attacking/using/waiting are left alone).

    This is a practical safety net for watching the agent play, layered on
    top of — not hidden inside — Laya's own decision: the overridden step
    is logged with ``overridden=True`` and ``proposed_action`` set to what
    Laya actually chose, so raw model behaviour is still fully visible in
    the logs. Set ``enabled=False`` (or pass ``--no-stuck-recovery`` on the
    CLI) to measure Laya completely unassisted.
    """

    enabled: bool = True
    no_progress_threshold: int = 2
    min_progress_units: float = 3.0


@dataclass
class ThreatResponseConfig:
    """When the previous step's result was "took_damage" and no enemy is
    currently visible (the THREAT line in state_encoder.py — an attacker
    outside perception.py's FOV), turn instead of letting Laya's decision
    stand if it wasn't already a turn.

    Ported over from the sibling Needle project, where this safety net was
    added because the textual THREAT signal alone (there, a `turn` tool
    trigger) wasn't enough — tested directly against a real near-death
    state (HEALTH 3, just took -15 damage, no visible enemy), that model
    still chose move_forward with the THREAT line present. Laya has no
    trigger mechanism at all (see actions.py/laya_agent.py — Laya picks
    among fixed choice labels via one forward pass, nothing regex-based),
    so this specific failure mode hasn't been independently re-verified
    here; kept on by default as the same logged-not-hidden precaution
    either way. Overridden steps set `overridden=True` and keep
    `proposed_action` as what Laya actually picked. `--no-threat-response`
    disables it.
    """

    enabled: bool = True


@dataclass
class StepRecord:
    step: int
    tic: int
    x: float
    y: float
    encoded_state: str
    action: str
    keys_held: tuple[str, ...] = ()
    proposed_action: str = ""
    overridden: bool = False
    tool_calls: list = field(default_factory=list)
    confidence: float | None = None
    latency_ms: float = 0.0
    reasoning: str | None = None
    result: str = ""
    health: int = 0
    armor: int = 0
    ammo: int = 0
    reward: float = 0.0


@dataclass
class EpisodeResult:
    episode: int
    steps: int
    survival_tics: int
    distance_travelled: float
    kills: int
    damage_given: int
    damage_taken: int
    health_remaining: int
    ammo_used: int
    items_collected: int
    total_reward: float
    died: bool
    completed: bool = False
    action_counts: dict[str, int] = field(default_factory=dict)
    mean_confidence: float | None = None
    mean_latency_ms: float = 0.0


def _infer_picked_up_key(before: Perception, after: Perception | None) -> str | None:
    """ViZDoom doesn't expose a key-inventory game variable, so this infers
    which key (if any) was just picked up: itemcount went up, and the
    nearest thing in `before.pickups` was a "*_key" kind at point-blank
    range. Heuristic, not a ground-truth read — good enough to surface a
    KEYS hint in the world-state text, not to gate any door logic on."""
    if after is None or after.itemcount <= before.itemcount or not before.pickups:
        return None
    nearest = before.pickups[0]
    if nearest.kind.endswith("_key") and nearest.distance == "very-near":
        return nearest.kind
    return None


def _should_override_for_threat(
    threat_response: ThreatResponseConfig, last_result: str | None, perception: Perception, proposed_action: str
) -> bool:
    return (
        threat_response.enabled
        and last_result == "took_damage"
        and not perception.enemies
        and proposed_action not in _TURN_ACTION_NAMES
    )


def _recovery_turn(perception: Perception, action_set: str, step: int) -> str:
    if perception.open_left and not perception.open_right:
        direction = "left"
    elif perception.open_right and not perception.open_left:
        direction = "right"
    else:
        direction = "left" if step % 2 == 0 else "right"
    return f"turn_{direction}" if action_set == "stage1" else f"turn_{direction}_small"


def run_episode(
    env: DoomEnv,
    agent: Agent,
    encoder: StateEncoder,
    perception_config: PerceptionConfig | None = None,
    max_steps: int = 500,
    episode_index: int = 0,
    on_step: Callable[[Perception, Decision, StepRecord], None] | None = None,
    stuck_recovery: StuckRecoveryConfig | None = None,
    threat_response: ThreatResponseConfig | None = None,
) -> tuple[EpisodeResult, list[StepRecord]]:
    perception_config = perception_config or PerceptionConfig()
    stuck_recovery = stuck_recovery if stuck_recovery is not None else StuckRecoveryConfig()
    threat_response = threat_response if threat_response is not None else ThreatResponseConfig()

    env.new_episode()
    agent.reset()
    encoder.reset()

    prev_perception: Perception | None = None
    last_action: str | None = None
    last_result: str | None = None
    starting_ammo: int | None = None
    no_progress_steps = 0
    keys_held: set[str] = set()

    records: list[StepRecord] = []
    action_counts: dict[str, int] = {}
    step = 0

    while not env.is_finished() and step < max_steps:
        state = env.get_state()
        perception = perceive(state, perception_config)
        if perception is None:
            break
        if starting_ammo is None:
            starting_ammo = perception.ammo

        encoded = encoder.encode(perception, prev_perception, last_action, last_result, frozenset(keys_held))
        decision = agent.decide(perception, encoded)

        overridden = False
        final_action = decision.action
        if _should_override_for_threat(threat_response, last_result, perception, decision.action):
            final_action = _recovery_turn(perception, env.config.action_set, step)
            overridden = True
        elif (
            stuck_recovery.enabled
            and no_progress_steps >= stuck_recovery.no_progress_threshold
            and decision.action in _MOVE_ACTION_NAMES
        ):
            final_action = _recovery_turn(perception, env.config.action_set, step)
            overridden = True

        reward = env.execute(final_action)

        new_state = env.get_state() if not env.is_finished() else None
        new_perception = perceive(new_state, perception_config) if new_state is not None else None
        result = summarize_result(perception, new_perception)
        encoder.record(final_action, result)

        picked_up_key = _infer_picked_up_key(perception, new_perception)
        if picked_up_key is not None:
            keys_held.add(picked_up_key)

        if final_action in _MOVE_ACTION_NAMES and new_perception is not None:
            progressed = (
                (new_perception.x - perception.x) ** 2 + (new_perception.y - perception.y) ** 2
            ) ** 0.5 >= stuck_recovery.min_progress_units
            no_progress_steps = 0 if progressed else no_progress_steps + 1
        else:
            no_progress_steps = 0

        record = StepRecord(
            step=step,
            tic=perception.tic,
            x=perception.x,
            y=perception.y,
            encoded_state=encoded,
            action=final_action,
            keys_held=tuple(sorted(keys_held)),
            proposed_action=decision.action,
            overridden=overridden,
            tool_calls=list((decision.raw or {}).get("function_calls") or []),
            confidence=decision.confidence,
            latency_ms=decision.latency_ms,
            reasoning=decision.reasoning,
            result=result,
            health=perception.health,
            armor=perception.armor,
            ammo=perception.ammo,
            reward=reward,
        )
        records.append(record)
        action_counts[final_action] = action_counts.get(final_action, 0) + 1

        if on_step is not None:
            on_step(perception, replace(decision, action=final_action), record)

        prev_perception = perception
        last_action = final_action
        last_result = result
        step += 1

    distance = 0.0
    for prev, cur in zip(records, records[1:]):
        distance += ((cur.x - prev.x) ** 2 + (cur.y - prev.y) ** 2) ** 0.5

    final = prev_perception
    confidences = [r.confidence for r in records if r.confidence is not None]
    latencies = [r.latency_ms for r in records]

    # ViZDoom has no "you won" game variable (checked the full GameVariable
    # enum — there isn't one), so `completed` is inferred: the episode ended
    # on its own (a scenario's built-in win condition — an exit trigger, a
    # killed target, a grabbed vest — not our own --max-steps budget running
    # out) and the player didn't die. A margin below the configured timeout
    # guards against a coincidental last-tic timeout looking like a win.
    died = env.is_player_dead()
    timeout_tics = env.episode_timeout_tics()
    completed = (
        not died
        and env.is_finished()
        and final is not None
        and final.tic < max(timeout_tics - 5, 0)
    )

    result_obj = EpisodeResult(
        episode=episode_index,
        steps=step,
        survival_tics=final.tic if final else 0,
        distance_travelled=distance,
        kills=final.killcount if final else 0,
        damage_given=final.damagecount if final else 0,
        damage_taken=final.damage_taken if final else 0,
        health_remaining=final.health if final else 0,
        ammo_used=max(0, (starting_ammo or 0) - (final.ammo if final else 0)),
        items_collected=final.itemcount if final else 0,
        total_reward=env.total_reward(),
        died=died,
        completed=completed,
        action_counts=action_counts,
        mean_confidence=sum(confidences) / len(confidences) if confidences else None,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
    )
    return result_obj, records
