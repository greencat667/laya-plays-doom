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

from . import wayfinding
from .doom_env import DoomEnv
from .laya_agent import Decision
from .perception import EnemyPercept, Perception, PerceptionConfig, perceive
from .state_encoder import StateEncoder, summarize_result


class Agent(Protocol):
    def decide(self, perception: Perception, encoded_state: str) -> Decision: ...

    def reset(self) -> None: ...


_MOVE_ACTION_NAMES = frozenset({"move_forward", "move_backward", "strafe_left", "strafe_right"})
# Real bugs found on long (3,000+ step) real --scenario level runs (see
# StuckRecoveryConfig's docstring for both): with a PICKUP visible,
# ExplorationNudgeConfig never engages (by design -- something legitimate
# to react to), and nothing else was watching for Laya proposing the SAME
# non-move, non-turn action over and over at a spot where it accomplishes
# nothing:
#   - "use" 2,339/3,000 times in one run, frozen at one position (a
#     genuine no-op there -- no door/switch).
#   - "wait" 3,919/4,000 times in a longer run: once the encoded state
#     reaches a fixed point (LAST_ACTION wait -> LAST_RESULT no_change,
#     forever, since nothing else in the state ever changes), Laya's own
#     top-probability label stays "wait" every single call -- a stable
#     trap with no mechanism to ever break out of it.
# Both "use" and "attack"/"shoot" are still never second-guessed for a
# SINGLE occurrence elsewhere in this module (a live combat/interaction
# decision), which is correct; the problem is specifically sustained
# repetition with zero real consequence. Treated the same as a stalled
# move for StuckRecoveryConfig's purposes only (not turn_loop_recovery's
# separate combat/use exclusion, which is about a different failure mode).
_STALL_ACTION_NAMES = _MOVE_ACTION_NAMES | frozenset({"use", "wait"})
_LEFT_OF_CENTER_BEARINGS = frozenset({"left", "far-left", "front-left"})
_RIGHT_OF_CENTER_BEARINGS = frozenset({"right", "far-right", "front-right"})
_TURN_ACTION_NAMES = frozenset(
    {"turn_left", "turn_right", "turn_left_small", "turn_right_small", "turn_left_large", "turn_right_large"}
)


@dataclass
class StuckRecoveryConfig:
    """If the agent tries to *move OR use* but doesn't actually make
    progress for ``no_progress_threshold`` consecutive steps (walking into
    a wall, or repeatedly pressing use against something that isn't
    actually interactive), force a turn toward whichever side has more
    open room instead of letting it repeat forever. Never overrides a
    decision that wasn't already trying to move or use (turning/attacking/
    waiting are left alone).

    This is a practical safety net for watching the agent play, layered on
    top of — not hidden inside — Laya's own decision: the overridden step
    is logged with ``overridden=True`` and ``proposed_action`` set to what
    Laya actually chose, so raw model behaviour is still fully visible in
    the logs. Set ``enabled=False`` (or pass ``--no-stuck-recovery`` on the
    CLI) to measure Laya completely unassisted.

    The no-progress streak itself (``run_episode``'s ``no_progress_steps``)
    only resets on genuine positional progress, never merely because a
    given step's final action happened not to be a move — a real
    ``--scenario level`` run showed why that distinction matters: a naive
    reset-on-non-move version let the override's own turn erase the
    streak every time it fired, so the very next move proposal got a
    fresh 2-strike allowance before overriding again, alternating
    move/forced-turn forever with zero net escape.

    A second, real "wall-hugging" freeze was found later by watching a
    fresh ``--scenario level`` run with ``override_reason`` logging added
    (see the README's "Real bug: wall-hugging" section for the exact
    step-by-step data): at one fixed position, ``PATH left``/``PATH
    right`` kept reporting the same tied value (open/open, so
    ``_recovery_turn``'s "which side has more open room" fallback had
    nothing to prefer and fell back to alternating by ``step % 2``) —
    and since the global step counter's parity flips every single call,
    it alternated ``turn_right_large``/``turn_left_large`` every step for
    79 consecutive steps, netting exactly zero rotation forever. Fixed by
    ``run_episode`` caching ONE ``_recovery_turn`` result per stuck event
    (``no_progress_steps`` streak) instead of recomputing — and therefore
    re-tie-breaking — it on every single firing; the cache only clears on
    real positional progress, so a tied fallback now commits to a single
    direction and actually accumulates rotation across repeated firings,
    the same way a real player would keep turning the same way rather
    than sawing back and forth. See TurnLoopRecoveryConfig's own docstring
    for the other half of this same real bug (why Laya's own proposal
    never being a turn also blocked the *other* escape mechanism).

    A third and fourth real bug, found on much longer (3,000+ step) real
    ``--scenario level`` runs while validating the frontier/secret-search
    work above, are the same underlying failure twice: with a ``PICKUP``
    visible, ExplorationNudgeConfig deliberately never engages (there's
    something legitimate to react to — see its own docstring), so nothing
    at all was watching for Laya proposing the SAME non-move action over
    and over at a spot where it accomplished nothing. ``use``/``attack``/
    ``shoot``/``wait`` were all excluded from every other recovery net's
    trigger in this module (a live combat/interaction/wait decision is
    normally never second-guessed), which is right for a single
    occurrence but wrong once it's repeating indefinitely with the exact
    same encoded state feeding back the exact same top-probability label
    every single call:

    - A 3,000-step run logged 2,339 real ``use`` decisions (78% of the
      episode) frozen at one position (``WALL ahead far``, both ``PATH``
      sides reported open, yet position never actually changed — a
      genuine no-op there, no door/switch).
    - A separate 4,000-step run logged 3,919 real ``wait`` decisions (98%
      of the episode) — once the encoded state hit a fixed point
      (``LAST_ACTION wait`` → ``LAST_RESULT no_change``, forever, since
      nothing else in the state ever changed either), Laya's own returned
      probabilities kept ``wait`` as the top label every time
      (``wait:0.396`` vs. the next-highest ``turn_left_small:0.141`` in
      the actual logged reasoning), with no mechanism to ever break the
      loop.

    Fixed by treating a stalled ``use`` or ``wait`` the same as a stalled
    move for this net's purposes only (see ``_STALL_ACTION_NAMES``) —
    after enough consecutive no-progress attempts at any of them, this
    now forces the same recovery turn a stalled move would get, rather
    than leaving the spam unaddressed forever.
    """

    enabled: bool = True
    no_progress_threshold: int = 2
    min_progress_units: float = 3.0


@dataclass
class TurnLoopRecoveryConfig:
    """Force one ``move_forward`` attempt after ``max_consecutive_turns``
    consecutive *executed* turn actions, regardless of what Laya itself
    proposed on that step, and regardless of whether Laya ever proposed a
    turn at all.

    Originally built (and originally checked ``decision.action``, Laya's
    own raw proposal, for being a turn) after watching a real
    ``--scenario level`` run get stuck in a corner where Laya kept
    choosing ``turn_left_small`` turn after turn on its own. That
    original condition missed a second, distinct real freeze found later
    by re-running with ``override_reason`` logging added (see the
    README's "Real bug: wall-hugging" section for the actual data): at one
    fixed (x, y), Laya kept proposing ``move_forward``/``strafe_left`` —
    never a turn — every single step for 79 consecutive steps, while
    StuckRecoveryConfig's own no-progress check *overrode* every one of
    those proposals into a turn (correctly, since neither move nor strafe
    ever made real progress there). Because this net's old condition only
    looked at Laya's *proposed* action, and Laya never proposed a turn in
    that scenario, it could never fire — StuckRecoveryConfig kept
    re-firing every step forever instead, with no mechanism to ever force
    an actual move attempt once turning had gone on long enough. Fixed by
    watching ``consecutive_turn_steps`` (already tracked against the
    *executed* ``final_action``, not the proposal) and moving this net
    ahead of StuckRecoveryConfig in run_episode's priority chain, so
    enough consecutive executed turns — however they were produced —
    forces a real move_forward test regardless of what Laya is currently
    proposing, except attack/shoot/use (a live combat or interaction
    decision is never overridden here). See StuckRecoveryConfig's own
    updated docstring for the other half of the same real bug (the
    turn direction itself flip-flopping and cancelling out). Logged the
    same way as the other nets: ``overridden=True``, ``proposed_action``
    kept as what Laya actually chose.
    """

    enabled: bool = True
    max_consecutive_turns: int = 6


@dataclass
class ExplorationNudgeConfig:
    """Circling *detector*: once every visible enemy is dead and every
    reachable pickup collected, there's nothing left in the world-state
    text to react to (no ``ENEMY``, no ``PICKUP``). Uses the AREA
    new/revisited signal (``state_encoder.StateEncoder.last_area_new``)
    already computed for the world-state text: if the current cell keeps
    coming back "revisited" for ``streak_threshold`` consecutive
    decisions, with no enemy or pickup to legitimately explain staying
    put, force a heading change. Only engages when every higher-priority
    safety net above isn't already handling this step.
    ``--no-exploration-nudge`` disables it.

    Originally a *blind* nudge (pick whichever open side, or alternate by
    step parity — the same ``_recovery_turn`` a wall-stuck agent gets)
    with an explicit caveat: it "can't know which heading actually leads
    toward unexplored space" because perception.py deliberately never
    trusts ViZDoom's ANGLE sign convention. That constraint no longer
    applies to *this* mechanism specifically — see FrontierExplorationConfig
    below, which empirically verified ANGLE's real convention and uses it,
    plus the existing visited-cells grid, to choose a directed heading
    instead. The detector logic above (when to fire) is unchanged; only
    the *action chosen once it fires* is now directed rather than blind.
    """

    enabled: bool = True
    streak_threshold: int = 15


@dataclass
class FrontierExplorationConfig:
    """Makes ExplorationNudgeConfig's override directed instead of blind,
    using ViZDoom's ANGLE game variable — verified empirically first (see
    ``wayfinding.py``'s module docstring and the README's "Verified: ANGLE's
    real sign convention" section for the actual printed evidence: real
    ``turn_left``/``turn_right`` actions against a real DoomEnv reliably
    increased/decreased ANGLE by ~10.5 degrees per action, wrapping
    0-360, and real POSITION_X/Y deltas after ``move_forward`` matched
    ``(cos(angle), sin(angle))`` to within measurement noise). Not trusted
    anywhere else in this codebase — bearing/aiming stay screen-space, per
    perception.py's own module docstring — this is a narrow, verified
    exception used only for this one override.

    When the circling detector fires, this projects several candidate
    headings (the current heading plus the offsets in
    ``wayfinding.best_exploration_heading``'s default set) forward by
    ``lookahead_cells`` grid cells and picks the smallest-turn candidate
    whose projected cell is NOT already in the encoder's visited-cells
    set — a real, directed guess at "which way is unexplored", not a
    guarantee (the projected cell could easily be on the far side of a
    wall this pipeline has no way of knowing about — same caveat as
    every other safety net here). Falls back to the old blind
    ``_recovery_turn`` if wayfinding has no visited-cell information to
    work from (area hint disabled) or is otherwise unable to compute a
    heading. ``--no-frontier-exploration`` reverts to the old blind nudge
    for comparison/regression.
    """

    enabled: bool = True
    lookahead_cells: float = 2.0


@dataclass
class SecretSearchConfig:
    """Escalates ExplorationNudgeConfig's override from a heading nudge to
    an active turn-and-use search once nudging has fired
    ``escalate_after`` times in a row without ever landing on genuinely
    new territory (``StateEncoder.last_area_new`` never going True in
    between) — the same "no reachable new territory nearby" state
    ExplorationNudgeConfig's revisited-streak already detects, just
    sustained rather than a single blip.

    Real design note from the user, grounded in the actual game: secret
    doors in Doom/Freedoom are visually identical to ordinary walls (no
    distinct texture or any cue perception.py could detect), so the only
    real way to find one is to press ``use`` against several nearby wall
    segments in turn, not just whichever single wall the player happens
    to be facing (which is all DoorUseConfig above ever tries). A plain
    directed nudge toward "the nearest cell my own grid hasn't marked
    visited yet" — which is nearly always available even in a small,
    fully-toured room, since ``wayfinding.nearest_unvisited_cell``'s
    fallback has no idea a wall is blocking the way — can keep reporting
    "go there" forever without ever concluding the area is a dead end
    worth searching; watching *how many nudges in a row produced no real
    new ground*, rather than waiting for that fallback to run dry, is
    what actually catches "stuck circling a small explored room" in
    practice.

    Once escalated, this turns to face each heading in ``headings_deg``
    (relative to the heading at the moment it triggered — the default
    ``(0, 90, 180, 270)`` covers "the wall dead ahead, then a quarter-turn
    each way, then the wall behind") and presses ``use`` once facing each
    one, before giving up and letting ordinary nudging resume. Genuinely
    a search, not a guaranteed find — if there's no secret door nearby,
    this is four wasted turn+use attempts, same "nudge, not a guarantee"
    caveat as every other net here. Checked at the same priority as
    ExplorationNudgeConfig (only engages when that circling detector would
    otherwise fire), and a sequence already in progress is checked ahead
    of a fresh circling-detector decision so it runs to completion (or
    gets pre-empted by a genuine higher-priority net, e.g. an enemy
    showing up mid-search) rather than restarting. ``--no-secret-search``
    disables it, falling back to plain nudging indefinitely.
    """

    enabled: bool = True
    headings_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    escalate_after: int = 2


@dataclass
class WallFollowConfig:
    """Last-resort deterministic fallback, one rung above SecretSearchConfig:
    the classic maze "right-hand rule" (or left-hand) — keep a wall at a
    fixed relative side and slide along it, turning to hug it at corners
    instead of turning away from it the way every other recovery net here
    does. Formally guaranteed to reach a simply-connected maze's exit with
    zero map memory; real Doom levels aren't pure mazes, but many are close
    enough (linear branching corridors) for this to be worth trying once
    plain nudging AND a full secret-door sweep have both come up empty
    ``activate_after_searches`` times in a row without landing on genuinely
    new territory.

    Uses only perception.py's existing screen-space wall/open-path flags
    (``open_left``/``open_right``/``open_forward``) — no ANGLE, no
    visited-cells grid, nothing this mechanism needs FrontierExploration/
    SecretSearch's machinery for. Deliberately distinct from the real
    "wall-hugging" bug this session found and fixed (StuckRecoveryConfig/
    TurnLoopRecoveryConfig's docstrings): that was *aimless* repeated
    facing/bumping with zero net rotation, going nowhere; this is
    *purposeful* wall contact — always the same hand, always sliding
    forward along it — which is a completely different thing despite the
    similar-sounding name. Runs for up to ``max_follow_steps`` consecutive
    decisions (bailing out early on real progress into new territory, or
    if any higher-priority net pre-empts it, e.g. an enemy appears) before
    giving up and letting the nudge/secret-search ladder try again from
    scratch. ``--no-wall-follow`` disables it.
    """

    enabled: bool = True
    hand: str = "right"
    activate_after_searches: int = 1
    max_follow_steps: int = 60


@dataclass
class DoorUseConfig:
    """``use`` (open a door / flip a switch) never wins Laya's crowded
    multi-way ``choice`` question — the same class of bug the shoot gate
    was built for, confirmed with real logged data before writing any
    code: across a real 2,980-step ``--scenario level`` run
    (``logs/laya_level_full.steps.jsonl``), ``use``'s own probability
    inside Laya's returned distribution never exceeded 0.169 (mean 0.071,
    2,950 real decisions where it was scored) and was picked as the final
    action exactly 0 times, regardless of how close the player was to a
    wall.

    Unlike the shoot gate, this doesn't build a matching `noul`
    "should_use" question — there's no game variable exposing "is there an
    actual door/switch here" for a probabilistic gate to be calibrated
    against, and pressing USE against a plain, non-interactive Doom wall
    is a real no-op (confirmed against the actual game rules, not
    assumed), so the simpler, safer lever is used instead: fire `use`
    deterministically, once, the moment ``WALL ahead near`` has held for
    ``stall_threshold`` consecutive steps — an edge trigger (fires exactly
    when the streak *reaches* the threshold, not every step after), so a
    dead-end wall gets exactly one wasted "use" attempt before falling
    through to StuckRecoveryConfig/TurnLoopRecoveryConfig's own turn-based
    recovery, rather than spamming `use` forever or blocking those nets
    from ever running. Ranked just above StuckRecoveryConfig in the
    priority chain: if the wall genuinely is a door, trying to use it
    should win over turning away from it, but a genuine attack/shoot
    decision (or a `use` Laya already chose on its own) is left alone.
    Logged the same way as every other net here: `overridden=True`,
    `proposed_action` kept as what Laya actually chose.
    """

    enabled: bool = True
    stall_threshold: int = 3


@dataclass
class ThreatResponseConfig:
    """When the previous step's result was "took_damage" and no enemy is
    currently visible (the THREAT line in state_encoder.py — an attacker
    outside perception.py's FOV), turn instead of letting Laya's decision
    stand if it wasn't already a turn.

    This safety net exists because the textual THREAT signal alone isn't
    guaranteed to change the model's decision — a purely textual cue can
    be present in the state and still lose out to a competing action.
    Laya has no trigger mechanism at all (see actions.py/laya_agent.py —
    Laya picks among fixed choice labels via one forward pass, nothing
    regex-based), so this specific failure mode hasn't been independently
    verified against Laya itself; kept on by default as a logged-not-hidden
    precaution either way. Overridden steps set `overridden=True` and keep
    `proposed_action` as what Laya actually picked. `--no-threat-response`
    disables it.
    """

    enabled: bool = True


@dataclass
class ThreatEngagementConfig:
    """When a real, visible, close-enough enemy is off to one side (not
    already bearing exactly ``front``) and the player has ammo, turn
    toward it instead of letting Laya's own movement choice stand — even
    if that choice was a perfectly reasonable-looking ``move_forward`` or
    ``strafe_left``.

    Real motivation, not a hypothetical: a full `--scenario level` run
    (see the README's "Verified behaviour") escaped the corner-spinning
    bug and travelled 10,121 distance units, one real kill, 4 items
    collected — and then died anyway, `damage_given=35` vs
    `damage_taken=120`, chipped from 51 health to 0 by one ordinary
    `zombieman`. It was visible almost the entire time
    (`front-left`/`left`/`far-left`/`front-right`), never quite exactly
    `front` for long, so `LayaAgent`'s shoot gate (which requires bearing
    ``front`` — see laya_agent.py) never fired on it, and the *existing*
    `ThreatResponseConfig` above never fired either, because that net is
    specifically for an *unseen* attacker — this one was visible the
    whole time, just off-center. Nothing in the pipeline was telling the
    movement choice "there's a real, nearby, currently-hostile enemy;
    weight turning toward it over exploring" — Laya kept picking
    `move_forward`/`strafe_left` while getting shot from the side.

    Scoped deliberately narrow: only enemies at ``engage_distances``
    (excludes ``far`` — a distant enemy glimpsed at the edge of view
    shouldn't hijack navigation, which is exactly the over-triggering
    risk `TurnLoopRecoveryConfig` was built to avoid on the *other* side
    of this same tension) and only when ammo remains (turning to face a
    threat you can't shoot doesn't help). Logged the same way as every
    other net here: `overridden=True`, `proposed_action` kept as what
    Laya actually chose.
    """

    enabled: bool = True
    engage_distances: frozenset[str] = frozenset({"very-near", "near", "medium"})


def _nearest_off_center_threat(
    perception: Perception, engage_distances: frozenset[str]
) -> EnemyPercept | None:
    for enemy in perception.enemies:
        if enemy.bearing == "front":
            continue
        if enemy.bearing not in (_LEFT_OF_CENTER_BEARINGS | _RIGHT_OF_CENTER_BEARINGS):
            continue
        if enemy.distance in engage_distances:
            return enemy
    return None


def _should_override_for_threat_engagement(
    threat_engagement: ThreatEngagementConfig, perception: Perception, proposed_action: str
) -> EnemyPercept | None:
    # Never second-guess a decision that's already attack/shoot: it means
    # some enemy (possibly a different one from the off-center threat
    # found below) is already lined up exactly front and the shoot gate
    # fired correctly — don't redirect away from a live kill to face a
    # less-aligned target instead.
    if not threat_engagement.enabled or perception.ammo <= 0 or proposed_action in ("attack", "shoot"):
        return None
    return _nearest_off_center_threat(perception, threat_engagement.engage_distances)


def _engagement_turn(enemy: EnemyPercept, action_set: str) -> str:
    direction = "left" if enemy.bearing in _LEFT_OF_CENTER_BEARINGS else "right"
    # "_small", not "_large": this is fine-aiming toward a known target
    # (mirrors the "small turn to fine-aim" criteria already given to
    # Laya for these labels — see laya_agent._ACTION_CRITERIA), not the
    # blind escape/scan the other recovery nets use turn_*_large for.
    return f"turn_{direction}" if action_set == "stage1" else f"turn_{direction}_small"


@dataclass
class LowHealthRetreatConfig:
    """When a *visible* enemy is present and health is at or below
    ``health_threshold``, retreat (``move_backward``) instead of letting
    Laya's decision stand — unless it already chose to attack/shoot or
    retreat itself, which are left alone. Below ``emergency_health_threshold``,
    retreat overrides even an attack decision.

    The real gap in practice was specifically the *absence* of deliberate
    disengagement when badly hurt. The two-threshold design is grounded
    in docs/tiny-doom-runtime-policy-200-rules.md rule 199 ("nearly dead
    enemy does not justify remaining in lethal geometry") paired with rule
    119 ("health below 10% → survival, health and exit dominate
    everything"). Directly complementary to ``ThreatEngagementConfig``
    above, not a duplicate of it: that one turns to *fight* an off-center
    threat while health is still reasonable; this one disengages once
    health drops low regardless of bearing — including the exact tail of
    the real death this README documents (health 51 → 10 → 4 → 0 against
    one zombieman, never disengaging even once truly critical).
    """

    enabled: bool = True
    health_threshold: int = 20
    emergency_health_threshold: int = 10


def _should_retreat_from_visible_threat(
    config: LowHealthRetreatConfig, perception: Perception, proposed_action: str
) -> bool:
    if not config.enabled or not perception.enemies or proposed_action == "move_backward":
        return False
    if perception.health <= config.emergency_health_threshold:
        return True  # emergency: overrides even attack/shoot, see class docstring
    return perception.health <= config.health_threshold and proposed_action not in ("attack", "shoot")


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
    # Which safety net fired, or "" when overridden is False. With only
    # `overridden` logged, a batch run showed 121 large-turn overrides with
    # no way to tell whether they came from stuck_recovery or
    # exploration_nudge (both can produce the same turn_*_large action).
    # Set at each override site below, one string per mechanism.
    override_reason: str = ""
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



# TurnLoopRecoveryConfig never overrides a live combat decision or a
# use-a-door attempt Laya chose on its own — see its docstring for why the
# old, narrower "only when Laya itself proposed a turn" condition missed a
# real freeze where Laya kept proposing move_forward/strafe_left instead.
_NEVER_OVERRIDE_TURN_LOOP_FOR = frozenset({"attack", "shoot", "use"})


def _should_override_for_turn_loop(
    turn_loop_recovery: TurnLoopRecoveryConfig, consecutive_turn_steps: int, proposed_action: str
) -> bool:
    return (
        turn_loop_recovery.enabled
        and consecutive_turn_steps >= turn_loop_recovery.max_consecutive_turns
        and proposed_action not in _NEVER_OVERRIDE_TURN_LOOP_FOR
    )


def _should_try_use(
    config: DoorUseConfig, perception: Perception, wall_near_streak: int, proposed_action: str
) -> bool:
    return (
        config.enabled
        and perception.wall_near
        and wall_near_streak == config.stall_threshold
        and proposed_action not in ("use", "attack", "shoot")
    )


def _should_nudge_exploration(
    config: ExplorationNudgeConfig, perception: Perception, revisited_streak: int, proposed_action: str
) -> bool:
    return (
        config.enabled
        and revisited_streak >= config.streak_threshold
        and not perception.enemies
        and not perception.pickups
        and proposed_action not in _TURN_ACTION_NAMES
    )


def _recovery_turn(perception: Perception, action_set: str, step: int) -> str:
    if perception.open_left and not perception.open_right:
        direction = "left"
    elif perception.open_right and not perception.open_left:
        direction = "right"
    else:
        direction = "left" if step % 2 == 0 else "right"
    # "_large" for the full action set, not "_small": this fires only when
    # already stuck, so a bigger rotation escapes a corner faster than the
    # small increments that got the agent stuck turning in place to begin
    # with (see TurnLoopRecoveryConfig above for the real run that showed
    # small turns alone repeating for 6+ steps without escaping).
    return f"turn_{direction}" if action_set == "stage1" else f"turn_{direction}_large"


def _exploration_turn(
    perception: Perception,
    encoder: StateEncoder,
    frontier_exploration: FrontierExplorationConfig,
    action_set: str,
    step: int,
) -> str:
    """The action ExplorationNudgeConfig's override actually executes once
    it fires. Directed when FrontierExplorationConfig is enabled and
    wayfinding can compute a real heading toward the nearest known-
    unvisited cell (see that config's docstring); falls back to the old
    blind ``_recovery_turn`` otherwise (no visited-cell information yet,
    or wayfinding genuinely has nothing better to offer)."""
    if frontier_exploration.enabled:
        target_heading = wayfinding.best_exploration_heading(
            perception.x,
            perception.y,
            perception.angle,
            encoder.visited_cells,
            encoder.config.area_cell_size,
            lookahead_distance=encoder.config.area_cell_size * frontier_exploration.lookahead_cells,
        )
        if target_heading is not None:
            return wayfinding.turn_action_for_heading(perception.angle, target_heading, action_set)
    return _recovery_turn(perception, action_set, step)


def _secret_search_action(
    perception: Perception, targets: tuple[float, ...], index: int, action_set: str
) -> tuple[str, int | None]:
    """Advances one step of a SecretSearchConfig sequence. ``targets`` is
    the fixed list of absolute headings (degrees) to visit, ``index`` is
    which one is currently being pursued. Returns ``(action,
    next_index)``: turns toward ``targets[index]`` while not yet facing it
    (``next_index`` unchanged), or presses ``use`` once it is faced
    (``next_index`` advances to the next target, or ``None`` once the last
    one has just been used — the caller should end the sequence then).
    Pure and independently testable, unlike the run_episode loop state
    that tracks which sequence is in progress across steps."""
    target = targets[index]
    if abs(wayfinding.angular_diff(perception.angle, target)) <= wayfinding.FACING_TOLERANCE_DEG:
        next_index = index + 1
        return "use", (next_index if next_index < len(targets) else None)
    return wayfinding.turn_action_for_heading(perception.angle, target, action_set), index


def _start_exploration_override(
    perception: Perception,
    encoder: StateEncoder,
    frontier_exploration: FrontierExplorationConfig,
    secret_search: SecretSearchConfig,
    nudge_without_new_area: int,
    action_set: str,
    step: int,
) -> tuple[str, tuple[float, ...] | None, int]:
    """What ExplorationNudgeConfig's override does the FIRST step it
    fires (see ``_secret_search_action`` above for continuing an
    already-started sequence). Returns ``(action, secret_search_targets,
    secret_search_index)`` — ``secret_search_targets`` is None unless a
    new search sequence was just started.

    Three cases, checked in order: (1) secret search is enabled and
    ``nudge_without_new_area`` has already reached
    ``secret_search.escalate_after`` — plain nudging has fired that many
    times in a row without ever landing on genuinely new territory, so
    escalate to a systematic turn-and-use sequence (see
    SecretSearchConfig's docstring for why "keep nudging" alone doesn't
    reliably converge: `wayfinding.best_exploration_heading`'s
    nearest-unvisited-cell fallback has no idea a wall might be blocking
    the way, so it can keep proposing "go there" forever without a
    sustained no-progress signal to escalate on instead). (2) frontier
    exploration (see FrontierExplorationConfig) finds real, reachable
    unexplored territory nearby — turn directly toward it, same as
    ``_exploration_turn``. (3) frontier disabled or wayfinding has
    nothing to offer — the old blind ``_recovery_turn``.
    """
    if secret_search.enabled and nudge_without_new_area >= secret_search.escalate_after:
        targets = tuple((perception.angle + offset) % 360.0 for offset in secret_search.headings_deg)
        action, next_index = _secret_search_action(perception, targets, 0, action_set)
        return action, (targets if next_index is not None else None), (next_index or 0)
    if frontier_exploration.enabled:
        target_heading = wayfinding.best_exploration_heading(
            perception.x,
            perception.y,
            perception.angle,
            encoder.visited_cells,
            encoder.config.area_cell_size,
            lookahead_distance=encoder.config.area_cell_size * frontier_exploration.lookahead_cells,
        )
        if target_heading is not None:
            return wayfinding.turn_action_for_heading(perception.angle, target_heading, action_set), None, 0
    return _recovery_turn(perception, action_set, step), None, 0


def _wall_follow_action(perception: Perception, hand: str, action_set: str) -> str:
    """One step of the classic "keep a wall on one side and slide along
    it" maze rule (see WallFollowConfig's docstring), using only
    perception.py's existing screen-space open_left/open_right/
    open_forward flags:

    1. If the follow-side is open (no wall there right now), turn toward
       it — the wall curved away at an outer corner; turning to face it
       keeps it at roughly the same relative bearing instead of drifting
       away from it entirely.
    2. Else, if forward is open, move forward — sliding along the wall.
    3. Else (blocked ahead AND on the follow side — an inner corner or
       dead end), turn away from the followed wall to round the corner.
    """
    follow_open = perception.open_right if hand == "right" else perception.open_left
    other_hand = "left" if hand == "right" else "right"
    if follow_open:
        return f"turn_{hand}" if action_set == "stage1" else f"turn_{hand}_small"
    if perception.open_forward:
        return "move_forward"
    return f"turn_{other_hand}" if action_set == "stage1" else f"turn_{other_hand}_small"


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
    turn_loop_recovery: TurnLoopRecoveryConfig | None = None,
    threat_engagement: ThreatEngagementConfig | None = None,
    low_health_retreat: LowHealthRetreatConfig | None = None,
    exploration_nudge: ExplorationNudgeConfig | None = None,
    door_use: DoorUseConfig | None = None,
    frontier_exploration: FrontierExplorationConfig | None = None,
    secret_search: SecretSearchConfig | None = None,
    wall_follow: WallFollowConfig | None = None,
) -> tuple[EpisodeResult, list[StepRecord]]:
    perception_config = perception_config or PerceptionConfig()
    stuck_recovery = stuck_recovery if stuck_recovery is not None else StuckRecoveryConfig()
    threat_response = threat_response if threat_response is not None else ThreatResponseConfig()
    turn_loop_recovery = turn_loop_recovery if turn_loop_recovery is not None else TurnLoopRecoveryConfig()
    threat_engagement = threat_engagement if threat_engagement is not None else ThreatEngagementConfig()
    low_health_retreat = low_health_retreat if low_health_retreat is not None else LowHealthRetreatConfig()
    exploration_nudge = exploration_nudge if exploration_nudge is not None else ExplorationNudgeConfig()
    door_use = door_use if door_use is not None else DoorUseConfig()
    frontier_exploration = frontier_exploration if frontier_exploration is not None else FrontierExplorationConfig()
    secret_search = secret_search if secret_search is not None else SecretSearchConfig()
    wall_follow = wall_follow if wall_follow is not None else WallFollowConfig()

    env.new_episode()
    agent.reset()
    encoder.reset()

    prev_perception: Perception | None = None
    last_action: str | None = None
    last_result: str | None = None
    starting_ammo: int | None = None
    no_progress_steps = 0
    consecutive_turn_steps = 0
    revisited_streak = 0
    wall_near_streak = 0
    stuck_recovery_turn: str | None = None
    secret_search_targets: tuple[float, ...] | None = None
    secret_search_index = 0
    nudge_without_new_area = 0
    secret_search_failures = 0
    wall_follow_steps_left = 0
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
        if perception.enemies or perception.pickups or encoder.last_area_new is not False:
            revisited_streak = 0
        else:
            revisited_streak += 1
        # Tracks SecretSearchConfig's and WallFollowConfig's escalation
        # triggers: only genuinely new ground (encoder.last_area_new is
        # True) resets these, not an enemy/pickup appearing — those
        # already reset revisited_streak above (something to react to),
        # but don't mean the area itself has actually grown, which is the
        # specific thing "nudging isn't working" needs to track. See
        # SecretSearchConfig's/WallFollowConfig's docstrings.
        if encoder.last_area_new is True:
            nudge_without_new_area = 0
            secret_search_failures = 0
        wall_near_streak = wall_near_streak + 1 if perception.wall_near else 0
        decision = agent.decide(perception, encoded)

        overridden = False
        override_reason = ""
        final_action = decision.action
        # Priority order for the first four nets: survival (low-health
        # retreat) outranks engaging a threat, which outranks reacting to
        # an unseen attacker, which outranks plain navigation recovery —
        # matching the playbook's own
        # "survival > threat control > navigation" hierarchy
        # (docs/tiny-doom-runtime-policy-200-rules.md's "core priority"
        # section). door_use sits just above the two turn-recovery nets:
        # trying `use` against a wall that turns out to be a real door
        # should win over turning away from it, but a genuine
        # threat/survival response still wins over either.
        #
        # turn_loop_recovery now outranks stuck_recovery (swapped from the
        # original order) — see TurnLoopRecoveryConfig's docstring for the
        # real bug this fixes: stuck_recovery's own no-progress condition
        # can stay true indefinitely (Laya kept proposing move_forward/
        # strafe_left every step at one stuck position), which under the
        # old order meant it always won the elif chain and turn_loop_recovery
        # never got a chance to force an actual move attempt, no matter how
        # many consecutive turns had actually been executed.
        if _should_retreat_from_visible_threat(low_health_retreat, perception, decision.action):
            final_action = "move_backward"
            overridden = True
            override_reason = "low_health_retreat"
        elif (
            engaging_enemy := _should_override_for_threat_engagement(threat_engagement, perception, decision.action)
        ) is not None:
            final_action = _engagement_turn(engaging_enemy, env.config.action_set)
            overridden = True
            override_reason = "threat_engagement"
        elif _should_override_for_threat(threat_response, last_result, perception, decision.action):
            final_action = _recovery_turn(perception, env.config.action_set, step)
            overridden = True
            override_reason = "threat_response"
        elif _should_try_use(door_use, perception, wall_near_streak, decision.action):
            final_action = "use"
            overridden = True
            override_reason = "door_use"
        elif _should_override_for_turn_loop(turn_loop_recovery, consecutive_turn_steps, decision.action):
            final_action = "move_forward"
            overridden = True
            override_reason = "turn_loop_recovery"
        elif (
            stuck_recovery.enabled
            and no_progress_steps >= stuck_recovery.no_progress_threshold
            and decision.action in _STALL_ACTION_NAMES
        ):
            # Cache one _recovery_turn result per stuck event instead of
            # recomputing (and re-tie-breaking) it on every firing — see
            # StuckRecoveryConfig's docstring for the real oscillation bug
            # this fixes (step-parity tie-break flip-flopping every call,
            # netting zero rotation). Cleared below only on real progress.
            if stuck_recovery_turn is None:
                stuck_recovery_turn = _recovery_turn(perception, env.config.action_set, step)
            final_action = stuck_recovery_turn
            overridden = True
            override_reason = "stuck_recovery"
        elif wall_follow_steps_left > 0:
            # WallFollowConfig already engaged — keep sliding along the
            # followed wall rather than letting a fresh exploration-nudge
            # decision interrupt it (same "run to completion unless a
            # higher-priority net pre-empts it" pattern as secret search).
            final_action = _wall_follow_action(perception, wall_follow.hand, env.config.action_set)
            wall_follow_steps_left -= 1
            overridden = True
            override_reason = "wall_follow"
        elif secret_search_targets is not None:
            # A SecretSearchConfig sequence already in progress — continue
            # it rather than letting a fresh exploration-nudge decision
            # restart or interrupt it (see SecretSearchConfig's docstring
            # for why it's checked ahead of _should_nudge_exploration).
            final_action, next_index = _secret_search_action(
                perception, secret_search_targets, secret_search_index, env.config.action_set
            )
            secret_search_index = next_index if next_index is not None else 0
            if next_index is None:
                secret_search_targets = None
                secret_search_failures += 1  # a full sweep just finished with nothing new to show for it
            overridden = True
            override_reason = "secret_search"
        elif _should_nudge_exploration(exploration_nudge, perception, revisited_streak, decision.action):
            if wall_follow.enabled and secret_search_failures >= wall_follow.activate_after_searches:
                # Plain nudging AND a full secret-door sweep have both
                # already come up empty enough times — try the classic
                # wall-following maze rule instead of nudging again (see
                # WallFollowConfig's docstring).
                final_action = _wall_follow_action(perception, wall_follow.hand, env.config.action_set)
                wall_follow_steps_left = wall_follow.max_follow_steps - 1
                overridden = True
                override_reason = "wall_follow"
                secret_search_failures = 0  # ladder consumed -- give it a fresh chance later
            else:
                final_action, secret_search_targets, secret_search_index = _start_exploration_override(
                    perception, encoder, frontier_exploration, secret_search, nudge_without_new_area,
                    env.config.action_set, step,
                )
                overridden = True
                if secret_search_targets is not None:
                    override_reason = "secret_search"
                    nudge_without_new_area = 0  # escalation consumed -- start counting fresh next time
                else:
                    override_reason = "exploration_nudge"
                    nudge_without_new_area += 1
            revisited_streak = 0

        reward = env.execute(final_action)

        new_state = env.get_state() if not env.is_finished() else None
        new_perception = perceive(new_state, perception_config) if new_state is not None else None
        result = summarize_result(perception, new_perception)
        encoder.record(final_action, result)

        picked_up_key = _infer_picked_up_key(perception, new_perception)
        if picked_up_key is not None:
            keys_held.add(picked_up_key)

        # Only real positional progress resets this — NOT merely "this
        # step's final_action happened not to be a move" (which includes
        # the recovery override's own turn). A real live --scenario level
        # run showed why the naive version breaks: strafe_left (no
        # progress) -> strafe_left (no progress, threshold hit) ->
        # forced turn_right_large override -> no_progress_steps reset to
        # 0 right there -> next decision proposes strafe_left again with
        # a fresh 2-strike allowance -> same override fires again -> the
        # cycle repeats forever, alternating move/forced-turn, distance
        # frozen. The fix: only real position change resets the streak;
        # any override still counts against it regardless of what
        # happened in between.
        progressed = new_perception is not None and (
            (new_perception.x - perception.x) ** 2 + (new_perception.y - perception.y) ** 2
        ) ** 0.5 >= stuck_recovery.min_progress_units
        if progressed:
            no_progress_steps = 0
            stuck_recovery_turn = None
        elif final_action in _STALL_ACTION_NAMES:
            no_progress_steps += 1

        consecutive_turn_steps = consecutive_turn_steps + 1 if final_action in _TURN_ACTION_NAMES else 0

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
            override_reason=override_reason,
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

    # ViZDoom has no "you won" game *variable* (checked the full
    # GameVariable enum — there isn't one), so `completed` was originally
    # inferred purely from timing: the episode ended on its own (a
    # scenario's built-in win condition, not our own --max-steps budget
    # running out) and the player didn't die, with a margin below the
    # configured timeout guarding against a coincidental last-tic timeout
    # looking like a win.
    #
    # For scenario="level" there's since a real, documented signal
    # instead: ViZDoom's own bundled scenarios/freedoom2.cfg sets
    # `map_exit_reward = 1`, "Reward for completing the level (exiting
    # through the exit)" — config/level.cfg now sets it too. Cross-checked
    # against total_reward() when available, since it's ground truth
    # rather than an inference; kept scoped to "level" specifically
    # because map_exit_reward means nothing on basic/my_way_home, and
    # basic's own built-in ACS reward script (a bonus for a kill) would
    # otherwise read as a false "completed" there.
    died = env.is_player_dead()
    timeout_tics = env.episode_timeout_tics()
    completed_by_timing = (
        not died
        and env.is_finished()
        and final is not None
        and final.tic < max(timeout_tics - 5, 0)
    )
    completed_by_exit_reward = not died and env.config.scenario == "level" and env.total_reward() >= 1
    completed = completed_by_timing or completed_by_exit_reward

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
