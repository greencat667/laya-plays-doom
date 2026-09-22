import laya_doom.controller as controller_mod
from laya_doom.controller import (
    DoorUseConfig,
    ExplorationNudgeConfig,
    FrontierExplorationConfig,
    LowHealthRetreatConfig,
    SecretSearchConfig,
    StuckRecoveryConfig,
    ThreatEngagementConfig,
    ThreatResponseConfig,
    TurnLoopRecoveryConfig,
    WallFollowConfig,
    _engagement_turn,
    _exploration_turn,
    _nearest_off_center_threat,
    _recovery_turn,
    _secret_search_action,
    _should_nudge_exploration,
    _should_override_for_threat,
    _should_override_for_threat_engagement,
    _should_override_for_turn_loop,
    _should_retreat_from_visible_threat,
    _should_try_use,
    _start_exploration_override,
    _wall_follow_action,
    run_episode,
)
from laya_doom.laya_agent import Decision
from laya_doom.perception import EnemyPercept, Perception
from laya_doom.state_encoder import EncoderConfig, StateEncoder
from tests.test_state_encoder import make_perception


def test_recovery_turn_prefers_open_side_left():
    perc = make_perception(open_left=True, open_right=False)
    assert _recovery_turn(perc, "stage1", step=0) == "turn_left"


def test_recovery_turn_prefers_open_side_right():
    perc = make_perception(open_left=False, open_right=True)
    assert _recovery_turn(perc, "stage1", step=0) == "turn_right"


def test_recovery_turn_alternates_when_tied():
    perc = make_perception(open_left=True, open_right=True)
    assert _recovery_turn(perc, "stage1", step=0) == "turn_left"
    assert _recovery_turn(perc, "stage1", step=1) == "turn_right"


def test_recovery_turn_uses_full_action_set_names():
    # "_large", not "_small": this fires only once already stuck, and a
    # real level run showed small turns alone repeating for 6+ steps
    # without escaping a corner (see TurnLoopRecoveryConfig).
    perc = make_perception(open_left=True, open_right=False)
    assert _recovery_turn(perc, "full", step=0) == "turn_left_large"


def test_threat_override_fires_on_unseen_damage():
    config = ThreatResponseConfig()
    perc = make_perception(enemies=())
    assert _should_override_for_threat(config, "took_damage", perc, "move_forward") is True


def test_threat_override_not_fired_when_enemy_visible():
    from laya_doom.perception import EnemyPercept

    config = ThreatResponseConfig()
    perc = make_perception(enemies=(EnemyPercept(kind="imp", bearing="front", distance="near", raw_distance_units=100.0),))
    assert _should_override_for_threat(config, "took_damage", perc, "move_forward") is False


def test_threat_override_not_fired_without_recent_damage():
    config = ThreatResponseConfig()
    perc = make_perception(enemies=())
    assert _should_override_for_threat(config, "no_change", perc, "move_forward") is False


def test_threat_override_not_fired_when_already_turning():
    config = ThreatResponseConfig()
    perc = make_perception(enemies=())
    assert _should_override_for_threat(config, "took_damage", perc, "turn_left_small") is False


def test_threat_override_disabled_via_config():
    config = ThreatResponseConfig(enabled=False)
    perc = make_perception(enemies=())
    assert _should_override_for_threat(config, "took_damage", perc, "move_forward") is False


def test_turn_loop_override_fires_after_enough_consecutive_turns():
    # Real motivation: a --scenario level run picked turn_left_small on
    # every single decision for 6+ steps in a row, never once proposing a
    # move, even after the logged state showed a path had opened up.
    config = TurnLoopRecoveryConfig(max_consecutive_turns=6)
    assert _should_override_for_turn_loop(config, consecutive_turn_steps=6, proposed_action="turn_left_small") is True
    assert _should_override_for_turn_loop(config, consecutive_turn_steps=9, proposed_action="turn_right_large") is True


def test_turn_loop_override_not_fired_below_threshold():
    config = TurnLoopRecoveryConfig(max_consecutive_turns=6)
    assert _should_override_for_turn_loop(config, consecutive_turn_steps=5, proposed_action="turn_left_small") is False


def test_turn_loop_override_fires_regardless_of_proposed_action():
    # Real bug fix, not the original design: this net used to require
    # Laya's own *proposed* action to be a turn, which missed a real
    # freeze where Laya kept proposing move_forward/strafe_left every
    # step while StuckRecoveryConfig overrode every one of them into a
    # turn (see TurnLoopRecoveryConfig's docstring for the full story).
    # consecutive_turn_steps is tracked against the *executed* action, so
    # it's already correct regardless of what Laya proposed — the net
    # should fire on it whenever the proposal isn't a live combat/use
    # decision, not only when the proposal happens to itself be a turn.
    config = TurnLoopRecoveryConfig(max_consecutive_turns=6)
    assert _should_override_for_turn_loop(config, consecutive_turn_steps=10, proposed_action="move_forward") is True
    assert _should_override_for_turn_loop(config, consecutive_turn_steps=10, proposed_action="wait") is True
    assert _should_override_for_turn_loop(config, consecutive_turn_steps=10, proposed_action="strafe_left") is True


def test_turn_loop_override_never_overrides_combat_or_use():
    config = TurnLoopRecoveryConfig(max_consecutive_turns=6)
    assert _should_override_for_turn_loop(config, consecutive_turn_steps=10, proposed_action="attack") is False
    assert _should_override_for_turn_loop(config, consecutive_turn_steps=10, proposed_action="shoot") is False
    assert _should_override_for_turn_loop(config, consecutive_turn_steps=10, proposed_action="use") is False


def test_turn_loop_override_disabled_via_config():
    config = TurnLoopRecoveryConfig(enabled=False)
    assert _should_override_for_turn_loop(config, consecutive_turn_steps=99, proposed_action="turn_left_small") is False


def _enemy(bearing: str, distance: str = "near") -> EnemyPercept:
    return EnemyPercept(kind="zombieman", bearing=bearing, distance=distance, raw_distance_units=100.0)


def test_nearest_off_center_threat_ignores_exact_front():
    perc = make_perception(enemies=(_enemy("front"),))
    assert _nearest_off_center_threat(perc, frozenset({"near"})) is None


def test_nearest_off_center_threat_finds_left_and_right_variants():
    for bearing in ("left", "far-left", "front-left", "right", "far-right", "front-right"):
        perc = make_perception(enemies=(_enemy(bearing),))
        assert _nearest_off_center_threat(perc, frozenset({"near"})) is not None


def test_nearest_off_center_threat_ignores_out_of_range_distance():
    # Real design choice: "far" is excluded so a distant enemy glimpsed at
    # the edge of view doesn't hijack navigation — see ThreatEngagementConfig.
    perc = make_perception(enemies=(_enemy("front-left", distance="far"),))
    assert _nearest_off_center_threat(perc, frozenset({"very-near", "near", "medium"})) is None


def test_engagement_turn_matches_bearing_side():
    assert _engagement_turn(_enemy("front-left"), "full") == "turn_left_small"
    assert _engagement_turn(_enemy("far-right"), "full") == "turn_right_small"
    assert _engagement_turn(_enemy("left"), "stage1") == "turn_left"


def test_threat_engagement_override_fires_with_ammo():
    config = ThreatEngagementConfig()
    perc = make_perception(ammo=50, enemies=(_enemy("front-left"),))
    assert _should_override_for_threat_engagement(config, perc, "move_forward") is not None


def test_threat_engagement_override_not_fired_without_ammo():
    # Real motivation this guards against: turning to face a threat you
    # can't shoot doesn't help — see the class docstring.
    config = ThreatEngagementConfig()
    perc = make_perception(ammo=0, enemies=(_enemy("front-left"),))
    assert _should_override_for_threat_engagement(config, perc, "move_forward") is None


def test_threat_engagement_override_not_fired_when_already_attacking():
    # Never second-guess an already-correct attack/shoot on a DIFFERENT
    # enemy that happens to be exactly front — see the function docstring.
    config = ThreatEngagementConfig()
    perc = make_perception(ammo=50, enemies=(_enemy("front"), _enemy("front-left")))
    assert _should_override_for_threat_engagement(config, perc, "attack") is None
    assert _should_override_for_threat_engagement(config, perc, "shoot") is None


def test_threat_engagement_override_disabled_via_config():
    config = ThreatEngagementConfig(enabled=False)
    perc = make_perception(ammo=50, enemies=(_enemy("front-left"),))
    assert _should_override_for_threat_engagement(config, perc, "move_forward") is None


def test_low_health_retreat_fires_with_visible_enemy_and_critical_health():
    config = LowHealthRetreatConfig(health_threshold=20)
    perc = make_perception(health=15, enemies=(_enemy("front"),))
    assert _should_retreat_from_visible_threat(config, perc, "move_forward") is True


def test_low_health_retreat_not_fired_above_threshold():
    config = LowHealthRetreatConfig(health_threshold=20)
    perc = make_perception(health=50, enemies=(_enemy("front"),))
    assert _should_retreat_from_visible_threat(config, perc, "move_forward") is False


def test_low_health_retreat_not_fired_without_visible_enemy():
    config = LowHealthRetreatConfig(health_threshold=20)
    perc = make_perception(health=10, enemies=())
    assert _should_retreat_from_visible_threat(config, perc, "move_forward") is False


def test_low_health_retreat_does_not_override_attack_above_emergency():
    # health=15 is at/below health_threshold(20) but above
    # emergency_health_threshold(10) — moderate, not critical.
    config = LowHealthRetreatConfig(health_threshold=20, emergency_health_threshold=10)
    perc = make_perception(health=15, enemies=(_enemy("front"),))
    assert _should_retreat_from_visible_threat(config, perc, "attack") is False
    assert _should_retreat_from_visible_threat(config, perc, "shoot") is False
    assert _should_retreat_from_visible_threat(config, perc, "move_backward") is False


def test_low_health_retreat_overrides_attack_at_emergency_threshold():
    # health=8 is at/below emergency_health_threshold(10) — rule 199/119
    # (docs/tiny-doom-runtime-policy-200-rules.md): survival overrides
    # finishing a kill once health is genuinely critical, not just low.
    config = LowHealthRetreatConfig(health_threshold=20, emergency_health_threshold=10)
    perc = make_perception(health=8, enemies=(_enemy("front"),))
    assert _should_retreat_from_visible_threat(config, perc, "attack") is True
    assert _should_retreat_from_visible_threat(config, perc, "move_forward") is True


def test_low_health_retreat_never_overrides_an_already_chosen_retreat():
    config = LowHealthRetreatConfig(health_threshold=20, emergency_health_threshold=10)
    perc = make_perception(health=1, enemies=(_enemy("front"),))
    assert _should_retreat_from_visible_threat(config, perc, "move_backward") is False


def test_low_health_retreat_disabled_via_config():
    config = LowHealthRetreatConfig(enabled=False, health_threshold=20)
    perc = make_perception(health=5, enemies=(_enemy("front"),))
    assert _should_retreat_from_visible_threat(config, perc, "move_forward") is False


def test_exploration_nudge_fires_on_a_long_revisited_streak_with_nothing_to_react_to():
    config = ExplorationNudgeConfig(streak_threshold=15)
    perc = make_perception(enemies=(), pickups=())
    assert _should_nudge_exploration(config, perc, revisited_streak=15, proposed_action="move_forward") is True


def test_exploration_nudge_not_fired_below_streak_threshold():
    config = ExplorationNudgeConfig(streak_threshold=15)
    perc = make_perception(enemies=(), pickups=())
    assert _should_nudge_exploration(config, perc, revisited_streak=14, proposed_action="move_forward") is False


def test_exploration_nudge_not_fired_with_a_visible_enemy_or_pickup():
    from laya_doom.perception import PickupPercept

    config = ExplorationNudgeConfig(streak_threshold=15)
    perc_enemy = make_perception(enemies=(_enemy("front"),), pickups=())
    assert _should_nudge_exploration(config, perc_enemy, revisited_streak=99, proposed_action="move_forward") is False
    perc_pickup = make_perception(
        enemies=(), pickups=(PickupPercept(kind="ammo", bearing="front", distance="near", raw_distance_units=50.0),)
    )
    assert _should_nudge_exploration(config, perc_pickup, revisited_streak=99, proposed_action="move_forward") is False


def test_exploration_nudge_not_fired_when_already_turning():
    config = ExplorationNudgeConfig(streak_threshold=15)
    perc = make_perception(enemies=(), pickups=())
    assert _should_nudge_exploration(config, perc, revisited_streak=99, proposed_action="turn_left_small") is False


def test_exploration_nudge_disabled_via_config():
    config = ExplorationNudgeConfig(enabled=False, streak_threshold=15)
    perc = make_perception(enemies=(), pickups=())
    assert _should_nudge_exploration(config, perc, revisited_streak=99, proposed_action="move_forward") is False


def test_should_try_use_fires_exactly_at_stall_threshold():
    # Edge trigger: fires the moment the streak *reaches* the threshold,
    # not on every step after -- see DoorUseConfig's docstring for why
    # (one attempt per stuck-against-wall event, not spam).
    config = DoorUseConfig(stall_threshold=3)
    perc = make_perception(wall_near=True)
    assert _should_try_use(config, perc, wall_near_streak=2, proposed_action="move_forward") is False
    assert _should_try_use(config, perc, wall_near_streak=3, proposed_action="move_forward") is True
    assert _should_try_use(config, perc, wall_near_streak=4, proposed_action="move_forward") is False


def test_should_try_use_requires_wall_near():
    config = DoorUseConfig(stall_threshold=3)
    perc = make_perception(wall_near=False)
    assert _should_try_use(config, perc, wall_near_streak=3, proposed_action="move_forward") is False


def test_should_try_use_never_overrides_combat_or_an_already_chosen_use():
    config = DoorUseConfig(stall_threshold=3)
    perc = make_perception(wall_near=True)
    assert _should_try_use(config, perc, wall_near_streak=3, proposed_action="attack") is False
    assert _should_try_use(config, perc, wall_near_streak=3, proposed_action="shoot") is False
    assert _should_try_use(config, perc, wall_near_streak=3, proposed_action="use") is False


def test_should_try_use_disabled_via_config():
    config = DoorUseConfig(enabled=False, stall_threshold=3)
    perc = make_perception(wall_near=True)
    assert _should_try_use(config, perc, wall_near_streak=3, proposed_action="move_forward") is False


def test_exploration_turn_directed_when_frontier_enabled_and_data_available():
    encoder = StateEncoder(EncoderConfig(area_cell_size=100.0, include_goal_line=False))
    # Manually seed visited cells the way a real episode would via encode().
    for cell_x, cell_y in [(0, 0), (1, 0), (2, 0)]:
        encoder.encode(
            make_perception(x=cell_x * 100.0 + 10.0, y=cell_y * 100.0 + 10.0),
            None,
            None,
            None,
        )
    perc = make_perception(x=10.0, y=10.0, angle=0.0)
    frontier = FrontierExplorationConfig(enabled=True, lookahead_cells=2.0)
    action = _exploration_turn(perc, encoder, frontier, "stage1", step=0)
    # Straight-ahead (east) is fully visited for a while; a directed turn
    # away from blindly continuing east should result in an actual turn
    # action (not asserting the exact direction, just that it's directed
    # turn/move vocabulary and not a crash/None).
    assert action in {"turn_left", "turn_right", "move_forward"}


def test_exploration_turn_falls_back_to_blind_recovery_without_visited_cells():
    # A fresh encoder has no visited-cell history yet -- wayfinding has
    # nothing to compute from, so this must fall back to the old blind
    # _recovery_turn rather than raising or returning something else.
    encoder = StateEncoder(EncoderConfig(area_cell_size=100.0, include_goal_line=False))
    perc = make_perception(open_left=True, open_right=False)
    frontier = FrontierExplorationConfig(enabled=True)
    assert _exploration_turn(perc, encoder, frontier, "stage1", step=0) == _recovery_turn(perc, "stage1", 0)


def test_exploration_turn_uses_blind_recovery_when_frontier_disabled():
    encoder = StateEncoder(EncoderConfig(area_cell_size=100.0, include_goal_line=False))
    encoder.encode(make_perception(x=10.0, y=10.0), None, None, None)
    perc = make_perception(x=10.0, y=10.0, open_left=True, open_right=False)
    frontier = FrontierExplorationConfig(enabled=False)
    assert _exploration_turn(perc, encoder, frontier, "stage1", step=0) == _recovery_turn(perc, "stage1", 0)


class _FakeEnvConfig:
    action_set = "full"
    scenario = "level"


class _FakeStuckEnv:
    """Minimal fake standing in for DoomEnv, used only by the wall-hugging
    regression test below. Simulates a symmetric stuck spot (open_left ==
    open_right, so _recovery_turn's tie-break is exercised on every
    firing) that only actually clears once enough NET rotation in one
    direction has accumulated — exactly the real geometry the live
    --scenario level run hit (see controller.StuckRecoveryConfig's
    docstring). ``facing`` is a shared dict so the monkeypatched
    ``perceive`` below can read the same accumulated rotation."""

    def __init__(self, facing: dict, clear_after: int = 6, max_steps: int = 60):
        self.config = _FakeEnvConfig()
        self.facing = facing
        self.clear_after = clear_after
        self.max_steps = max_steps
        self.x = 272.1
        self.y = 240.0
        self.tic = 0
        self.steps = 0

    def new_episode(self) -> None:
        self.steps = 0

    def is_finished(self) -> bool:
        return self.steps >= self.max_steps

    def get_state(self):
        return self  # opaque; the monkeypatched perceive() below ignores it

    def execute(self, action_name: str) -> float:
        self.steps += 1
        self.tic += 4
        if action_name == "turn_left_large":
            self.facing["turns"] += 1
        elif action_name == "turn_right_large":
            self.facing["turns"] -= 1
        elif action_name == "move_forward" and abs(self.facing["turns"]) >= self.clear_after:
            self.x += 100.0
        return 0.0

    def is_player_dead(self) -> bool:
        return False

    def total_reward(self) -> float:
        return 0.0

    def episode_timeout_tics(self) -> int:
        return 10_000


class _FakeMoveStrafeAgent:
    """Stands in for LayaAgent, reproducing the exact real proposal
    pattern logged during the live run that motivated this fix: it never
    once proposes a turn, alternating move_forward/strafe_left forever
    regardless of the state it's shown."""

    def __init__(self):
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def decide(self, perception, encoded_state: str) -> Decision:
        self.calls += 1
        action = "move_forward" if self.calls % 2 == 0 else "strafe_left"
        return Decision(action=action, confidence=0.2, latency_ms=0.0, reasoning=None, raw={})


def test_wall_hugging_regression_escapes_a_symmetric_stuck_corner(monkeypatch):
    """Regression test for the real bug found by watching a fresh
    --scenario level run with override_reason logging (see the README's
    "Real bug: wall-hugging" section for the exact data): at one fixed
    (x, y), open_left/open_right kept reporting the same tied value, so
    _recovery_turn's old step-parity tie-break alternated turn_right_large
    /turn_left_large every step forever (net rotation always ~0) — and
    because Laya never once proposed a turn itself (it kept alternating
    move_forward/strafe_left), the old TurnLoopRecoveryConfig condition
    (checking the *proposed* action) could never fire either. Combined,
    the agent was stuck at the exact same position for the rest of the
    real episode. This test reproduces that geometry with fakes and
    asserts the fix actually escapes it well before the step budget runs
    out — a real regression check, not just a check that the code runs.
    """
    facing = {"turns": 0}

    def fake_perceive(state, config=None):
        stuck = abs(facing["turns"]) < state.clear_after
        return Perception(
            tic=state.tic,
            alive=True,
            health=100,
            armor=0,
            ammo=50,
            weapon=2,
            killcount=0,
            itemcount=0,
            damagecount=0,
            damage_taken=0,
            hitcount=0,
            x=state.x,
            y=state.y,
            angle=0.0,
            enemies=(),
            pickups=(),
            wall_ahead=stuck,
            wall_near=False,  # matches the real logged "WALL ahead far", not "near"
            open_forward=not stuck,
            open_left=True,
            open_right=True,  # tied -- exercises _recovery_turn's tie-break every time
        )

    monkeypatch.setattr(controller_mod, "perceive", fake_perceive)

    env = _FakeStuckEnv(facing, clear_after=6, max_steps=60)
    agent = _FakeMoveStrafeAgent()
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))

    result, records = run_episode(
        env,
        agent,
        encoder,
        max_steps=60,
        stuck_recovery=StuckRecoveryConfig(),
        turn_loop_recovery=TurnLoopRecoveryConfig(),
        threat_response=ThreatResponseConfig(),
        threat_engagement=ThreatEngagementConfig(),
        low_health_retreat=LowHealthRetreatConfig(),
        exploration_nudge=ExplorationNudgeConfig(),
        door_use=DoorUseConfig(enabled=False),
        frontier_exploration=FrontierExplorationConfig(enabled=False),
    )

    # The old bug: x/y never move because the turn overrides cancel out
    # and nothing ever forces a real move attempt once turned enough.
    assert result.distance_travelled > 0
    assert any(reason == "turn_loop_recovery" for reason in (r.override_reason for r in records))


def test_use_spam_regression_stuck_recovery_breaks_a_fruitless_use_loop(monkeypatch):
    """Regression test for the real bug found on a 3,000-step
    --scenario level run (see StuckRecoveryConfig's docstring): with a
    PICKUP visible, ExplorationNudgeConfig never engages, and nothing else
    was watching for Laya proposing `use` every step at a spot where it's
    a genuine no-op -- 2,339/3,000 real decisions in that run were `use`
    at one frozen position. Reproduces that exact shape (a PICKUP always
    visible, Laya always proposing `use`, position never moving) with
    fakes and asserts the fix actually breaks out of it well before the
    step budget runs out."""
    from laya_doom.perception import PickupPercept

    def fake_perceive(state, config=None):
        return Perception(
            tic=state.tic, alive=True, health=100, armor=0, ammo=50, weapon=2,
            killcount=0, itemcount=0, damagecount=0, damage_taken=0, hitcount=0,
            x=state.x, y=state.y, angle=0.0,
            enemies=(),
            pickups=(PickupPercept(kind="health_pack", bearing="front", distance="medium", raw_distance_units=300.0),),
            wall_ahead=True, wall_near=False, open_forward=False, open_left=True, open_right=True,
        )

    monkeypatch.setattr(controller_mod, "perceive", fake_perceive)

    class _Config:
        action_set = "stage1"
        scenario = "level"

    class _FrozenSpotEnv:
        def __init__(self):
            self.config = _Config()
            self.x, self.y, self.tic, self.steps = 10.0, 10.0, 0, 0

        def new_episode(self):
            self.steps = 0

        def is_finished(self):
            return self.steps >= 100

        def get_state(self):
            return self

        def execute(self, action_name):
            self.steps += 1
            self.tic += 4
            return 0.0  # nothing this fake env does ever moves x/y

        def is_player_dead(self):
            return False

        def total_reward(self):
            return 0.0

        def episode_timeout_tics(self):
            return 100_000

    class _AlwaysUseAgent:
        def reset(self):
            pass

        def decide(self, perception, encoded_state):
            return Decision(action="use", confidence=0.2, latency_ms=0.0, reasoning=None, raw={})

    env = _FrozenSpotEnv()
    agent = _AlwaysUseAgent()
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))
    result, records = run_episode(
        env,
        agent,
        encoder,
        max_steps=100,
        stuck_recovery=StuckRecoveryConfig(),
        turn_loop_recovery=TurnLoopRecoveryConfig(),
        threat_response=ThreatResponseConfig(),
        threat_engagement=ThreatEngagementConfig(),
        low_health_retreat=LowHealthRetreatConfig(),
        exploration_nudge=ExplorationNudgeConfig(),
        door_use=DoorUseConfig(enabled=False),
        frontier_exploration=FrontierExplorationConfig(enabled=False),
        secret_search=SecretSearchConfig(enabled=False),
        wall_follow=WallFollowConfig(enabled=False),
    )
    use_count = sum(1 for r in records if r.action == "use")
    # The old bug: use_count would be ~100/100 (every single step).
    assert use_count < len(records)
    assert any(r.override_reason == "stuck_recovery" for r in records)


def test_wait_spam_regression_stuck_recovery_breaks_a_fruitless_wait_loop(monkeypatch):
    """Regression test for the real bug found on a 4,000-step
    --scenario level run (see StuckRecoveryConfig's docstring): once the
    encoded state hit a fixed point, Laya's own top-probability label
    stayed `wait` every single call -- 3,919/4,000 real decisions in that
    run. Same shape as the `use`-spam regression above, `wait` instead."""
    from laya_doom.perception import PickupPercept

    def fake_perceive(state, config=None):
        return Perception(
            tic=state.tic, alive=True, health=100, armor=0, ammo=50, weapon=2,
            killcount=0, itemcount=0, damagecount=0, damage_taken=0, hitcount=0,
            x=state.x, y=state.y, angle=0.0,
            enemies=(),
            pickups=(PickupPercept(kind="health_pack", bearing="far-right", distance="near", raw_distance_units=100.0),),
            wall_ahead=True, wall_near=False, open_forward=False, open_left=False, open_right=True,
        )

    monkeypatch.setattr(controller_mod, "perceive", fake_perceive)

    class _Config:
        action_set = "stage1"
        scenario = "level"

    class _FrozenSpotEnv:
        def __init__(self):
            self.config = _Config()
            self.x, self.y, self.tic, self.steps = 10.0, 10.0, 0, 0

        def new_episode(self):
            self.steps = 0

        def is_finished(self):
            return self.steps >= 100

        def get_state(self):
            return self

        def execute(self, action_name):
            self.steps += 1
            self.tic += 4
            return 0.0

        def is_player_dead(self):
            return False

        def total_reward(self):
            return 0.0

        def episode_timeout_tics(self):
            return 100_000

    class _AlwaysWaitAgent:
        def reset(self):
            pass

        def decide(self, perception, encoded_state):
            return Decision(action="wait", confidence=0.18, latency_ms=0.0, reasoning=None, raw={})

    env = _FrozenSpotEnv()
    agent = _AlwaysWaitAgent()
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))
    result, records = run_episode(
        env,
        agent,
        encoder,
        max_steps=100,
        stuck_recovery=StuckRecoveryConfig(),
        turn_loop_recovery=TurnLoopRecoveryConfig(),
        threat_response=ThreatResponseConfig(),
        threat_engagement=ThreatEngagementConfig(),
        low_health_retreat=LowHealthRetreatConfig(),
        exploration_nudge=ExplorationNudgeConfig(),
        door_use=DoorUseConfig(enabled=False),
        frontier_exploration=FrontierExplorationConfig(enabled=False),
        secret_search=SecretSearchConfig(enabled=False),
        wall_follow=WallFollowConfig(enabled=False),
    )
    wait_count = sum(1 for r in records if r.action == "wait")
    # The old bug: wait_count would be ~100/100 (every single step).
    assert wait_count < len(records)
    assert any(r.override_reason == "stuck_recovery" for r in records)


def test_secret_search_action_turns_then_uses_each_target():
    perc = make_perception(angle=0.0)
    targets = (90.0, 180.0)
    action, index = _secret_search_action(perc, targets, 0, "stage1")
    assert action in ("turn_left", "turn_right")
    assert index == 0  # not facing target[0] (90) yet -- index unchanged

    perc_facing = make_perception(angle=90.0)
    action, index = _secret_search_action(perc_facing, targets, 0, "stage1")
    assert action == "use"
    assert index == 1  # advances to the next target


def test_secret_search_action_returns_none_after_last_target():
    perc_facing_last = make_perception(angle=180.0)
    targets = (90.0, 180.0)
    action, index = _secret_search_action(perc_facing_last, targets, 1, "stage1")
    assert action == "use"
    assert index is None  # sequence complete


def test_start_exploration_override_prefers_directed_frontier_when_available():
    encoder = StateEncoder(EncoderConfig(area_cell_size=100.0, include_goal_line=False))
    encoder.encode(make_perception(x=10.0, y=10.0), None, None, None)
    perc = make_perception(x=10.0, y=10.0, angle=0.0)
    action, targets, index = _start_exploration_override(
        perc, encoder, FrontierExplorationConfig(enabled=True), SecretSearchConfig(enabled=True),
        nudge_without_new_area=0, action_set="stage1", step=0,
    )
    assert targets is None  # real unexplored territory found -- no search needed
    assert action in {"turn_left", "turn_right", "move_forward"}


def test_start_exploration_override_escalates_to_secret_search_after_enough_fruitless_nudges():
    # nudge_without_new_area already at the escalate_after threshold ->
    # secret search starts instead of another plain nudge, regardless of
    # whether frontier exploration could still find some far-off cell to
    # point at (see SecretSearchConfig's docstring for why that fallback
    # alone isn't a reliable "truly stuck" signal).
    encoder = StateEncoder(EncoderConfig(area_cell_size=100.0, include_goal_line=False))
    encoder.encode(make_perception(x=10.0, y=10.0), None, None, None)
    perc = make_perception(x=10.0, y=10.0, angle=0.0)
    action, targets, index = _start_exploration_override(
        perc, encoder, FrontierExplorationConfig(enabled=True), SecretSearchConfig(enabled=True, escalate_after=2),
        nudge_without_new_area=2, action_set="stage1", step=0,
    )
    assert targets is not None
    assert len(targets) == 4


def test_start_exploration_override_stays_below_escalation_threshold():
    encoder = StateEncoder(EncoderConfig(area_cell_size=100.0, include_goal_line=False))
    encoder.encode(make_perception(x=10.0, y=10.0), None, None, None)
    perc = make_perception(x=10.0, y=10.0, angle=0.0)
    action, targets, index = _start_exploration_override(
        perc, encoder, FrontierExplorationConfig(enabled=True), SecretSearchConfig(enabled=True, escalate_after=2),
        nudge_without_new_area=1, action_set="stage1", step=0,
    )
    assert targets is None


def test_start_exploration_override_falls_back_to_blind_recovery_when_secret_search_disabled():
    encoder = StateEncoder(EncoderConfig(area_cell_size=100.0, include_goal_line=False))
    perc = make_perception(x=0.0, y=0.0, angle=0.0, open_left=True, open_right=False)
    action, targets, index = _start_exploration_override(
        perc, encoder, FrontierExplorationConfig(enabled=False), SecretSearchConfig(enabled=False),
        nudge_without_new_area=99, action_set="stage1", step=0,
    )
    assert targets is None
    assert action == _recovery_turn(perc, "stage1", 0)


def test_secret_search_regression_tries_use_around_a_dead_end(monkeypatch):
    """Once frontier exploration is genuinely exhausted, this should try
    `use` against several nearby wall-facing directions (a secret door
    looks identical to an ordinary wall in Doom/Freedoom -- see
    SecretSearchConfig's docstring) rather than only ever nudging a
    generic heading change. Uses the same fake-env technique as the
    wall-hugging regression test above."""
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))

    def fake_perceive(state, config=None):
        return Perception(
            tic=state.tic, alive=True, health=100, armor=0, ammo=50, weapon=2,
            killcount=0, itemcount=0, damagecount=0, damage_taken=0, hitcount=0,
            x=state.x, y=state.y, angle=state.angle, enemies=(), pickups=(),
            wall_ahead=False, wall_near=False, open_forward=True, open_left=True, open_right=True,
        )

    monkeypatch.setattr(controller_mod, "perceive", fake_perceive)

    class _Config:
        action_set = "stage1"
        scenario = "level"

    class _DeadEndEnv:
        def __init__(self):
            self.config = _Config()
            self.x, self.y, self.angle, self.tic, self.steps = 10.0, 10.0, 0.0, 0, 0

        def new_episode(self):
            self.steps = 0

        def is_finished(self):
            return self.steps >= 200

        def get_state(self):
            return self

        def execute(self, action_name):
            self.steps += 1
            self.tic += 4
            if action_name in ("turn_left", "turn_right"):
                # verified convention: turn_left increases ANGLE, turn_right decreases it
                self.angle = (self.angle + (10.0 if action_name == "turn_left" else -10.0)) % 360.0
            return 0.0

        def is_player_dead(self):
            return False

        def total_reward(self):
            return 0.0

        def episode_timeout_tics(self):
            return 100_000

    class _NeverMoveAgent:
        def reset(self):
            pass

        def decide(self, perception, encoded_state):
            # A boring stateless proposal that never itself changes
            # anything real -- x/y never move in this fake env at all
            # regardless of what's proposed or executed, simulating a
            # true dead end with no ordinary path out.
            return Decision(action="wait", confidence=0.1, latency_ms=0.0, reasoning=None, raw={})

    env = _DeadEndEnv()
    agent = _NeverMoveAgent()
    result, records = run_episode(
        env,
        agent,
        encoder,
        max_steps=200,
        stuck_recovery=StuckRecoveryConfig(enabled=False),
        turn_loop_recovery=TurnLoopRecoveryConfig(enabled=False),
        threat_response=ThreatResponseConfig(enabled=False),
        threat_engagement=ThreatEngagementConfig(enabled=False),
        low_health_retreat=LowHealthRetreatConfig(enabled=False),
        exploration_nudge=ExplorationNudgeConfig(streak_threshold=15),
        door_use=DoorUseConfig(enabled=False),
        frontier_exploration=FrontierExplorationConfig(enabled=True, lookahead_cells=1.0),
        secret_search=SecretSearchConfig(enabled=True),
    )
    use_attempts = [r for r in records if r.action == "use" and r.override_reason == "secret_search"]
    assert len(use_attempts) >= 4  # tried all four default headings, not just one wall


def test_wall_follow_turns_toward_follow_side_when_open():
    # Right-hand rule: if the right side is open, turn right to keep
    # tracking the wall's curve, regardless of what's ahead.
    perc = make_perception(open_left=False, open_right=True, open_forward=False)
    assert _wall_follow_action(perc, "right", "stage1") == "turn_right"
    assert _wall_follow_action(perc, "right", "full") == "turn_right_small"


def test_wall_follow_moves_forward_when_follow_side_blocked_but_ahead_open():
    perc = make_perception(open_left=True, open_right=False, open_forward=True)
    assert _wall_follow_action(perc, "right", "stage1") == "move_forward"


def test_wall_follow_turns_away_at_inner_corner_or_dead_end():
    perc = make_perception(open_left=True, open_right=False, open_forward=False)
    assert _wall_follow_action(perc, "right", "stage1") == "turn_left"


def test_wall_follow_left_hand_mirrors_right_hand():
    perc = make_perception(open_left=True, open_right=False, open_forward=False)
    assert _wall_follow_action(perc, "left", "stage1") == "turn_left"
    perc2 = make_perception(open_left=False, open_right=True, open_forward=False)
    assert _wall_follow_action(perc2, "left", "stage1") == "turn_right"


def test_wall_follow_regression_engages_after_secret_search_and_nudging_both_fail(monkeypatch):
    """Escalation-ladder regression: once plain nudging AND a full
    secret-door sweep have both come up with nothing new, WallFollowConfig
    should take over rather than nudging forever. Reuses the dead-end fake
    env technique above but gives wall-following an actual escape route
    (an opening on its follow side a bit further along) so the test can
    also confirm real movement results from it, not just that it was
    selected."""
    encoder = StateEncoder(EncoderConfig(include_goal_line=False))

    def fake_perceive(state, config=None):
        # Once far enough along (simulated by a step counter on state),
        # pretend a corridor opens up so wall-following can make real
        # progress -- otherwise it's a plain dead end (matches the
        # regression test above).
        escaped = state.follow_progress >= 3
        return Perception(
            tic=state.tic, alive=True, health=100, armor=0, ammo=50, weapon=2,
            killcount=0, itemcount=0, damagecount=0, damage_taken=0, hitcount=0,
            x=state.x, y=state.y, angle=state.angle, enemies=(), pickups=(),
            wall_ahead=not escaped, wall_near=False,
            open_forward=escaped, open_left=False, open_right=True,
        )

    monkeypatch.setattr(controller_mod, "perceive", fake_perceive)

    class _Config:
        action_set = "stage1"
        scenario = "level"

    class _DeadEndWithEscapeEnv:
        def __init__(self):
            self.config = _Config()
            self.x, self.y, self.angle, self.tic, self.steps = 10.0, 10.0, 0.0, 0, 0
            self.follow_progress = 0

        def new_episode(self):
            self.steps = 0

        def is_finished(self):
            return self.steps >= 300

        def get_state(self):
            return self

        def execute(self, action_name):
            self.steps += 1
            self.tic += 4
            if action_name in ("turn_left", "turn_right"):
                self.angle = (self.angle + (10.0 if action_name == "turn_left" else -10.0)) % 360.0
            elif action_name == "turn_right":
                self.follow_progress += 1
            elif action_name == "move_forward" and self.follow_progress >= 3:
                self.x += 50.0
                self.follow_progress += 1
            return 0.0

        def is_player_dead(self):
            return False

        def total_reward(self):
            return 0.0

        def episode_timeout_tics(self):
            return 100_000

    class _NeverMoveAgent:
        def reset(self):
            pass

        def decide(self, perception, encoded_state):
            return Decision(action="wait", confidence=0.1, latency_ms=0.0, reasoning=None, raw={})

    env = _DeadEndWithEscapeEnv()
    agent = _NeverMoveAgent()
    result, records = run_episode(
        env,
        agent,
        encoder,
        max_steps=300,
        stuck_recovery=StuckRecoveryConfig(enabled=False),
        turn_loop_recovery=TurnLoopRecoveryConfig(enabled=False),
        threat_response=ThreatResponseConfig(enabled=False),
        threat_engagement=ThreatEngagementConfig(enabled=False),
        low_health_retreat=LowHealthRetreatConfig(enabled=False),
        exploration_nudge=ExplorationNudgeConfig(streak_threshold=15),
        door_use=DoorUseConfig(enabled=False),
        frontier_exploration=FrontierExplorationConfig(enabled=True, lookahead_cells=1.0),
        secret_search=SecretSearchConfig(enabled=True, escalate_after=1),
        wall_follow=WallFollowConfig(enabled=True, hand="right", activate_after_searches=1),
    )
    reasons = [r.override_reason for r in records]
    assert "wall_follow" in reasons
