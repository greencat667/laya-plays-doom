from laya_doom.perception import EnemyPercept, PickupPercept
from tests.test_state_encoder import make_perception

from experiments.heuristic_agent import HeuristicAgent


def test_shoots_enemy_dead_ahead_with_ammo():
    agent = HeuristicAgent(action_set="stage1")
    perc = make_perception(
        ammo=10, enemies=(EnemyPercept(kind="imp", bearing="front", distance="near", raw_distance_units=100.0),)
    )
    assert agent._choose(perc) == "shoot"


def test_turns_toward_enemy_not_ahead():
    agent = HeuristicAgent(action_set="stage1")
    perc = make_perception(enemies=(EnemyPercept(kind="imp", bearing="right", distance="near", raw_distance_units=100.0),))
    assert agent._choose(perc) == "turn_right"


def test_enemy_takes_priority_over_pickup():
    agent = HeuristicAgent(action_set="stage1")
    perc = make_perception(
        ammo=10,
        enemies=(EnemyPercept(kind="imp", bearing="front", distance="near", raw_distance_units=50.0),),
        pickups=(PickupPercept(kind="health_pack", bearing="front", distance="near", raw_distance_units=60.0),),
    )
    assert agent._choose(perc) == "shoot"


def test_moves_toward_pickup_dead_ahead_when_no_enemy():
    agent = HeuristicAgent(action_set="stage1")
    perc = make_perception(
        pickups=(PickupPercept(kind="health_pack", bearing="front", distance="medium", raw_distance_units=300.0),)
    )
    assert agent._choose(perc) == "move_forward"


def test_turns_toward_pickup_off_to_one_side():
    agent = HeuristicAgent(action_set="stage1")
    perc = make_perception(
        pickups=(PickupPercept(kind="armor", bearing="left", distance="medium", raw_distance_units=300.0),)
    )
    assert agent._choose(perc) == "turn_left"


def test_falls_back_to_wall_avoidance_with_no_enemy_or_pickup():
    agent = HeuristicAgent(action_set="stage1")
    perc = make_perception(wall_near=True, open_left=False, open_right=True)
    assert agent._choose(perc) == "turn_right"


def test_default_is_move_forward():
    agent = HeuristicAgent(action_set="stage1")
    perc = make_perception()
    assert agent._choose(perc) == "move_forward"
