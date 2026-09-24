"""Ad-hoc probe for tuning Laya's `choice`/`noul` question wording against
8 hand-built, characteristic Doom world-states (the source of the 0/8 and
shoot-gate numbers quoted in laya_agent.py and the README). Loads the real model once and reports
REAL returned probabilities/confidence for each variant — nothing here is
invented or hand-adjusted after the fact.

Not part of the test suite (it loads the real ~800MiB model — see
tests/test_laya_agent.py for the mocked version). Run directly:

    python scripts/probe_criteria.py
"""

from __future__ import annotations

import time

import laya

from laya_doom.perception import EnemyPercept, Perception
from laya_doom.state_encoder import EncoderConfig, StateEncoder


def _perception(**overrides) -> Perception:
    base = dict(
        tic=100,
        alive=True,
        health=100,
        armor=0,
        ammo=50,
        weapon=3,
        killcount=0,
        itemcount=0,
        damagecount=0,
        damage_taken=0,
        hitcount=0,
        x=0.0,
        y=0.0,
        angle=0.0,
        enemies=(),
        pickups=(),
        wall_ahead=False,
        wall_near=False,
        open_forward=True,
        open_left=True,
        open_right=True,
    )
    base.update(overrides)
    return Perception(**base)


STATES: dict[str, Perception] = {
    "enemy_front_near": _perception(enemies=(EnemyPercept("imp", "front", "near", 150.0),)),
    "enemy_front_veryn": _perception(enemies=(EnemyPercept("imp", "front", "very-near", 50.0),)),
    "enemy_front_left": _perception(enemies=(EnemyPercept("imp", "front-left", "near", 150.0),)),
    "enemy_right": _perception(enemies=(EnemyPercept("imp", "right", "near", 150.0),)),
    "wall_ahead_near": _perception(wall_near=True, wall_ahead=True, open_forward=False),
    "open_path_no_enemy": _perception(),
    "no_ammo_enemy_front": _perception(ammo=0, enemies=(EnemyPercept("imp", "front", "near", 150.0),)),
    "low_health_enemy_far": _perception(health=15, enemies=(EnemyPercept("imp", "front", "medium", 400.0),)),
}

_encoder = StateEncoder(EncoderConfig(memory_mode="stateless"))
ENCODED: dict[str, str] = {name: _encoder.encode(p, None, None, None) for name, p in STATES.items()}


def run_choice(agent, criteria: dict[str, str], label: str) -> dict[str, str]:
    questions = {
        "action": {
            "type": "choice",
            "instructions": "Given the current Doom game world-state text below, which action should the player take right now?",
            "criteria": criteria,
        }
    }
    print(f"\n=== {label} (choice, {len(criteria)} labels: {sorted(criteria)}) ===")
    dist: dict[str, int] = {}
    results = {}
    for name, state in ENCODED.items():
        t0 = time.perf_counter()
        resp = agent.predict(state, questions)
        ms = (time.perf_counter() - t0) * 1000.0
        ans = resp["answers"]["action"]
        choice = ans["choice"]
        conf = ans["confidence"]
        probs = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])[:3]
        top3 = ", ".join(f"{k}:{v:.3f}" for k, v in probs)
        print(f"{name:22s} -> {choice:14s} conf={conf:.3f}  ({ms:.1f}ms)  top3=[{top3}]")
        dist[choice] = dist.get(choice, 0) + 1
        results[name] = choice
    print(f"distribution: {dist}")
    return results


def run_noul(agent, instructions: str, label: str) -> dict[str, float]:
    questions = {"should_shoot": {"type": "noul", "instructions": instructions}}
    print(f"\n=== {label} (noul gate) ===")
    out = {}
    for name, state in ENCODED.items():
        resp = agent.predict(state, questions)
        p = resp["answers"]["should_shoot"]["noul"]
        print(f"{name:22s} -> P(shoot)={p:.3f}")
        out[name] = p
    return out


def main() -> None:
    print("loading laya...")
    agent = laya.load("convaiinnovations/laya")

    # V1 — shipped criteria (laya_agent._ACTION_CRITERIA), stage1 subset.
    v1 = {
        "move_forward": "the path ahead isn't blocked and no enemy is lined up to shoot — the default action for making progress and exploring",
        "turn_left": "the path ahead is blocked or a wall is near, or to face an enemy that isn't directly ahead, turning left",
        "turn_right": "the path ahead is blocked or a wall is near, or to face an enemy that isn't directly ahead, turning right",
        "shoot": "an ENEMY is reported with bearing front — lined up to hit",
    }
    run_choice(agent, v1, "V1 shipped (reproduce baseline)")

    # V3 — a single binary reframe within `choice` itself: two labels only,
    # "engage" vs "reposition", to test whether a crowding fix (fewer
    # competing labels) has any effect on Laya's single-pass softmax,
    # without changing wording style at all (V1's
    # shoot wording, verbatim, just with move/turn collapsed into one
    # competing label instead of three).
    v3 = {
        "shoot": v1["shoot"],
        "reposition": "no enemy is lined up to shoot right now — move or turn to make progress, find one, or get unstuck",
    }
    run_choice(agent, v3, "V3 two-label collapse (crowding test)")

    # V4 — a wording-bias hypothesis: is `shoot` losing because the
    # OTHER labels are worded more concretely
    # and literally, catching more of the model's attention share? Make
    # every non-shoot label as terse as possible (single clause) instead
    # of shoot being the only concrete one among vaguer competitors.
    v4 = {
        "move_forward": "path open, no target",
        "turn_left": "blocked or wall — turn left",
        "turn_right": "blocked or wall — turn right",
        "shoot": "ENEMY at bearing front — shoot it now",
    }
    run_choice(agent, v4, "V4 terse-symmetric wording")

    # The `noul` gate — Laya's own separate documented primitive for
    # exactly this kind of independent binary judgment (P(true)): not
    # invented, one of Laya's three documented question types. Tests
    # whether taking shoot OUT of the crowded multi-way choice entirely
    # and asking it as its own calibrated boolean changes anything.
    run_noul(
        agent,
        "Is there an enemy directly ahead (bearing 'front') and ammo available, such that the player should shoot right now?",
        "should_shoot gate, v1 wording",
    )
    run_noul(
        agent,
        "ENEMY reported at bearing front and AMMO greater than 0: should the player shoot?",
        "should_shoot gate, literal wording",
    )

    # Crowding-hypothesis test on the `full` action set (11 labels) vs a
    # trimmed 5-label version (move_forward, turn_left_small,
    # turn_right_small, attack, use), dropping to <=5 competing labels.
    # Single state: enemy front + open path, full vocabulary.
    full_11 = {
        "move_forward": "the path ahead isn't blocked and no enemy is lined up to shoot",
        "move_backward": "retreat, e.g. overwhelmed at close range",
        "strafe_left": "sidestep left to dodge or reposition",
        "strafe_right": "sidestep right to dodge or reposition",
        "turn_left_small": "small turn left to fine-aim at an enemy not directly ahead",
        "turn_right_small": "small turn right to fine-aim at an enemy not directly ahead",
        "turn_left_large": "large turn left to scan when blocked",
        "turn_right_large": "large turn right to scan when blocked",
        "attack": "an ENEMY is reported with bearing front — lined up to hit",
        "use": "interact with a door or switch directly ahead",
        "wait": "nothing else applies",
    }
    full_5 = {
        "move_forward": full_11["move_forward"],
        "turn_left_small": full_11["turn_left_small"],
        "turn_right_small": full_11["turn_right_small"],
        "attack": full_11["attack"],
        "use": full_11["use"],
    }
    state = ENCODED["enemy_front_near"]
    questions_11 = {"action": {"type": "choice", "instructions": "Which action should the player take right now?", "criteria": full_11}}
    questions_5 = {"action": {"type": "choice", "instructions": "Which action should the player take right now?", "criteria": full_5}}
    r11 = agent.predict(state, questions_11)["answers"]["action"]
    r5 = agent.predict(state, questions_5)["answers"]["action"]
    print("\n=== crowding hypothesis: full action set, enemy front + open path ===")
    print(f"11 labels -> {r11['choice']:12s} conf={r11['confidence']:.3f}  attack_prob={r11['probabilities'].get('attack', 0):.3f}")
    print(f"5 labels  -> {r5['choice']:12s} conf={r5['confidence']:.3f}  attack_prob={r5['probabilities'].get('attack', 0):.3f}")

    check_combined_gate(agent)


def check_combined_gate(agent, threshold: float = 0.45) -> None:
    """Sanity-check the actual design going into laya_agent.py: noul gate
    decides shoot vs not, with a deterministic AMMO>0 guard layered on
    top — a deterministic safety net layered on a model decision,
    always visible, never hidden."""
    v1_movement = {
        "move_forward": "the path ahead isn't blocked and no enemy is lined up to shoot — the default action for making progress and exploring",
        "turn_left": "the path ahead is blocked or a wall is near, or to face an enemy that isn't directly ahead, turning left",
        "turn_right": "the path ahead is blocked or a wall is near, or to face an enemy that isn't directly ahead, turning right",
    }
    gate_q = {"should_shoot": {"type": "noul", "instructions": "ENEMY reported at bearing front and AMMO greater than 0: should the player shoot?"}}
    move_q = {"action": {"type": "choice", "instructions": "Which action should the player take right now?", "criteria": v1_movement}}
    ammo_by_state = {name: p.ammo for name, p in STATES.items()}
    print(f"\n=== combined gate (threshold={threshold}) + movement choice, ammo guard ===")
    for name, state in ENCODED.items():
        p_shoot = agent.predict(state, gate_q)["answers"]["should_shoot"]["noul"]
        ammo_ok = ammo_by_state[name] > 0
        want_shoot = p_shoot >= threshold
        if want_shoot and ammo_ok:
            final = "shoot"
        else:
            mv = agent.predict(state, move_q)["answers"]["action"]
            final = mv["choice"]
            if want_shoot and not ammo_ok:
                final += "  [ammo-guard blocked shoot]"
        print(f"{name:22s} p_shoot={p_shoot:.3f}  ammo_ok={ammo_ok!s:5s} -> {final}")


if __name__ == "__main__":
    main()
