"""Empirically verify ViZDoom's ANGLE game-variable sign convention before
trusting it for anything (see laya_doom/wayfinding.py's module docstring
and the README's "Verified: ANGLE's real sign convention" section for how
the printed output below is used).

Starts a real headless DoomEnv (``basic`` scenario, so turning is
unobstructed), reads ANGLE before/after executing several real
turn_left/turn_right actions via ``env.execute(...)``, and prints the real
deltas. Nothing here is assumed or invented — every number below is real,
measured output from an actual ViZDoom process.

Run (from the repo root, with the venv active or via .venv/bin/python):

    python -m scripts.verify_angle
"""

from __future__ import annotations

from laya_doom import actions as actions_mod
from laya_doom.doom_env import DoomEnv, DoomEnvConfig
from laya_doom.perception import PerceptionConfig, perceive


def main() -> None:
    env = DoomEnv(DoomEnvConfig(scenario="basic", action_set="full", window_visible=False))
    env.new_episode()
    pc = PerceptionConfig()

    def read_angle():
        p = perceive(env.get_state(), pc)
        return p.angle, p.x, p.y

    print("=== initial ===")
    a0, x0, y0 = read_angle()
    print(f"angle={a0} x={x0} y={y0}")

    print("\n=== 5x turn_left (env.execute, real ViZDoom actions) ===")
    prev = a0
    for i in range(5):
        env.execute("turn_left")
        a, x, y = read_angle()
        print(f"step {i}: angle={a:.3f} delta={a - prev:+.3f}")
        prev = a

    print("\n=== 5x turn_right (env.execute, real ViZDoom actions) ===")
    prev, _, _ = read_angle()
    for i in range(5):
        env.execute("turn_right")
        a, x, y = read_angle()
        print(f"step {i}: angle={a:.3f} delta={a - prev:+.3f}")
        prev = a

    print("\n=== raw 1-tic button press (bypassing actions.py's tic counts) ===")
    spec = actions_mod.build_action_table()["turn_left"]
    a_before, _, _ = read_angle()
    env.game.make_action(spec.button_vector(), 1)
    a_after, _, _ = read_angle()
    print(f"1 tic turn_left: angle {a_before:.3f} -> {a_after:.3f} (delta {a_after - a_before:+.3f})")

    spec = actions_mod.build_action_table()["turn_right"]
    a_before, _, _ = read_angle()
    env.game.make_action(spec.button_vector(), 1)
    a_after, _, _ = read_angle()
    print(f"1 tic turn_right: angle {a_before:.3f} -> {a_after:.3f} (delta {a_after - a_before:+.3f})")

    env.close()
    print("\ndone")


if __name__ == "__main__":
    main()
