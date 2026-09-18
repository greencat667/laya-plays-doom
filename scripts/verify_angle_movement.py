"""Second half of the ANGLE verification (see scripts/verify_angle.py):
confirms that real POSITION_X/Y deltas after ``move_forward`` match the
``(cos(angle_deg), sin(angle_deg))`` convention wayfinding.py relies on,
rather than just assuming it once the turn-direction sign is known.

Run:

    python -m scripts.verify_angle_movement
"""

from __future__ import annotations

import math

from laya_doom.doom_env import DoomEnv, DoomEnvConfig
from laya_doom.perception import PerceptionConfig, perceive


def main() -> None:
    env = DoomEnv(DoomEnvConfig(scenario="basic", action_set="full", window_visible=False))
    env.new_episode()
    pc = PerceptionConfig()

    def read():
        p = perceive(env.get_state(), pc)
        return p.angle, p.x, p.y

    a0, x0, y0 = read()
    print(f"start: angle={a0:.2f} x={x0:.2f} y={y0:.2f}")

    for _ in range(9):
        env.execute("turn_left")
    a1, x1, y1 = read()
    print(f"after turning left ~9x: angle={a1:.2f} x={x1:.2f} y={y1:.2f}")

    env.execute("move_forward")
    a2, x2, y2 = read()
    dx, dy = x2 - x1, y2 - y1
    mag = max((dx * dx + dy * dy) ** 0.5, 1e-9)
    pred_dx, pred_dy = math.cos(math.radians(a1)), math.sin(math.radians(a1))
    print(
        f"after move_forward: dx={dx:+.2f} dy={dy:+.2f}  "
        f"predicted unit dir=({pred_dx:+.3f},{pred_dy:+.3f})  actual unit dir=({dx/mag:+.3f},{dy/mag:+.3f})"
    )

    for _ in range(9):
        env.execute("turn_left")
    a3, x3, y3 = read()
    print(f"\nafter turning left another ~9x: angle={a3:.2f} x={x3:.2f} y={y3:.2f}")
    env.execute("move_forward")
    a4, x4, y4 = read()
    dx2, dy2 = x4 - x3, y4 - y3
    mag2 = max((dx2 * dx2 + dy2 * dy2) ** 0.5, 1e-9)
    pred_dx2, pred_dy2 = math.cos(math.radians(a3)), math.sin(math.radians(a3))
    print(
        f"after move_forward: dx={dx2:+.2f} dy={dy2:+.2f}  "
        f"predicted unit dir=({pred_dx2:+.3f},{pred_dy2:+.3f})  actual unit dir=({dx2/mag2:+.3f},{dy2/mag2:+.3f})"
    )

    env.close()
    print("\ndone")


if __name__ == "__main__":
    main()
