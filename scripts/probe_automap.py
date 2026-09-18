"""Real investigation of ViZDoom's automap buffer -- enable it on a real
``level`` scenario, capture real buffers before/after moving and turning,
and do simple numpy analysis of what's actually in them (see the README's
"Automap buffer investigation" section for the conclusion this fed into:
enabling it, real captured images, real per-pixel analysis, and an honest
decision NOT to fold it into the decision pipeline).

Dumps PNGs under the path given by --out-dir so they can actually be
looked at, not just summarized numerically -- this project's own "verified,
not assumed" rule applies to this investigation too. Needs Pillow, which
is NOT a shipped dependency of this project (only this probe script uses
it): ``uv pip install --python .venv/bin/python pillow`` first, or the
image-saving step is skipped gracefully.

Run:

    python -m scripts.probe_automap --out-dir /tmp/automap_probe
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import vizdoom as vzd

from laya_doom import actions as actions_mod
from laya_doom.doom_env import DoomEnv, DoomEnvConfig


def _save_png(buf: np.ndarray, path: str) -> None:
    try:
        from PIL import Image
    except ImportError:
        print(f"(Pillow not installed -- skipping image save for {path})")
        return
    Image.fromarray(buf).save(path)
    print(f"saved {path}  shape={buf.shape} dtype={buf.dtype}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="/tmp/automap_probe")
    p.add_argument(
        "--mode",
        choices=["NORMAL", "WHOLE", "OBJECTS", "OBJECTS_WITH_SIZE"],
        default="NORMAL",
        help="real vzd.AutomapMode values, confirmed against the installed vizdoom package -- not guessed",
    )
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    env = DoomEnv(
        DoomEnvConfig(
            scenario="level",
            action_set="full",
            window_visible=False,
            automap_buffer_enabled=True,
            automap_mode=getattr(vzd.AutomapMode, args.mode),
        )
    )
    env.new_episode()
    table = actions_mod.build_action_table()

    def dump(label: str) -> None:
        state = env.get_state()
        buf = state.automap_buffer
        if buf is None:
            print(f"{label}: automap_buffer is None")
            return
        _save_png(buf, os.path.join(args.out_dir, f"{label}.png"))
        arr = buf.astype(np.int32)
        # The background fill color (top-left pixel is always background
        # in these captures) is NOT literally black -- see the README:
        # revealed wall-line geometry is drawn in a different color, but
        # unrevealed regions use the SAME fill as revealed-but-empty
        # floor space. "black_frac" below is kept only as a sanity check
        # that this fill isn't RGB (0,0,0), which it isn't.
        black_frac = (arr.sum(axis=-1) == 0).mean() if arr.ndim == 3 else (arr == 0).mean()
        bg = tuple(arr[0, 0].tolist()) if arr.ndim == 3 else int(arr[0, 0])
        nonbg_frac = (arr != bg).any(axis=-1).mean() if arr.ndim == 3 else (arr != bg).mean()
        print(f"{label}: literal-black-pixel fraction={black_frac:.4f}  bg-color={bg}  non-bg (drawn) fraction={nonbg_frac:.4f}")

    dump("00_start")
    moves = (["move_forward"] * 8 + ["turn_left"] * 3) * 6
    for m in moves:
        if env.is_finished():
            break
        env.execute(m)
    dump("01_after_moving")
    for _ in range(10):
        env.execute("turn_left")
    dump("02_after_turning")

    # Real, printed negative result kept here rather than hidden: ViZDoom's
    # console does not implement the classic Doom "iddt" cheat.
    try:
        env.game.send_game_command("iddt")
        dump("03_after_iddt_attempt")
    except Exception as exc:  # pragma: no cover -- investigation script
        print(f"send_game_command('iddt') raised: {exc}")

    env.close()
    print("\ndone")


if __name__ == "__main__":
    main()
