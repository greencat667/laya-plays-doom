"""Thin wrapper around vizdoom.DoomGame: scenario/buffer/button setup, plus
``execute(action_name)`` that turns a semantic action (see actions.py) into
the right ``make_action()`` call.

No commercial Doom files are required: ViZDoom ships its own scenario WADs
(under ``vizdoom.scenarios_path``) and falls back to the bundled Freedoom2
IWAD automatically when doom2.wad isn't present on the system. The same
Freedoom2 IWAD is also used directly (no add-on scenario pwad) for
``scenario="level"``, which loads a real multi-room Freedoom map instead of
one of ViZDoom's small purpose-built scenario wads — see config/level.cfg.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import vizdoom as vzd

from . import actions as actions_mod
from .perception import TRACKED_GAME_VARIABLES

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Bundled directly in the vizdoom package (sibling to its scenarios/ dir),
# not inside vizdoom.scenarios_path — confirmed by inspecting the installed
# package rather than assumed.
FREEDOOM2_PATH = os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad")

LEVEL_SCENARIO = "level"
DEFAULT_LEVEL_MAP = "MAP01"


@dataclass
class DoomEnvConfig:
    scenario: str = "basic"
    action_set: actions_mod.ActionSet = "stage1"
    window_visible: bool = False
    uniform_tics: int | None = None
    seed: int | None = None
    doom_map: str | None = None
    screen_resolution: vzd.ScreenResolution = vzd.ScreenResolution.RES_320X240


class DoomEnv:
    def __init__(self, config: DoomEnvConfig | None = None):
        self.config = config or DoomEnvConfig()
        self.action_table = actions_mod.build_action_table(self.config.uniform_tics)
        self.game = vzd.DoomGame()
        self._configure()
        self.game.init()

    def _configure(self) -> None:
        game = self.game
        cfg_path = CONFIG_DIR / f"{self.config.scenario}.cfg"
        if cfg_path.exists():
            game.load_config(str(cfg_path))

        # Always set the scenario/map ourselves (using vizdoom's own
        # scenarios_path, or the bundled Freedoom2 IWAD for a real level) —
        # this is correct regardless of cwd/install location and overrides
        # whatever load_config guessed above.
        if self.config.scenario == LEVEL_SCENARIO:
            game.set_doom_game_path(FREEDOOM2_PATH)
            game.set_doom_map(self.config.doom_map or DEFAULT_LEVEL_MAP)
        else:
            scenario_wad = os.path.join(vzd.scenarios_path, f"{self.config.scenario}.wad")
            game.set_doom_scenario_path(scenario_wad)
            if self.config.doom_map:
                game.set_doom_map(self.config.doom_map)

        game.set_screen_resolution(self.config.screen_resolution)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_depth_buffer_enabled(True)
        game.set_labels_buffer_enabled(True)
        game.set_automap_buffer_enabled(False)
        game.set_audio_buffer_enabled(False)
        game.set_objects_info_enabled(False)
        game.set_sectors_info_enabled(False)

        game.set_available_buttons(list(actions_mod.ALL_BUTTONS))
        game.set_available_game_variables(list(TRACKED_GAME_VARIABLES))

        game.set_window_visible(self.config.window_visible)
        game.set_mode(vzd.Mode.PLAYER)
        if self.config.seed is not None:
            game.set_seed(self.config.seed)

    def new_episode(self) -> None:
        self.game.new_episode()

    def get_state(self):
        return self.game.get_state()

    def is_finished(self) -> bool:
        return self.game.is_episode_finished()

    def is_player_dead(self) -> bool:
        return self.game.is_player_dead()

    def total_reward(self) -> float:
        return self.game.get_total_reward()

    def episode_timeout_tics(self) -> int:
        return self.game.get_episode_timeout()

    def execute(self, action_name: str) -> float:
        """Execute a canonical semantic action (see actions.py) for its
        configured number of tics. Falls back to "wait" for unknown names.
        Returns the reward ViZDoom reports for those tics."""
        spec = self.action_table.get(action_name, self.action_table["wait"])
        return self.game.make_action(spec.button_vector(), spec.tics)

    def close(self) -> None:
        self.game.close()

    def __enter__(self) -> "DoomEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
