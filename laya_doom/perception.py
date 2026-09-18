"""ViZDoom GameState -> structured Perception.

Deliberately reads only what ViZDoom already computes for us (game
variables, the labels buffer, the depth buffer) — no screen-buffer pixels,
no CNN. Enemies and pickups are limited to what's in ``state.labels``, i.e.
what's currently visible/in-FOV; this keeps perception honest about partial
observability instead of giving Laya god's-eye knowledge of the level, and
sidesteps ambiguity in ViZDoom's player-angle sign convention entirely by
using screen-space bounding-box position for bearing instead of trig.

Depth-buffer calibration: per the ViZDoom maintainers (Farama-Foundation/
ViZDoom#192), depth_buffer values are ~linear in distance at roughly
14 buffer units per 100 game units in the center of the screen, i.e.
1 depth-buffer unit ~= 7.14 game units. Distance thresholds below are
specified in game units and converted with that constant; they are
approximate (bilinear falloff towards screen edges) and meant to be
re-tuned by eye using the dashboard, not treated as exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

GAME_UNITS_PER_DEPTH_UNIT = 100.0 / 14.0

BEARING_BUCKETS: tuple[str, ...] = (
    "far-left",
    "left",
    "front-left",
    "front",
    "front-right",
    "right",
    "far-right",
)

DISTANCE_BUCKETS: tuple[str, ...] = ("very-near", "near", "medium", "far")

# object_name (as reported by ViZDoom labels) -> compact perception token.
# Covers the Freedoom/Doom actor names that show up in the bundled
# scenarios; anything unmatched falls back to a snake_cased object_name
# rather than being silently dropped.
_MONSTER_KINDS: dict[str, str] = {
    "Zombieman": "zombieman",
    "ShotgunGuy": "shotgun_guy",
    "ChaingunGuy": "chaingunner",
    "Imp": "imp",
    "Demon": "demon",
    "Spectre": "spectre",
    "Cacodemon": "cacodemon",
    "LostSoul": "lost_soul",
    "HellKnight": "hell_knight",
    "BaronOfHell": "baron_of_hell",
    "Arachnotron": "arachnotron",
    "PainElemental": "pain_elemental",
    "Revenant": "revenant",
    "Fatso": "mancubus",
    "Vile": "arch_vile",
    "Cyberdemon": "cyberdemon",
    "SpiderMastermind": "spider_mastermind",
    "WolfensteinSS": "wolf_ss",
}

_PICKUP_KINDS: dict[str, str] = {
    "Stimpack": "health_pack",
    "Medikit": "health_pack",
    "HealthBonus": "health_bonus",
    "Soulsphere": "soulsphere",
    "Megasphere": "megasphere",
    "ArmorBonus": "armor_bonus",
    "GreenArmor": "armor",
    "BlueArmor": "armor",
    "Clip": "ammo",
    "ClipBox": "ammo",
    "Shell": "ammo",
    "ShellBox": "ammo",
    "RocketAmmo": "ammo",
    "RocketBox": "ammo",
    "Cell": "ammo",
    "CellPack": "ammo",
    "Backpack": "backpack",
    "Shotgun": "shotgun_pickup",
    "SuperShotgun": "shotgun_pickup",
    "Chaingun": "chaingun_pickup",
    "RocketLauncher": "rocket_launcher_pickup",
    "PlasmaRifle": "plasma_rifle_pickup",
    "BFG9000": "bfg_pickup",
    "Chainsaw": "chainsaw_pickup",
    "Berserk": "berserk",
    "InvulnerabilitySphere": "invulnerability",
    "BlurSphere": "invisibility",
    "RadSuit": "rad_suit",
    "Infrared": "light_amp",
    "BlueCard": "blue_key",
    "RedCard": "red_key",
    "YellowCard": "yellow_key",
    "BlueSkull": "blue_key",
    "RedSkull": "red_key",
    "YellowSkull": "yellow_key",
    "DoomSkull": "yellow_key",
}

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _snake(name: str) -> str:
    return _CAMEL_RE.sub("_", name).lower()


def kind_of(object_name: str, object_category: str | None) -> tuple[str, str] | None:
    """Classify a label. Returns (role, kind) where role is "enemy" or
    "pickup", or None if the object is neither (walls, decorations, the
    player itself, dead corpses, etc.)."""
    if object_name in _MONSTER_KINDS:
        return "enemy", _MONSTER_KINDS[object_name]
    if object_name in _PICKUP_KINDS:
        return "pickup", _PICKUP_KINDS[object_name]
    if object_category == "Monsters":
        return "enemy", _snake(object_name)
    if object_category in ("Items", "Weapons", "Powerups", "Ammo", "Health", "Armor"):
        return "pickup", _snake(object_name)
    return None


@dataclass(frozen=True)
class EnemyPercept:
    kind: str
    bearing: str
    distance: str
    raw_distance_units: float


@dataclass(frozen=True)
class PickupPercept:
    kind: str
    bearing: str
    distance: str
    raw_distance_units: float


@dataclass(frozen=True)
class Perception:
    tic: int
    alive: bool
    health: int
    armor: int
    ammo: int
    weapon: int
    killcount: int
    itemcount: int
    damagecount: int
    damage_taken: int
    hitcount: int
    x: float
    y: float
    angle: float
    enemies: tuple[EnemyPercept, ...]
    pickups: tuple[PickupPercept, ...]
    wall_ahead: bool
    wall_near: bool
    open_forward: bool
    open_left: bool
    open_right: bool


@dataclass
class PerceptionConfig:
    max_enemies: int = 3
    max_pickups: int = 2
    distance_thresholds_game_units: dict[str, float] = field(
        default_factory=lambda: {"very-near": 64.0, "near": 192.0, "medium": 512.0}
    )
    # Row/columns for the wall/open-path depth scan, as fractions of buffer size.
    scan_row_frac: float = 0.5
    scan_col_fracs: dict[str, float] = field(
        default_factory=lambda: {"forward": 0.5, "left": 0.15, "right": 0.85}
    )
    wall_near_game_units: float = 48.0
    wall_far_game_units: float = 256.0

    def distance_bucket(self, raw_depth_value: int) -> tuple[str, float]:
        game_units = raw_depth_value * GAME_UNITS_PER_DEPTH_UNIT
        t = self.distance_thresholds_game_units
        if game_units <= t["very-near"]:
            bucket = "very-near"
        elif game_units <= t["near"]:
            bucket = "near"
        elif game_units <= t["medium"]:
            bucket = "medium"
        else:
            bucket = "far"
        return bucket, game_units


def bearing_bucket(center_x: float, screen_width: int) -> str:
    ratio = min(max(center_x / max(screen_width, 1), 0.0), 0.999999)
    index = int(ratio * len(BEARING_BUCKETS))
    return BEARING_BUCKETS[index]


def _wall_scan(depth_buffer, config: PerceptionConfig):
    """Returns (wall_ahead, wall_near, open_forward, open_left, open_right)."""
    if depth_buffer is None:
        return False, False, True, True, True

    height, width = depth_buffer.shape[:2]
    row = min(max(int(height * config.scan_row_frac), 0), height - 1)

    def sample(col_frac: float) -> float:
        col = min(max(int(width * col_frac), 0), width - 1)
        return float(depth_buffer[row, col]) * GAME_UNITS_PER_DEPTH_UNIT

    forward_units = sample(config.scan_col_fracs["forward"])
    left_units = sample(config.scan_col_fracs["left"])
    right_units = sample(config.scan_col_fracs["right"])

    wall_ahead = forward_units <= config.wall_far_game_units
    wall_near = forward_units <= config.wall_near_game_units
    open_forward = forward_units > config.wall_far_game_units
    open_left = left_units > config.wall_near_game_units
    open_right = right_units > config.wall_near_game_units
    return wall_ahead, wall_near, open_forward, open_left, open_right


def perceive(state, config: PerceptionConfig | None = None) -> Perception | None:
    """Convert a vizdoom.GameState (or a duck-typed stand-in, for tests)
    into a Perception. Returns None if state is None (episode finished)."""
    if state is None:
        return None
    config = config or PerceptionConfig()

    gv = {var.name: value for var, value in zip(_TRACKED_VARS, state.game_variables)}

    screen_width = getattr(state.depth_buffer, "shape", [0, 0])[1] if state.depth_buffer is not None else 320

    enemies: list[EnemyPercept] = []
    pickups: list[PickupPercept] = []
    for label in state.labels or []:
        classified = kind_of(label.object_name, getattr(label, "object_category", None))
        if classified is None:
            continue
        role, kind = classified
        center_x = label.x + label.width / 2.0
        bearing = bearing_bucket(center_x, screen_width)

        raw_depth = None
        if state.depth_buffer is not None:
            cy = min(max(int(label.y + label.height / 2.0), 0), state.depth_buffer.shape[0] - 1)
            cx = min(max(int(center_x), 0), state.depth_buffer.shape[1] - 1)
            raw_depth = state.depth_buffer[cy, cx]
        distance, game_units = (
            config.distance_bucket(raw_depth) if raw_depth is not None else ("medium", 0.0)
        )

        if role == "enemy" and len(enemies) < config.max_enemies:
            enemies.append(EnemyPercept(kind=kind, bearing=bearing, distance=distance, raw_distance_units=game_units))
        elif role == "pickup" and len(pickups) < config.max_pickups:
            pickups.append(PickupPercept(kind=kind, bearing=bearing, distance=distance, raw_distance_units=game_units))

    enemies.sort(key=lambda e: e.raw_distance_units)
    pickups.sort(key=lambda p: p.raw_distance_units)

    wall_ahead, wall_near, open_forward, open_left, open_right = _wall_scan(state.depth_buffer, config)

    return Perception(
        tic=state.tic if hasattr(state, "tic") else 0,
        alive=gv.get("DEAD", 0) == 0,
        health=int(gv.get("HEALTH", 0)),
        armor=int(gv.get("ARMOR", 0)),
        ammo=int(gv.get("SELECTED_WEAPON_AMMO", 0)),
        weapon=int(gv.get("SELECTED_WEAPON", 0)),
        killcount=int(gv.get("KILLCOUNT", 0)),
        itemcount=int(gv.get("ITEMCOUNT", 0)),
        damagecount=int(gv.get("DAMAGECOUNT", 0)),
        damage_taken=int(gv.get("DAMAGE_TAKEN", 0)),
        hitcount=int(gv.get("HITCOUNT", 0)),
        x=float(gv.get("POSITION_X", 0.0)),
        y=float(gv.get("POSITION_Y", 0.0)),
        angle=float(gv.get("ANGLE", 0.0)),
        enemies=tuple(enemies[: config.max_enemies]),
        pickups=tuple(pickups[: config.max_pickups]),
        wall_ahead=wall_ahead,
        wall_near=wall_near,
        open_forward=open_forward,
        open_left=open_left,
        open_right=open_right,
    )


# Populated lazily so this module is importable without vizdoom for tests
# that only exercise pure classification/bucketing logic.
class _LazyVar:
    def __init__(self, name: str):
        self.name = name


try:
    import vizdoom as vzd

    TRACKED_GAME_VARIABLES = [
        vzd.GameVariable.HEALTH,
        vzd.GameVariable.ARMOR,
        vzd.GameVariable.DEAD,
        vzd.GameVariable.SELECTED_WEAPON,
        vzd.GameVariable.SELECTED_WEAPON_AMMO,
        vzd.GameVariable.KILLCOUNT,
        vzd.GameVariable.ITEMCOUNT,
        vzd.GameVariable.DAMAGECOUNT,
        vzd.GameVariable.DAMAGE_TAKEN,
        vzd.GameVariable.HITCOUNT,
        vzd.GameVariable.POSITION_X,
        vzd.GameVariable.POSITION_Y,
        vzd.GameVariable.ANGLE,
    ]
    _TRACKED_VARS = TRACKED_GAME_VARIABLES
except ImportError:  # pragma: no cover
    TRACKED_GAME_VARIABLES = []
    _TRACKED_VARS = [
        _LazyVar(n)
        for n in (
            "HEALTH",
            "ARMOR",
            "DEAD",
            "SELECTED_WEAPON",
            "SELECTED_WEAPON_AMMO",
            "KILLCOUNT",
            "ITEMCOUNT",
            "DAMAGECOUNT",
            "DAMAGE_TAKEN",
            "HITCOUNT",
            "POSITION_X",
            "POSITION_Y",
            "ANGLE",
        )
    ]
