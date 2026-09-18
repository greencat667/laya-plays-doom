# AI Doom Playbook: A Decision Manual for Completing Classic Doom

## Executive Summary

This report operationalizes the supplied brief: the objective is not to make an agent knowledgeable about Doom, but to maximize its probability of **surviving, navigating, conserving resources, solving levels, recovering from mistakes, and ultimately reaching exits** in *Doom*, *The Ultimate Doom*, and *Doom II*. fileciteturn0file0

The most important conclusion is that classic Doom should not be treated primarily as an aiming problem. It is a **continuous risk-management and navigation problem** in which movement, geometry, threat triage, resource preservation, and memory usually matter more than precision shooting. Doom's engine gives the player unusually high mobility relative to monsters; simultaneous forward and strafe movement is faster than straight running, and monster infighting can turn enemy firepower into a resource. Conversely, hitscan enemies, Arch-viles, Pain Elementals, close Lost Souls, and bad positioning can punish an agent before raw enemy health becomes relevant. citeturn20view1turn19view1turn2view0turn9view0

The recommended top-level control hierarchy is therefore:

> **AVOID IMMINENT DAMAGE → ESCAPE BAD GEOMETRY → STOP COMPOUNDING THREATS → REMOVE RELIABLE DAMAGE SOURCES → STABILIZE SPACE → SECURE ESSENTIAL RESOURCES → UPDATE ORIENTATION → ADVANCE PROGRESSION → OPTIONAL KILLS/SECRETS**

This ordering deliberately puts **movement and geometry before target elimination**. Against an Arch-vile, for example, reaching cover before its attack completes is often more important than continuing to shoot it; against a chaingunner or Spider Mastermind, breaking line of sight can stop sustained hitscan fire; against a Cyberdemon, maintaining lateral space and avoiding nearby walls takes precedence over maximizing damage. citeturn2view0turn9view0turn14view0turn13view1

The agent should operate at three timescales:

| Layer | Horizon | Main responsibility |
|---|---:|---|
| **Reflex** | frames to ~1 second | Dodge, move away from walls, break line of sight, interrupt a charging Lost Soul, avoid self-damage, react to unseen damage |
| **Tactical** | ~1–15 seconds | Choose target and weapon, reposition, retreat through a doorway, create infighting, exploit a pillar, decide whether to fight or bypass |
| **Strategic** | tens of seconds to minutes | Conserve ammunition, remember keys/doors/switches, manage pickup caches, prevent navigation loops, choose unexplored branches |

The evidence base favors **engine/source-code behavior first**, DoomWiki's mechanics and tactical analyses second, and map walkthroughs/community and speedrunning knowledge as supplementary evidence. id Software's released source exposes the weapon, monster, pickup, ammunition, and sector-special logic directly, including monster behavior, weapon code, interaction rules and line/sector triggers. citeturn0search3turn0search5turn0search17turn0search19

Speedrunning techniques should be treated selectively. Basic diagonal movement and efficient routing generalize well; SR50, extreme point-blank Cyberdemon SSG fighting, glides, Arch-vile jumps and other execution-sensitive tricks should **not** be baseline behavior. SR40 already produces about 128% of straight running speed, while SR50 reaches about 141% but constrains turning in vanilla-style controls; DoomWiki explicitly places these techniques in the straferunning/speedrunning tradition. citeturn20view1

Finally, the policy must be **ruleset-aware**. On Nightmare, Doom/Doom II use Ultra-Violence monster placement but double ammunition, fast monsters and monster respawning. That makes methodical extermination less valuable and rapid progression, bypassing and route security more important. citeturn19view0


## Decision Architecture for an Autonomous Doom Agent

A Doom agent should maintain a compact internal state rather than reacting independently to every frame.

**Combat state**

`health, armor_value, armor_type, ammo[bullets,shells,rockets,cells], weapons_owned, current_weapon`

**Threat state**

`visible_enemy(type, distance, bearing, LOS, attack_state), incoming_projectile(trajectory,time_to_contact), recent_damage_direction, nearby_enemy_count`

**Geometry state**

`lateral_space, rear_space, nearest_cover, nearest_corner, doorway, choke_point, wall_distance, damaging_floor, escape_routes`

**Progress state**

`keys_owned, locked_doors_seen, switches_seen/pressed, lifts, teleporters, unexplored_frontiers, visible_inaccessible_items, exit_candidate`

**Memory state**

`topological_location, visited_edges, last_progress_event, recently_changed_geometry, resource_caches, failed_routes, death_signatures`

This is preferable to merely tracking visible enemies because Doom progression frequently requires remembering a locked door after acquiring its key, revisiting an area after a remote switch changes it, or detecting that repeated traversal has produced no state change. Stock and community maps repeatedly use sequences such as “find locked door → obtain key elsewhere → backtrack,” or “press switch → previously inaccessible geometry changes.” citeturn22search0turn22search2turn22search6

**The refined priority hierarchy**

| Priority | Question | Default decision |
|---|---|---|
| **Imminent survival** | Will something hit within ~1 second? | Dodge, break LOS or create distance before aiming |
| **Position** | Am I cornered, against a wall, crossing damage, or surrounded? | Move toward known open/covered space |
| **Snowball control** | Can an enemy rapidly make the fight worse? | Prioritize Arch-vile/Pain Elemental or escape their influence |
| **Reliable damage** | Who can damage me despite ordinary strafing? | Remove exposed chaingunners, shotgun guys and sustained hitscan threats |
| **Pressure control** | What restricts movement? | Interrupt Lost Souls/revenants; thin close melee bodies |
| **Fight efficiency** | Can geometry/infighting solve this more cheaply? | Reposition before spending scarce ammo |
| **Resources** | Is there a safe, useful pickup? | Collect or mark for later |
| **Progress** | What unexplored or newly unlocked objective is most promising? | Move to highest-value frontier |
| **Cleanup** | Is optional combat/secret hunting worth the cost? | Only if expected benefit exceeds risk |

A simple multiplicative formula such as `damage × proximity × mobility` is insufficient. An Arch-vile may not be the closest or highest-DPS enemy but can resurrect corpses and delivers a high-damage attack unless line of sight is broken at the critical moment; a Pain Elemental can continuously increase enemy count; a chaingunner's hitscan stream is much harder to evade than a baron's individually dodgeable fireballs. citeturn2view0turn17search4turn9view0turn14view3

A better AI ranking is lexicographic:

```text
THREAT_CLASS =
    imminent_collision_or_attack
    > compounding_special_ability
    > exposed_hitscan
    > homing_or_fast_pressure
    > movement-denial_projectiles
    > ordinary_projectile_tank
    > isolated_melee_enemy_at_distance

Within a class rank by:
    time_to_damage
    × expected_damage
    × exposure_duration
    × ability_to_restrict_escape

Then discount targets that:
    are currently blocked by geometry,
    are safely infighting,
    require dangerous exposure to engage,
    can be bypassed without compromising the route.
```

**Movement policy.** During active combat, standing still should require a positive reason rather than be the default. Doom's fast player movement makes lateral repositioning especially powerful against projectile attacks and mixed groups. Infighting guidance explicitly notes the favorable player-to-monster speed ratio, and simultaneous forward/strafe input is faster than straight running. citeturn20view0turn20view1

The direction of movement matters more than “always move.” Against ordinary projectiles, lateral movement generally changes the intercept point fastest. Against hitscan, movement in the open does not magically dodge a shot already resolved; the robust counter is **breaking line of sight, shortening exposure, or killing/stunning the shooter**. Heavy weapon dudes continue firing until sight is broken, they are hurt/stunned, or they die; the Spider Mastermind has similar sustained-fire behavior. citeturn9view0turn14view0

Use these geometry rules:

| Situation | Preferred behavior |
|---|---|
| Projectile approaching | Strafe across its trajectory; preserve a second escape direction |
| Homing revenant missile | Use a corner/pillar; otherwise late side movement followed by forward movement can cause it to miss |
| Hitscan enemy exposed | Break LOS or aggressively eliminate it |
| Door into unknown room | Open, back into known territory, observe what activates |
| Narrow corridor | Control one direction; do not let melee enemies reach both sides |
| Large arena | Maintain an orbit with escape lanes rather than stop in the center |
| Cyberdemon arena | Stay away from walls that can convert a miss into splash damage |
| Arch-vile present | Track nearest hard LOS blocker continuously |
| Surrounded | Stop shooting unless firing opens an escape lane; move toward the least-blocked sector |
| Damaging floor | Cross by the shortest safe path; use radiation suit if available |

Revenant homing missiles are specifically vulnerable to geometry; DoomWiki recommends walls/pillars or a side-then-forward dodge near impact. Mancubi have a recognizable three-burst attack pattern and can punish excessive dodging because their shots spread across lanes. Cyberdemon rockets add blast damage when they impact nearby geometry. citeturn15view1turn16view0turn13view1

Damaging sectors are mechanically independent of their floor artwork, although liquids, lava, nukage and environmental markings frequently signal them. Damage is periodic, so the correct general strategy is to minimize exposure time rather than explore hazardous terrain slowly. Radiation suits provide temporary protection. citeturn21view7turn3view3

**Difficulty adaptation.** On Nightmare, ordinary “clear room, then explore safely” assumptions break because monsters respawn and attacks/projectiles are faster. The strategic objective should shift from kill percentage toward **route opening and exit acquisition**; ammunition is doubled, so spending more to force quick progress is acceptable. citeturn19view0


## Full Doom Strategy Manual

**Combat should be fought from favorable geometry, not wherever contact happens.** When a door opens onto multiple monsters, the agent should normally retreat through it rather than commit immediately. This converts a room-wide encounter into a smaller number of enemies sharing a choke point, reduces simultaneous line of sight and gives projectile attacks fewer useful angles. The exception is when retreat would lead into an already compromised area or allow a compounding threat such as an Arch-vile to resurrect a large corpse field. citeturn2view0turn19view1

Corners and pillars are disproportionately valuable. They defeat Arch-vile line-of-sight attacks, absorb revenant missiles and can interrupt sustained chaingunner/Spider Mastermind fire. Against Cyberdemons, however, a pillar is useful only if the player is not so close to it that rocket splash still reaches them. citeturn2view0turn15view1turn13view1

Circle-strafing is good when the arena is known to be open and the player controls the outer path. It is bad when the perimeter contains unseen monsters, damaging floors, walls close enough for Cyberdemon splash, or crossfire from multiple directions. Thus the AI rule should be **“circle only after validating the circle.”** The same principle applies to diagonal high-speed movement: ordinary SR40-style movement is useful for crossing exposed ground and creating separation, but execution-heavy SR50 and speedrun-specific maneuvers should not displace reliable steering. citeturn20view1turn13view1

**Hitscan and projectile enemies require different control policies.** Imps, cacodemons, Hell knights and barons become comparatively manageable when the agent has open lateral space because their ordinary projectile attacks can be evaded. By contrast, chaingunners can continue applying immediate hitscan damage while visible; shotgun guys become dangerous at close range; the Spider Mastermind sustains hitscan fire until sight is broken or its attack is interrupted. This is why enemy priority must not correlate simply with hit points. citeturn9view0turn9view1turn10view1turn14view2turn14view3turn14view0

**Special enemies must be judged by how quickly they worsen the future state.** An Arch-vile can resurrect most normal corpses and has a high-damage LOS-dependent attack. Its very low pain chance makes “just keep shooting and hope it flinches” unreliable. A Pain Elemental's attacks add Lost Souls, so delaying the kill can literally increase the number of threats. Revenants create persistent homing pressure and are fast, while chaingunners make extended exposure expensive. citeturn2view0turn17search4turn15view1turn9view0

This gives a practical enemy-order default:

> **Arch-vile or immediate escape requirement → Pain Elemental creating new souls → exposed chaingunner/other hitscan → charging Lost Soul / dangerous revenant → sustained Arachnotron/Mancubus pressure → nearby melee blockers → ordinary projectile monsters → distant health tanks.**

This is context-dependent. An Arch-vile behind total cover may temporarily rank below a chaingunner already hitting the player; a Pain Elemental at long range may rank below a Lost Soul entering point-blank range.

**Infighting should be an active but conditional capability.** Most projectile and hitscan monsters can accidentally damage other species and trigger retaliation. Repeating, inaccurate and spread attacks from chaingunners, mancubi, arachnotrons, Spider Masterminds and Cyberdemons are especially useful for provoking it. Same-species projectiles normally do not cause damage, and Hell knights/barons are treated as one species for their green projectiles in classic PC Doom. Pain Elementals cannot initiate conventional infighting because the Lost Soul is the attacking entity, while Arch-vile damage does not provoke retaliation against the Arch-vile. citeturn19view1turn20view0

The correct infighting policy is:

```text
IF mixed powerful enemies
AND player has ample movement space
AND no immediate hitscan/special threat requires attention
THEN place one enemy's firing line through another,
     dodge the attack,
     disengage while they fight.

ELSE fight normally.
```

Infighting is particularly valuable against ammunition-heavy targets such as Cyberdemons and Spider Masterminds. Doom II's MAP20 is a canonical example in which the Cyberdemon and Spider Mastermind can be induced to fight, while E3M8 allows the Spider Mastermind to become distracted by other monsters. citeturn20view0turn12search6

Do **not** force infighting by standing exposed to a chaingunner for several seconds, crossing a Cyberdemon firing lane without lateral room, or circling a pack when health is already critical. Infighting is an efficiency tactic, not a survival obligation.

**Weapon choice should minimize expected total cost rather than maximize instantaneous power.**

| Weapon | Default role | Prefer when | Avoid / conserve when |
|---|---|---|---|
| **Fist** | Last resort; excellent with berserk | Berserk active, isolated manageable target | Normal fist versus serious enemy |
| **Chainsaw** | Ammo-free suppression/melee | Isolated demon/pinky, safe doorway, ammo emergency | Open mixed fights, powerful melee enemy |
| **Pistol** | Low-power precision/finishing | Isolated weak distant enemy; extreme conservation | Any serious time-critical fight |
| **Shotgun** | General-purpose efficiency | Weak/medium individual targets; mid-range shots | Dense close crowd where SSG is superior |
| **Super shotgun** | Close/mid burst damage | Tough enemies and dense close groups | Long range; lone trivial enemy; when reload creates danger |
| **Chaingun** | Fast reaction and suppression | Chaingunners, revenants, Lost Souls, weak packs; tap for distance | Long bullet dump into high-HP tanks |
| **Rocket launcher** | Efficient area/tough-target damage | Clusters, long lanes, mancubi/barons/knights | Close enemies, nearby wall, Lost Souls rushing player |
| **Plasma rifle** | Emergency high DPS/suppression | Arch-vile, dangerous revenant/arachnotron, dense pressure | Safe low-threat cleanup |
| **BFG9000** | Extreme burst/crowd/boss solution | Dense high-value targets, bosses, crisis reset | Weak isolated enemies |

The shotgun fires seven pellets and remains useful at close-to-medium range; the super shotgun spends two shells but delivers roughly three times the standard shotgun's theoretical damage, trading accuracy and reload time for burst power. citeturn4view0turn4view1

The chaingun is particularly valuable for AI control because its first pair of shots is accurate and rapid fire can exploit monster pain states. Short bursts therefore combine accuracy with quick interruption better than continuous indiscriminate spraying. citeturn5view0

Rockets combine direct damage with a 128-unit blast radius, making them efficient against clusters but dangerous near walls and close enemies. Lost Souls are a particularly bad rocket target when near the player because they can abruptly charge into point-blank range. citeturn5view1turn10view0

The plasma rifle's rapid projectiles make it excellent for suppressing revenants, arachnotrons, mancubi and other dangerous targets, but firing long uncontrolled streams wastes cells when enemies can move out of the stream. citeturn5view2

The BFG's important mechanic is not simply “large projectile = huge explosion.” It consumes 40 cells, has a firing delay, and after the projectile event creates 40 tracer attacks based on the player's later position and original firing direction. The practical inference is that **where the agent is when the BFG's secondary tracers occur matters**, so controlled closing/repositioning can greatly improve damage. This is an advanced tactic; safe survival still overrides maximizing tracer contact. citeturn5view3

**Ammunition policy.** Vanilla maximum capacities are 200 bullets, 50 shells, 50 rockets and 300 cells before backpack capacity expansion. citeturn0search19 A conservative synthesized reserve policy—not an engine rule—is:

| Ammo | Enter conservation mode when below | Preferred conservation response |
|---|---:|---|
| Bullets | ~40 | Shotgun/chainsaw where safe; chaingun only for interruption |
| Shells | ~10–15 | Stop SSG-ing trivial targets |
| Rockets | ~8–10 | Reserve for dense/tough fights unless rockets solve immediate danger |
| Cells | ~60–80 | Prefer conventional weapons; once BFG owned, try to retain ~80 cells for two BFG shots |

Percentage-based thresholds are preferable if a backpack has doubled capacity. A backpack is normally worth collecting because it increases ammunition capacity, while large resources should sometimes be deliberately left in safe, memorable locations for later use. citeturn3view3

**Health and armor are strategic resources, not compulsory pickups.** Stimpacks and medikits restore ordinary health up to 100, while soulspheres can push health to 200. Green armor provides 100 armor at lower protection than blue combat armor, which provides 200 with greater damage absorption. Berserk sets health to 100 if lower and massively increases fist damage; invulnerability, radiation suits, partial invisibility and megaspheres can radically change encounter economics. citeturn3view3

Therefore:

```text
Large health pickup + near full health + safely revisitable
→ mark pickup and leave it.

Berserk + health well below 100
→ strongly consider taking for heal even if melee is not planned.

Invulnerability + major encounter immediately ahead
→ activate encounter first or take immediately before committing;
  do not waste most of its duration exploring.

Radiation suit + known hazardous route
→ take immediately before entering the hazard.

Backpack
→ normally take immediately.

Megasphere/soulsphere + already near 200 effective durability
→ preserve if the area can safely be revisited.
```

Partial invisibility deserves a special exception. It scatters both hitscan and ordinary projectile aim and is strongly useful against distant hitscan fire, but the resulting projectile paths become less predictable; it does not disrupt homing missiles or Arch-vile attacks. For an AI with strong trajectory prediction, that unpredictability can make some projectile-heavy encounters harder to model. citeturn20view2turn20view3

**Navigation must be treated as graph search with event memory.** Every meaningful area becomes a node and traversable passage a graph edge. The agent should record:

```text
NODE:
  visual_landmark
  exits/frontiers
  locked doors + colors
  switches
  lifts
  teleporters
  hazards
  useful pickups left behind

EVENT:
  key acquired
  switch pressed
  pickup trigger
  door unlocked
  lift activated
  teleporter used
  wall/closet opened
  sound suggesting remote movement
```

Teleporters often use a distinctive floor area in classic Doom, but that appearance is conventional rather than technically mandatory, and a teleporter can send the player to different destinations depending on implementation. Teleportation also temporarily prevents player movement after arrival, so unknown teleport destinations deserve elevated caution. citeturn21view3

Secret doors are commonly hinted at by an off-color or otherwise anomalous wall, but stock maps are generally designed so that secrets are not necessary for ordinary completion. Thus secret-hunting should be **secondary to finding genuine progression**. citeturn21view4

When stuck, use this deterministic search sequence:

1. **Inventory-change check:** has a new blue/red/yellow key been obtained? Route directly to remembered matching locks.
2. **Trigger-change check:** revisit the area around the most recent switch/key/major pickup and inspect for opened walls, lowered lifts, stairs or bars.
3. **Frontier check:** choose the nearest unvisited passage, doorway or distinct room boundary.
4. **Vertical-access check:** inspect lifts, lowering platforms, stairs and reachable ledges.
5. **Inaccessible-item check:** visible pickups behind barriers imply that some route, lift or switch may exist.
6. **Anomaly check:** examine conspicuously different/misaligned walls, lighting changes and isolated switch-like panels.
7. **Remote-change check:** revisit previously blocked geometry after significant switches.
8. **Loop break:** if the same route has been traversed twice without acquiring a key, pressing a new switch, changing geometry, discovering an area or collecting a strategically useful item, temporarily blacklist it.
9. **Systematic expansion:** search outward from the last confirmed progression event rather than wandering globally.

Stock and community maps repeatedly demonstrate switches opening remote geometry, keys requiring backtracking to colored doors, lifts exposing routes, and key pickups triggering ambushes or previously inaccessible areas. citeturn22search0turn22search4turn22search6turn22search11

**Trap recognition must be probabilistic.** A key, BFG, rocket launcher, megasphere or similar conspicuous item centered in an otherwise suspiciously empty room should raise the prior probability of an ambush. Doom maps commonly make pickups or switches coincide with new monster closets, teleporting enemies, closing routes or opened walls; examples across established WAD walkthroughs show precisely these trigger patterns. citeturn22search1turn22search4turn22search11

Before triggering such an object:

> clear visible enemies → identify escape direction → avoid standing against a wall → choose a weapon appropriate for an ambush → trigger while moving → immediately scan rear/flanks → retreat to known geometry if the room changes.

A “safe-looking empty room” is therefore not evidence of safety.


## Machine-Friendly Enemy and Weapon Reference

The enemy table uses **priority as a default**, not an immutable ranking. Position overrides taxonomy. A nearby pinky blocking the only exit is more urgent than a distant Arch-vile currently separated by hard cover.

| Enemy | Attack / speed | Default threat | Engagement policy | Weapon / range | Delay or ignore when… |
|---|---|---|---|---|---|
| **Zombieman** | Weak hitscan; slow | Low alone, attritional in groups | Remove when exposed; exploit dropped bullets | Shotgun, pistol or brief chaingun; any range | Another serious threat dominates and LOS can be broken. citeturn8view0 |
| **Shotgun Guy** | Hitscan spread; slow | Medium-high close | Kill quickly if close/exposed; do not melee unnecessarily | Shotgun/chaingun | Distant and covered. citeturn8view1 |
| **Chaingunner** | Sustained rapid hitscan; moderate | **Very high** | Immediate LOS break or fast kill/suppression | Chaingun, shotgun/SSG, plasma; rockets for groups | Only when fully occluded or useful infighting is already controlling it. citeturn9view0 |
| **Imp** | Slow projectile + melee; moderate | Low-medium | Strafe; shotgun efficiently | Shotgun midrange; SSG/rocket for groups | Distant with ample lateral space. citeturn9view1 |
| **Demon / Pinky** | Melee only; fairly fast | Low at range, high when blocking | Maintain distance; funnel through choke | SSG; shotgun; chainsaw if safely isolated | Far away and unable to restrict movement. citeturn9view2 |
| **Spectre** | Demon-class melee; visually difficult to track | Medium near player | Treat as pinky with larger uncertainty margin; keep more space | Shotgun/SSG/chaingun | Clearly distant and tracked. This policy is an inference from demon behavior plus the spectre-like partial-invisibility rendering. citeturn9view2turn20view2 |
| **Lost Soul** | Extremely fast charge; flying | High when charging/close | Interrupt charge immediately; create distance | Chaingun ideal reaction weapon; shotgun/SSG | Far away and another special/hitscan threat dominates. Never casually rocket one at close range. citeturn10view0 |
| **Cacodemon** | Slow projectile + close bite; flying | Moderate | Strafe in open; avoid letting flyers surround | Shotgun/SSG/chaingun/plasma | Distant with open movement space. citeturn10view1 |
| **Pain Elemental** | Produces Lost Souls; flying | **Very high / compounding** | Focus early before enemy count grows | SSG, rockets if safe, plasma | Only when inactive/blocked, or immediate survival requires another target. Vanilla's 21-Lost-Soul cap can suppress spawning, but ports may alter it. citeturn17search4turn17search0turn17search18 |
| **Revenant** | Fast; melee + normal/homing missile | High | Use cover; chaingun can exploit high pain chance; do not let it crowd player | Chaingun, SSG, plasma; rockets for groups | Missile lanes are blocked and higher-class threat exists. citeturn15view1 |
| **Hell Knight** | Strong projectile/melee; moderate | Medium | Maintain space; efficient heavy weapon fire | SSG, rocket, plasma | Distant with easy-to-dodge fireballs. citeturn14view2 |
| **Baron of Hell** | Same style as knight, 1000 HP | Medium despite durability | Do not waste emergency ammo solely because of health; kite or infight | Rockets/plasma/SSG | Often safe to postpone in open space. citeturn14view3 |
| **Arachnotron** | Sustained fast plasma; reasonably mobile | High in exposed lanes | Broad lateral movement plus LOS interruption; suppress | Plasma/rocket/chaingun/SSG | Hard cover blocks it. Blur sphere can make its projectile pattern less predictable. citeturn16view2 |
| **Mancubus** | Three volleys of paired fireballs; slow | High lane-control threat | Keep distance; recognize volley spread; avoid over-dodging | Rocket/plasma/SSG | Behind cover or safely infighting. citeturn16view0 |
| **Arch-vile** | Delayed high-damage LOS attack; resurrects corpses; very fast | **Critical** | First locate cover; then kill rapidly. Do not rely on pain-stun | BFG/plasma/SSG/rocket | Only while hard cover guarantees attack failure and another immediate threat must be handled. citeturn2view0 |
| **Cyberdemon** | Three-rocket volleys; fast, 4000 HP; blast-immune | Boss-level | Continuous lateral control; avoid walls; use infighting; preserve distance | BFG best, plasma/rockets next | Frequently bypassable if map allows. Do not insist on killing merely for completeness. citeturn13view1 |
| **Spider Mastermind** | Sustained powerful hitscan; large | Boss-level | Hard-cover cycle: expose, damage, hide; infighting valuable | BFG, plasma, rockets | Safely distracted by infighting or bypassable. citeturn14view0turn12search6 |

Four special rules matter enough to memorize:

**Arch-vile:** cover is the counter. The attack has a long cast and requires the relevant line-of-sight condition around the blast; the Arch-vile also resurrects many corpses and has a very low pain chance. Rapid heavy damage beats hopeful suppression. citeturn2view0

**Pain Elemental:** time is a resource. Every delayed kill risks extra Lost Souls, so target priority increases rather than decreases as the fight continues. Vanilla behavior caps new Lost Souls once 21 exist, but this engine-specific limit is removed or configurable in some ports; an AI intended to generalize should not depend on it. citeturn17search0turn17search4

**Chaingunner:** exposure time is the enemy. Its firing sequence continues until LOS is lost, it is interrupted or killed. Move to cover immediately rather than attempting a long damage race from a bad position. citeturn9view0

**Revenant:** geometry beats raw speed. Homing missiles can be absorbed by walls and pillars or defeated with well-timed lateral/forward movement; the revenant's high pain chance also makes rapid weapons unusually effective. citeturn15view1

Weapon automation can therefore use:

```text
IF current target is weak + isolated + no immediate danger
    → shotgun/pistol/chainsaw depending range and ammo.

IF several weak/medium enemies overlap at close-mid range
    → SSG.

IF charging Lost Soul OR exposed chaingunner OR single revenant needs interruption
    → chaingun.

IF dense enemies AND safe blast clearance
    → rocket launcher.

IF high-priority enemy must die immediately
    → plasma.

IF dense high-value fight OR boss OR imminent tactical collapse
    → BFG.

IF enemy is near player OR wall is within blast radius
    → inhibit rocket launcher.

IF engagement is long range
    → prefer shotgun/chaingun tap fire over SSG.

IF target can safely be bypassed
    → compare ammunition cost with strategic value before firing.
```

These rules follow the documented characteristics of Doom's shotgun spread, SSG burst, chaingun pain-lock potential, rocket splash, plasma rate of fire and BFG mechanics. citeturn4view0turn4view1turn5view0turn5view1turn5view2turn5view3

**Bosses.** For a Cyberdemon, the robust approach is lateral movement with generous space, awareness of each three-rocket volley, no wall-hugging, and BFG/plasma/rockets as resources permit. DoomWiki estimates roughly 45 direct rocket hits are required, while BFG kills are much faster; point-blank SSG pain manipulation exists but is explicitly risky and associated with high-skill/speedrunning play, so it should not be a default AI policy. citeturn13view1

For the Spider Mastermind, hard cover is more important than conventional dodging because its primary threat is sustained hitscan fire. Infighting is especially useful, and BFG, rockets or plasma are the preferred high-power weapons. Point-blank BFG attacks can kill very rapidly, but the general-purpose AI should only close when the attack route is controlled. citeturn14view0turn12search6

For Doom II's Icon of Sin, **do not try to clear the endlessly spawning battlefield**. The stock MAP30 solution is to climb the lift-steps, activate the upper switch to raise the central pillar, lower/ride that pillar and time rockets through the opening in the face; approximately three successful rocket blasts to the hidden boss target are normally sufficient. Arch-viles, Pain Elementals, Arachnotrons and Revenants are particularly disruptive around the firing platform, so clear or divert those that directly interfere, not the entire horde. citeturn21view6turn14view1


## Failure Modes, Recovery, and Anti-Loop Control

The following table converts likely visually controlled AI failures into detectable control conditions. The corrective policies synthesize the monster, geometry, weapon and resource mechanics above. citeturn5view1turn9view0turn15view1turn19view1turn21view7

| Failure | Why / detection | Corrective action |
|---|---|---|
| **Standing still while aiming** | Combat active but movement input near zero | Cancel aim fixation; select safe lateral or cover-directed movement first |
| **Backing into a wall** | Rear distance repeatedly decreases to zero | Switch to lateral/diagonal escape; rotate briefly to validate rear space |
| **Corner entrapment** | Two nearby blocking surfaces + enemies closing | Stop fighting tanky target; kill/block only escape-lane enemy and leave |
| **Continuous wall fire** | Shots produce no enemy hit/target occluded | Cease fire immediately; reacquire LOS before next burst |
| **Wrong weapon persistence** | Damage rate poor while safer/better weapon available | Recompute weapon choice after ~2 ineffective attack cycles |
| **Rocketing trivial targets** | Low-threat enemy and no splash value | Switch shotgun/chaingun |
| **Point-blank rocket fire** | Enemy/wall inside safety radius | Weapon inhibit; switch SSG/plasma/chaingun |
| **Ignoring rear attacks** | Health loss without visible source | Move immediately; perform rapid 90°/180° threat scan |
| **Target tunnel vision** | Same target tracked while closer/new high-priority threat enters view | Interrupt target lock and rerank all threats |
| **Chasing fleeing/low-priority enemy** | Distance traveled increases without progression or danger reduction | Break pursuit; restore route objective |
| **Aggressive room entry** | Multiple new LOS contacts immediately after threshold | Retreat through entry and reduce engagement aperture |
| **Failure to retreat** | Incoming DPS exceeds damage dealt and escape remains open | Invoke disengagement trigger regardless of target health |
| **Navigation loop** | Same node-edge sequence repeated twice without progress event | Blacklist route until state change; select different frontier |
| **Missed switch** | Dead end + unusual wall/panel + no frontier | Deliberate wall/switch scan before leaving |
| **Forgotten locked door** | New key obtained but exploration continues randomly | Make matching remembered lock the default objective |
| **Health-pickup waste** | Large heal consumed near cap | Cache for later unless route may become inaccessible |
| **Ignoring armor** | Armor low/zero while accessible armor known | Raise pickup priority; effective durability matters |
| **Infighting missed** | Mixed powerful enemies overlap firing lanes and movement space is good | Reposition to induce crossfire before spending heavy ammo |
| **Hazard-floor wandering** | Health ticks while no combat explains damage | Shortest route to safe floor; stop exploratory scanning until safe |
| **Missed monster closet** | Progress trigger followed by opening sound/new damage | Immediate rear/flank scan and reposition |
| **Obstacle-blocked firing** | Repeated impacts close to player | Stop fire; sidestep for clean lane |
| **Weapon-switch thrashing** | Multiple switches within a few seconds without tactical change | Add commitment window; switch only for changed target/range/risk |
| **Unnecessary extermination** | Enemies isolated from progression and resources low | Bypass and continue objective |
| **No unwinnable-state recognition** | Health falling rapidly, surrounded, no target dying | Survival override: abandon DPS goal, create one escape lane |
| **Trap freeze** | Room changes suddenly and agent stops to classify | Move toward preselected escape/cover while classification continues |

**Low-health policy**

`Health 25–50%:` reduce optional exposure. Fight from doors/corners, collect efficient nearby health/armor, avoid unnecessary melee and stop trading damage for speed.

`Health 10–25%:` survival dominates ammunition efficiency. Disengage from optional combat, spend plasma/SSG if it creates a safe exit, prioritize known health caches, and avoid infighting setups requiring open exposure.

`Health <10%:` enter emergency mode. Navigate toward a known safe health source or exit; fire only to remove immediate blockers/high-reliability attackers; never attempt a point-blank rocket shot; avoid speculative exploration.

Large health items need not be taken at maximum health because soulspheres exceed the normal 100-health cap, while medikits/stimpacks do not. Armor further changes effective durability. citeturn3view3

**Low-ammunition policy**

```text
1. Stop using SSG against trivial lone enemies.
2. Stop rockets unless splash value or threat level justifies them.
3. Preserve cells for high-priority threats.
4. Use ordinary shotgun for routine targets.
5. Use chaingun in short control bursts rather than long sprays.
6. Use chainsaw/berserk against safely isolated suitable targets.
7. Seek infighting in mixed groups.
8. Bypass enemies disconnected from progression.
9. Route toward remembered ammo caches and dropped weapons/ammo.
```

Infighting specifically reduces ammunition expenditure and has long been an established tactic in ammo-constrained Doom play. citeturn20view0

**Surrounded recovery**

```text
DO NOT try to kill the toughest enemy first.

1. Identify widest reachable gap.
2. Identify body blocking that gap.
3. Use high-stagger/high-DPS weapon only on that blocker.
4. Move toward gap immediately while firing.
5. Prefer diagonal/lateral escape over straight backward retreat.
6. Once outside encirclement, seek doorway/corner.
7. Rerank threats only after mobility is restored.
```

**Unknown-damage recovery**

```text
Damage received + attacker absent from central view
→ MOVE immediately;
→ break current target lock;
→ head toward nearest known cover/open escape area;
→ scan 90° left/right, then rear;
→ identify hitscan/special attacker;
→ rerank;
→ do not remain stationary while searching.
```

This is particularly important because rapid hitscan enemies can continue firing during a visual search. citeturn9view0turn14view0

**Lost / unable to progress**

```text
IF key acquired since last route decision:
    revisit matching known lock.

ELSE IF switch/major trigger occurred recently:
    inspect its room;
    inspect adjacent geometry;
    revisit known blocked route most likely connected to it.

ELSE:
    select nearest unvisited frontier.

IF no frontiers:
    inspect lifts/platforms and inaccessible visible objects;
    scan anomalous/misaligned walls in dead ends;
    revisit major switch-controlled areas.

IF same edge sequence repeats twice without:
    new area,
    new key,
    new switch,
    new geometry,
    strategically relevant pickup:
        mark sequence NONPRODUCTIVE;
        forbid it until another state variable changes.
```

Secret-wall clues such as off-color walls exist, but secrets are generally optional in stock Doom, so wall-hunting should come **after** systematic progression checks rather than before them. citeturn21view4

**Repeated-death recovery**

Store an attempt signature:

```text
{
  entry_route,
  first_cover,
  first_target,
  opening_weapon,
  movement_direction,
  trigger_timing,
  resource_state,
  death_cause
}
```

After one repeat death, change the most causally relevant element. After two similar deaths, change at least **two** of route, target priority, weapon, initial movement or trigger timing. After three, abandon the encounter formulation: seek additional resources, engineer infighting, use a different entrance, trigger the fight then retreat farther, or bypass the enemy if progression allows.

The agent should never interpret “I almost succeeded” as sufficient evidence to repeat an identical policy indefinitely.


## AI Doom Playbook

The following is the context-ready operational artifact. It intentionally removes most explanation. The 80 rules are divided conceptually across reflex, tactical and strategic control, but all use the same `CONDITION → ACTION` form. They encode the mechanics and tactics established above, especially Doom's movement advantage, hitscan behavior, special enemies, weapon properties, infighting and progression structure. citeturn20view1turn9view0turn2view0turn19view1turn5view1

| ID | CONDITION → ACTION |
|---|---|
| **R01** | Active combat + no reason to hold position → maintain movement. |
| **R02** | Projectile approaching → strafe across its trajectory rather than retreat directly along it. |
| **R03** | Projectile dodge selected → preserve a second escape direction; do not dodge into a wall. |
| **R04** | Taking damage from unknown direction → move first, scan second. |
| **R05** | Rear space uncertain → do not continue long backward movement. |
| **R06** | Back nearly against wall → move diagonally/laterally toward open space. |
| **R07** | Two walls constrain movement → treat position as emergency even before health falls. |
| **R08** | Surrounded → prioritize escape lane over damaging highest-HP enemy. |
| **R09** | One enemy blocks escape → focus that blocker with rapid/high-burst damage. |
| **R10** | Door opens onto multiple enemies → retreat into known space unless retreat itself is unsafe. |
| **R11** | Corner/pillar available against ranged enemy → use it to reduce simultaneous LOS. |
| **R12** | Hitscan enemy firing → break LOS or suppress/kill immediately; ordinary strafing alone is insufficient. |
| **R13** | Chaingunner visible and unobstructed → elevate to very high priority. |
| **R14** | Spider Mastermind firing → seek hard cover before attempting sustained return fire. |
| **R15** | Arch-vile begins attack → locate and reach hard LOS cover before blast completion. |
| **R16** | Arch-vile visible among corpses → raise priority because delay can restore killed enemies. |
| **R17** | Arch-vile covered and another enemy is causing immediate damage → solve immediate danger before re-engaging vile. |
| **R18** | Pain Elemental actively producing Lost Souls → prioritize before threat count grows. |
| **R19** | Lost Soul begins close charge → interrupt rapidly with chaingun/shotgun; do not prepare a rocket. |
| **R20** | Revenant missile homing → route it into wall/pillar or perform late lateral-plus-forward dodge. |
| **R21** | Revenant alone and exposed → rapid chaingun fire is a strong suppression option. |
| **R22** | Revenant reaches close range → create distance unless a deliberate expert melee strategy is required. |
| **R23** | Mancubus begins volley → keep distance and avoid huge direction changes that enter its spread lanes. |
| **R24** | Arachnotron maintains plasma stream → broad lateral movement + periodic LOS break. |
| **R25** | Demon/Pinky distant → deprioritize unless it blocks future movement. |
| **R26** | Demon/Pinky reaches corridor choke → SSG/shotgun; prevent it body-blocking retreat. |
| **R27** | Spectre expected nearby → enlarge safety margin because visual tracking is less reliable. |
| **R28** | Cacodemon distant in open area → ordinary lateral movement usually sufficient; save emergency ammo. |
| **R29** | Baron/Hell Knight distant with lateral space → dodge; do not prioritize solely because of high HP. |
| **R30** | Multiple enemy types visible → prioritize attack reliability/special ability over raw health. |
| **R31** | Current target behind hard cover → cease firing instead of shooting obstacle. |
| **R32** | Shots repeatedly hit nearby wall/door frame → stop fire and reposition. |
| **R33** | Weak isolated enemy + safe encounter → use ammunition-efficient weapon. |
| **R34** | Lone weak/medium enemy at mid range → standard shotgun preferred. |
| **R35** | Dense close/mid group → super shotgun favored if reload window is safe. |
| **R36** | Long-range target → prefer shotgun or controlled chaingun over SSG. |
| **R37** | Fast interrupt required → chaingun favored. |
| **R38** | Chaingun used at range → tap short bursts rather than hold trigger indefinitely. |
| **R39** | Several enemies clustered at safe distance → rocket launcher favored. |
| **R40** | Enemy or wall close enough for blast danger → inhibit rocket launcher. |
| **R41** | Lost Souls are close → inhibit rocket launcher. |
| **R42** | High-priority enemy must die quickly → plasma favored when cells permit. |
| **R43** | Safe weak cleanup → do not spend plasma merely because it is strongest. |
| **R44** | Large high-value enemy cluster or boss → consider BFG. |
| **R45** | BFG fired → maintain awareness that secondary tracer geometry depends on subsequent player position. |
| **R46** | BFG positioning would require suicidal exposure → abandon optimal tracer positioning and survive. |
| **R47** | Berserk active + safely isolated suitable enemy → fist becomes ammunition-saving option. |
| **R48** | Chainsaw + isolated Pinky/Demon in controlled geometry → use to conserve ammunition. |
| **R49** | Strong melee enemy or mixed group → do not chainsaw merely to save ammo. |
| **R50** | Shells low → stop SSG use on lone weak targets. |
| **R51** | Rockets low → reserve them for clusters, tough targets or emergency lane clearing. |
| **R52** | Cells low → reserve plasma/BFG for special threats and difficult fights. |
| **R53** | BFG owned + future danger unknown → try to preserve roughly two shots' worth of cells when feasible. |
| **R54** | Ammo generally low → increase priority of infighting, bypassing and remembered ammo caches. |
| **R55** | Mixed powerful monsters + open movement room → attempt infighting before committing heavy ammo. |
| **R56** | Infighting has begun → disengage enough to let monsters damage each other. |
| **R57** | Monster is safely infighting → lower its immediate target priority unless it threatens the route. |
| **R58** | Same-species projectile monsters only → do not rely on them damaging one another with their normal projectiles. |
| **R59** | Infighting setup requires extended exposure to hitscan/Arch-vile fire → abandon setup. |
| **R60** | Large health pickup + health near cap + safe return route → leave and mark it. |
| **R61** | Berserk available + health substantially below 100 → value it as both healing and melee upgrade. |
| **R62** | Backpack accessible → normally collect. |
| **R63** | Invulnerability accessible immediately before high-risk fight → take close to commitment time, not long beforehand. |
| **R64** | Radiation suit available + damaging-floor route ahead → delay pickup until immediately before crossing. |
| **R65** | Partial invisibility active → expect ordinary projectile directions to be less predictable; do not use standard “dodge exact aim line” assumptions. |
| **R66** | Damaging floor under player → minimize time on it; exploration becomes secondary. |
| **R67** | High-value pickup isolated in suspicious room → assume possible trap; preselect escape route before collection. |
| **R68** | Key/weapon/switch just triggered → immediately scan rear/flanks and listen/look for changed geometry. |
| **R69** | New key obtained → highest-value navigation target becomes remembered matching door. |
| **R70** | Switch pressed → record location and inspect nearby/previously blocked geometry for change. |
| **R71** | Teleporter entered → expect unknown orientation/contact; prepare movement and rapid threat scan on arrival. |
| **R72** | Visible but inaccessible useful item → record as navigation clue rather than repeatedly running into barrier. |
| **R73** | Unvisited branch exists → explore it before repeatedly traversing completed corridors. |
| **R74** | Same route traversed twice without progress → temporarily blacklist it. |
| **R75** | No unexplored branch → revisit newest key/lock or newest switch/change relationship. |
| **R76** | Still stuck → inspect lifts, lowering surfaces, switches and anomalous walls near meaningful rooms. |
| **R77** | Secret-like wall suspected but ordinary progression frontiers remain → prioritize normal progression; secrets are usually optional. |
| **R78** | Exit path identified and survival/resources sufficient → favor completion over unnecessary extermination. |
| **R79** | Same encounter kills agent repeatedly → alter route, opening target, weapon or trigger timing; do not repeat identical policy. |
| **R80** | Current combat state has become unwinnable from present geometry → disengage immediately even if target is nearly dead. |

**Strategic state machine**

```text
SURVIVAL
    ↓
POSITIONAL CONTROL
    ↓
THREAT CONTROL
    ↓
RESOURCE STABILITY
    ↓
ORIENTATION / MEMORY UPDATE
    ↓
PROGRESSION
    ↓
OPTIONAL VALUE
```

Never allow a lower state to block a higher one. For example, do not finish collecting ammunition while being shot, do not continue exploration while surrounded, and do not continue shooting a baron while an Arch-vile attack requires immediate cover.

**Trap protocol**

```text
BEFORE conspicuous key / strong weapon / major switch / suspicious teleporter:
    identify retreat direction
    identify nearest hard cover
    select crowd-capable weapon
    ensure rocket safety if rocket selected

TRIGGER while moving

AFTER trigger:
    scan rear
    scan left/right
    detect new doors/closets/teleports
    create distance
    engage only once escape space is preserved
```

**Open-room protocol**

```text
ENTER only far enough to reveal threats.
DO NOT run to room center automatically.
Prefer perimeter/cover that preserves retreat.
Classify:
    hitscan?
    Arch-vile/Pain Elemental?
    incoming projectile?
    melee pressure?
    possible infighting?
Then choose fight geometry.
```

**Doorway protocol**

```text
OPEN
→ step/back away
→ let enemies expose themselves
→ remove highest-priority exposed threat
→ prevent Pinkies from blocking doorway
→ re-enter only when simultaneous threat count is manageable.
```

**Resource protocol**

```text
Health:
    <50 → become conservative
    <25 → disengage from optional combat
    <10 → survival/health/exit only

Ammo:
    low shells → shotgun efficiency
    low rockets → no routine single-target rockets
    low cells → reserve plasma/BFG
    broadly low → infight, chainsaw/berserk where safe, bypass

Armor:
    low armor → accessible armor gains high priority,
                especially before unavoidable encounter.
```

**Navigation memory**

Maintain:

```text
known_keys
known_locked_doors[color → location]
pressed_switches
known_lifts
known_teleporters[source → destination]
unexplored_frontiers
recent_geometry_changes
resource_caches
route_visit_counts
failed_route_patterns
exit_candidates
```

Use **event-driven re-planning**. A key, switch, teleporter, large pickup, monster-closet opening or moving sector can invalidate the previous map model.

**Boss policies**

```text
CYBERDEMON:
    preserve lateral space
    avoid nearby walls
    track three-rocket volleys
    BFG > plasma/rockets
    use infighting when safe
    bypass if map permits
    never default to expert point-blank SSG tactics

SPIDER MASTERMIND:
    break LOS frequently
    avoid long exposed damage races
    use BFG/plasma/rockets
    exploit infighting
    close-range BFG only with controlled approach

ICON OF SIN:
    objective race, not extermination
    activate top switch
    use rising central platform
    time rockets into exposed brain
    remove only monsters that stop firing cycle,
      especially Arch-viles, Pain Elementals,
      Revenants and Arachnotrons
    lure excess monsters away rather than clearing endlessly
```

The Cyberdemon/Spider policies follow their documented rocket and sustained-hitscan behavior; Doom II MAP30 explicitly spawns monsters indefinitely and requires attacking the hidden boss target with timed rockets, which is why ordinary “clear then progress” logic is pathological there. citeturn13view1turn14view0turn21view6


## Ultra-Compressed Runtime Policy and Decision Pseudocode

**Artifact C — runtime policy**

```text
PRIMARY ORDER
SURVIVE → GET SPACE/COVER → STOP SNOWBALL THREATS → REMOVE RELIABLE
DAMAGE → STABILIZE → RESOURCES → ORIENT → PROGRESS → OPTIONAL KILLS.

MOVEMENT
Stay moving during active combat unless cover/aim timing requires otherwise.
Preserve lateral space.
Projectile incoming → strafe across trajectory.
Unknown damage → move first, scan second.
Do not back blindly for long.
Near wall/corner → escape diagonally before continuing damage.
Surrounded → kill/move through weakest blocking lane; do not duel tanks.
Open unknown door → open, retreat, observe, then enter.
Do not circle-strafe unless the circle is known clear.
Damaging floor → shortest safe crossing.
Cyberdemon present → do not hug walls.

THREATS
Arch-vile attack → reach hard LOS cover first; then kill rapidly.
Pain Elemental spawning souls → kill early.
Chaingunner exposed → break LOS or kill/suppress immediately.
Charging Lost Soul → interrupt immediately.
Revenant missile → use wall/pillar or late lateral+forward dodge.
Arachnotron stream → lateral movement + LOS break.
Mancubus volley → keep distance; do not over-dodge.
Pinky/Spectre → dangerous mainly when close or blocking movement.
Baron/Hell Knight/Cacodemon at range + open space → often lower priority.
Rank enemies by immediacy, reliability, special ability and movement denial,
not by HP.

WEAPONS
Weak lone target → efficient weapon.
Shotgun → default weak/medium target.
SSG → close/mid tough target or dense group.
Chaingun → interruption, Lost Souls, revenants, hitscan threats; tap at range.
Rockets → clusters/tough enemies with blast clearance.
Enemy/wall close → never rocket.
Plasma → emergency DPS / dangerous special target.
BFG → boss, dense high-value group or tactical emergency.
Do not waste cells/rockets when shotgun safely solves fight.
Berserk/chainsaw → ammo saving only when melee geometry is controlled.
Stop firing if shot is obstructed.

INFIGHTING
Mixed powerful groups + open maneuvering space → try to make one attack another.
Once infighting starts → disengage and let it work.
Do not force infighting while exposed to immediate hitscan/special danger.
Do not assume same-species projectiles will hurt one another.

RESOURCES
Large health + nearly full + safely revisitable → leave and remember.
Health <50 → reduce optional risk.
Health <25 → seek health/armor and disengage aggressively.
Health <10 → survival/escape/health/exit dominates everything.
Backpack → normally take.
Invulnerability/radsuit → take immediately before relevant danger.
Low ammo → shotgun efficiency, controlled chaingun, infighting, melee if safe,
bypass optional enemies.
Preserve roughly two BFG shots of cells when future danger is unknown if feasible.

TRAPS
Conspicuous key/weapon/powerup in suspicious room → assume ambush possible.
Before trigger: choose escape direction and weapon.
After key/switch/major pickup: move and scan rear/flanks immediately.
New closet/teleport attack → retreat toward known geometry, not deeper unknown space.

NAVIGATION
Track visited areas, unexplored branches, keys, locks, switches, lifts,
teleporters, hazards, useful cached pickups and exit candidates.
Key acquired → revisit matching known door.
Switch pressed → inspect environment and previously blocked routes for changes.
Visible inaccessible item → record clue; do not repeatedly collide with barrier.
Prefer unvisited frontier over already-cleared corridor.
Same route twice with no state change → blacklist temporarily.
Stuck → new key/lock → recent switch/change → unexplored branch → lift/platform
→ inaccessible item clue → anomalous wall.
Secrets are usually optional; do not secret-hunt while ordinary progression remains.
Exit found + survivable route → complete level rather than demand 100% kills.

FAILSAFE
Never stand still merely to think.
Never keep firing at an obstacle.
Never persist with a poor weapon after repeated ineffective cycles.
Never chase a low-priority enemy away from progression.
Never repeat an identical failed encounter indefinitely.
After repeated death change position, target priority, weapon or entry route.
If present position becomes unwinnable → disengage immediately.
```

The highest-value runtime assumptions behind this compressed policy are the documented advantages of player mobility and infighting, the special mechanics of Arch-viles/revenants, the sustained danger of hitscan attackers, and the safety/resource characteristics of Doom's weapons. citeturn20view0turn20view1turn2view0turn15view1turn9view0turn5view1turn5view2

**Decision pseudocode**

```text
state:
    health
    armor
    ammo[4]
    weapons
    keys

    current_node
    visited_edges
    frontier_nodes
    locked_doors_by_color
    switches
    lifts
    teleport_links
    resource_caches
    hazard_regions
    exit_candidates

    recent_progress_events
    failed_routes
    encounter_failures

loop forever:

    OBSERVE_FRAME()

    # ----- REFLEX LAYER -----

    detect:
        visible_enemies
        enemy_attack_states
        projectiles
        recent_damage
        nearest_walls
        lateral_space
        cover
        floor_hazard
        escape_lanes

    if imminent_self_rocket_risk():
        cancel_fire()
        select_non_explosive_weapon()
        move_away_from_blast_geometry()
        continue

    if imminent_projectile_collision():
        execute_best_lateral_dodge()
        continue_observation_while_moving()

    if archvile_attack_about_to_complete():
        if reachable_hard_cover():
            move_to_cover()
        else:
            execute_emergency_damage_or_escape()
        continue

    if taking_damage_from_unknown_direction():
        move_toward_known_safe_space()
        perform_fast_peripheral_and_rear_scan()
        clear_target_lock()

    if surrounded_or_cornered():
        lane = safest_escape_lane()
        blocker = enemy_blocking(lane)

        if blocker:
            choose_fast_lane_clearing_weapon(blocker)
            attack_while_moving_toward(lane)
        else:
            move_through(lane)

        continue

    if on_damaging_floor():
        prioritize_shortest_safe_exit_from_hazard()

    # Prevent stationary combat pathology
    if combat_active() and movement_near_zero() and not deliberate_cover_hold():
        initiate_safe_lateral_or_cover_motion()


    # ----- THREAT MODEL -----

    for enemy in visible_enemies:

        enemy.priority_class =
            IMMINENT if attack_about_to_land(enemy)
            else COMPOUNDING if enemy is ARCHVILE or active_PAIN_ELEMENTAL
            else HITSCAN if exposed_hitscan(enemy)
            else PRESSURE if enemy is charging_LOST_SOUL
                              or dangerous_REVENANT
            else DENIAL if sustained_projectile_pressure(enemy)
            else ORDINARY

        enemy.score =
            priority_class_weight(enemy)
            + time_to_damage_factor(enemy)
            + expected_damage_factor(enemy)
            + proximity_factor(enemy)
            + escape_denial_factor(enemy)
            - cover_factor(enemy)
            - safe_infighting_factor(enemy)

    target = highest_scoring_enemy()


    # ----- POSITION BEFORE DAMAGE -----

    if current_position_is_bad():
        destination = best_reposition_location(
            cover=True,
            lateral_space=True,
            retreat_route=True,
            low_hazard=True
        )
        move_toward(destination)

    if exposed_hitscan(target) and hard_cover_reachable():
        favor_cover_over_damage_race()


    # ----- INFIGHTING OPPORTUNITY -----

    if mixed_powerful_enemy_groups()
       and adequate_movement_space()
       and no_immediate_special_or_hitscan_emergency():

        if safe_infighting_geometry_exists():
            position_enemy_attack_through_other_enemy()
            dodge_attack()
            if infighting_confirmed():
                disengage_to_safe_observation_position()


    # ----- WEAPON SELECTION -----

    candidate_weapon = choose_weapon(
        target_type=target.type,
        target_distance=distance(target),
        enemy_density=local_enemy_density(),
        ammo=ammo,
        blast_clearance=rocket_clearance(),
        urgency=target.score,
        future_reserve=desired_ammo_reserve()
    )

    if candidate_weapon is ROCKET_LAUNCHER:
        if close_enemy() or close_wall() or close_lost_soul():
            candidate_weapon = best_safe_alternative()

    if candidate_weapon is BFG:
        if target_value_too_low():
            candidate_weapon = efficient_alternative()

    if current_weapon_is_effective()
       and weapon_switch_has_no_meaningful_gain():
        keep_current_weapon()
    else:
        switch(candidate_weapon)


    # ----- ATTACK CONTROL -----

    if clear_line_of_fire(target):
        fire_appropriate_burst(target)
    else:
        stop_firing()
        reposition_for_line_of_fire()

    if repeated_shots_hit_geometry():
        stop_firing()
        mark_current_fire_lane_blocked()
        sidestep_or_reacquire()

    if target_dies_or_becomes_occluded_or_low_priority():
        rerank_immediately()


    # ----- RESOURCE MANAGEMENT -----

    update_health_armor_ammo()

    if health < 10:
        mode = CRITICAL_SURVIVAL
        suppress_optional_combat()
        target_nearest_known_health_or_exit()

    elif health < 25:
        mode = EMERGENCY
        favor_cover_and_known_health()
        allow_heavy_ammo_for_escape()

    elif health < 50:
        mode = CONSERVATIVE
        reduce_optional_exposure()

    else:
        mode = NORMAL

    if ammo_broadly_low():
        increase_infighting_weight()
        increase_bypass_weight()
        increase_cached_ammo_priority()
        decrease_heavy_weapon_cleanup_use()

    for pickup in visible_pickups:
        value = pickup_value(pickup, state)

        if large_health_pickup(pickup)
           and health_near_cap()
           and safely_revisitable(pickup):
            remember_as_resource_cache(pickup)
        elif pickup_trigger_risk_high(pickup):
            prepare_escape_and_ambush_weapon()
            collect_while_moving(pickup)
            trigger_post_pickup_scan()
        elif value_high:
            collect(pickup)


    # ----- EVENT / MAP MEMORY -----

    if key_acquired():
        remember_key()
        set_navigation_goal(nearest_known_matching_locked_door())

    if switch_pressed():
        remember_switch_and_timestamp()
        mark_nearby_and_known_blocked_geometry_for_reinspection()

    if geometry_changed():
        update_topological_map()
        clear_related_route_blacklists()

    if teleported():
        record_source_destination_link()
        perform_arrival_threat_scan()

    if conspicuous_progress_trigger():
        perform_rear_and_flank_scan()


    # ----- STRATEGIC NAVIGATION -----

    if exit_accessible() and route_risk_acceptable():
        navigation_goal = EXIT

    elif newly_acquired_key_matches_known_door():
        navigation_goal = best_matching_door()

    elif recent_switch_likely_changed_known_area():
        navigation_goal = highest_probability_changed_area()

    elif unvisited_frontiers_exist():
        navigation_goal = best_frontier(
            novelty=True,
            distance=True,
            hazard=False,
            progression_likelihood=True
        )

    else:
        navigation_goal = systematic_stuck_search()


    # ----- LOOP PREVENTION -----

    edge = current_navigation_edge()
    visited_edges[edge] += 1

    if same_route_sequence_repeated()
       and no_progress_event_since_previous_cycle():

        blacklist_route_temporarily(edge)
        navigation_goal = next_best_distinct_frontier()

    if repeated_collision_with_inaccessible_object():
        mark_object_as_currently_inaccessible()
        stop_retrying_until_state_change()

    if route_failed_repeatedly():
        change_route_or_required_precondition()


    # ----- REPEATED DEATH / ENCOUNTER LEARNING -----

    if encounter_has_prior_failures():
        signature = current_encounter_signature()

        if signature_matches_previous_failure(signature):
            force_change_at_least_one_of(
                entry_route,
                first_target,
                first_cover,
                opening_weapon,
                trigger_timing,
                movement_direction
            )

        if same_encounter_failed_three_times():
            force_strategic_replan(
                seek_resources=True,
                alternative_route=True,
                infighting=True,
                bypass_if_possible=True
            )


    # ----- UNWINNABLE-STATE DETECTOR -----

    if expected_time_to_death()
       < expected_time_to_stabilize_current_position():

        cancel_nonessential_attack()
        choose_best_escape_route()
        spend_high_value_ammo_if_needed_to_open_escape()
        disengage()


    ACT()
    OBSERVE_AGAIN()
```

This loop deliberately prevents the characteristic AI pathologies in the brief: movement is checked before attack optimization; firing requires a valid line; heavy ammunition is gated by target value and blast safety; target ranking is continuously interruptible; navigation edges have visit counts and blacklists; key/switch events trigger map-memory updates; attacks from outside the visual center break target fixation; and repeated deaths force policy variation instead of identical retries. fileciteturn0file0

The central principle is simple: **classic Doom rewards an agent that preserves options**. Space, cover, ammunition, health, remembered routes and unexplored frontiers are all forms of optionality. The best action is usually not the one that deals the most damage this instant; it is the one that leaves the agent in the strongest state for the next several decisions. That conclusion follows consistently from Doom's movement mechanics, monster behavior, resource model, infighting system and level-trigger architecture. citeturn20view1turn19view1turn0search5turn0search17turn0search19