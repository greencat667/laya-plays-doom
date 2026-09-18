# Tiny Doom Runtime Policy — 200 Rules

## Core priority

1. Imminent lethal damage → evade first.
2. Bad position → escape before attacking.
3. Lost movement space → restore space.
4. Dangerous special enemy → neutralise or avoid.
5. Exposed hitscan enemy → kill or break line of sight.
6. Immediate blocker → remove it.
7. Stable fight → optimise weapon and ammunition.
8. Safe resources → collect when useful.
9. No active danger → update orientation.
10. Safe and stable → pursue level progress.
11. Progress beats unnecessary kills.
12. Survival overrides every lower priority.

## Movement

13. Active combat → keep moving.
14. Do not stand still merely to aim.
15. Projectile approaching → strafe across its path.
16. Do not dodge directly backwards unless rear space is known clear.
17. Preserve lateral movement space.
18. Near wall → move away before continuing prolonged combat.
19. Near corner → escape before becoming trapped.
20. Surrounded → escape through the least-blocked direction.
21. Enemy blocks escape → attack that enemy first.
22. Open arena → move around threats, not directly toward them.
23. Unknown room → enter cautiously rather than rushing to centre.
24. Door reveals enemies → retreat into known territory.
25. Narrow corridor → prevent enemies reaching both sides.
26. Hard cover available → use it against ranged threats.
27. Pillar available → orbit it against projectiles when safe.
28. Circle-strafe only when the full movement path is known safe.
29. Do not circle-strafe beside damaging floors or unseen corridors.
30. Taking damage from unknown direction → move immediately.
31. Unknown attacker → scan while moving, never while stationary.
32. Damaging floor → take shortest viable path to safety.
33. Crossing exposed ground → move diagonally when practical.
34. Retreating → retreat toward known safe geometry.
35. Never retreat deeper into unexplored space unless necessary.
36. Large enemy nearby → preserve enough space to sidestep.
37. Cyberdemon nearby → stay away from walls.
38. Teleport arrival → move and scan immediately.

## Threat selection

39. Rank threats by danger, not hit points.
40. Attack about to land → highest priority.
41. Arch-vile attacking → reach hard cover immediately.
42. Arch-vile exposed → usually kill rapidly.
43. Arch-vile among corpses → increase priority.
44. Pain Elemental spawning Lost Souls → kill early.
45. Chaingunner with line of sight → kill or break LOS immediately.
46. Spider Mastermind firing → hard cover before damage race.
47. Charging Lost Soul → interrupt immediately.
48. Revenant missile tracking → use wall/pillar if possible.
49. Revenant close → create distance.
50. Arachnotron firing continuously → strafe widely or break LOS.
51. Mancubus firing → respect spread; do not over-dodge into another projectile.
52. Shotgun Guy close → eliminate quickly.
53. Pinky close and blocking movement → high priority.
54. Spectre close → treat as Pinky with extra safety margin.
55. Cacodemon distant in open space → usually low urgency.
56. Imp distant in open space → usually low urgency.
57. Hell Knight distant with room to strafe → usually lower priority.
58. Baron distant with room to strafe → usually lower priority.
59. High-health enemy behind cover → deprioritise.
60. Enemy safely infighting → lower priority.
61. New dangerous enemy appears → immediately rerank targets.
62. Current target retreats away from objective → do not chase automatically.

## Geometry

63. Reduce the number of enemies that can see you simultaneously.
64. Fight from doorways when they reduce enemy exposure.
65. Do not remain in doorway if melee enemies can trap you there.
66. Open door → step back and observe.
67. Corner → use to interrupt hitscan fire.
68. Pillar → use to absorb projectiles.
69. Narrow passage → funnel enemies.
70. Wide room → maintain escape routes.
71. High ground is useful only if it does not trap movement.
72. Do not fire explosives into nearby walls.
73. Shot blocked by geometry → stop firing.
74. Repeated shots hit obstacle → reposition.
75. Unknown space behind you → avoid long backward movement.

## Weapon choice

76. Weak isolated enemy → use efficient weapon.
77. Standard shotgun → default for routine weak/medium targets.
78. Super shotgun → close or medium tough target.
79. Super shotgun → dense close group.
80. Long range → prefer shotgun or controlled chaingun over SSG.
81. Chaingun → use for rapid interruption.
82. Chaingun → useful against Lost Souls.
83. Chaingun → useful against Revenants.
84. Chaingun at distance → short bursts.
85. Dense enemies at safe range → rockets.
86. Tough enemy at safe range → rockets are acceptable.
87. Enemy close → do not use rockets.
88. Wall close → do not use rockets.
89. Lost Soul approaching → do not use rockets.
90. Urgent dangerous enemy → use plasma if available.
91. Safe weak enemy → conserve plasma.
92. Boss or dense high-value group → consider BFG.
93. Tactical collapse imminent → BFG is acceptable.
94. Do not spend BFG on trivial targets.
95. Chainsaw → isolated manageable melee enemy only.
96. Berserk fist → useful for safe ammo conservation.
97. Mixed dangerous group → avoid melee weapons.
98. Current weapon works well → avoid unnecessary switching.
99. Weapon repeatedly ineffective → change weapon or position.
100. Stronger weapon is not automatically the better weapon.

## Ammunition

101. Low shells → stop using SSG on trivial enemies.
102. Low rockets → reserve for clusters and dangerous targets.
103. Low cells → reserve plasma/BFG for serious threats.
104. BFG available → preserve emergency cells when practical.
105. Broadly low ammo → seek infighting.
106. Broadly low ammo → bypass optional enemies.
107. Broadly low ammo → favour shotgun efficiency.
108. Ammo scarcity → value dropped weapons/ammunition more highly.
109. Do not expend ammunition merely to achieve 100% kills.

## Infighting

110. Mixed strong enemies + space → attempt to provoke infighting.
111. Place one monster's attack path through another when safe.
112. Once monsters fight each other → disengage and let them work.
113. Do not enter danger merely to create infighting.
114. Hitscan exposure too high → abandon infighting attempt.
115. Arch-vile threatening → survival takes priority over infighting.
116. Infighting enemy blocks progression → reassess rather than wait indefinitely.

## Health, armour and pickups

117. Health below 50% → reduce optional risk.
118. Health below 25% → actively seek health/armour and disengage more readily.
119. Health below 10% → survival, health and exit dominate everything.
120. Low health → heavy ammunition may be spent to create escape.
121. Armour accessible and armour low → collect it.
122. Large health pickup + nearly full health → leave it if safely revisitable.
123. Remember useful pickups deliberately left behind.
124. Backpack accessible → normally collect.
125. Berserk + health below 100 → value strongly.
126. Radiation suit + hazard ahead → collect immediately before entering hazard.
127. Invulnerability + major fight ahead → collect just before committing.
128. Do not waste timed power-ups while exploring.
129. Partial invisibility → expect less predictable projectile paths.

## Traps

130. Conspicuous key in suspiciously empty room → assume possible ambush.
131. Conspicuous weapon → assume possible ambush.
132. Conspicuous power-up → assume possible ambush.
133. Major switch → expect geometry or enemy state to change.
134. Before suspicious pickup → identify escape route.
135. Before suspicious trigger → choose combat-ready weapon.
136. Trigger suspicious event while moving.
137. After key/pickup/switch → scan rear and flanks.
138. New door or wall opens → assume monsters may emerge.
139. Teleport sounds/new enemies → retreat toward known geometry if overwhelmed.
140. Apparently empty room → do not assume safety.

## Navigation

141. Maintain memory of visited areas.
142. Maintain memory of unexplored branches.
143. Maintain memory of locked doors.
144. Record locked-door colour.
145. Maintain memory of switches.
146. Maintain memory of lifts.
147. Maintain memory of teleporters.
148. Maintain memory of useful resource caches.
149. Maintain memory of hazards.
150. Maintain memory of probable exit locations.
151. New key acquired → revisit matching remembered door.
152. Switch pressed → inspect nearby changes.
153. Switch pressed → reconsider previously blocked routes.
154. Geometry changes → update map memory immediately.
155. Visible inaccessible item → treat as navigation clue.
156. Do not repeatedly collide with inaccessible item/barrier.
157. Unexplored branch exists → prefer it to repeated cleared corridors.
158. Same route twice without progress → mark temporarily unproductive.
159. Progress means new area, key, switch, useful item or changed geometry.
160. No progress → choose different frontier.
161. Stuck → check unused key doors first.
162. Still stuck → inspect effects of recent switches.
163. Still stuck → explore unvisited branches.
164. Still stuck → inspect lifts and moving platforms.
165. Still stuck → inspect visible inaccessible areas.
166. Still stuck → inspect unusual walls/textures near meaningful locations.
167. Secrets are secondary to normal progression.
168. Exit found + survivable route → favour completing the level.

## Combat failure prevention

169. Never keep firing at a wall.
170. Never stand still because target identification is uncertain.
171. Never blindly reverse for long.
172. Never let aiming override movement.
173. Never tunnel-vision one target.
174. Never chase a trivial enemy into unknown space.
175. Never use rockets at point-blank range.
176. Never spend scarce ammunition merely because it is available.
177. Never fight every monster by default.
178. Never collect every pickup immediately by default.
179. Never assume high HP means high priority.
180. Never assume an empty area is safe.
181. Never repeatedly attempt a blocked route without state change.
182. Never allow optional combat to prevent reaching an exit.

## Recovery

183. Unknown damage → move, seek cover, scan all directions.
184. Surrounded → identify escape lane before selecting kill target.
185. Cornered → use high DPS on the enemy blocking escape.
186. Low ammo → bypass, infight, conserve.
187. Low health → shorten exposure and seek known resources.
188. Lost → return to last meaningful progression event.
189. Repeated corridor loop → blacklist that route temporarily.
190. Same encounter death twice → change opening tactic.
191. Repeated death → change at least one of route, target, weapon or movement.
192. Three similar deaths → radically re-plan encounter.
193. Seek additional resources after repeated failure.
194. Try a different entrance after repeated failure.
195. Trigger encounter then retreat farther if previous position failed.
196. Engineer infighting if direct combat repeatedly fails.
197. Bypass encounter if progression permits.
198. Position becoming unwinnable → disengage immediately.
199. Nearly dead enemy does not justify remaining in lethal geometry.
200. When uncertain between damage and preserving options → preserve options.
