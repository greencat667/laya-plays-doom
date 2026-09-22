# Laya Plays Doom

Can [Laya](https://github.com/NandhaKishorM/laya) — a fast, **non-autoregressive**
"System 1" decision model (`convaiinnovations/laya` on Hugging Face) that
answers typed questions over a state in a single forward pass, with no
text generation at all — act as an autonomous game-playing agent in Doom,
using nothing but a compact textual description of the world and a fixed
set of `choice` labels?

```
ViZDoom → perception → compact world-state text → Laya.predict() → choice label → ViZDoom
```

This project reuses a Doom-playing pipeline — ViZDoom setup, perception,
world-state text encoding, metrics, the stuck-recovery/threat-response
safety nets — that was originally built and validated in a related but
separate experiment, which asked the same question of Cactus Compute's
Needle 3, an autoregressive tool-calling model, as the decision engine
instead of Laya. That earlier project is **not part of this repository**
and isn't published alongside it, so where this README cites specific
Needle numbers below (latency, confidence, crowding-fix behaviour), treat
them as prior/comparative results measured on that other setup, not
something you can click through and verify here. The **only** thing that
changes in this repo is the decision engine. This project is fully
self-contained (own venv, own `requirements.txt`, own git repo) and has
no dependency on that earlier project or on `cactus-needle`.

**Status: Stage 1 (four-action proof of concept) is built and verified
running** on Apple Silicon macOS with real ViZDoom + the real
`convaiinnovations/laya` checkpoint — see [Verified
behaviour](#verified-behaviour) below for actual measured output, not a
projection. **This checkpoint is fine-tuned for email routing, moderation,
and support-ticket triage — not Doom — and out of the box it showed:**
with `shoot`/`attack` competing directly against move/turn inside a
single `choice` question, it never once fired on 8 hand-built states, and
a real 150-step episode picked `turn_right` on literally every decision,
including 7 times with an enemy dead ahead. **That's since been fixed**
(see [The real fix](#the-real-fix-pull-combat-out-into-its-own-noul-question)
below): pulling combat out into its own `noul` "should_shoot" question —
one of Laya's own documented primitives, used the way the Needle-based
experiment used Needle's `triggers=` — plus two deterministic guards (`AMMO>0`,
bearing exactly `front`) checked in code rather than trusted from the
model, took it from worse-than-random (0 kills/5 episodes) to matching
the hand-written heuristic baseline: real, verified 10-episode numbers,
`mean_kills=1.00`, `completion_rate=1.00`, `mean_latency_ms=22.8` — see
[Controller comparison, real run](#controller-comparison-real-run).
**Since then**, real wayfinding was added on top (verified `ANGLE`
convention, frontier-directed exploration, a secret-door search, a
wall-following fallback) and three more real bugs were found and fixed on
long real `--scenario level` runs (a wall-hugging freeze, and two
"repeat the same fruitless action forever" stalls) — see
[Stage 4/5](#stage-45-real-wayfinding-secret-search-and-three-more-real-bugs)
for the full real before/after data. Exploration coverage, survival, and
distance travelled all improved measurably on real runs; **a real level
exit (`completed=True`) was still never observed**, reported plainly
rather than glossed over.

## Demo

[![Laya plays Doom](https://img.youtube.com/vi/-Vp0UT_nzz8/0.jpg)](https://www.youtube.com/watch?v=-Vp0UT_nzz8)

## Architecture

```
ViZDoom (vizdoom.DoomGame)
    │  GameState: game variables, labels buffer, depth buffer
    ▼
perception.py           — GameState → Perception (health/armor/ammo, visible
    │                      enemies & pickups with bearing/distance buckets,
    │                      wall/open-path flags). No screen pixels, no CNN.
    ▼
state_encoder.py         — Perception (+ previous tick, + memory mode) →
    │                      compact text block, e.g.:
    │                        HEALTH 74
    │                        ARMOR 20
    │                        ENEMY imp front-right near
    │                        WALL ahead near
    │                        LAST_ACTION turn_right
    │                        LAST_RESULT enemy_closer
    ▼
laya_agent.py            — LayaAgent.decide(): laya.Agent.predict(state, questions)
    │                      → per-label probabilities + calibrated confidence,
    │                      in ONE forward pass, no decode loop
    ▼
actions.py               — canonical action name → ViZDoom button presses + tics
    │                      (Laya's chosen label IS the canonical action name —
    │                      no tool-name/arguments indirection to resolve)
    ▼
doom_env.py               — DoomEnv.execute(action_name) → game.make_action(...)
    ▼
ViZDoom (repeat)
```

`controller.py` is the same explicit per-control-cycle loop as that
earlier pipeline — one `agent.decide()` call per Doom step, no autonomous loop.
`random_agent.py`, `heuristic_agent.py` and `laya_agent.py` (in
`experiments/`) all implement the same tiny interface —
`decide(perception, encoded_state) -> Decision`, `reset()` — and run
through the *exact same* `perception.py`/`state_encoder.py` pipeline via
`controller.run_episode()`, which is what makes them comparable.

## No commercial Doom files needed

Same setup as that earlier pipeline, reused as-is: for `basic`/`my_way_home`
scenarios, ViZDoom ships its own small scenario WADs and falls back
automatically to the **Freedoom2** IWAD bundled inside the `vizdoom` pip
package when `doom2.wad` isn't present. `doom_env.py` never sets
`doom_game_path` for those. Zero manual asset setup, verified on this
machine.

## Verified behaviour

Real output from this repository, run on this machine (Apple Silicon
macOS, Python 3.11, `vizdoom==1.3.0`, `laya==0.1.6`, `torch==2.14.0`, the
`convaiinnovations/laya` checkpoint fetched from Hugging Face on first
run). Nothing below is projected or invented — every number came from an
actual command run against this repo.

### Installing and loading Laya

```bash
$ pip install laya
# ... resolved 43 packages, including torch==2.14.0, transformers==5.17.0 ...
# Installed cleanly, no build-from-source steps, no extra system deps.

$ python -c "import laya, time; t0=time.perf_counter(); a=laya.load('convaiinnovations/laya'); print(time.perf_counter()-t0, a.device, a.dtype)"
Fetching 15 files: 100%|██████████| 15/15 [02:48<00:00, 11.22s/it]
189.65422325002146 s   mps   torch.float32
```

First-run load (weights not yet cached) took **189.7s**, almost all of it
downloading — the actual weights blob (`model.safetensors`) is
**842,609,210 bytes (~803MiB / ~843MB)**, not the "small" footprint the
project's own marketing framing might suggest at a glance; the rest of the
~3.6MB in the snapshot directory (tokenizer, encoder config, eval scripts)
is negligible next to that. A second `laya.load()` with the weights
already in `~/.cache/huggingface/hub/` took **19.1s** (model
construction + moving weights onto the device, no network). Device
resolution is automatic and worked correctly on this Apple Silicon
machine with **no configuration**: `laya.Agent.__init__` checks
`torch.cuda.is_available()` then `torch.backends.mps.is_available()` then
falls back to CPU — here it picked `mps`, and forces `float32` on `mps`/
`cpu` (`fp16`/`bf16` only apply on `cuda`). This was confirmed by reading
`laya.Agent.__init__`'s actual source in the installed package, not
assumed from the README.

### A smoke test: real probabilities on a hand-built world-state string

```
$ python -c "... agent.predict(state, questions) ..."
state:
GOAL stay_alive kill_threats explore collect find_exit
HEALTH 100
ARMOR 0
AMMO 50
ENEMY cacodemon front near
PATH left open
PATH right open

result (first call after a fresh model construction — MPS kernel warmup, see below):
{
  "model": "laya-rl-agent",
  "answers": {
    "action": {
      "type": "choice",
      "choice": "turn_right",
      "probabilities": {
        "move_forward": 0.0588, "turn_left": 0.2632, "turn_right": 0.4553,
        "shoot": 0.0981, "wait": 0.1246
      },
      "confidence": 0.1529
    }
  },
  "usage": {"input_tokens": 174, "output_tokens": 0}
}
latency: 1486.02 ms   # first call after loading — MPS graph/kernel warmup, not representative
```

The very first `predict()` call after a fresh `laya.load()` took 1.49s
(one-time MPS warmup cost); the next two calls dropped to 173ms and
190ms; a run of 30 more calls settled at a **mean of 21.3ms** (p50
21.3ms, p95 22.1ms, min 20.1ms, max 22.4ms) — all measured on this
machine, isolated, nothing else competing for the GPU/CPU.

### Does Laya need a `reset()` between decisions? Tested, not assumed

The related Needle-based experiment needed `NeedleAgent.reset()`/`reset_every`
specifically because a long streak of autoregressive `complete()` calls
without resetting could make a *later* decode pathologically slow on
repetitive input (confirmed there: 38 calls in, 57s+ and still climbing).
Laya has no `reset()` method at all — `dir(laya.Agent)` on the installed
package shows only `predict` and `system_one` — and no persistent
conversation state between calls for that kind of staleness to
accumulate in. Rather than assert "so this failure class doesn't apply
here," it was tested directly: **200 consecutive `predict()` calls with
the byte-identical repetitive input** (same world-state string, same
questions dict, every time):

```
mean=21.56ms p50=21.56ms p95=22.18ms min=20.43ms max=25.56ms
first half (calls 1-100) mean=21.58ms, second half (calls 101-200) mean=21.54ms
```

No trend, no blowup, no outlier beyond ordinary jitter. `LayaAgent.reset()`
is therefore a documented no-op (see `laya_doom/laya_agent.py`) — a
genuine architectural difference from Needle, verified empirically on
this machine rather than inferred from the API shape alone.

### The honest finding: the base checkpoint doesn't play Doom sensibly

`convaiinnovations/laya`'s public benchmark numbers (routing/moderation/
triage accuracy) are for exactly those domains — nothing in the model
card claims Doom-playing competence, and this project didn't fine-tune it
for that. Tested directly against 8 hand-built, characteristic Doom
world-states (enemy dead ahead near/very-near, enemy off to a side, wall
ahead, open path with no enemy, no ammo with an enemy ahead, low health
with a far enemy) using the shipped criteria wording
(`laya_agent._ACTION_CRITERIA`, adapted from that earlier pipeline's Needle
tool docstrings):

```
enemy_front_near       -> turn_right   conf=0.111  top3=[turn_right:0.421, turn_left:0.246, shoot:0.136]
enemy_front_veryn      -> turn_left    conf=0.126  top3=[turn_left:0.385, turn_right:0.325, wait:0.120]
enemy_front_left       -> turn_left    conf=0.143  top3=[turn_left:0.422, turn_right:0.294, wait:0.137]
enemy_right            -> turn_right   conf=0.185  top3=[turn_right:0.524, turn_left:0.200, wait:0.108]
wall_ahead_near        -> turn_right   conf=0.068  top3=[turn_right:0.319, move_forward:0.257, turn_left:0.231]
open_path_no_enemy     -> turn_left    conf=0.098  top3=[turn_left:0.363, move_forward:0.247, turn_right:0.233]
no_ammo_enemy_front    -> turn_right   conf=0.113  top3=[turn_right:0.419, turn_left:0.256, shoot:0.126]
low_health_enemy_far   -> turn_right   conf=0.129  top3=[turn_right:0.411, turn_left:0.291, wait:0.122]
action distribution: {turn_right: 5, turn_left: 3}
```

Zero `shoot`s, including on `enemy_front_near` (an enemy reported exactly
`front`, ammo available — precisely the condition the `shoot` criteria
describes) and `no_ammo_enemy_front` (where `shoot` should score *lower*
still, and does, but only barely). Confidence never exceeded 0.19 across
any of the 8 states — well below any reasonable "safe to execute
automatically" threshold, which is itself a coherent, informative
signal: the model's own calibration is correctly reporting that it is
guessing near the 5-way-uniform baseline (0.20) for this out-of-domain
task, not confidently wrong.

**One improvement pass was tried, per this project's own no-repeated-
tuning rule** — the Needle-based experiment's approach was to rewrite Needle's
tool triggers/docstrings until behaviour improved; the equivalent lever
here is criteria wording. A second version
(`CRITERIA_V2` in the test script, not shipped) made every condition
maximally literal — spelling out the exact tokens to match (`'WALL ahead
near'`, `'ENEMY ... front '` with a trailing space to exclude
front-left/front-right, an explicit `AMMO > 0` precondition for `shoot`)
instead of prose description. Same 8 states, same model:

```
enemy_front_near       -> shoot        conf=0.045  top3=[shoot:0.292, turn_left:0.269, turn_right:0.193]
enemy_front_veryn      -> shoot        conf=0.057  top3=[shoot:0.310, turn_left:0.274, turn_right:0.194]
enemy_front_left       -> turn_left    conf=0.059  top3=[turn_left:0.330, shoot:0.263, wait:0.162]
enemy_right            -> shoot        conf=0.047  top3=[shoot:0.284, turn_left:0.271, turn_right:0.211]
wall_ahead_near        -> turn_left    conf=0.021  top3=[turn_left:0.259, move_forward:0.250, shoot:0.188]
open_path_no_enemy     -> move_forward conf=0.036  top3=[move_forward:0.315, turn_left:0.202, shoot:0.201]
no_ammo_enemy_front    -> shoot        conf=0.039  top3=[shoot:0.283, turn_left:0.264, turn_right:0.194]
low_health_enemy_far   -> shoot        conf=0.051  top3=[shoot:0.341, turn_left:0.226, turn_right:0.184]
action distribution: {shoot: 5, turn_left: 2, move_forward: 1}
```

This is **not reported as an improvement** — it swapped one bias for
another. `shoot` now fires on `enemy_right` and `low_health_enemy_far`
(neither has bearing exactly `front`) and on `no_ammo_enemy_front`
(explicitly told `AMMO > 0` was required, `AMMO 0` in the state, still
chose `shoot`). Confidence dropped further (0.02–0.06, i.e. *more*
uniform, not less). Being more literal about the tokens to match did not
make the model attend to them any more reliably — consistent with this
being a capability gap for this exact checkpoint on this exact task, not
a wording problem.

A third pass (`CRITERIA_V4` in `scripts/probe_criteria.py`) looked like a
clean win in isolation — every non-shoot label rewritten to a single terse
clause, symmetric with `shoot`'s own wording. Same 8 states:

```
enemy_front_near       -> shoot   conf=0.535  top3=[shoot:0.823, turn_left:0.081, turn_right:0.072]
open_path_no_enemy     -> shoot   conf=0.538  top3=[shoot:0.825, turn_left:0.093, turn_right:0.050]
wall_ahead_near        -> shoot   conf=0.433  top3=[shoot:0.766, turn_left:0.113, turn_right:0.078]
action distribution: {shoot: 8}   # every single one of the 8 states, including the 2 above
```

`shoot` won all 8/8 — including `open_path_no_enemy` (no enemy anywhere in
the state) and `wall_ahead_near`. That's not discrimination, it's a
keyword-bait bias toward whichever label reads as the most concrete,
literal command — worse than useless, not reported as a fix, and not
shipped.

None of these three wording passes are what's shipped. **`CRITERIA_V1`
is still the criteria text in `laya_agent.py`** — but `shoot`/`attack` is
no longer a competing label inside that `choice` question at all. See the
next section for the actual fix.

### The real fix: pull combat out into its own `noul` question

Needle's fix for the equivalent problem in that other experiment ("attack
was being crowded out") was to shrink the competing-tool count, using Needle's own
`triggers=`/tool-count guidance. That lever doesn't exist for Laya's
`choice` type, and shrinking the label set didn't reproduce the effect
here either (see the crowding-analogy test below). But Laya has a
*different* documented primitive that turns out to fit this exact
problem: `noul`, a standalone calibrated `P(true)` question, independent
of any other competing label. Pulling `shoot`/`attack` out of the 4-way
`choice` entirely and asking it as its own `noul` question — "ENEMY
reported at bearing front and AMMO greater than 0: should the player
shoot?" — on the same 8 states, same model:

```
enemy_front_near      -> P=0.682   (correct: shoot)
enemy_front_veryn     -> P=0.482   (correct: shoot)
enemy_front_left      -> P=0.460   (imperfect: not exactly bearing front, fires anyway)
enemy_right           -> P=0.427   (correct: below threshold, no shoot)
low_health_enemy_far  -> P=0.422   (correct: enemy too far, no shoot)
open_path_no_enemy    -> P=0.378   (correct: no enemy, no shoot)
wall_ahead_near       -> P=0.366   (correct: no enemy, no shoot)
no_ammo_enemy_front   -> P=0.512   (WRONG on the gate alone — AMMO>0 precondition ignored)
```

A real, monotonic-ish signal instead of noise — not perfect (the model
still doesn't reliably enforce its own `AMMO>0` precondition, and
`front-left` scores above threshold when strictly it shouldn't), but a
genuine improvement over 0/8. `laya_agent.py` ships this as the default:
`LayaAgent._decide_with_shoot_gate()` calls the `noul` gate first; if
`P >= 0.45` (picked directly from the spread above, not tuned against a
held-out set) **and** a deterministic `AMMO > 0` check passes — code, not
the model, since the model doesn't reliably enforce it — the combat
action fires; otherwise a `choice` call over the remaining movement
labels decides, with combat no longer competing in it at all. This
mirrors that earlier pipeline's own pattern of a deterministic safety net
layered visibly on top of a model decision (`StuckRecoveryConfig`/
`ThreatResponseConfig`), not hidden inside it. `--no-shoot-gate` restores
the single-`choice`-call design for comparison.

The Needle-crowding analogy was also tested directly, since the
methodology should be checked even where the hypothesis turns out not to
transfer: the `full` action set's `attack` label, one state (enemy front,
open path), 11 competing labels vs. a trimmed 5 (mirroring that other
experiment's fix):

```
11 labels -> attack   conf=0.998  attack_prob=0.999
5 labels  -> use      conf=0.002  attack_prob=0.213
```

The opposite of the Needle result — trimming labels made `attack` score
*worse*, not better, on this one trial. Reported as a single-state,
single-trial data point, not a generalizable claim; the honest takeaway
is that Needle's crowding fix (a retrieval-based tool-rendering
mechanism) has no clear analogue in Laya's flat, no-retrieval `choice`
softmax, so it isn't the lever that helped here — the `noul` gate is.

**A second real bug, found only by wiring the gate into an actual
episode, not the 8 isolated states**: with the default `--memory
prev_state`, the gate *under*-fired on the tactically correct case and
*over*-fired on the wrong one. A 3-episode run logged 120 real `shoot`
decisions, all 120 coinciding with a real `ENEMY` line (so it wasn't
firing blind) — but 100% of them at bearing `front-left`/`front-right`
(0 damage, 0 kills) and 0% at exact `front`. Directly inspecting a real
logged step where the enemy *was* at bearing `front`:

```
ENEMY cacodemon front medium
LAST_ACTION turn_right
LAST_RESULT enemy_farther
...
reasoning: should_shoot=0.305  turn_right:0.437 turn_left:0.331 move_forward:0.232 (confidence=0.029)
```

`P=0.305`, below the 0.45 threshold calibrated against the 8 hand-built
states — because those states used `--memory stateless` text (no
`LAST_ACTION`/`LAST_RESULT`/`*_CHANGE` lines), and the real episode's
default `prev_state` text carries several extra lines the calibration
never saw. Confirmed by re-running the identical scenario with `--memory
stateless`: real kills appeared (`mean_kills=0.67`, `mean_damage_given=5.0`
over 3 episodes) where the default-memory run had none. **Fix**: the gate
now builds its own `stateless`-mode encoding directly from the
`Perception` object (`LayaAgent._gate_encoder`, reset once per episode by
`LayaAgent.reset()`), independent of whatever `--memory` mode the caller
is using for the movement choice or logging — decoupling "what the gate
was calibrated against" from "what the experimenter is varying" instead
of quietly overfitting the threshold to one memory mode. After this fix,
with the default `--memory prev_state` (no special flag needed):

```
$ python -m experiments.run --controller laya --episodes 5 --scenario basic --max-steps 80
action_distribution: {'shoot': 184, 'move_forward': 23, 'turn_left': 11}
mean_kills: 0.8   mean_damage_given: 7.0   completion_rate: 0.8
```

Real, substantial, reproducible — not a single lucky run; see the
updated controller-comparison table below. Still not perfect: mean
confidence on the gate path is 0.31–0.59 depending on the run, `AMMO>0`
is enforced by code rather than the model, and `front-left`/`front-right`
still fire somewhat more readily than strict "front" alone would justify
— reported as the current real state, not as fully solved.

### A real 150-step episode, before the fix: stuck turning, not just biased in isolated tests

This ran with the original single-`choice` design (`--no-shoot-gate`) —
kept here as the real evidence that motivated [The real
fix](#the-real-fix-pull-combat-out-into-its-own-noul-question) above, not
as the current default behaviour:

```
$ python -m experiments.run --controller laya --episodes 1 --scenario basic --max-steps 150 --no-shoot-gate
episode 0: steps=150 kills=0 health=100 died=False reward=-450.0 mean_conf=0.135 mean_latency_ms=28.8

action_distribution: {'turn_right': 150}
mean_distance_travelled: 0.0
```

Every single one of 150 real decisions, driven by real ViZDoom state, was
`turn_right` — including **7 separate steps** where the encoded state
read exactly `ENEMY cacodemon front medium` (bearing exactly `front`, the
`shoot` precondition), e.g.:

```
--- ENCODED STATE ---
GOAL stay_alive kill_threats explore collect find_exit
HEALTH 100
ARMOR 0
AMMO 50
ENEMY cacodemon front medium
PATH left open
PATH right open
AREA revisited
LAST_ACTION turn_right
LAST_RESULT no_change
HEALTH_CHANGE +0
AMMO_CHANGE +0
VISIBLE_ENEMIES_CHANGE +0
MOVED no
--- DECISION ---
action: turn_right
confidence: 0.142
reasoning: turn_right:0.529 move_forward:0.188 turn_left:0.187 shoot:0.096 (confidence=0.142)
```

`shoot` was the *lowest*-probability option, not a close second. This is
the same class of finding reported for base Needle in the related
experiment ("mostly turned and almost never chose shoot") — the difference is that
Needle's fix (`triggers=`/tool-count) doesn't transfer to Laya's `choice`
mechanism, but Laya's separate `noul` primitive turned out to be an
equally real, if different, fix — see above.

### Controller comparison, real run

**Before the fix** (single `choice` call including combat, i.e.
`--no-shoot-gate`):

```
$ python -m experiments.compare --controllers random heuristic laya --episodes 5 --scenario basic --no-shoot-gate

                  Controller comparison
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ metric                ┃ random  ┃ heuristic ┃ laya     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ episodes              │ 5       │ 5         │ 5        │
│ mean_steps            │ 324.80  │ 38.20     │ 500.00   │
│ mean_survival_tics    │ 879.80  │ 54.00     │ 1511.00  │
│ mean_kills            │ 0.40    │ 1.00      │ 0.00     │
│ mean_damage_given     │ 6.00    │ 11.00     │ 0.00     │
│ mean_damage_taken     │ 0.00    │ 0.00      │ 0.00     │
│ mean_health_remaining │ 100.00  │ 100.00    │ 100.00   │
│ mean_items_collected  │ 0.00    │ 0.00      │ 0.00     │
│ mean_total_reward     │ -979.80 │ 49.00     │ -1500.00 │
│ death_rate            │ 0.00    │ 0.00      │ 0.00     │
│ completion_rate       │ 0.40    │ 1.00      │ 0.00     │
│ mean_confidence       │ n/a     │ 1.00      │ 0.14     │
│ mean_latency_ms       │ 0.00    │ 0.00      │ 30.92    │
└───────────────────────┴─────────┴───────────┴──────────┘
```

`laya` never once landed a kill or dealt damage across 5 episodes (2,500
real decisions), never completed an episode, and its `mean_total_reward`
was worse than even the `random` baseline.

**After the fix** (default: `noul` shoot gate + deterministic `AMMO>0`/
bearing==`front` guards), same command, no flags:

```
$ python -m experiments.compare --controllers random heuristic laya --episodes 10 --scenario basic

                 Controller comparison
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ metric                ┃ random  ┃ heuristic ┃ laya   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ episodes              │ 10      │ 10        │ 10     │
│ mean_steps            │ 271.90  │ 14.50     │ 23.60  │
│ mean_survival_tics    │ 758.30  │ 30.50     │ 41.00  │
│ mean_kills            │ 0.50    │ 1.00      │ 1.00   │
│ mean_damage_given     │ 6.00    │ 11.50     │ 11.00  │
│ mean_damage_taken     │ 0.00    │ 0.00      │ 0.00   │
│ mean_health_remaining │ 100.00  │ 100.00    │ 100.00 │
│ mean_items_collected  │ 0.00    │ 0.00      │ 0.00   │
│ mean_total_reward     │ -824.40 │ 81.00     │ 67.50  │
│ death_rate            │ 0.00    │ 0.00      │ 0.00   │
│ completion_rate       │ 0.50    │ 1.00      │ 1.00   │
│ mean_confidence       │ n/a     │ 1.00      │ 0.47   │
│ mean_latency_ms       │ 0.00    │ 0.00      │ 22.76  │
└───────────────────────┴─────────┴───────────┴────────┘
```

`laya` now matches the hand-written `heuristic` baseline on every combat
metric that matters (`mean_kills` 1.00 vs 1.00, `completion_rate` 1.00 vs
1.00, `mean_damage_given` 11.0 vs 11.5) at roughly a third of its own
earlier latency and a fraction of Needle's (22.8ms vs Needle's real
115–356ms, see [Laya vs Needle](#laya-vs-needle) below) — while `random`
still only manages 0.5 kills / 50% completion. Repeated runs varied
(separate 5-episode batches showed 0.8 and 1.0 mean kills/episode; this
isn't a single lucky run, but N=10 is still small — worth a larger batch
before trusting the exact numbers). This is a genuine fix, not a
re-tuned-until-it-looked-good number: it came from two real, independently
verified findings (the `noul` gate helps at all; the gate's calibration
doesn't survive extra `--memory` context; off-center bearings were firing
and missing) each fixed with its own targeted, logged mechanism — a
`noul` question, a stateless-mode encoder for the gate only, and a
deterministic bearing check — not from repeatedly rewording criteria
until a number looked right.

## Stage 4/5: real wayfinding, secret search, and three more real bugs

Both this project's own Roadmap and the related Needle-based experiment's
explicitly called out the same gap: perception.py deliberately never trusts ViZDoom's
`ANGLE` sign convention (see its module docstring), so neither project had
ever built real "which way is unexplored" wayfinding — only a blind
circling detector (`ExplorationNudgeConfig`). This section closes that gap
for real: `ANGLE` is verified empirically before being trusted for
anything, ViZDoom's automap buffer is investigated and (honestly)
rejected, real frontier-directed exploration is built and measured, and
three more real bugs were found and fixed along the way by running long
episodes rather than assuming the fixes above were the end of the story.

### Verified: ANGLE's real sign convention

Before using `ANGLE` for anything, its actual behaviour was measured
directly against a real `DoomEnv`, the same way every other fact in this
README is established — see `scripts/verify_angle.py` and
`scripts/verify_angle_movement.py` (both runnable, not one-off snippets
thrown away after use). Real output from this repository:

```
$ python -m scripts.verify_angle
=== initial ===
angle=0.0 x=-384.0 y=32.0

=== 5x turn_left (env.execute, real ViZDoom actions) ===
step 0: angle=5.273 delta=+5.273
step 1: angle=12.305 delta=+7.031
step 2: angle=22.852 delta=+10.547
step 3: angle=33.398 delta=+10.547
step 4: angle=43.945 delta=+10.547

=== 5x turn_right (env.execute, real ViZDoom actions) ===
step 0: angle=33.398 delta=-10.547
step 1: angle=22.852 delta=-10.547
step 2: angle=12.305 delta=-10.547
step 3: angle=1.758 delta=-10.547
step 4: angle=351.211 delta=+349.453   # wraps 0/360, i.e. -10.547 mod 360

=== raw 1-tic button press (bypassing actions.py's tic counts) ===
1 tic turn_left: angle 351.211 -> 354.727 (delta +3.516)
1 tic turn_right: angle 354.727 -> 351.211 (delta -3.516)
```

```
$ python -m scripts.verify_angle_movement
after turning left ~9x: angle=86.13 x=-384.00 y=32.00
after move_forward: dx=+0.47 dy=+7.10
  predicted unit dir=(+0.067,+0.998)  actual unit dir=(+0.067,+0.998)

after turning left another ~9x: angle=172.27
after move_forward: dx=-7.01 dy=+1.55
  predicted unit dir=(-0.991,+0.135)  actual unit dir=(-0.976,+0.216)
```

Conclusion, backed by real printed numbers, not the ViZDoom docs alone:
`ANGLE` is degrees, wraps 0–360, `turn_left` reliably **increases** it
(~10.5°/action once turning speed ramps up, ~3.5°/tic), `turn_right`
reliably **decreases** it by the same magnitude, and the real forward
movement direction matches `(cos(angle), sin(angle))` in
`POSITION_X`/`POSITION_Y` space — the standard math convention, close
enough at both tested headings (exact at 86°, same sign and rough
magnitude at 172° — collision/wall drag plausibly explains the small
residual there). `laya_doom/wayfinding.py` is built directly on this,
and only this — bearing/aiming everywhere else in the pipeline stays
screen-space, unchanged, per perception.py's own module docstring.

### Automap buffer investigation — tried, and honestly not adopted

`vzd.AutomapMode` really does exist with exactly `NORMAL`, `OBJECTS`,
`OBJECTS_WITH_SIZE`, `WHOLE` (confirmed against the installed `vizdoom`
package, not assumed from docs). `doom_env.py`'s `DoomEnvConfig` now
exposes `automap_buffer_enabled`/`automap_mode` knobs (still `False` by
default — see below for why), and `scripts/probe_automap.py` exercises
them through the same `DoomEnv` real runs use, dumping real PNGs and
doing simple numpy analysis. Real captured output from this repository:

```
$ python -m scripts.probe_automap --out-dir /tmp/automap_probe
00_start: bg-color=(111, 87, 67)  non-bg (drawn) fraction=0.0001
01_after_moving: bg-color=(111, 87, 67)  non-bg (drawn) fraction=0.0322
02_after_turning: bg-color=(111, 87, 67)  non-bg (drawn) fraction=0.0349
send_game_command('iddt') -> "Unknown command "iddt"" (real console output)
03_after_iddt_attempt: bg-color=(111, 87, 67)  non-bg (drawn) fraction=0.0349
```

Looking at the actual saved images (320×240 RGB): the background fill is a
uniform brownish colour — **not** literal black, and critically, **the
same colour for both revealed and un-revealed regions**. Only wall-line
geometry the player has actually been near gets drawn (a thin darker
outline plus a small white player-position arrow); open floor space the
player hasn't walked near yet is indistinguishable, pixel-for-pixel, from
open floor space fully inside an already-toured room. This matches real
Doom automap behaviour (undiscovered areas simply aren't drawn, per the
Freedoom manual's own "grey/not normally shown" description of the
in-game Tab-map) but means a naive "black-pixel fraction = unexplored"
heuristic — the first thing tried — doesn't work: there's no black to
count. The non-bg (drawn-line) fraction does grow with real exploration
(0.0001 → 0.0322 → 0.0349 above), so it's a real, if coarse, "how much
wall geometry have I revealed" signal, but it can't distinguish "open
floor I haven't reached yet" from "open floor already fully explored" —
exactly the distinction frontier-directed exploration needs, and exactly
what the verified-`ANGLE` + visited-cells-grid approach below already
gives directly and precisely, from data perception.py already computes.
The classic Doom `iddt` "reveal full map" console cheat was also tried
purely as a potential ground-truth validation aid for this investigation
(never intended for the shipped agent) — real result: ViZDoom's console
doesn't implement it (`Unknown command "iddt"`), so that avenue is closed
too. **Conclusion: investigated for real, with real images and numbers,
and honestly not folded into the decision pipeline** — it doesn't add
anything the verified-ANGLE approach doesn't already give more directly,
and costs an extra rendered buffer every tic for it. `automap_buffer_enabled`
stays `False` by default; the config knob and probe script are kept for
anyone who wants to look further.

One more thing checked while here, per a specific follow-up question: does
`perception.py`'s pickup table recognise Doom's "Computer Area Map"
power-up (reveals the whole level's map for the rest of that level, which
would be a legitimate assist if the agent ever walks over one)? Its
common actor/class name across Doom-engine ports is `Allmap` — searched
for that string (and `AreaMap`/`ComputerMap`) across the installed
`vizdoom` package's bundled files and found no occurrence, so it wasn't
added to `_PICKUP_KINDS` speculatively. If it exists in Freedoom2's MAP01
under a different label name, `perception.py`'s existing fallback (any
unmatched `Items`/`Powerups`-category label gets a snake-cased token
instead of being silently dropped — see `kind_of()`) means it would still
show up in the world-state text under its raw class name, just not
recognised by its intended English name; not independently verified with
a real pickup of one in this session.

### Frontier-directed exploration (`FrontierExplorationConfig`)

`laya_doom/wayfinding.py` uses the verified `ANGLE` convention plus the
player's `(x, y)` and the existing visited-cells grid
(`StateEncoder._visited_cells`, now exposed as a public `visited_cells`
property) to compute, for a handful of candidate headings around the
current one, which one leads to a cell the encoder hasn't marked visited
yet — `best_exploration_heading()` — and converts that into a concrete
`turn_left`/`turn_right`/`move_forward` action —
`turn_action_for_heading()`. `ExplorationNudgeConfig`'s override (the
circling detector — unchanged trigger logic) now calls this instead of
the old blind "pick whichever open side, or alternate by step parity"
guess, falling back to that only when wayfinding has no visited-cell
information to work from. This is a real, directed guess at "which way is
unexplored", grounded in ground-truth position data the pipeline already
tracks — not a guarantee (the projected cell could be on the far side of
a wall this pipeline has no way of knowing about), same "nudge, not a
guarantee" caveat as every other safety net here. `--no-frontier-exploration`
reverts to the old blind guess.

### Secret search and wall-following: an escalation ladder, not a bigger hammer

A plain frontier nudge alone doesn't reliably converge in a small, fully
toured room: `best_exploration_heading`'s own fallback (aim at the
nearest grid cell not yet marked visited, searched out to a 20-cell
radius) has no idea a wall might be blocking the way, so it can keep
proposing "go there" forever without any signal that nudging isn't
actually working. This project's own `docs/doom-strategy-research.md`
independently describes exactly this shape of problem and exactly this
fix, cited here rather than invented from scratch:

> "select nearest unvisited frontier. IF no frontiers: inspect
> lifts/platforms and inaccessible visible objects; scan anomalous/
> misaligned walls in dead ends... IF same edge sequence repeats twice
> without: new area, new key, new switch, new geometry, strategically
> relevant pickup: mark sequence NONPRODUCTIVE" (lines 445–460), and
> "Secret-like wall suspected but ordinary progression frontiers remain →
> prioritize normal progression; secrets are usually optional" (R77) —
> matched by `docs/tiny-doom-runtime-policy-200-rules.md` rules 157/160/
> 163/167 (prefer an unexplored branch; no progress → choose a different
> frontier; still stuck → explore unvisited branches; secrets are
> secondary to normal progression).

`ExplorationNudgeConfig`'s override now tracks `nudge_without_new_area` —
how many nudges in a row produced no genuinely new visited cell (reset
the moment one does) — and escalates through two tiers, each ranked
above plain nudging but only engaging once nudging has demonstrably
stopped working, matching the doc's own "twice without new area → mark
NONPRODUCTIVE" framing (`SecretSearchConfig.escalate_after` defaults to
exactly **2**, taken directly from that line, not tuned):

1. **`SecretSearchConfig`** — Doom/Freedoom secret doors are visually
   identical to ordinary walls (no texture/colour cue perception.py could
   detect even if it sampled screen pixels, which it deliberately
   doesn't — see perception.py's own docstring), so the only real way to
   find one is to press `use` against several nearby wall-facing
   directions, not just whichever wall happens to be faced right now
   (all `DoorUseConfig` below ever tries). Once escalated, this turns to
   face each of four headings relative to the heading at trigger time
   (dead ahead, left, right, behind) and presses `use` at each, before
   giving up.
2. **`WallFollowConfig`** — the classic maze "right-hand rule": keep a
   wall at a fixed relative side and slide along it, turning to hug it at
   corners, using only perception.py's existing screen-space
   `open_left`/`open_right`/`open_forward` flags (no `ANGLE`, no visited
   cells at all). Formally guaranteed to reach a simply-connected maze's
   exit with zero map memory; real Doom levels aren't pure mazes, but many
   are close enough (linear branching corridors) for this to be worth
   trying once nudging *and* a full secret-door sweep have both come up
   empty. Deliberately distinct from the wall-hugging **bug** below
   despite the similar name — that was aimless, zero-net-rotation wall
   contact going nowhere; this is purposeful, directional wall contact
   (always the same hand, always sliding forward).

Both are real, verified to actually fire in real play, not just unit
tested in isolation. A real 1,200-step `--scenario level` run
(`logs/laya_frontier_on.steps.jsonl`) and a real 4,000-step run
(`logs/laya_level_exit3.steps.jsonl`, see below) both show real
`secret_search`/`wall_follow` entries in `override_reason`: the
4,000-step run logged 276 `secret_search`-attributed steps and 810
`wall_follow`-attributed steps (see the measured-results table below for
the full breakdown) — these mechanisms engage substantially in real play,
not just in the unit tests (`tests/test_controller.py`'s
`test_secret_search_regression_tries_use_around_a_dead_end` and
`test_wall_follow_regression_engages_after_secret_search_and_nudging_both_fail`
cover the pure logic with fakes; the numbers above are the real-run
confirmation). `--no-secret-search`/`--no-wall-follow` disable each independently.

### Real bug: wall-hugging (the literal one the user watched live)

Watching a real `--render` run surfaced the exact behaviour this whole
section was scoped to fix: the agent survived and explored a lot, then
spent long stretches "hugging/facing the wall". Diagnosing this needed
real data, not speculation — `StepRecord.override_reason` was added first
(ported from that earlier pipeline's own identical fix, for the identical
reason: `overridden=True` alone couldn't say *which* net fired), then a
fresh real `--scenario level` episode was run and the logs inspected
directly. Real finding, `logs/laya_level_diag1.steps.jsonl`, steps
721–799 (all 79 of them, run only stopped because `--max-steps` ran out):

```
721 x=272.1 y=240.0 proposed=move_forward final=turn_right_large
722 x=272.1 y=240.0 proposed=strafe_left  final=turn_left_large
723 x=272.1 y=240.0 proposed=move_forward final=turn_right_large
724 x=272.1 y=240.0 proposed=strafe_left  final=turn_left_large
...  (repeats identically for 79 consecutive steps)
```

with the encoded state showing `WALL ahead far` and `PATH left`/`PATH
right` both `open` on the tied steps. Two compounding root causes, both
real:

1. **`_recovery_turn`'s tie-break oscillated.** When `open_left ==
   open_right`, the fallback picked a direction by `step % 2` — and
   since the global step counter's parity flips every single call, it
   alternated `turn_right_large`/`turn_left_large` every step, netting
   exactly zero rotation forever. **Fix**: `run_episode` now caches ONE
   `_recovery_turn` result per stuck event (only cleared on real
   positional progress) instead of re-deriving — and re-tie-breaking — it
   on every firing.
2. **`TurnLoopRecoveryConfig` couldn't rescue it.** It's designed to
   force a `move_forward` attempt after too many consecutive turns — but
   its original trigger checked Laya's own *proposed* action for being a
   turn, and here Laya kept proposing `move_forward`/`strafe_left` (never
   a turn) every single step, while `StuckRecoveryConfig` overrode every
   one of those into a turn. Since `StuckRecoveryConfig`'s own condition
   stayed true indefinitely and was checked first, `TurnLoopRecoveryConfig`
   never even got a chance, no matter how many turns had actually been
   *executed*. **Fix**: relaxed its trigger to watch
   `consecutive_turn_steps` (already tracked against the executed action,
   not the proposal) and moved it ahead of `StuckRecoveryConfig` in the
   priority chain, excluding only a live combat/use proposal.

Verified fixed with a fresh real run on the same scenario
(`logs/laya_level_diag2.steps.jsonl`): the longest `stuck_recovery`-only
run dropped from 79 steps to exactly **6** (capped by
`TurnLoopRecoveryConfig`'s own `max_consecutive_turns=6`), and — the part
that actually matters — that 6-step run is now `['turn_right_large'] * 6`,
a single consistent direction, not an oscillation. A regression test
(`test_wall_hugging_regression_escapes_a_symmetric_stuck_corner`)
reproduces the exact tied-open-sides geometry with fakes and asserts the
fix actually escapes it, not just that the code runs.

### Real bug: `use` never fires at all (before `DoorUseConfig`)

Before building anything for it, checked with real logged data whether
`use` was ever actually being chosen, the same way the shoot-gate section
above checked `shoot`: across a real 2,980-step `--scenario level` run
(`logs/laya_level_full.steps.jsonl`), `use`'s own probability inside
Laya's returned distribution never exceeded **0.169** (mean **0.071**,
2,950 real decisions where it was scored) and was picked as the final
action **0 times**, no matter how close the player was to a wall — the
same class of bug the shoot gate fixed for `attack`, now for `use`. No
`noul`-style gate was built for it (there's no game variable exposing
"is there a real door/switch here" to calibrate one against, unlike the
`should_shoot` case); instead `DoorUseConfig` fires `use` deterministically
once `WALL ahead near` has held for `stall_threshold` (default 3)
consecutive steps — a real no-op against a plain wall in Doom, so a low
false-positive cost — ranked just above `StuckRecoveryConfig` (try the
door before turning away from it). Verified actually firing in real play:
a real 1,200-step run logged 16 real `use` actions, every one of them
attributed to `door_use` in `override_reason` (Laya itself still never
chose `use` on its own in that run either) — the same 0-in-isolation
finding still held even with the fix layered on top, confirming the fix
is doing real work rather than papering over a problem that had already
gone away.

### Real bug: `use`/`wait` spam (a second and third stall pattern StuckRecoveryConfig missed)

Validating the work above meant running much longer episodes (3,000–4,000
steps) than anything tried earlier in this project, and that surfaced two
more real, previously-invisible bugs — both the same underlying shape:
Laya proposing the identical non-move action forever once the encoded
state hit a fixed point, with nothing watching for it because
`attack`/`shoot`/`use`/`wait` are all deliberately "never second-guess a
live decision" everywhere else in this module.

- **`use` spam**: a 3,000-step run (`logs/laya_level_exit1.steps.jsonl`)
  logged **2,339/3,000 (78%)** real `use` decisions, frozen at one
  position, with a real `PICKUP` visible the whole time (which is exactly
  why `ExplorationNudgeConfig` correctly never engaged — something
  legitimate to react to — but nothing else was watching either).
- **`wait` spam**: a separate 4,000-step run
  (`logs/laya_level_exit2.steps.jsonl`) logged **3,919/4,000 (98%)** real
  `wait` decisions. The real logged reasoning at that frozen spot:
  `wait:0.396` vs. the next-highest `turn_left_small:0.141` — `wait` won
  comfortably and stayed the top label every single call, since nothing
  in the encoded state ever changed to shift it.

**Fix**: `StuckRecoveryConfig`'s stall detection (`_STALL_ACTION_NAMES`,
which already fed `no_progress_steps` and its own trigger condition) now
also counts a stalled `use` or `wait`, not just a stalled move — after
enough consecutive no-progress attempts at any of them, it forces the
same recovery turn a stalled move gets. Verified with a fresh real run,
same seed as the `wait`-spam run
(`logs/laya_level_exit3.steps.jsonl`): **0** `wait` actions anywhere in
4,000 real decisions (down from 3,919), and a second run without a fixed
seed (`logs/laya_level_exit4.steps.jsonl`) likewise shows **0** `wait`
actions across 4,000 steps. Two regression tests
(`test_use_spam_regression_...`/`test_wait_spam_regression_...`)
reproduce both exact shapes with fakes.

### Real measured results: before vs. after this section's work

Same machine, same `--scenario level`, `--action-set full`. "Before" is
this session's diagnostic run with `override_reason` logging added but
before the wall-hugging/frontier/secret-search/wall-follow/stall fixes;
"after" is the final state, same seed where noted:

| Run | Steps | Distance | Kills | Items | Died | `completed` | Notable |
|---|---|---|---|---|---|---|---|
| `laya_level_diag1` (before) | 800 | 7,135 | 2 | 2 | No | False | 79-step wall-hugging freeze found here |
| `laya_level_diag2` (after wall-hugging fix) | 800 | — | 2 | — | No | False | longest stuck_recovery run: 79 → 6 steps |
| `laya_level_exit1`, seed 42 (before use/wait-spam fix) | 3,000 | 6,732 | 2 | 4 | No | False | 2,339/3,000 steps were `use` spam |
| `laya_level_exit2`, seed 42 (before wait-spam fix) | 4,000 | 1,013 | 0 | 0 | No | False | 3,919/4,000 steps were `wait` spam |
| `laya_level_exit3`, seed 42 (after all fixes) | 2,898 | **22,391** | 9 | 10 | **Yes** (overwhelmed while retreating) | False | 0 `wait`/no long freeze; wall_follow fired 240×, secret_search 68× |
| `laya_level_exit4`, no seed (after all fixes) | 4,000 (full budget) | **34,261** | 7 | 12 | No | False | wall_follow fired 810×, secret_search 276×; 64 distinct visited cells |

Real, substantial, measured improvement over both this session's own
earlier diagnostic runs and the prior result reported for the related
Needle-based experiment (a 674-step episode that died at 10,121 distance,
per that experiment's own "Verified behaviour" findings): longer survival, far more distance
covered, more kills and items, and the specific pathological freezes this
section set out to fix are gone from real logged data, not just
theoretically addressed.

**Honest limitation, not hidden**: raw distance travelled is not the same
as exploration coverage, and the `exit4` numbers make that concrete —
despite covering *more* raw distance than `exit3` (34,261 vs. 22,391),
`exit4` visited *fewer* distinct 128-unit grid cells (64 vs. 83). With
`wall_follow` firing 810 times in that run (20% of all 4,000 steps), a
real chunk of that distance is plausibly sliding back and forth along
already-known wall segments rather than reaching new ground — a genuine
trade-off of the escalation ladder, not something this README is
glossing over.

### Did a real run ever reach `completed=True`?

**No — not in any run tried this session**, including the two final
4,000-step runs above with every mechanism from this section engaged.
`completed` for `--scenario level` is `map_exit_reward >= 1` (see
`run_episode`'s own comment on `completed_by_exit_reward`) — a real,
ground-truth ViZDoom signal, not an inferred proxy, so this isn't a
measurement gap; the agent genuinely never reached MAP01's exit in any of
these episodes, despite substantial real exploration (up to 34,261
distance units, 64–83 distinct grid cells, multiple real secret-door
sweeps and wall-following excursions). Reported plainly rather than
reframed as a partial win: exploration coverage and survival improved
measurably and are backed by real before/after data above, but the actual
success bar this section was given — get a real, honest exit — was not
met. Plausible reasons, none independently confirmed: MAP01
(`Hydroelectric Plant`) may be large/branching enough that 4,000 decision
steps (at 3–6 tics each) still isn't enough real playtime to reach a
distant exit; the wall-following/secret-search ladder, while real and
firing often, spends a meaningful fraction of the budget on searches and
corner-following rather than pure forward progress (see the coverage
caveat above); and `--decision-tics`/`--max-steps` were not swept as part
of this work, which would be the next thing to try before concluding the
mechanism itself is insufficient.

## Laya vs Needle

Real, measured numbers from both projects on this same machine — not
marketing figures from either vendor. The Needle numbers were measured in
the related, separate experiment described above (not part of this
repository); the Laya numbers are from this repository.

| | Needle 3 (related experiment) | Laya (this project) |
|---|---|---|
| Mechanism | Autoregressive tool-calling: decode loop produces a structured function call + free-text reasoning | Non-autoregressive: one forward pass over typed `choice`/`score`/`noul` questions, no generation at all |
| Persistent state between decisions | Yes — internal state that needed periodic `reset()` (`reset_every`, default 15) to avoid a real, confirmed decode stall (38 calls in, 57s+) on repetitive input | None — `laya.Agent` has no `reset()` method; verified empirically here (200 identical calls, no latency trend, see above) rather than just inferred |
| System/goal-priority injection | No documented `system=` prose mechanism (facts-only); `GOAL` line injected into world-state text instead | No `system`/instructions parameter of any kind on `predict()`; identical `GOAL`-line-in-state-text approach used here for the same reason |
| Tool/label count constraint | Own guidance: 5 or fewer tools render directly before retrieval filters — the related experiment found `full`'s original 6 tools crowding out `attack` (0/2257 real decisions) and dropped to exactly 5, which helped (0/10 -> 5/10 on a combined test) | Tested the same crowding hypothesis here (11 labels vs. a trimmed 5, one state): the *opposite* happened — `attack_prob` dropped from 0.999 to 0.213 with fewer labels. No equivalent mechanism transfers; see [The real fix](#the-real-fix-pull-combat-out-into-its-own-noul-question) |
| Measured latency, base/default weights (this machine) | mean 356ms (basic-scenario compare run); separately, mean 115ms / p50 100ms / p95 247ms (isolated `level`-scenario benchmark) | mean 21.3–30.9ms warmed (isolated smoke test / real in-loop run), p95 ≤ 30.4ms in the 150-step real run; ~23ms mean with the (now two-call) shoot gate enabled in a real 10-episode comparison run |
| Measured latency, smallest tested variant | `rung_6.cact` (6-layer model-ladder rung, built via `needle build`): mean 35ms, p50 31ms, p95 86ms | N/A — Laya ships one checkpoint, no depth ladder to build a smaller rung from |
| Confidence calibration on this task | Reported ~0.72 mean in a real basic-scenario run; vendor's own guidance suggests ~0.7 as an execute band | Reported 0.02–0.19 with the original single-`choice` design; 0.31–0.59 with the shoot gate (a `noul` P(true), not a `choice` top-label probability — different calibration semantics, not directly comparable to Needle's number) |
| Base-checkpoint Doom behaviour, first honest look | "mostly turned and almost never chose shoot" (fixed afterward via triggers/docstrings) | Never chose shoot at all with combat competing inside one `choice` question (0/8 hand-built states); one 150-step run was 100% `turn_right` |
| Fix mechanism available | Needle's own documented `triggers=` regex + docstrings — used successfully in the related experiment to fix the shoot/move_forward/wall/attack-crowding issues | No `triggers=`-equivalent for `choice`, and criteria-wording alone didn't help (one pass made it worse — see "keyword-bait bias" above) — but Laya's separate `noul` primitive, pulling combat out of the `choice` entirely, plus deterministic `AMMO>0`/bearing guards in code, did: real 10-episode result now matches the `heuristic` baseline (`mean_kills` 1.00, `completion_rate` 1.00) |

The latency gap is the clearest win for Laya's architecture on this
machine — even Needle's *smallest* tested rung (35ms mean) is slower than
Laya's steady-state mean (21.3ms), and the two-call shoot-gate design
still lands around 23ms in real episodes, well under Needle's base
latency. The behavioural gap has narrowed since the first look above:
both projects hit essentially the same underlying problem (the model
doesn't reliably fire the combat action) and fixed it with a mechanism
specific to their own architecture — Needle's `triggers=`/tool-count
guidance for its retrieval-filtered tool rendering; Laya's separate
`noul` primitive plus deterministic guards for its flat, no-retrieval
`choice`/`noul` design (Needle's own crowding fix was tested here too and
didn't transfer — see the table). Neither fix came from repeatedly
rewording a single prompt/criteria string until a number looked good;
both came from isolating the actual mechanism (a trigger regex, a
separate calibrated question type) that the vendor's own docs describe
for exactly this kind of problem. Whether a fine-tuned Laya checkpoint
(the project explicitly supports fine-tuning on a T4 GPU via its Colab
notebook) would close the remaining gaps (calibration semantics, the
bearing guard still being enforced in code rather than learned) is an
open question this project didn't attempt to answer — that would be
training a model for this task, out of scope here, same as the related
experiment stayed out of RL training for Needle.

## Installation

Requires Python 3.11+. On Apple Silicon, use Python **3.11** specifically
if you have a choice — as of this writing, `vizdoom`'s prebuilt
macOS-arm64 wheels lag behind the newest CPython releases.

```bash
brew install python@3.11   # if you don't already have it
./scripts/setup.sh          # creates .venv/, installs requirements.txt
source .venv/bin/activate
```

`scripts/setup.sh` uses [`uv`](https://github.com/astral-sh/uv) if it's on
your PATH (fast), otherwise falls back to `python3.11 -m venv` + `pip`. No
manual Doom asset download is required (see above). The first `laya`
controller run downloads the `convaiinnovations/laya` checkpoint from
Hugging Face (~843MB weights; ~190s on this machine's connection, cached
under `~/.cache/huggingface/hub/` afterward — not inside this repo, so
it's not something `.gitignore` needs to cover, though a `.cache/` guard
is included anyway in case `HF_HOME` is ever pointed here).

## Running the demo

```bash
# random baseline — proves the ViZDoom + perception + metrics pipeline works
python -m experiments.run --controller random --episodes 3 --scenario basic

# the hand-written heuristic baseline
python -m experiments.run --controller heuristic --episodes 3 --scenario basic

# Laya itself — the actual experiment
python -m experiments.run --controller laya --episodes 3 --scenario basic --render --dashboard
```

`--dashboard` shows a live terminal view (health/armor/ammo, nearest
enemy, Laya's decision/confidence/latency, a short action history,
running episode stats).

## Comparing controllers

```bash
python -m experiments.compare --controllers random heuristic laya --episodes 100 --scenario basic
```

Runs all three through identical episodes and prints a side-by-side table
(and logs each to `logs/compare_<controller>_<scenario>_<action-set>.episodes.jsonl`).

## Viewing results

Same JSONL logging scheme as that earlier pipeline: every run writes
`<run_name>.steps.jsonl` and `<run_name>.episodes.jsonl` under `logs/`
(pass `--run-name` to control the name; the default **appends** across
repeated invocations). Each step row has the full world-state text Laya
saw, the executed action, confidence, latency, and the resulting state —
plus a `reasoning` field that is always a real rendering of Laya's
returned probability distribution (see `laya_agent.py`), never invented
text, since Laya generates no free-text reasoning of its own. Load with:

```python
from laya_doom.metrics import load_jsonl, summarize_episodes
rows = load_jsonl("logs/laya_basic_stage1.episodes.jsonl")
print(summarize_episodes(rows))
```

## Configuration knobs

All exposed as CLI flags on `experiments/run.py` and `experiments/compare.py`:

| Flag | What it controls |
|---|---|
| `--scenario {basic,my_way_home,level}` | Same three scenarios as that earlier pipeline. Only `basic` has been run and verified here (see Roadmap). |
| `--action-set {stage1,full}` | `stage1`: 4 labels. `full`: 11 labels (10 canonical actions + `wait`). Unlike the related Needle-based experiment, there is no "5 or fewer" tool-count guidance to work around here — Laya's `choice` type takes the whole `criteria` dict in one pass regardless of size. |
| `--decision-tics N` | Same control-frequency knob as that earlier pipeline. |
| `--memory {stateless,prev_state,rolling}` (or `0`/`1`/`2`) | Same three memory modes, unchanged from `state_encoder.py`. |
| `--confidence-mode {always_execute,confidence_threshold}` + `--confidence-threshold F` | `confidence_threshold` substitutes `wait` when Laya's own calibrated confidence is below `F`. Defaults to **0.15**, not the 0.6 used for Needle in the related experiment — see [Verified behaviour](#verified-behaviour): this checkpoint's confidence on Doom states runs far lower than a tool-calling model's, so a tool-calling-model-appropriate threshold would gate almost every decision to `wait` here. |
| `--model-id ID` | HuggingFace model id or local path passed to `laya.load()` — the closest equivalent to the related experiment's `--weights` model-ladder lever, except Laya ships one checkpoint, not a depth ladder to sweep. |
| `--device {cuda,mps,cpu}` | Forces a device instead of auto-detection. Omit it — auto-detection correctly picked `mps` on this machine with no configuration. |
| `--no-goal-line` | Omit the `GOAL` line from the world-state text — same toggle as that earlier pipeline, same open empirical question (does it change anything at all). |
| `--no-stuck-recovery` / `--no-threat-response` | Same two controller-level safety nets, ported unchanged from `controller.py` — see that module for what each does. |
| `--no-shoot-gate` | Use the original single-`choice`-call design (combat competes directly against move/turn) instead of the default `noul` shoot gate + `AMMO>0`/bearing guards — see [The real fix](#the-real-fix-pull-combat-out-into-its-own-noul-question). Mainly useful for regression/comparison runs against the pre-fix behaviour. |
| `--shoot-gate-threshold F` | `P(should_shoot)` cutoff for the gate (default **0.45**) — picked directly from a real 8-state spread (see the shoot-gate section), not a calibrated cutoff against a held-out set. |
| `--no-turn-loop-recovery` | Disable the safety net that forces a `move_forward` attempt after too many consecutive *executed* turns (`controller.TurnLoopRecoveryConfig`) — found on a real `--scenario level` run stuck turning in a corner; later broadened to fix the real wall-hugging bug (see [Stage 4/5](#stage-45-real-wayfinding-secret-search-and-three-more-real-bugs)). |
| `--no-threat-engagement` | Disable the safety net that turns toward a visible, near-enough, off-center enemy instead of letting Laya's movement choice stand (`controller.ThreatEngagementConfig`) — found on a real run that died to a zombieman it never turned to face. |
| `--no-low-health-retreat` / `--low-health-threshold N` / `--emergency-health-threshold N` | Disable/tune the safety net that retreats (or, below the emergency threshold, overrides even an attack) when a visible enemy is present and health is low (`controller.LowHealthRetreatConfig`) — ported from the related Needle-based experiment. |
| `--no-exploration-nudge` / `--exploration-streak-threshold N` | Disable/tune the circling detector (`controller.ExplorationNudgeConfig`) — see [Stage 4/5](#stage-45-real-wayfinding-secret-search-and-three-more-real-bugs) for how its override action changed from a blind guess to directed frontier-seeking. |
| `--no-frontier-exploration` / `--frontier-lookahead-cells F` | Revert ExplorationNudgeConfig's override to the old blind guess, or tune how many grid cells ahead each candidate heading is projected (`controller.FrontierExplorationConfig`) — see [Stage 4/5](#stage-45-real-wayfinding-secret-search-and-three-more-real-bugs). |
| `--no-door-use` / `--door-use-stall-threshold N` | Disable/tune the safety net that tries `use` once after `WALL ahead near` holds for N consecutive steps (`controller.DoorUseConfig`) — added after real logged data showed `use` never wins Laya's movement choice on its own (max probability 0.169 over 2,950 real decisions, 0 times chosen). |
| `--no-secret-search` | Disable the systematic turn-and-use sequence that engages once frontier exploration has stopped finding new territory (`controller.SecretSearchConfig`) — see [Stage 4/5](#stage-45-real-wayfinding-secret-search-and-three-more-real-bugs). |
| `--no-wall-follow` / `--wall-follow-hand {left,right}` | Disable/pick the hand for the classic maze wall-following fallback (`controller.WallFollowConfig`) — engages once plain nudging *and* a full secret-door sweep have both come up empty; see [Stage 4/5](#stage-45-real-wayfinding-secret-search-and-three-more-real-bugs). |
| `--window-scale {1,2,3}` | Doom window size with `--render`: 1=320×240, 2=640×480 (exact 2×), 3=1024×768 (closest 4:3 preset to 3× — ViZDoom has no exact 3×). Purely a display size; perception.py samples by fraction of width/height, so behaviour is unaffected — ported from the related Needle-based experiment. |

## Why there's no system prompt

`laya.Agent.predict(state, questions)` takes exactly two arguments: no
`system`, `instructions`, or persona parameter of any kind exists to even
consider using — there's nothing analogous to Needle's facts-only
`system=` to invent workarounds for. The brief's priority list is
injected the same way as in that earlier pipeline: one `GOAL` line at the
top of the world-state text itself (`state_encoder.GOAL_LINE`), since
that's the only text Laya actually reads. Whether it changes anything is
the same open, measurable question as before (`--no-goal-line`).

## Why `LayaAgent.reset()` is mostly a no-op

Covered in depth in [Verified behaviour](#verified-behaviour) above:
`laya.Agent` (0.1.6) exposes no `reset` method at all, and 200 consecutive
identical `predict()` calls showed no latency trend on this machine —
there's no persistent model state to reset and no confirmed staleness
failure mode to guard against, unlike the Needle wrapper used in the
related experiment. The one thing `LayaAgent.reset()` actually does is clear its own
internal `_gate_encoder`'s AREA visited-cells bookkeeping once per episode
(see [The real fix](#the-real-fix-pull-combat-out-into-its-own-noul-question)
for why the shoot gate has its own separate `StateEncoder`) — not a Laya
API call, just this wrapper's own per-episode state.

## Testing

```bash
pytest              # 140 tests, pure logic — no vizdoom process needed, and
                     # no real Laya model loaded (LayaAgent's own tests fake
                     # out laya.load() to test its label->action mapping,
                     # the shoot gate's AMMO/bearing guards, confidence
                     # gating, and Decision field population in isolation —
                     # see tests/test_laya_agent.py; tests/test_wayfinding.py
                     # covers the ANGLE-based heading/turn-choice pure logic;
                     # tests/test_controller.py includes regression tests that
                     # reproduce the real wall-hugging/use-spam/wait-spam bugs
                     # and secret-search/wall-follow escalation with fakes —
                     # plus scripts/probe_criteria.py, scripts/verify_angle.py,
                     # scripts/verify_angle_movement.py and
                     # scripts/probe_automap.py for the real-model/real-ViZDoom
                     # experiments, not part of the automated suite)
```

## Project structure

```
laya-doom/
    config/basic.cfg,        ViZDoom scenario configs, ported unchanged
    my_way_home.cfg,         from that earlier pipeline
    level.cfg
    laya_doom/
        doom_env.py          vizdoom.DoomGame wrapper + execute(action_name)
        perception.py        GameState -> Perception (no screen pixels)
        state_encoder.py      Perception -> compact text, 3 memory modes
        laya_agent.py          laya.Agent wrapper, predict()-based, no tool schema
        actions.py             semantic action vocabulary <-> button tables
        wayfinding.py            verified-ANGLE frontier/heading helpers (Stage 4/5)
        controller.py          the explicit per-tic run_episode() loop
        metrics.py              JSONL logging + summary stats
        dashboard.py             live terminal view
    experiments/
        random_agent.py, heuristic_agent.py   baselines, ported unchanged
        run.py                  single-controller CLI
        compare.py               multi-controller comparison CLI
    scripts/
        setup.sh                 automated venv + install
        probe_criteria.py        real-model criteria-wording/shoot-gate probes
        verify_angle.py,         real ANGLE sign-convention verification
        verify_angle_movement.py (see Stage 4/5)
        probe_automap.py         real automap-buffer capture + analysis
    tests/                  actions/perception/state_encoder/metrics/controller/
                             heuristic_agent/laya_agent/wayfinding unit tests
    logs/                   JSONL output (gitignored)
```

## Research questions this is built to answer

Same list as the related Needle-based experiment, minus the model-ladder-specific one
(Laya has no ladder to sweep):

1. Does Laya react sensibly to enemies at all? — **tested, answer is no**
   for this checkpoint (see Verified behaviour).
2. Do repeated local decisions produce anything resembling persistent
   behaviour, or just noise? (watch `--dashboard`; the 150-step
   all-`turn_right` run above is itself an answer of sorts — persistent,
   but not purposeful)
3. Can it navigate without an explicit planner? (`--scenario my_way_home` —
   not yet run here, see Roadmap)
4. How little state does it actually need? (`--memory stateless` vs. others)
5. Does remembering the last action help? (`--memory prev_state` vs. `stateless`)
6. Does outcome feedback (`LAST_RESULT`) help? (same axis)
7. What decision frequency works best? (`--decision-tics`)
8. How does it compare to the tiny heuristic? — **measured**, see the
   comparison table above; heuristic wins decisively.
9. How does Laya's architecture compare to Needle's for this task? — see
   [Laya vs Needle](#laya-vs-needle).

## Roadmap

Built:

- **Stage 1** (four-action proof of concept, `basic` scenario) — working
  end to end, including the shoot-gate fix, see Verified behaviour above.
- **Stage 2/3** (`my_way_home` navigation, and a real Freedoom level via
  `--scenario level` with the `full` action set) — extensively run this
  session (multiple real episodes from 674 up to 4,000 steps), covered in
  detail by the safety nets in `controller.py` and the
  [Stage 4/5](#stage-45-real-wayfinding-secret-search-and-three-more-real-bugs)
  section above.
- **Stage 4/5, partially** (real wayfinding, exit-seeking) — ViZDoom's
  `ANGLE` convention verified empirically and used for real
  frontier-directed exploration; the automap buffer investigated and
  honestly rejected; a systematic secret-door search and a classic
  wall-following fallback built and verified firing in real play; three
  more real bugs found on long real runs and fixed. **Not** achieved:
  `completed=True` was never observed in any real run this session — see
  that section's own honest conclusion for exactly what was tried and
  what's still missing. The two Doom strategy documents this project's
  folder already had (`docs/doom-strategy-research.md`,
  `docs/tiny-doom-runtime-policy-200-rules.md`) turned out to directly
  describe the same frontier→secret-search escalation shape built here
  (cited directly in that section, not just sitting unused as this
  Roadmap previously said) — a genuinely useful starting point for a
  future hierarchical agent, which this project still doesn't attempt.

Not built:

- **Fine-tuning Laya for this task** — the project explicitly supports
  this (a T4-GPU Colab notebook for fine-tuning on custom data); doing so
  would be training a model for Doom-playing specifically, which is
  out of scope here the same way RL training was out of scope for the
  related Needle-based experiment. Would also plausibly fix the calibration gaps
  the shoot gate currently papers over in code (the `AMMO>0` precondition,
  bearing discrimination).
- **`score` question type** — only `choice` and (as of the shoot-gate fix)
  `noul` were used here; Laya's third typed-question primitive wasn't
  explored for this task.
- **A hierarchical agent with real per-ammo-type/enemy-taxonomy/navigation-
  memory state** — the two strategy documents above assume a materially
  richer game state than this project's `Perception`/world-state text
  implements (they were used to *design* the escalation ladder's ordering
  and thresholds, not to add that richer state itself); building that
  state and a higher-level goal/skill layer on top of it remains unbuilt.
- **Sweeping `--decision-tics`/`--max-steps` specifically to chase
  `completed=True`** — flagged as the most likely next lever in the
  Stage 4/5 section's own honest conclusion, not attempted this session
  to avoid exactly the kind of "kept tuning knobs until a number looked
  good" pattern this project's methodology avoids elsewhere.
- **Browser dashboard, multi-agent, multi-map progression** — same
  unbuilt items as the related Needle-based experiment's own roadmap, for the same reasons.

## Known quirks

- Everything under perception.py/state_encoder.py's own "Known quirks" in
  the related Needle-based experiment's README applies here unchanged (depth-buffer
  calibration is approximate, `basic`'s built-in reward script vs.
  tracked game variables can disagree, `EpisodeResult.completed` is an
  inferred proxy) — this project didn't re-verify or re-derive any of
  that, since none of it depends on the decision engine.
- Laya's `confidence` field is described as "calibrated" in its own
  README via ECE (expected calibration error) numbers on its own
  benchmark tasks (routing, moderation, etc.) — nothing here re-verifies
  that calibration claim on those tasks; what's measured here is only
  that confidence stayed uniformly low (0.02–0.19) on Doom states, which
  is at least consistent with a well-calibrated model correctly reporting
  low confidence on an out-of-distribution task, not proof of it.
- The one 150-step real run reported above (pre-fix, `--no-shoot-gate`)
  is a single episode, not a large-sample result — it's reported because
  it's a vivid, reproducible illustration of the same 100%-one-action
  pattern the isolated 8-state test also showed, not because n=1
  episodes are being treated as statistically definitive. The 5-episode
  pre-fix `compare` table is the closer-to-decisive number, and it agrees
  with it (0 kills, 0 damage, 0% completion across all 5). The post-fix
  10-episode table is likewise not a huge sample — separate 5-episode
  batches with the fix showed 0.8–1.0 mean kills/episode, so treat the
  exact decimal as indicative, not precise.
- The shoot gate's bearing guard requires an enemy at exactly bearing
  `front` (matching `perception.py`'s existing bearing buckets) before
  code will ever let the combat action fire, regardless of what the
  `noul` gate says — this is deliberately conservative (an enemy at
  `front-left`/`front-right` will never get shot at, even close up, until
  a `turn` lines it up to exactly `front`), which is what fixed the real
  0-damage bug but is also a real behavioural constraint worth knowing
  about, not just a bug fix with no trade-off.
- The real wall-hugging bug's own logged data (see
  [Stage 4/5](#stage-45-real-wayfinding-secret-search-and-three-more-real-bugs))
  showed `WALL ahead far` — not `near` — at a position where the player
  was completely, persistently stuck. `wall_ahead`/`wall_near` come from a
  3-ray depth scan (forward/left/right at fixed screen fractions, one row)
  that can under-detect nearby blocking geometry outside those exact rays
  (a thin pillar, an off-axis corner) — consistent with perception.py's
  own docstring calling the depth-scan calibration "approximate...meant to
  be re-tuned by eye", not something this session re-derived or fixed.
  `DoorUseConfig` only triggers on `wall_near`, so it correctly never
  fired at this particular spot; the fix that mattered there was
  StuckRecoveryConfig/TurnLoopRecoveryConfig's own turn-direction logic,
  independent of this quirk.
- ViZDoom's automap buffer background fill colour is the same for
  revealed and undiscovered regions (see the automap investigation above)
  — a naive "count black pixels" heuristic for "how much is unexplored"
  does not work on this buffer, and this wasn't obvious without actually
  capturing and looking at real images.
- `WallFollowConfig` inflates raw distance-travelled without a
  proportional increase in exploration coverage (see the measured-results
  table above: `exit4` covered more distance than `exit3` but fewer
  distinct grid cells) — a real trade-off of the escalation ladder, not
  a free improvement.
