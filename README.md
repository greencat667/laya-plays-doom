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

This is the sibling experiment to
[`058 - Needle Plays Doom`](../058%20-%20Needle%20Plays%20Doom/README.md),
which asked the same question of Cactus Compute's Needle 3, an
autoregressive tool-calling model. Everything upstream of the decision —
ViZDoom setup, perception, world-state text encoding, metrics, the
stuck-recovery/threat-response safety nets — is the same code, ported
directly from that project. The **only** thing that changes here is the
decision engine. This project is fully self-contained (own venv, own
`requirements.txt`, own git repo) and has no dependency on the sibling
project or on `cactus-needle`.

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
one of Laya's own documented primitives, used the way the sibling project
used Needle's `triggers=` — plus two deterministic guards (`AMMO>0`,
bearing exactly `front`) checked in code rather than trusted from the
model, took it from worse-than-random (0 kills/5 episodes) to matching
the hand-written heuristic baseline: real, verified 10-episode numbers,
`mean_kills=1.00`, `completion_rate=1.00`, `mean_latency_ms=22.8` — see
[Controller comparison, real run](#controller-comparison-real-run).

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

`controller.py` is the same explicit per-control-cycle loop as the sibling
project — one `agent.decide()` call per Doom step, no autonomous loop.
`random_agent.py`, `heuristic_agent.py` and `laya_agent.py` (in
`experiments/`) all implement the same tiny interface —
`decide(perception, encoded_state) -> Decision`, `reset()` — and run
through the *exact same* `perception.py`/`state_encoder.py` pipeline via
`controller.run_episode()`, which is what makes them comparable.

## No commercial Doom files needed

Same setup as the sibling project, reused as-is: for `basic`/`my_way_home`
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

The sibling Needle project needed `NeedleAgent.reset()`/`reset_every`
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
(`laya_agent._ACTION_CRITERIA`, adapted from the sibling project's Needle
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
tuning rule** — the sibling project's approach was to rewrite Needle's
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

Needle's fix for the equivalent problem (058, "attack was being crowded
out") was to shrink the competing-tool count, using Needle's own
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
mirrors the sibling project's own pattern of a deterministic safety net
layered visibly on top of a model decision (`StuckRecoveryConfig`/
`ThreatResponseConfig`), not hidden inside it. `--no-shoot-gate` restores
the single-`choice`-call design for comparison.

The Needle-crowding analogy was also tested directly, since the
methodology should be checked even where the hypothesis turns out not to
transfer: the `full` action set's `attack` label, one state (enemy front,
open path), 11 competing labels vs. a trimmed 5 (mirroring 058's fix):

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
the same class of finding the sibling README reported for base Needle
("mostly turned and almost never chose shoot") — the difference is that
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

## Laya vs Needle

Real, measured numbers from both projects on this same machine — not
marketing figures from either vendor.

| | Needle 3 (058) | Laya (059, this project) |
|---|---|---|
| Mechanism | Autoregressive tool-calling: decode loop produces a structured function call + free-text reasoning | Non-autoregressive: one forward pass over typed `choice`/`score`/`noul` questions, no generation at all |
| Persistent state between decisions | Yes — internal state that needed periodic `reset()` (`reset_every`, default 15) to avoid a real, confirmed decode stall (38 calls in, 57s+) on repetitive input | None — `laya.Agent` has no `reset()` method; verified empirically here (200 identical calls, no latency trend, see above) rather than just inferred |
| System/goal-priority injection | No documented `system=` prose mechanism (facts-only); `GOAL` line injected into world-state text instead | No `system`/instructions parameter of any kind on `predict()`; identical `GOAL`-line-in-state-text approach used here for the same reason |
| Tool/label count constraint | Own guidance: 5 or fewer tools render directly before retrieval filters — 058 found `full`'s original 6 tools crowding out `attack` (0/2257 real decisions) and dropped to exactly 5, which helped (0/10 -> 5/10 on a combined test) | Tested the same crowding hypothesis here (11 labels vs. a trimmed 5, one state): the *opposite* happened — `attack_prob` dropped from 0.999 to 0.213 with fewer labels. No equivalent mechanism transfers; see [The real fix](#the-real-fix-pull-combat-out-into-its-own-noul-question) |
| Measured latency, base/default weights (this machine) | mean 356ms (basic-scenario compare run); separately, mean 115ms / p50 100ms / p95 247ms (isolated `level`-scenario benchmark) | mean 21.3–30.9ms warmed (isolated smoke test / real in-loop run), p95 ≤ 30.4ms in the 150-step real run; ~23ms mean with the (now two-call) shoot gate enabled in a real 10-episode comparison run |
| Measured latency, smallest tested variant | `rung_6.cact` (6-layer model-ladder rung, built via `needle build`): mean 35ms, p50 31ms, p95 86ms | N/A — Laya ships one checkpoint, no depth ladder to build a smaller rung from |
| Confidence calibration on this task | Reported ~0.72 mean in a real basic-scenario run; vendor's own guidance suggests ~0.7 as an execute band | Reported 0.02–0.19 with the original single-`choice` design; 0.31–0.59 with the shoot gate (a `noul` P(true), not a `choice` top-label probability — different calibration semantics, not directly comparable to Needle's number) |
| Base-checkpoint Doom behaviour, first honest look | "mostly turned and almost never chose shoot" (fixed afterward via triggers/docstrings) | Never chose shoot at all with combat competing inside one `choice` question (0/8 hand-built states); one 150-step run was 100% `turn_right` |
| Fix mechanism available | Needle's own documented `triggers=` regex + docstrings — used successfully in 058 to fix the shoot/move_forward/wall/attack-crowding issues | No `triggers=`-equivalent for `choice`, and criteria-wording alone didn't help (one pass made it worse — see "keyword-bait bias" above) — but Laya's separate `noul` primitive, pulling combat out of the `choice` entirely, plus deterministic `AMMO>0`/bearing guards in code, did: real 10-episode result now matches the `heuristic` baseline (`mean_kills` 1.00, `completion_rate` 1.00) |

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
training a model for this task, out of scope here, same as 058 stayed
out of RL training for Needle.

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

Same JSONL logging scheme as the sibling project: every run writes
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
| `--scenario {basic,my_way_home,level}` | Same three scenarios as the sibling project. Only `basic` has been run and verified here (see Roadmap). |
| `--action-set {stage1,full}` | `stage1`: 4 labels. `full`: 11 labels (10 canonical actions + `wait`). Unlike the sibling project, there is no "5 or fewer" tool-count guidance to work around here — Laya's `choice` type takes the whole `criteria` dict in one pass regardless of size. |
| `--decision-tics N` | Same control-frequency knob as the sibling project. |
| `--memory {stateless,prev_state,rolling}` (or `0`/`1`/`2`) | Same three memory modes, unchanged from `state_encoder.py`. |
| `--confidence-mode {always_execute,confidence_threshold}` + `--confidence-threshold F` | `confidence_threshold` substitutes `wait` when Laya's own calibrated confidence is below `F`. Defaults to **0.15**, not the sibling project's 0.6 — see [Verified behaviour](#verified-behaviour): this checkpoint's confidence on Doom states runs far lower than a tool-calling model's, so a tool-calling-model-appropriate threshold would gate almost every decision to `wait` here. |
| `--model-id ID` | HuggingFace model id or local path passed to `laya.load()` — the closest equivalent to the sibling project's `--weights` model-ladder lever, except Laya ships one checkpoint, not a depth ladder to sweep. |
| `--device {cuda,mps,cpu}` | Forces a device instead of auto-detection. Omit it — auto-detection correctly picked `mps` on this machine with no configuration. |
| `--no-goal-line` | Omit the `GOAL` line from the world-state text — same toggle as the sibling project, same open empirical question (does it change anything at all). |
| `--no-stuck-recovery` / `--no-threat-response` | Same two controller-level safety nets, ported unchanged from `controller.py` — see that module for what each does. |
| `--no-shoot-gate` | Use the original single-`choice`-call design (combat competes directly against move/turn) instead of the default `noul` shoot gate + `AMMO>0`/bearing guards — see [The real fix](#the-real-fix-pull-combat-out-into-its-own-noul-question). Mainly useful for regression/comparison runs against the pre-fix behaviour. |
| `--shoot-gate-threshold F` | `P(should_shoot)` cutoff for the gate (default **0.45**) — picked directly from a real 8-state spread (see the shoot-gate section), not a calibrated cutoff against a held-out set. |

## Why there's no system prompt

`laya.Agent.predict(state, questions)` takes exactly two arguments: no
`system`, `instructions`, or persona parameter of any kind exists to even
consider using — there's nothing analogous to Needle's facts-only
`system=` to invent workarounds for. The brief's priority list is
injected the same way as in the sibling project: one `GOAL` line at the
top of the world-state text itself (`state_encoder.GOAL_LINE`), since
that's the only text Laya actually reads. Whether it changes anything is
the same open, measurable question as before (`--no-goal-line`).

## Why `LayaAgent.reset()` is mostly a no-op

Covered in depth in [Verified behaviour](#verified-behaviour) above:
`laya.Agent` (0.1.6) exposes no `reset` method at all, and 200 consecutive
identical `predict()` calls showed no latency trend on this machine —
there's no persistent model state to reset and no confirmed staleness
failure mode to guard against, unlike the sibling project's Needle
wrapper. The one thing `LayaAgent.reset()` actually does is clear its own
internal `_gate_encoder`'s AREA visited-cells bookkeeping once per episode
(see [The real fix](#the-real-fix-pull-combat-out-into-its-own-noul-question)
for why the shoot gate has its own separate `StateEncoder`) — not a Laya
API call, just this wrapper's own per-episode state.

## Testing

```bash
pytest              # 78 tests, pure logic — no vizdoom process needed, and
                     # no real Laya model loaded (LayaAgent's own tests fake
                     # out laya.load() to test its label->action mapping,
                     # the shoot gate's AMMO/bearing guards, confidence
                     # gating, and Decision field population in isolation —
                     # see tests/test_laya_agent.py — plus scripts/probe_criteria.py
                     # for the real-model wording/gate experiments, not part
                     # of the automated suite)
```

## Project structure

```
laya-doom/
    config/basic.cfg,        ViZDoom scenario configs, ported unchanged
    my_way_home.cfg,         from the sibling project
    level.cfg
    laya_doom/
        doom_env.py          vizdoom.DoomGame wrapper + execute(action_name)
        perception.py        GameState -> Perception (no screen pixels)
        state_encoder.py      Perception -> compact text, 3 memory modes
        laya_agent.py          laya.Agent wrapper, predict()-based, no tool schema
        actions.py             semantic action vocabulary <-> button tables
        controller.py          the explicit per-tic run_episode() loop
        metrics.py              JSONL logging + summary stats
        dashboard.py             live terminal view
    experiments/
        random_agent.py, heuristic_agent.py   baselines, ported unchanged
        run.py                  single-controller CLI
        compare.py               multi-controller comparison CLI
    scripts/
        setup.sh                 automated venv + install
    tests/                  actions/perception/state_encoder/metrics/controller/
                             heuristic_agent/laya_agent unit tests
    logs/                   JSONL output (gitignored)
```

## Research questions this is built to answer

Same list as the sibling project, minus the model-ladder-specific one
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

Not built (all ported over structurally from the sibling project's
decision-agnostic pieces, so nothing below needs new perception/encoding
work — only running it and reporting what happens):

- **Stage 2 (`my_way_home` navigation)** and **Stage 3 (a real Freedoom
  level, `--scenario level`)** — both scenarios and the `full` action set
  are wired and covered by the ported tests, but not yet run against the
  real Laya checkpoint for this README. Now that combat actually works on
  `basic`, this is worth checking rather than assuming it inherits the
  same fix — `full`'s `attack` and `move_backward` weren't part of the
  crowding-analogy test that gave the opposite-of-Needle result above.
- **Fine-tuning Laya for this task** — the project explicitly supports
  this (a T4-GPU Colab notebook for fine-tuning on custom data); doing so
  would be training a model for Doom-playing specifically, which is
  out of scope here the same way RL training was out of scope for the
  sibling Needle project. Would also plausibly fix the calibration gaps
  the shoot gate currently papers over in code (the `AMMO>0` precondition,
  bearing discrimination).
- **`score` question type** — only `choice` and (as of the shoot-gate fix)
  `noul` were used here; Laya's third typed-question primitive wasn't
  explored for this task.
- **A richer threat/priority model** — two full-game Doom strategy
  documents (`deep-research-report (4).md`, a detailed AI Doom playbook
  covering enemy-specific tactics, weapon selection, infighting, and
  navigation memory; `tiny-doom-runtime-policy-200-rules.md`, a compressed
  200-rule version of the same) were dropped into this project folder —
  not used to build anything here, since they assume a much richer game
  state (per-ammo-type tracking, enemy taxonomy beyond bearing/distance,
  navigation memory, weapon selection) than this Stage 1 proof of concept
  implements. They're a plausible starting point for Stage 4/5 (a
  hierarchical agent issuing higher-level goals/skills, per both
  projects' own roadmaps) rather than something this project's current
  scope calls for.
- **Browser dashboard, multi-agent, multi-map progression** — same
  unbuilt items as the sibling project's own roadmap, for the same reasons.

## Known quirks

- Everything under perception.py/state_encoder.py's own "Known quirks" in
  the sibling project's README applies here unchanged (depth-buffer
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
