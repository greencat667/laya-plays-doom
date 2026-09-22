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
safety nets — that was originally built for a related but separate
experiment, which asked the same question of Cactus Compute's Needle 3,
an autoregressive tool-calling model, as the decision engine instead of
Laya. That earlier project is **not part of this repository**, so where
this README cites Needle numbers below (latency, confidence,
crowding-fix behaviour), they're comparative results measured on that
other setup, not something you can click through and verify here. The
**only** thing that changes in this repo is the decision engine. This
project is fully self-contained (own venv, own `requirements.txt`, own
git repo) and has no dependency on that earlier project or on
`cactus-needle`.

**Status**: the four-action proof of concept (`basic` scenario) works
end to end on Apple Silicon macOS with real ViZDoom and the
`convaiinnovations/laya` checkpoint. That checkpoint is fine-tuned for
email routing, moderation, and support-ticket triage, not Doom, so combat
needed a dedicated mechanism (see [How the decision engine
works](#how-the-decision-engine-works)): pulling `shoot`/`attack` out of
the main `choice` call into its own `noul` "should_shoot" question, plus
deterministic `AMMO>0`/bearing-front guards in code. With that in place,
Laya matches the hand-written heuristic baseline on combat metrics
(`mean_kills=1.00`, `completion_rate=1.00`, `mean_latency_ms=22.8`).
Frontier-directed exploration, a secret-door search, and a
wall-following fallback handle navigating a real Freedoom level. A level
exit (`completed=True`) has not yet been observed in testing.

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

## How the decision engine works

`LayaAgent.decide()` calls `laya.Agent.predict(state, questions)` once per
control cycle, passing the encoded world-state text and a dict of typed
questions. Two of Laya's question types are used:

- **`choice`** — a softmax over a fixed label set (the movement actions:
  `move_forward`, `turn_left`, `turn_right`, `wait`, etc.), returned as
  per-label probabilities plus a calibrated top-label confidence.
- **`noul`** — a standalone calibrated `P(true)` over one yes/no
  question, independent of any other label.

**Combat is not part of the `choice` call.** The `convaiinnovations/laya`
checkpoint is fine-tuned for email routing, moderation, and
support-ticket triage, not Doom, and testing showed it essentially never
picks `shoot`/`attack` when combat competes directly against movement
inside one `choice` question (0/8 hand-built states; a 150-step episode
picked `turn_right` on every single decision, including with an enemy
directly ahead). Instead, `shoot`/`attack` is asked as its own `noul`
question — "should the player shoot?" — and only fires when the model's
`P(should_shoot)` clears a threshold (`--shoot-gate-threshold`, default
**0.45**) *and* a deterministic code-level check passes (`AMMO > 0` and
the enemy's bearing is exactly `front`). The model's own confidence on
this out-of-domain task is low enough (0.02–0.19 on the plain `choice`
call) that the deterministic guards, not the model, carry the precision
here. `--no-shoot-gate` reverts to the single-`choice`-call design for
comparison.

**Navigation** uses ViZDoom's `ANGLE` game variable (verified empirically
to be degrees, wrapping 0–360, with `turn_left` increasing it) plus the
player's `(x, y)` position and a visited-cells grid to compute a
frontier-directed heading (`FrontierExplorationConfig`,
`laya_doom/wayfinding.py`). When frontier-directed movement stops finding
new territory, `SecretSearchConfig` sweeps `use` against nearby walls
(Doom secret doors are visually identical to ordinary walls), and
`WallFollowConfig` falls back to a classic maze wall-following rule.
`--no-frontier-exploration`, `--no-secret-search`, and `--no-wall-follow`
disable each independently. A handful of deterministic safety nets in
`controller.py` (`StuckRecoveryConfig`, `TurnLoopRecoveryConfig`,
`ThreatEngagementConfig`, `LowHealthRetreatConfig`, `DoorUseConfig`) sit
above all of this to recover from repeated no-progress decisions — see
[Configuration knobs](#configuration-knobs) for what each one does.

With the shoot gate and wayfinding both enabled, a 10-episode comparison
run on the `basic` scenario gives:

```
                 Controller comparison
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ metric                ┃ random  ┃ heuristic ┃ laya   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ mean_kills            │ 0.50    │ 1.00      │ 1.00   │
│ mean_damage_given     │ 6.00    │ 11.50     │ 11.00  │
│ completion_rate       │ 0.50    │ 1.00      │ 1.00   │
│ mean_confidence       │ n/a     │ 1.00      │ 0.47   │
│ mean_latency_ms       │ 0.00    │ 0.00      │ 22.76  │
└───────────────────────┴─────────┴───────────┴────────┘
```

Laya matches the hand-written heuristic on the combat metrics that
matter. On longer runs against a real Freedoom level (`--scenario
level`), exploration coverage and survival improve substantially with
wayfinding enabled (up to ~34,000 distance units and 64–83 distinct
visited grid cells over a 4,000-step run), but a level exit
(`completed=True`, ViZDoom's own `map_exit_reward >= 1` signal) has not
been observed in any run tried so far.

## Laya vs Needle

Measured numbers from both projects on the same machine, not vendor
marketing figures. The Needle numbers come from the separate, unlinked
experiment described above; the Laya numbers are from this repository.

| | Needle 3 (related experiment) | Laya (this project) |
|---|---|---|
| Mechanism | Autoregressive tool-calling: decode loop produces a structured function call + free-text reasoning | Non-autoregressive: one forward pass over typed `choice`/`score`/`noul` questions, no generation at all |
| Persistent state between decisions | Yes — internal state that needed periodic `reset()` (`reset_every`, default 15) to avoid a decode stall on repetitive input | None — `laya.Agent` has no `reset()` method and no persistent state between calls |
| System/goal-priority injection | No `system=` prose mechanism (facts-only); `GOAL` line injected into world-state text instead | No `system`/instructions parameter on `predict()`; same `GOAL`-line-in-state-text approach |
| Tool/label count constraint | 5 or fewer tools render directly before retrieval filters; dropping `full`'s original 6 tools to 5 fixed `attack` being crowded out (0/2257 → helped) | Tested the same crowding hypothesis (11 labels vs. a trimmed 5): the opposite happened — `attack_prob` dropped from 0.999 to 0.213 with fewer labels. Doesn't transfer; see [How the decision engine works](#how-the-decision-engine-works) |
| Measured latency, base/default weights | mean 356ms (basic-scenario), mean 115ms / p95 247ms (level-scenario) | mean 21–31ms, ~23ms with the shoot gate enabled |
| Measured latency, smallest tested variant | `rung_6.cact` (6-layer model-ladder rung): mean 35ms | N/A — Laya ships one checkpoint, no depth ladder |
| Confidence calibration on this task | ~0.72 mean; vendor guidance suggests ~0.7 as an execute band | 0.02–0.19 with the plain `choice` call; 0.31–0.59 with the shoot gate's `noul` P(true) — different calibration semantics, not directly comparable |
| Base-checkpoint Doom behaviour, first look | Mostly turned, almost never chose shoot (fixed via triggers/docstrings) | Never chose shoot with combat competing inside one `choice` question (0/8 hand-built states) |
| Fix mechanism available | `triggers=` regex + docstrings on the tool schema | No `triggers=` equivalent for `choice`; fixed instead with a separate `noul` question plus deterministic `AMMO>0`/bearing guards |

The clearest architectural win here is latency: even Needle's smallest
tested rung (35ms mean) is slower than Laya's steady-state mean (21ms).
Both projects hit the same underlying problem — the model not reliably
firing the combat action — and fixed it with a mechanism specific to
their own architecture (Needle's `triggers=`/tool-count guidance vs.
Laya's separate `noul` question type). Whether a fine-tuned Laya
checkpoint would close the remaining calibration gap is untested —
Laya supports fine-tuning via a T4-GPU Colab notebook, but training a
model for this task was out of scope here.

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
| `--scenario {basic,my_way_home,level}` | Which ViZDoom scenario to run. Only `basic` and `level` have been exercised so far — see Roadmap. |
| `--action-set {stage1,full}` | `stage1`: 4 labels. `full`: 11 labels (10 canonical actions + `wait`). Laya's `choice` type takes the whole `criteria` dict in one pass regardless of size, so there's no tool-count constraint to work around here. |
| `--decision-tics N` | Control-frequency knob: how many game tics pass between decisions. |
| `--memory {stateless,prev_state,rolling}` (or `0`/`1`/`2`) | Three memory modes for how much history goes into the encoded state text, implemented in `state_encoder.py`. |
| `--confidence-mode {always_execute,confidence_threshold}` + `--confidence-threshold F` | `confidence_threshold` substitutes `wait` when Laya's own calibrated confidence is below `F`. Defaults to **0.15** — this checkpoint's confidence on Doom states runs much lower than on its native tasks, so a higher threshold gates almost every decision to `wait`. |
| `--model-id ID` | HuggingFace model id or local path passed to `laya.load()`. Laya ships one checkpoint, no depth ladder to sweep. |
| `--device {cuda,mps,cpu}` | Forces a device instead of auto-detection (`laya.Agent` checks CUDA, then MPS, then falls back to CPU). |
| `--no-goal-line` | Omit the `GOAL` line from the world-state text. |
| `--no-stuck-recovery` / `--no-threat-response` | Two controller-level safety nets in `controller.py` — see that module for what each does. |
| `--no-shoot-gate` | Use the original single-`choice`-call design (combat competes directly against move/turn) instead of the default `noul` shoot gate + `AMMO>0`/bearing guards — see [How the decision engine works](#how-the-decision-engine-works). |
| `--shoot-gate-threshold F` | `P(should_shoot)` cutoff for the gate (default **0.45**). |
| `--no-turn-loop-recovery` | Disable the safety net that forces a `move_forward` attempt after too many consecutive executed turns (`controller.TurnLoopRecoveryConfig`). |
| `--no-threat-engagement` | Disable the safety net that turns toward a visible, near-enough, off-center enemy instead of letting Laya's movement choice stand (`controller.ThreatEngagementConfig`). |
| `--no-low-health-retreat` / `--low-health-threshold N` / `--emergency-health-threshold N` | Disable/tune the safety net that retreats (or, below the emergency threshold, overrides even an attack) when a visible enemy is present and health is low (`controller.LowHealthRetreatConfig`). |
| `--no-exploration-nudge` / `--exploration-streak-threshold N` | Disable/tune the circling detector (`controller.ExplorationNudgeConfig`), whose override defaults to directed frontier-seeking — see [How the decision engine works](#how-the-decision-engine-works). |
| `--no-frontier-exploration` / `--frontier-lookahead-cells F` | Revert `ExplorationNudgeConfig`'s override to a blind guess, or tune how many grid cells ahead each candidate heading is projected (`controller.FrontierExplorationConfig`). |
| `--no-door-use` / `--door-use-stall-threshold N` | Disable/tune the safety net that tries `use` once `WALL ahead near` has held for N consecutive steps (`controller.DoorUseConfig`) — Laya's own movement choice almost never picks `use` on its own. |
| `--no-secret-search` | Disable the systematic turn-and-use sequence that engages once frontier exploration has stopped finding new territory (`controller.SecretSearchConfig`). |
| `--no-wall-follow` / `--wall-follow-hand {left,right}` | Disable/pick the hand for the classic maze wall-following fallback (`controller.WallFollowConfig`), which engages once nudging and a secret-door sweep have both failed. |
| `--window-scale {1,2,3}` | Doom window size with `--render`: 1=320×240, 2=640×480, 3=1024×768. Purely a display size; behaviour is unaffected. |

## Why there's no system prompt

`laya.Agent.predict(state, questions)` takes exactly two arguments: no
`system`, `instructions`, or persona parameter exists. The priority list
is injected as one `GOAL` line at the top of the world-state text itself
(`state_encoder.GOAL_LINE`), since that's the only text Laya reads.
`--no-goal-line` omits it.

## Why `LayaAgent.reset()` is mostly a no-op

`laya.Agent` (0.1.6) exposes no `reset` method at all, and there's no
persistent model state between `predict()` calls to reset. The one thing
`LayaAgent.reset()` actually does is clear its own internal
`_gate_encoder`'s AREA visited-cells bookkeeping once per episode (the
shoot gate uses its own separate `StateEncoder` — see [How the decision
engine works](#how-the-decision-engine-works)) — not a Laya API call,
just this wrapper's own per-episode state.

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
        wayfinding.py            ANGLE-based frontier/heading helpers
        controller.py          the explicit per-tic run_episode() loop
        metrics.py              JSONL logging + summary stats
        dashboard.py             live terminal view
    experiments/
        random_agent.py, heuristic_agent.py   baselines, ported unchanged
        run.py                  single-controller CLI
        compare.py               multi-controller comparison CLI
    scripts/
        setup.sh                 automated venv + install
        probe_criteria.py        criteria-wording/shoot-gate probes against the real model
        verify_angle.py,         ANGLE sign-convention verification
        verify_angle_movement.py
        probe_automap.py         automap-buffer capture + analysis
    tests/                  actions/perception/state_encoder/metrics/controller/
                             heuristic_agent/laya_agent/wayfinding unit tests
    logs/                   JSONL output (gitignored)
```

## Research questions this is built to answer

1. Does Laya react sensibly to enemies at all? No, not without the shoot
   gate — see [How the decision engine works](#how-the-decision-engine-works).
2. Do repeated local decisions produce anything resembling persistent
   behaviour, or just noise? (watch `--dashboard`)
3. Can it navigate without an explicit planner? (`--scenario my_way_home` —
   not yet run here, see Roadmap)
4. How little state does it actually need? (`--memory stateless` vs. others)
5. Does remembering the last action help? (`--memory prev_state` vs. `stateless`)
6. Does outcome feedback (`LAST_RESULT`) help? (same axis)
7. What decision frequency works best? (`--decision-tics`)
8. How does it compare to the tiny heuristic? Matches it on combat
   metrics; see [How the decision engine works](#how-the-decision-engine-works).
9. How does Laya's architecture compare to Needle's for this task? — see
   [Laya vs Needle](#laya-vs-needle).

## Roadmap

Built:

- **Stage 1** (four-action proof of concept, `basic` scenario) — working
  end to end, including the shoot gate.
- **Stage 2/3** (`my_way_home` navigation, and a real Freedoom level via
  `--scenario level` with the `full` action set) — run over multiple
  episodes up to 4,000 steps, exercising the safety nets in
  `controller.py`.
- **Stage 4/5, partially** (real wayfinding, exit-seeking) — frontier-directed
  exploration, a secret-door search, and a wall-following fallback are
  built and working. Not achieved: a level exit (`completed=True`) has
  not been observed in any run so far. The project's own
  `docs/doom-strategy-research.md` and
  `docs/tiny-doom-runtime-policy-200-rules.md` describe the same
  frontier→secret-search escalation shape used here, and were a useful
  starting point for a future hierarchical agent, which this project
  doesn't attempt.

Not built:

- **Fine-tuning Laya for this task** — supported via a T4-GPU Colab
  notebook, but out of scope here. Would plausibly fix the calibration
  gaps the shoot gate currently papers over in code (the `AMMO>0`
  precondition, bearing discrimination).
- **`score` question type** — only `choice` and `noul` were used here;
  Laya's third typed-question primitive wasn't explored.
- **A hierarchical agent with richer per-ammo-type/enemy-taxonomy/
  navigation-memory state** — the two strategy documents above assume
  more game state than `Perception`/the world-state text currently
  implements.
- **Sweeping `--decision-tics`/`--max-steps` to chase `completed=True`** —
  the most likely next lever, not yet tried.
- **Browser dashboard, multi-agent, multi-map progression.**

## Known quirks

- Laya's `confidence` field is described as "calibrated" via ECE
  (expected calibration error) numbers on its own benchmark tasks
  (routing, moderation, etc.) — that calibration claim isn't re-verified
  here. What's measured is that confidence stayed low (0.02–0.19) on
  Doom states, consistent with a well-calibrated model correctly
  reporting low confidence on an out-of-distribution task.
- The shoot gate's bearing guard requires an enemy at exactly bearing
  `front` before the combat action can fire, regardless of the `noul`
  gate's output — an enemy at `front-left`/`front-right` won't get shot
  at until a turn lines it up exactly. This fixed a real 0-damage bug but
  is a real behavioural constraint, not a free improvement.
- `wall_ahead`/`wall_near` come from a 3-ray depth scan (forward/left/right
  at fixed screen fractions) that can under-detect nearby blocking
  geometry outside those exact rays (a thin pillar, an off-axis corner);
  `perception.py`'s own docstring calls this calibration approximate.
- ViZDoom's automap buffer background fill colour is the same for
  revealed and undiscovered regions, so a "count black pixels"
  heuristic for "how much is unexplored" doesn't work on it.
- `WallFollowConfig` inflates raw distance travelled without a
  proportional increase in exploration coverage — a real trade-off of
  the escalation ladder, not a free improvement.
