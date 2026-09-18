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
projection. The honest headline finding, reported plainly rather than
tuned away (see that section): **this checkpoint is fine-tuned for email
routing, moderation, and support-ticket triage — not Doom — and it shows.**
It never once fired a `shoot`/`attack` action across every real run in
this README, including states where an enemy was reported dead ahead with
ammo available, and on one real 150-step run it picked `turn_right` on
literally every single decision, with zero net distance travelled.

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
a wording problem. **`CRITERIA_V1` (the vaguer, Needle-docstring-derived
wording) is what's shipped in `laya_agent.py`** — there was no honest
basis to prefer V2's result, and re-tuning further to chase a
better-looking number would be exactly the "did we engineer it to look
like it can" trap the sibling project's own README warns against.

### A real 150-step episode: stuck turning, not just biased in isolated tests

```
$ python -m experiments.run --controller laya --episodes 1 --scenario basic --max-steps 150
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
("mostly turned and almost never chose shoot") — except here there was no
equivalent trigger/docstring mechanism to fix it with (see [Laya vs
Needle](#laya-vs-needle) below), and the one wording pass tried above
didn't help either. Reported plainly, per this project's own rule, rather
than hidden or re-tuned away.

### Controller comparison, real run

```
$ python -m experiments.compare --controllers random heuristic laya --episodes 5 --scenario basic

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
real decisions), never completed an episode (always ran out the
`--max-steps` budget instead of the scenario's own win/end condition
firing), and its `mean_total_reward` (-1500.0, i.e. the `basic` scenario's
`living_reward=-1` for 500 straight tics at `turn`'s 3-tic cost with no
compensating kill reward) is worse than even the `random` baseline. The
hand-written `heuristic` — a few lines of `if enemy front: shoot`
— comfortably wins on every combat metric. This is exactly the
"is it worth its latency/compute cost" bar `heuristic_agent.py`'s own
docstring describes, and on this checkpoint, for this task, the answer is
no.

## Laya vs Needle

Real, measured numbers from both projects on this same machine — not
marketing figures from either vendor.

| | Needle 3 (058) | Laya (059, this project) |
|---|---|---|
| Mechanism | Autoregressive tool-calling: decode loop produces a structured function call + free-text reasoning | Non-autoregressive: one forward pass over typed `choice`/`score`/`noul` questions, no generation at all |
| Persistent state between decisions | Yes — internal state that needed periodic `reset()` (`reset_every`, default 15) to avoid a real, confirmed decode stall (38 calls in, 57s+) on repetitive input | None — `laya.Agent` has no `reset()` method; verified empirically here (200 identical calls, no latency trend, see above) rather than just inferred |
| System/goal-priority injection | No documented `system=` prose mechanism (facts-only); `GOAL` line injected into world-state text instead | No `system`/instructions parameter of any kind on `predict()`; identical `GOAL`-line-in-state-text approach used here for the same reason |
| Tool/label count constraint | Own guidance: 5 or fewer tools render directly before retrieval filters; `full` action set (6 tools) already over that | None found or documented — Laya's `choice` type takes an arbitrary `criteria` dict in one pass; the `full` action set's 11 labels were used directly with no trimming |
| Measured latency, base/default weights (this machine) | mean 356ms (basic-scenario compare run); separately, mean 115ms / p50 100ms / p95 247ms (isolated `level`-scenario benchmark) | mean 21.3–30.9ms warmed (isolated smoke test / real in-loop run), p95 ≤ 30.4ms in the 150-step real run |
| Measured latency, smallest tested variant | `rung_6.cact` (6-layer model-ladder rung, built via `needle build`): mean 35ms, p50 31ms, p95 86ms | N/A — Laya ships one checkpoint, no depth ladder to build a smaller rung from |
| Confidence calibration on this task | Reported ~0.72 mean in a real basic-scenario run; vendor's own guidance suggests ~0.7 as an execute band | Reported 0.02–0.19 across every state tested here (isolated and in-loop) — well below Needle's band, but plausibly an honest signal that this checkpoint knows it's out of its training distribution, not miscalibration |
| Base-checkpoint Doom behaviour, first honest look | "mostly turned and almost never chose shoot" (fixed afterward via triggers/docstrings) | Never chose shoot at all in every test run in this README, including one 150-step run that was 100% turn_right; one wording pass tried, didn't clearly help (see above) |
| Fix mechanism available | Needle's own documented `triggers=` regex + docstrings — used successfully in 058 to fix the shoot/move_forward/wall issues | None equivalent exists — no trigger/schema mechanism for `choice` questions; the only lever is criteria wording, which was tried once here and did not clearly help |

The latency gap is the clearest win for Laya's architecture on this
machine — even Needle's *smallest* tested rung (35ms mean) is slower than
Laya's steady-state mean (21.3ms) with zero model-ladder engineering
required. The behavioural gap runs the other way: Needle's triggers gave
this project a concrete, documented lever to fix bad tool-selection
behaviour, and using it took Needle from "almost never shoots" to
"shoots reliably when an enemy is exactly ahead." Laya's single-pass
`choice` mechanism has no equivalent lever — criteria wording is the only
thing to tune, and one honest attempt at that here didn't move the
needle (so to speak). Whether a fine-tuned Laya checkpoint (the project
explicitly supports fine-tuning on a T4 GPU via its Colab notebook) would
close this gap is an open question this project didn't attempt to answer
— that would be training a model for this task, which is out of scope
for what was asked here, same as 058 stayed out of RL training for Needle.

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

## Why there's no system prompt

`laya.Agent.predict(state, questions)` takes exactly two arguments: no
`system`, `instructions`, or persona parameter of any kind exists to even
consider using — there's nothing analogous to Needle's facts-only
`system=` to invent workarounds for. The brief's priority list is
injected the same way as in the sibling project: one `GOAL` line at the
top of the world-state text itself (`state_encoder.GOAL_LINE`), since
that's the only text Laya actually reads. Whether it changes anything is
the same open, measurable question as before (`--no-goal-line`).

## Why `LayaAgent.reset()` is a no-op

Covered in depth in [Verified behaviour](#verified-behaviour) above:
`laya.Agent` (0.1.6) exposes no `reset` method at all, and 200 consecutive
identical `predict()` calls showed no latency trend on this machine —
there's no persistent state to reset and no confirmed staleness failure
mode to guard against, unlike the sibling project's Needle wrapper. Kept
as a real (if trivial) method purely so `LayaAgent` satisfies the same
`Agent` protocol `controller.run_episode()` expects from every controller.

## Testing

```bash
pytest              # 74 tests, pure logic — no vizdoom process needed, and
                     # no real Laya model loaded (LayaAgent's own tests fake
                     # out laya.load() to test its label->action mapping,
                     # confidence gating, and Decision field population in
                     # isolation — see tests/test_laya_agent.py)
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
  end to end, see Verified behaviour above.

Not built (all ported over structurally from the sibling project's
decision-agnostic pieces, so nothing below needs new perception/encoding
work — only running it and reporting what happens):

- **Stage 2 (`my_way_home` navigation)** and **Stage 3 (a real Freedoom
  level, `--scenario level`)** — both scenarios and the `full` action set
  are wired and covered by the ported tests, but not yet run against the
  real Laya checkpoint for this README; given the Stage 1 finding above
  (never shoots, gets stuck turning), there's no reason to expect
  materially different navigation behaviour without first establishing
  whether this checkpoint responds to `PATH`/`WALL` state at all any
  better than it responds to `ENEMY` state — worth checking, not assuming.
- **Fine-tuning Laya for this task** — the project explicitly supports
  this (a T4-GPU Colab notebook for fine-tuning on custom data); doing so
  would be training a model for Doom-playing specifically, which is
  out of scope here the same way RL training was out of scope for the
  sibling Needle project.
- **`score`/`noul` question types** — only `choice` was used here (the
  natural fit for "pick one action"); Laya's other two typed-question
  primitives weren't explored for this task.
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
- The one 150-step real run reported above is a single episode, not a
  large-sample result — it's reported because it's a vivid, reproducible
  illustration of the same 100%-one-action pattern the isolated 8-state
  test also showed, not because n=1 episodes are being treated as
  statistically definitive. The 5-episode `compare` table is the closer-
  to-decisive number, and it agrees with it (0 kills, 0 damage, 0%
  completion across all 5).
