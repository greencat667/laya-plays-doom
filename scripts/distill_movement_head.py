"""Distil a trained movement head for Laya from logged play, with stuntd.

    python -m scripts.distill_movement_head --out models/movement_head

Needs ``stuntd[train]`` and laya>=0.3.4 (stuntd's trainer uses laya.common);
the project's own venv pins laya 0.1.6, so run this from a separate env.

stuntd (github.com/bladedevoff/stuntd) fine-tunes Laya's own decision head
(``type_emb``/``head``/``scorer``) on a frozen encoder for one fixed question,
from (text, answer) pairs. Its Snake demo went from 0.70 to 11.40 points a
game by distilling a BFS oracle. Here the teacher is this project's own
logged play (``logs/*.steps.jsonl``), filtered to decisions worth copying:

- move/strafe/retreat actions whose result wasn't ``no_change``;
- turns only when a reactive safety net chose them (threat_engagement,
  threat_response, stuck_recovery) -- Laya's own turns are label-order
  noise (scripts/probe_direction_bias.py);
- nothing from map-dependent overrides (frontier_planner, wall_follow,
  secret_search, exploration_nudge): their choice depends on a map the
  state text doesn't carry, so it can't be learned from it -- except the
  frontier planner's, when the logged state has the FRONTIER line
  (``--frontier-hint``) pointing where the planner was going;
- no ``wait`` (the wait trap), no combat (the shoot gate decides that).

The teacher inherits Laya's left bias wherever the action was Laya's own
(455 strafe_left vs 1 strafe_right in the first build), so every example
is also added mirrored -- "left" and "right" swapped in both the state text
and the label, which Doom's symmetric geometry makes a valid example too.
The last ``--drop-before-death`` steps (default 30) before any death are
left out: whatever the agent did there is the behaviour not to copy, and
door-first runs die often (5 of 6 teacher episodes on seed 6200).
Identical state texts are merged by majority label. The report compares
held-out agreement with the teacher against zero-shot Laya asked the same
movement question LayaAgent asks.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
from pathlib import Path

from laya_doom.laya_agent import QUESTION_ID, build_movement_criteria

FIELD = "action"
LABELS = tuple(build_movement_criteria("full"))
_REACTIVE_TURN_NETS = frozenset({"threat_engagement", "threat_response", "stuck_recovery"})
_MAP_DEPENDENT_NETS = frozenset({"frontier_planner", "wall_follow", "secret_search", "exploration_nudge"})
_MOVES = frozenset({"move_forward", "move_backward", "strafe_left", "strafe_right"})


def teacher_label(row: dict) -> str | None:
    """The action to copy from one logged step, or None to skip it. The
    frontier planner's own moves and turns count too when the state text
    carries the FRONTIER line that explains them (EncoderConfig.
    include_frontier_hint); without it they can't be learned from the text."""
    action, reason = row.get("action"), row.get("override_reason") or ""
    planner_explained = reason == "frontier_planner" and "\nFRONTIER " in row.get("encoded_state", "")
    if action not in LABELS or action in ("wait", "use"):
        return None
    if reason in _MAP_DEPENDENT_NETS and not planner_explained:
        return None
    if action in _MOVES:
        return action if row.get("result") != "no_change" else None
    # a turn: only worth copying when a reactive net, or the explained planner, picked it
    return action if row.get("overridden") and (reason in _REACTIVE_TURN_NETS or planner_explained) else None


def mirror(text: str) -> str:
    """Swap every "left" and "right" (state text or action label)."""
    return text.replace("left", "\0").replace("right", "left").replace("\0", "right")


def _rows_before_death_dropped(rows: list[dict], window: int) -> list[dict]:
    """rows minus the ``window`` steps leading up to (and including) each death,
    per episode. ViZDoom ends the episode the moment the player dies, so the
    logged step records ``episode_ended`` (never ``died``); a step-budget end
    records an ordinary result. A level exit would also read as
    ``episode_ended`` -- none has happened yet, and losing its last steps is
    harmless."""
    if window <= 0:
        return rows
    doomed: set[tuple[object, int]] = set()
    for row in rows:
        if row.get("result") in ("died", "episode_ended"):
            episode, step = row.get("episode"), row.get("step", 0)
            doomed.update((episode, s) for s in range(step - window + 1, step + 1))
    return [r for r in rows if (r.get("episode"), r.get("step", 0)) not in doomed]


def build_examples(paths: list[str], mirrored: bool = True, drop_before_death: int = 30) -> list[tuple[str, str]]:
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for path in paths:
        with open(path) as fh:
            rows = _rows_before_death_dropped([json.loads(line) for line in fh], drop_before_death)
            for row in rows:
                label = teacher_label(row)
                if label is not None:
                    votes[row["encoded_state"]][label] += 1
                    if mirrored:
                        votes[mirror(row["encoded_state"])][mirror(label)] += 1
    return [(text, counter.most_common(1)[0][0]) for text, counter in votes.items()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--logs", nargs="+", default=sorted(glob.glob("logs/laya_level_*.steps.jsonl") + glob.glob("logs/distill_*.steps.jsonl"))
    )
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--drop-before-death", type=int, default=30)
    parser.add_argument("--out", default="models/movement_head")
    parser.add_argument("--model-id", default="convaiinnovations/laya")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-gib", type=float, default=8.0,
        help="stuntd caches encoder output up to this size; over it, it re-encodes every epoch (much slower)",
    )
    args = parser.parse_args(argv)

    import laya
    from stuntd.train.dataset import Item, SiteDataset
    from stuntd.train.trainer import LayaTrainer

    examples = build_examples(args.logs, mirrored=not args.no_mirror, drop_before_death=args.drop_before_death)
    # Split by mirror pair, so no held-out state is a flipped copy of a training one.
    keys = sorted({min(text, mirror(text)) for text, _ in examples})
    random.Random(args.seed).shuffle(keys)
    held = set(keys[int(len(keys) * (1 - args.holdout)) :])
    examples.sort(key=lambda ex: min(ex[0], mirror(ex[0])) in held)
    split = sum(min(text, mirror(text)) not in held for text, _ in examples)
    index = {label: i for i, label in enumerate(LABELS)}
    items = [Item(text, index[label], float(i)) for i, (text, label) in enumerate(examples)]
    dataset = SiteDataset(
        site="doom_movement", kind="choice", field=FIELD, labels=LABELS,
        train=tuple(items[:split]), holdout=tuple(items[split:]), deduplicated=0, dropped=0,
    )
    counts = collections.Counter(label for _, label in examples)
    print(f"{len(examples)} unique states from {len(args.logs)} logs; train {split}, holdout {len(items) - split}")
    print("label counts:", dict(counts.most_common()))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer = LayaTrainer(
        args.model_id, epochs=args.epochs, seed=args.seed, cache_max_bytes=int(args.cache_gib * 2**30)
    )
    logits = trainer(dataset, out / "head.safetensors")
    holdout = dataset.holdout
    trained = sum(max(range(len(LABELS)), key=row.__getitem__) == item.label for row, item in zip(logits, holdout))

    base = laya.load(args.model_id)
    criteria = build_movement_criteria("full")
    question = {QUESTION_ID: {"type": "choice", "instructions": (
        "Given the current Doom game world-state text below, which action should the player take right now?"
    ), "criteria": criteria}}
    zero_shot = sum(
        base.predict(item.text, question)["answers"][QUESTION_ID]["choice"] == LABELS[item.label] for item in holdout
    )
    majority = counts.most_common(1)[0]
    report = {
        "examples": len(examples), "train": split, "holdout": len(holdout), "labels": list(LABELS),
        "label_counts": dict(counts), "epochs": args.epochs, "model_id": args.model_id, "field": FIELD,
        "holdout_agreement_trained": trained / len(holdout),
        "holdout_agreement_zero_shot": zero_shot / len(holdout),
        "always_majority_label": majority[0],
        "holdout_agreement_majority": sum(LABELS[i.label] == majority[0] for i in holdout) / len(holdout),
    }
    (out / "meta.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k.startswith(("holdout", "always"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
