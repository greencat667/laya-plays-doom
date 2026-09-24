"""Does Laya's left/right choice come from the state, or from label order?

    python -m scripts.probe_direction_bias --steps-log logs/laya_level_exit4.steps.jsonl

Real model, real logged world-state texts. Three checks:

1. Label-order swap: ask the same movement `choice` question with the
   criteria dict in its normal order and reversed. If direction followed
   the state, the winners would match; with a position bias they flip.
2. State mirror: swap "left"<->"right" in the state text, keep label
   order. A state-sensitive model should flip direction; a biased one won't.
3. Order-balanced signal: average probabilities over both label orders,
   then compare P(left labels) - P(right labels) between each state and
   its mirror. A real directional signal shows up as a consistently
   positive difference; noise averages to ~0.

Result when this was written (60 / 40 sampled states): original order
chose left-side labels 42/60 and right-side 0/60; reversed order chose
strafe_right 48/60; mirroring the state produced 0 right-side choices; the
order-balanced mirror difference was +0.027 +/- 0.156 (n=40) -- no usable
directional signal. That's why LayaAgent(direction_mode="resolved") exists.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics

from laya_doom.laya_agent import QUESTION_ID, LayaAgent


def _mirror(text: str) -> str:
    return text.replace("left", "\0").replace("right", "left").replace("\0", "right")


def _side_margin(probabilities: dict[str, float]) -> float:
    left = sum(p for label, p in probabilities.items() if "left" in label)
    right = sum(p for label, p in probabilities.items() if "right" in label)
    return left - right


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps-log", default="logs/laya_level_exit4.steps.jsonl")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default="convaiinnovations/laya")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    agent = LayaAgent(action_set="full", model_id=args.model_id, device=args.device)
    instructions = agent._movement_questions[QUESTION_ID]["instructions"]
    forward = list(agent._movement_criteria.items())
    reverse = forward[::-1]

    def ask(state: str, items: list[tuple[str, str]]) -> dict:
        questions = {QUESTION_ID: {"type": "choice", "instructions": instructions, "criteria": dict(items)}}
        return agent.agent.predict(state, questions)["answers"][QUESTION_ID]

    def balanced(state: str) -> dict[str, float]:
        a, b = ask(state, forward)["probabilities"], ask(state, reverse)["probabilities"]
        return {label: (a[label] + b[label]) / 2 for label in a}

    with open(args.steps_log) as fh:
        rows = [json.loads(line) for line in fh]
    random.seed(args.seed)
    states = [r["encoded_state"] for r in random.sample(rows, min(args.samples, len(rows)))]

    winners = {"original order": collections.Counter(), "reversed order": collections.Counter(),
               "mirrored state": collections.Counter()}
    margins = []
    for state in states:
        winners["original order"][ask(state, forward)["choice"]] += 1
        winners["reversed order"][ask(state, reverse)["choice"]] += 1
        winners["mirrored state"][ask(_mirror(state), forward)["choice"]] += 1
        if "left" in state or "right" in state:
            margins.append(_side_margin(balanced(state)) - _side_margin(balanced(_mirror(state))))

    for name, counter in winners.items():
        print(f"{name:15s} {counter.most_common()}")
    if len(margins) > 1:
        print(
            f"order-balanced margin(state) - margin(mirror): mean {statistics.fmean(margins):+.4f} "
            f"sd {statistics.stdev(margins):.4f} n={len(margins)}  (a real signal would be clearly > 0)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
