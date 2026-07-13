#!/usr/bin/env python3
"""Practical Agentic Score (PAS) — single headline number for the beat.

PAS = mean of the core practical dimensions. Baseline (from the source card's
own numbers, with recovery/multi-turn = 0 because the baseline has neither):

  parseable 0.9933 | valid 0.9700 | expected 0.9267 | exact_args 0.6533
  arg_overlap 0.7517 | stopped 0.1500 | recovery 0 | multiturn 0
  => baseline PAS ~= 0.63

Yantra target PAS >= 0.88.
"""
from __future__ import annotations

import argparse
import json


def pas_from_summary(summary: dict, recovery: float = 0.0, multiturn: float = 0.0) -> float:
    comps = [
        summary.get("parseable", 0.0),
        summary.get("valid_name", 0.0),
        summary.get("expected_name", 0.0),
        summary.get("exact_args", 0.0),
        summary.get("arg_key_overlap", 0.0),
        summary.get("stopped_cleanly", 0.0),
        recovery,
        multiturn,
    ]
    return round(sum(comps) / len(comps), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="eval/results/latest.json")
    ap.add_argument("--recovery", type=float, default=0.0)
    ap.add_argument("--multiturn", type=float, default=0.0)
    args = ap.parse_args()

    data = json.load(open(args.results))
    summary = data["summary"]
    score = pas_from_summary(summary, args.recovery, args.multiturn)
    print(f"PAS = {score}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
