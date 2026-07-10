#!/usr/bin/env python3
"""Build a fixed, deterministic 300-case ToolACE eval set.

Mirrors the community model card's "300 examples derived from the untouched
Team-ACE/ToolACE dataset" used for the first-call evaluation. We pin the seed
and write the set to disk so every later run is reproducible and comparable.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_answers(answers: Any) -> list[dict]:
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except json.JSONDecodeError:
            return []
    if not isinstance(answers, list):
        return []
    out = []
    for a in answers:
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except json.JSONDecodeError:
                continue
        if isinstance(a, dict) and "name" in a:
            out.append({"name": a["name"], "arguments": a.get("arguments", {}) or {}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Team-ACE/ToolACE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    if Path(args.src).exists():
        with open(args.src) as f:
            raw = [json.loads(l) for l in f if l.strip()]
    else:
        from datasets import load_dataset

        raw = list(load_dataset(args.src, split=args.split))

    # Keep only single-turn, single-call, well-formed examples.
    clean = []
    for ex in raw:
        q = (ex.get("queries") or [])
        t = (ex.get("tools") or [])
        c = parse_answers(ex.get("answers"))
        if len(q) == 1 and t and len(c) >= 1:
            clean.append(
                {
                    "id": ex.get("id", len(clean)),
                    "query": q[0],
                    "tools": t,
                    "relevant_tools": ex.get("relevant_tools", []),
                    "gold": c,
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(clean)
    selected = clean[: args.n]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} eval cases -> {args.out}")


if __name__ == "__main__":
    main()
