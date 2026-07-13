#!/usr/bin/env python3
"""Build a fixed, deterministic 300-case ToolACE eval set.

Mirrors the community model card's "300 examples derived from the untouched
Team-ACE/ToolACE dataset" used for the first-call evaluation. We pin the seed
and write the set to disk so every later run is reproducible and comparable.

Supports both the live ShareGPT format (Team-ACE/ToolACE on the Hub:
`conversations` + tools in `system`) and the older `queries`/`tools`/`answers`
export. Single-call examples are selected for a clean first-call measurement.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.toolace_conversations import extract_example as extract_conversations


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


def _from_legacy(ex: dict) -> dict | None:
    q = (ex.get("queries") or [])
    t = (ex.get("tools") or [])
    c = parse_answers(ex.get("answers"))
    if len(q) == 1 and t and len(c) >= 1:
        return {
            "id": ex.get("id", 0),
            "query": q[0],
            "tools": t,
            "relevant_tools": ex.get("relevant_tools", []),
            "gold": c,
        }
    return None


def _normalize(rec: dict) -> dict | None:
    if not rec:
        return None
    gold = rec["gold"]
    if len(gold) != 1:  # keep first-call eval single-call
        return None
    return {
        "id": rec.get("id", 0),
        "query": rec["query"],
        "tools": rec["tools"],
        "relevant_tools": rec.get("relevant_tools", []),
        "gold": gold,
    }


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

    clean = []
    for ex in raw:
        # live ShareGPT format first, then legacy export
        rec = extract_conversations(ex) if "conversations" in ex else None
        if rec is None:
            rec = _from_legacy(ex)
        rec = _normalize(rec)
        if rec:
            clean.append(rec)

    rng = random.Random(args.seed)
    rng.shuffle(clean)
    selected = clean[: args.n]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} eval cases -> {args.out} (pool={len(clean)})")


if __name__ == "__main__":
    main()
