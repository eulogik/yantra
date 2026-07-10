#!/usr/bin/env python3
"""Build the RTE (Reflection-on-Tool-Error) synthetic corpus.

For each gold tool call we synthesize a realistic tool failure, then emit a
corrected call. Format:

    <bind tool="..."/>
    <args><param name="city">Pariz</param></args>
    <action_end/>
    <tool_error>404: unknown city 'Pariz'</tool_error>
    <reflect/>
    <bind tool="..."/>
    <args><param name="city">Paris</param></args>
    <action_end/>
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from convert_toolace_dtsa import convert_example, parse_answers, to_dtsa

ERROR_TEMPLATES = [
    ("404: unknown {param} '{value}'", "typo"),
    ("ValidationError: {param} must be one of [enum]; got '{value}'", "enum"),
    ("TypeError: {param} expected int, got str '{value}'", "type"),
    ("MissingRequiredArgument: {param} is required", "missing"),
    ("EmptyResult: no data returned for {param}='{value}'", "empty"),
]


def corrupt_arguments(arguments: dict, rng: random.Random):
    """Return (corrupted_args, error_msg, corrected_args)."""
    keys = list(arguments.keys())
    if not keys:
        return arguments, "MissingRequiredArgument: a parameter is required", arguments
    k = rng.choice(keys)
    v = arguments[k]
    kind = rng.choice(ERROR_TEMPLATES)
    if kind[1] == "missing":
        corrupted = {kk: vv for kk, vv in arguments.items() if kk != k}
        return corrupted, kind[0].format(param=k, value=""), arguments
    if kind[1] == "typo":
        bad = str(v) + "x" if v not in ("", None) else "zzz"
        corrupted = {**arguments, k: bad}
        return corrupted, kind[0].format(param=k, value=bad), arguments
    # enum / type / empty: keep value but flag it
    corrupted = {**arguments, k: v}
    return corrupted, kind[0].format(param=k, value=v), arguments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Team-ACE/ToolACE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.src, split="train")
    rng = random.Random(args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w") as f:
        for ex in ds:
            rec = convert_example(ex)
            if not rec:
                continue
            for call in rec["gold"]:
                corrupted, err, corrected = corrupt_arguments(call["arguments"], rng)
                block_bad = to_dtsa(call["name"], corrupted)
                block_good = to_dtsa(call["name"], corrected)
                f.write(
                    json.dumps(
                        {
                            "query": rec["query"],
                            "tools": rec["tools"],
                            "failed": block_bad,
                            "error": err,
                            "corrected": block_good,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                if written >= args.n:
                    break
            if written >= args.n:
                break
    print(f"wrote {written} RTE records -> {args.out}")


if __name__ == "__main__":
    main()
