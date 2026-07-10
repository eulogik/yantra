#!/usr/bin/env python3
"""Hold out entire tool FAMILIES for the schema-generalization benchmark.

ToolACE tool names are typically namespaced like `category.func`. We split by
the top-level category so the eval measures zero-shot generalization to tool
families the model never saw in SFT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict


def family_of(tool_name: str) -> str:
    return tool_name.split(".")[0] if "." in tool_name else tool_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True, help="toolace_300.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.eval_set) if l.strip()]
    by_fam = defaultdict(list)
    for c in cases:
        fams = {family_of(g["name"]) for g in c["gold"]}
        key = sorted(fams)[0]
        by_fam[key].append(c)

    # 20% of families -> held-out (generalization) split
    fams = sorted(by_fam.keys())
    n_hold = max(1, int(len(fams) * 0.2))
    hold_fams = set(fams[:n_hold])
    seen, held = [], []
    for c in cases:
        fams = {family_of(g["name"]) for g in c["gold"]}
        if fams & hold_fams:
            held.append(c)
        else:
            seen.append(c)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for c in seen:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(str(args.out).replace(".jsonl", ".heldout.jsonl"), "w") as f:
        for c in held:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"seen={len(seen)} heldout={len(held)} families={sorted(hold_fams)}")


if __name__ == "__main__":
    main()
