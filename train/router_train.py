#!/usr/bin/env python3
"""Router training — DTSA Tool Router cross-encoder.

Trains a small cross-encoder (query x tool_description -> score) so tool
selection is a routing problem, not an LM generation problem. This is what
makes valid_name ~1.0 by construction at 1B.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_triples(data_path: str):
    """Yield (query, positive_tool_desc, negative_tool_desc) triples."""
    for line in open(data_path):
        rec = json.loads(line)
        gold = {g["name"] for g in rec["gold"]}
        descs = {}
        for t in rec["tools"]:
            fn = t.get("function", t)
            descs[fn.get("name")] = fn.get("description", "")
        for pos in gold:
            negs = [n for n in descs if n not in gold]
            if negs:
                yield rec["query"], descs[pos], descs[negs[0]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="DTSA jsonl")
    ap.add_argument("--out", default="train/outputs/router")
    args = ap.parse_args()
    # TODO: train a cross-encoder (e.g. MiniLM) with contrastive/CE loss over
    # build_triples(); export for runtime/router.ToolRouter(model_name=...).
    raise NotImplementedError("wire router training here")


if __name__ == "__main__":
    main()
