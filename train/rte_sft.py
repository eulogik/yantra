#!/usr/bin/env python3
"""Stage 3 — RTE SFT: Reflection-on-Tool-Error training.

Fine-tunes the Stage-2 model on synthetic error-recovery traces so the small
model learns to consume <tool_error> and emit a corrected <bind>/<args> after a
<reflect> token. Reuses the Stage-1 SFT harness with RTE-formatted data.
"""
from __future__ import annotations

import argparse
import json

from sft_dtsa import build_messages  # reuse prompt builder


def build_rte_messages(rec: dict) -> tuple[str, str]:
    tool_json = json.dumps(
        [t.get("function", t) for t in rec["tools"]], ensure_ascii=False
    )
    prompt = (
        f"<user>{rec['query']}</user>\n"
        f"<tools>{tool_json}</tools>\n"
        f"<calls>{rec['failed']}\n"
        f"<tool_error>{rec['error']}</tool_error>\n"
        f"<reflect/>\n"
    )
    completion = rec["corrected"]
    return prompt, completion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data/rte.jsonl")
    ap.add_argument("--base", default="train/outputs/sft-dtsa")  # Stage-2 merged
    ap.add_argument("--out", default="train/outputs/rte")
    args = ap.parse_args()
    # TODO: load Stage-2 model, run SFT over build_rte_messages() samples.
    raise NotImplementedError("wire RTE SFT loop here")


if __name__ == "__main__":
    main()
