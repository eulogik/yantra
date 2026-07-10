#!/usr/bin/env python3
"""Stage 1 — DTSA SFT (QLoRA) on MiniCPM5-1B.

Teaches the model the DTSA format: given a `<bind tool="..."/>` prefix it emits
a schema-shaped `<args>...</args><action_end/>` block. Uses unsloth for fast
QLoRA on a single 24-48GB GPU (falls back to plain peft/transformers).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_messages(rec: dict) -> tuple[str, str]:
    """Return (prompt, completion) for SFT. The bind prefix is injected so the
    LM only learns argument generation, not tool-name recall."""
    tool_json = json.dumps(
        [t.get("function", t) for t in rec["tools"]], ensure_ascii=False
    )
    prompt = (
        f"<user>{rec['query']}</user>\n"
        f"<tools>{tool_json}</tools>\n"
        f"<calls>{rec['completion'].splitlines()[0]}\n"  # the <bind .../> line
    )
    completion = "\n".join(rec["completion"].splitlines()[1:])
    return prompt, completion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data/*.dtsa.jsonl")
    ap.add_argument("--base", default="openbmb/MiniCPM5-1B")
    ap.add_argument("--out", default="train/outputs/sft-dtsa")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--lora-r", type=int, default=64)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data) if l.strip()]

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer

        print("[sft_dtsa] using unsloth + TRL SFTTrainer")
        # TODO: load 4-bit model, apply LoRA (r=args.lora_r), build SFT dataset
        # from build_messages(), train, save_merged(args.out)
        raise NotImplementedError("wire unsloth load/train here")
    except ImportError:
        from transformers import Trainer  # noqa: F401

        print("[sft_dtsa] unsloth not found; use peft/transformers path")
        # TODO: peft LoraConfig + SFT with build_messages()
        raise NotImplementedError("wire peft/transformers training here")


if __name__ == "__main__":
    main()
