#!/usr/bin/env python3
"""Stage 1 DTSA SFT on Apple Silicon via MLX (mlx-lm).

Prepares DTSA records into mlx_lm training format and launches `mlx_lm.lora`.
This is the M4-native replacement for unsloth/peft (which are CUDA-only).

Usage:
  python3 train/mlx_sft.py --data data/toolace.dtsa.jsonl \
      --base openbmb/MiniCPM5-1B --out train/outputs/sft-dtsa --iters 2000
"""

# Project name constant (kept in sync with configs/config.yaml).
PROJECT = "yantra"
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from convert_toolace_dtsa import to_dtsa  # reuse DTSA serialization


def make_text(rec: dict) -> str:
    tool_json = json.dumps(
        [t.get("function", t) for t in rec["tools"]], ensure_ascii=False
    )
    lines = rec["completion"].splitlines()
    bind_line = lines[0]                      # <bind tool="..."/>
    args_block = "\n".join(lines[1:])
    prompt = (
        f"<user>{rec['query']}</user>\n"
        f"<tools>{tool_json}</tools>\n"
        f"<calls>{bind_line}\n"
    )
    return prompt + args_block + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="DTSA jsonl (data/convert_toolace_dtsa.py)")
    ap.add_argument("--base", default="openbmb/MiniCPM5-1B")
    ap.add_argument("--out", default="train/outputs/sft-dtsa")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lora-layers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    out_dir = Path(args.out) / "mlx_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps({"text": make_text(r)}, ensure_ascii=False) + "\n")
    print(f"[mlx_sft] prepared {len(records)} samples -> {out_dir}/train.jsonl")

    # Optionally convert base HF -> MLX if a local HF dir is given and not MLX yet.
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", args.base,
        "--data", str(out_dir),
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--lora-layers", str(args.lora_layers),
        "--learning-rate", str(args.lr),
        "--save-path", str(Path(args.out) / "adapters"),
    ]
    print("[mlx_sft] launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[mlx_sft] done -> {Path(args.out) / 'adapters'}")


if __name__ == "__main__":
    main()
