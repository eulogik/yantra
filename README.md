# NanoAgent

A practical sub/same-size agentic model that **beats `ewinregirgojr/MiniCPM5-1B-Agentic-Tooluse-GGUF`** on the weak points that model exhibits in real deployment:

- `exact_args` 65% → **≥85%** (Execution-Grounded On-Policy Distillation)
- `stopped_cleanly` 15% → **≥95%** (Self-Terminating Structured Action — no external parser)
- no error recovery → **≥80%** recovery (Reflection-on-Tool-Error)
- tool-name recall off the LM's critical path (Decoupled Tool Selection / Argument generation)

See `implementation-doc.md` (one level up) for the full method, and `HANDOFF.md` for the live plan + progress.

## Quick start (reproduce baseline first)

```bash
cd nanogent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Build the fixed 300-case ToolACE eval set (deterministic)
python eval/build_eval_set.py --out eval/data/toolace_300.jsonl --n 300 --seed 42

# 2) Serve the baseline model (or NanoAgent once trained), then evaluate
python eval/first_call_eval.py \
  --eval-set eval/data/toolace_300.jsonl \
  --base-url http://localhost:8000/v1 \
  --model ewinregirgojr/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q4_K_M
```

## Layout

```
data/      conversion + corpus builders (ToolACE->DTSA, RTE, holdout split)
train/     SFT/EG-OPD/RTE/boundary/router/distill stages
runtime/   standalone runtime: router + boundary decoder + verifier + lm server
eval/      first-call evaluator, 300-case builder, PAS score
configs/   LoRA / training / router configs
```
