#!/usr/bin/env bash
# serve_and_eval.sh — serve a GGUF (baseline or NanoAgent) and run the 300-case evaluator.
# Auto-detects server backend: llama-server (llama.cpp CLI) else python -m llama_cpp.server.
set -e
MODEL="${1:?usage: serve_and_eval.sh <model.gguf|hf_id> [port] [eval-set] [mode]}"
PORT="${2:-8000}"
EVAL_SET="${3:-eval/data/toolace_300.jsonl}"
MODE="${4:-legacy}"

echo ">> starting server for: $MODEL on :$PORT"
if command -v llama-server >/dev/null 2>&1; then
  SERVER=(llama-server -m "$MODEL" --port "$PORT")
else
  SERVER=(python3 -m llama_cpp.server --model "$MODEL" --port "$PORT")
fi
"${SERVER[@]}" > /tmp/nanogent_server.log 2>&1 &
SRV_PID=$!
echo "   server pid=$SRV_PID (log: /tmp/nanogent_server.log)"

# wait for health
for i in $(seq 1 60); do
  if curl -sS -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 2
done

echo ">> running evaluator (mode=$MODE)"
python3 eval/first_call_eval.py --eval-set "$EVAL_SET" --base-url "http://localhost:$PORT/v1" --model "$MODEL" --mode "$MODE"
python3 eval/pas.py --results eval/results/latest.json

echo ">> stopping server (pid=$SRV_PID)"
kill "$SRV_PID" 2>/dev/null || true
