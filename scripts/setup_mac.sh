#!/usr/bin/env bash
# setup_mac.sh — install everything needed to train + eval NanoAgent on Apple Silicon (M4 16GB).
set -e
cd "$(dirname "$0")/.."   # repo root
PIP="python3 -m pip"

if [ ! -d .venv ]; then
  echo ">> creating .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
PIP="pip"

echo ">> upgrading pip"
$PIP install -U pip

echo ">> core deps (eval + data)"
$PIP install -U "datasets" "huggingface_hub" "openai" "sentence-transformers"
echo ">> MLX (training student via mlx-lm)"
$PIP install -U "mlx" "mlx-lm"
echo ">> llama-cpp-python (serving GGUF for eval)"
$PIP install -U "llama-cpp-python"
echo "DONE. Activate with: source .venv/bin/activate"
