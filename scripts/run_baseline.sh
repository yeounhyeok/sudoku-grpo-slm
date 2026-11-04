#!/usr/bin/env bash
# Official evaluation protocol (same command before and after fine-tuning).
# Usage: scripts/run_baseline.sh [model ...]   (default: qwen3.5-2b qwen3.5-0.8b; a checkpoint dir also works)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${VLLM_PY:-/opt/vllm-venv/bin/python}
MODELS=${*:-"qwen3.5-2b qwen3.5-0.8b"}
for m in $MODELS; do
  echo "=== $m $(date +%H:%M)"
  $PY scripts/eval.py --model "$m" --mode think --backend vllm --eager --no-prefix-cache \
      --split test_id,test_ood --n 64 || echo "FAIL $m"
done
echo "EXIT=done $(date +%H:%M)"
