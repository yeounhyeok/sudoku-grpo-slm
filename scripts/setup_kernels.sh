#!/usr/bin/env bash
# Optimized Qwen3.5 Gated-DeltaNet kernels for the TRAINING venv (.venv). Without them transformers
# falls back to reference PyTorch code (correct but slow).
# Verified 2026-11-04 on eney (MI250X/MI250 gfx90a, ROCm 7.2, torch 2.14.1+rocm7.2):
#   2B, seq 1536 x batch 8, grad ckpt:  LoRA 2,578 -> 4,527 tok/s (x1.76), Full FT 2,709 -> 5,329 tok/s (x1.97)
#   same-batch forward loss 13.89671 / 13.8969 (bf16 noise) -> numerically equivalent
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PIP="$REPO/.venv/bin/pip"; PY="$REPO/.venv/bin/python"
# 1) flash-linear-attention: pure Triton (works with triton-rocm). Pulls only fla-core + einops.
"$PIP" install "flash-linear-attention==0.6.0"
# 2) causal-conv1d: no ROCm wheels upstream (v1.7.0 release assets are CUDA only) -> build with hipcc.
#    torch >= 2.14 headers require C++20, but setup.py hard-codes -std=c++17 -> patch before building.
WORK=${CC1D_WORK:-/var/tmp/cc1d}; mkdir -p "$WORK"; cd "$WORK"
URL=$(curl -s https://pypi.org/pypi/causal-conv1d/1.7.0/json | "$PY" -c "import sys,json;print([u['url'] for u in json.load(sys.stdin)['urls'] if u['packagetype']=='sdist'][0])")
curl -sL "$URL" -o cc.tar.gz && tar xzf cc.tar.gz && cd causal_conv1d-1.7.0
sed -i 's/-std=c++17/-std=c++20/g' setup.py
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=${MAX_JOBS:-12} "$PIP" install --no-build-isolation .
"$PY" -c "import torch, fla; from causal_conv1d import causal_conv1d_fn; print('fla', fla.__version__, 'causal_conv1d ok')"
