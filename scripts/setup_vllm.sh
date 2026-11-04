#!/usr/bin/env bash
# vLLM (ROCm) evaluation environment — separate from the training venv because vLLM pins its own torch.
# Verified 2026-11-04 on eney: MI250X/MI250 (gfx90a), Ubuntu 24.04 (glibc 2.39), system ROCm 7.2.0, Python 3.12.
#   -> vllm 0.33.0+rocm72, torch 2.13.0 (from the vLLM ROCm index).
# The venv is re-creatable, so it lives on the (non-persistent) overlay by default.
set -euo pipefail
VENV=${VLLM_VENV:-/opt/vllm-venv}
VLLM_VERSION=${VLLM_VERSION:-0.33.0}
VARIANT=${VLLM_ROCM_VARIANT:-rocm72}          # system ROCm 7.2 stack
export UV_CACHE_DIR=${UV_CACHE_DIR:-/var/cache/uv}

# torch from the vLLM ROCm index links against OpenMPI C++ bindings and elfutils.
if command -v apt-get >/dev/null; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends libopenmpi3t64 libdw1t64
fi
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q uv
"$VENV/bin/uv" pip install --python "$VENV/bin/python" "vllm==${VLLM_VERSION}" \
  --extra-index-url "https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VARIANT}"
"$VENV/bin/python" -c "import torch, vllm; print('torch', torch.__version__, 'hip', torch.version.hip, 'vllm', vllm.__version__, 'gpu', torch.cuda.is_available())"
# NOTE: on MI250 run vLLM with --eager (HIP graphs crash: HSA_STATUS_ERROR_INVALID_PACKET_FORMAT).
