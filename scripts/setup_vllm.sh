#!/usr/bin/env bash
# vLLM (ROCm) evaluation environment — separate from the training venv because vLLM pins its own torch.
# Verified 2026-11-04 on eney: MI250X/MI250 (gfx90a), Ubuntu 24.04 (glibc 2.39), system ROCm 7.2.0, Python 3.12.
#   -> vllm 0.33.0+rocm72, torch 2.13.0 (from the vLLM ROCm index).
# The venv lives on the persistent /root volume (a pod restart wipes the overlay incl. /opt; large overlay
# writes next to a running job preceded a pod loss on 11/5). /opt/vllm-venv is kept as a symlink for old scripts.
# WITH_TRL=1 also installs TRL/FLA/wandb so the same env can run GRPO with vLLM colocate rollouts.
set -euo pipefail
VENV=${VLLM_VENV:-/root/venvs/vllm}
VLLM_VERSION=${VLLM_VERSION:-0.33.0}
VARIANT=${VLLM_ROCM_VARIANT:-rocm72}          # system ROCm 7.2 stack
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}

# torch from the vLLM ROCm index links against OpenMPI C++ bindings and elfutils.
if command -v apt-get >/dev/null; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends python3.12-venv libopenmpi3t64 libdw1t64
fi
mkdir -p "$(dirname "$VENV")"
[ -x "$VENV/bin/pip" ] || { rm -rf "$VENV"; python3 -m venv "$VENV"; }
"$VENV/bin/pip" install -q uv
"$VENV/bin/uv" pip install --python "$VENV/bin/python" "vllm==${VLLM_VERSION}" \
  --extra-index-url "https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VARIANT}"
if [ "${WITH_TRL:-0}" = 1 ]; then
  "$VENV/bin/uv" pip install --python "$VENV/bin/python" trl==1.17.1 wandb flash-linear-attention==0.6.0
fi
rm -rf "$UV_CACHE_DIR"
[ -e /opt/vllm-venv ] || ln -s "$VENV" /opt/vllm-venv
"$VENV/bin/python" -c "import torch, vllm; print('torch', torch.__version__, 'hip', torch.version.hip, 'vllm', vllm.__version__, 'gpu', torch.cuda.is_available())"
# NOTE: on MI250 run vLLM with --eager (HIP graphs crash: HSA_STATUS_ERROR_INVALID_PACKET_FORMAT).
