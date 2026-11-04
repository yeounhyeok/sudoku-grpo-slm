#!/usr/bin/env bash
# Create .venv and install deps.   BACKEND=rocm (default) | cu128 | cpu
#   BACKEND=rocm ROCM_VER=7.2 bash scripts/setup_env.sh
# On the eney pod keep the repo under /root (persistent nbd0 volume), not / (overlay may be wiped on restart).
# python3-venv must exist: `apt-get install -y python3.12-venv` (system-level; lost on pod restart).
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND="${BACKEND:-rocm}"
ROCM_VER="${ROCM_VER:-7.2}"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
case "$BACKEND" in
  rocm) pip install torch --index-url "https://download.pytorch.org/whl/rocm${ROCM_VER}" ;;
  cu128) pip install torch --index-url https://download.pytorch.org/whl/cu128 ;;
  cpu) pip install torch --index-url https://download.pytorch.org/whl/cpu ;;
  *) echo "unknown BACKEND=$BACKEND"; exit 1 ;;
esac
pip install -r requirements.txt
python - <<'PY'
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "cuda", torch.version.cuda)
print("gpu available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
