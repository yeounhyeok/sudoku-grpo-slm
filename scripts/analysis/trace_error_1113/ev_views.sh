#!/bin/bash
# 11-13 diagnostic ablation: v4s60, test_id medium (same 24 problems as the _ctl run: seed0 n64 -> tier medium),
# prompt = grid + columns + boxes spelled out. Same decoding as _ctl. Waits for a free GPU (shared node).
cd ~/workspace/vi005/sudoku || exit 1
ST=$PWD/ev_views.status
st(){ echo "$(date '+%F %T') $*" >> "$ST"; }
st "start (waiting for free GPU: mem<60000MiB & util<30%)"
while true; do
  if pgrep -f "scripts/eval_k.py" >/dev/null; then sleep 120; continue; fi
  G=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<60000 && $3<30 {print $1; exit}')
  [ -n "$G" ] && break
  sleep 120
done
st "GPU $G free -> run"
export HF_HOME=$PWD/.hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$G
cd repo || exit 1
$PWD/../venv/bin/python scripts/eval_k.py --model ../merged/v4s60 --split test_id --n 64 --tier medium --prompt-variant views \
  --k 8 --max-tokens 32768 --presence-penalty 1.5 --top-k 512 --tag _ctl_views_med --gpu-mem 0.3 > ../evk_views_v4s60.log 2>&1
st "done rc=$?"
grep -a "^RESULT" ../evk_views_v4s60.log | cut -c1-2000 >> "$ST"
st "EXIT"
