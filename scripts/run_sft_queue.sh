#!/usr/bin/env bash
# SFT experiment queue (runs sequentially on one GPU; a failed step is logged and the queue continues).
# Every model is evaluated with the fixed protocol (scripts/run_baseline.sh settings).
#   nohup setsid scripts/run_sft_queue.sh > sft_queue.log 2>&1 &
# Progress markers in the log:  "### <name> ..." per finished step, "QUEUE_DONE" at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=${HF_HOME:-/root/.cache/huggingface}
TPY=.venv/bin/python
VPY=${VLLM_PY:-/opt/vllm-venv/bin/python}
ts() { date +%H:%M; }

evaluate() {  # evaluate MODEL_DIR NAME
  $VPY scripts/eval.py --model "$1" --mode think --backend vllm --eager --no-prefix-cache \
      --split test_id,test_ood --n 64 2>&1 | grep -a -E '^\[|^\{"run|Traceback|Error|HSA_STATUS' | cut -c1-700
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] && echo "### $2 eval ok $(ts)" || echo "### $2 FAIL eval rc=$rc $(ts)"
}

eval_lora() {  # eval_lora ADAPTER_DIR NAME
  local m="outputs/merged/$2"
  if $TPY scripts/merge_lora.py "$1" "$m"; then evaluate "$m" "$2"; else echo "### $2 FAIL merge $(ts)"; fi
  rm -rf "$m"   # merged weights are reproducible from the adapter; keep the disk free
}

train() {  # train NAME ARGS...  -> outputs/sft/<run>/final
  local name=$1; shift
  echo "=== train $name $(ts): $*"
  if $TPY scripts/train_sft.py "$@" > "logs/train_$name.log" 2>&1; then
    grep -a '^DONE' "logs/train_$name.log" | cut -c1-600; echo "### $name train ok $(ts)"; return 0
  fi
  tail -5 "logs/train_$name.log"; echo "### $name FAIL train $(ts)"; return 1
}

mkdir -p logs
# 0) wait for the main run S1 (started separately: run_sft_s1.sh -> sft_s1.log)
while ! grep -q "EXIT=" sft_s1.log 2>/dev/null; do sleep 60; done
grep -a '^DONE' sft_s1.log | cut -c1-600
echo "### S1 train $(grep -a -o 'EXIT=[0-9]*' sft_s1.log) $(ts)"

# 1) S1 main: L2, 2B, LoRA r64, all data (17,897)
eval_lora outputs/sft/sft-qwen3.5-2b-L2-lora/final S1_L2_2b_lora

# 2) RQ1 / data quality: trace format L0, L1 (same puzzles, same config)
train S2_L0 --level 0 && eval_lora outputs/sft/sft-qwen3.5-2b-L0-lora/final S2_L0
train S3_L1 --level 1 && eval_lora outputs/sft/sft-qwen3.5-2b-L1-lora/final S3_L1

# 3) data size: nested prefixes of the (tier-shuffled) train set; 17,897 = S1
for n in 1000 5000 10000; do
  k=$((n / 1000))k
  train D_$k --level 2 --n-train $n --eval-steps 25 --tag _n$k && eval_lora outputs/sft/sft-qwen3.5-2b-L2-lora_n$k/final D_$k
done

# 4) base model size: 0.8B
train S4_0.8b --level 2 --model qwen3.5-0.8b && eval_lora outputs/sft/sft-qwen3.5-0.8b-L2-lora/final S4_0.8b

# 5) LoRA rank at 5K rows (r=64 @5K = D_5k)
for r in 4 16; do
  train R_r$r --level 2 --n-train 5000 --eval-steps 25 --lora-r $r --tag _n5k_r$r \
    && eval_lora outputs/sft/sft-qwen3.5-2b-L2-lora_n5k_r$r/final R_r$r
done

# 6) fine-tuning method: full FT (LR 1e-5)
if train S5_full --level 2 --mode full; then
  F=outputs/sft/sft-qwen3.5-2b-L2-full/final
  for p in models/qwen3.5-2b/*; do  # processor/config files the trainer does not write
    case "$p" in *.safetensors|*safetensors.index.json) ;; *) [ -e "$F/$(basename "$p")" ] || cp "$p" "$F/";; esac
  done
  evaluate "$F" S5_full
fi

# 7) S1 learning curve: archived checkpoints (archive_ckpts.sh)
for s in 100 200 300 400 500; do
  c=outputs/sft/sft-qwen3.5-2b-L2-lora_ckpts/checkpoint-$s
  [ -d "$c" ] && eval_lora "$c" S1_step$s
done
echo "QUEUE_DONE $(ts)"
