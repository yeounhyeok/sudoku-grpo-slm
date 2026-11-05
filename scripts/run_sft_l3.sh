#!/bin/bash
# L3 (evidence first, digit last) SFT from the original 2B, small N, then a quick structured-output check on val.
# Usage: scripts/run_sft_l3.sh [N_TRAIN=1000]
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
N=${1:-1000}; K=$((N/1000))k
TPY=.venv/bin/python; VPY=/root/venvs/vllm/bin/python
NAME=L3_$K; RUN=outputs/sft/sft-qwen3.5-2b-L3-lora_n$K; M=outputs/grpo_init/$NAME
echo "=== train $NAME $(date +%H:%M)"
$TPY scripts/train_sft.py --level 3 --n-train $N --tag _n$K > logs/train_$NAME.log 2>&1 \
  || { tail -5 logs/train_$NAME.log; echo "EXIT=sft_l3 train_fail"; exit 1; }
grep -a '^DONE' logs/train_$NAME.log | cut -c1-400
echo "=== merge $(date +%H:%M)"
$TPY scripts/merge_lora.py $RUN/final $M || { echo "EXIT=sft_l3 merge_fail"; exit 1; }
echo "=== val check $(date +%H:%M)"
$VPY scripts/eval.py --model $M --mode think --backend vllm --eager --no-prefix-cache \
  --split val --n 64 --max-new-tokens 4096 --tag _fmt 2>&1 | grep -a -E '^\{"run|Traceback|Error' | cut -c1-500
$TPY scripts/struct_check.py --level 3 outputs/eval/*${NAME}*val_n64*/samples.jsonl
echo "EXIT=sft_l3 $(date +%H:%M)"
