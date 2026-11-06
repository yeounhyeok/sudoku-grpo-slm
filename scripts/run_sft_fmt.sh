#!/bin/bash
# Quick format SFT from the original 2B + structured-output check on val (not sudoku accuracy).
# Usage: scripts/run_sft_fmt.sh LEVEL N_TRAIN [MAX_LEN=2048]
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
LV=$1; N=$2; ML=${3:-2048}; K=$((N/1000))k
TPY=.venv/bin/python; VPY=/root/venvs/vllm/bin/python
NAME=L${LV}_$K; RUN=outputs/sft/sft-qwen3.5-2b-L$LV-lora_n$K; M=outputs/grpo_init/$NAME
echo "=== train $NAME $(date +%H:%M)"
$TPY scripts/train_sft.py --level $LV --n-train $N --max-len $ML --tag _n$K > logs/train_$NAME.log 2>&1 \
  || { tail -5 logs/train_$NAME.log; echo "### $NAME train_fail"; exit 1; }
grep -a '^DONE' logs/train_$NAME.log | cut -c1-300
grep -a -o 'runs/[a-z0-9]*' logs/train_$NAME.log | head -1
$TPY scripts/merge_lora.py $RUN/final $M || { echo "### $NAME merge_fail"; exit 1; }
echo "=== val check $NAME $(date +%H:%M)"
$VPY scripts/eval.py --model $M --mode think --backend vllm --eager --no-prefix-cache \
  --split val --n 64 --max-new-tokens 4096 --tag _fmt 2>&1 | grep -a -E '^\{"run|Traceback|Error' | cut -c1-400
$TPY scripts/struct_check.py --level $LV outputs/eval/*${NAME}*val_n64*/samples.jsonl
rm -rf $M   # merged weights are reproducible from the adapter (merge_lora.py ~1 min); keep /root free
echo "### $NAME done $(date +%H:%M)"
