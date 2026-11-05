#!/bin/bash
# GRPO v3.1: solution-free (sound-steps-only) reasoning reward (code judge), warm start = S1 (L2 SFT, highest reasoning score)
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
I=outputs/grpo_init/S1
[ -d $I ] || .venv/bin/python scripts/merge_lora.py outputs/sft/sft-qwen3.5-2b-L2-lora/final $I || { echo "EXIT=merge_fail"; exit 1; }
echo "=== v31 $(date +%H:%M)"
.venv/bin/python scripts/train_grpo.py --init $I --level 2 --reward reason --max-steps 100 --tag _v31 > logs/grpo_v31.log 2>&1
rc=$?; echo "### grpo_v31 train rc=$rc $(date +%H:%M)"; grep -a "^DONE" logs/grpo_v31.log | cut -c1-400
if [ $rc = 0 ]; then
  R=outputs/grpo/grpo-S1-L2-reason_v31; M=outputs/merged/G_v3
  .venv/bin/python scripts/merge_lora.py $R/final $M && \
  /opt/vllm-venv/bin/python scripts/eval.py --model $M --mode think --backend vllm --eager --no-prefix-cache \
     --split test_id,test_ood --n 64 2>&1 | grep -a -E '^\[|^\{"run|Traceback|Error|HSA' | cut -c1-700
  echo "### grpo_v31 eval rc=${PIPESTATUS[0]} $(date +%H:%M)"; rm -rf $M
fi
echo "EXIT=grpo_v31 $(date +%H:%M)"
