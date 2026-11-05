#!/bin/bash
# GRPO v3.3: gate x (0.75 reason*(0.5+0.5 precision) + 0.25 exact), level-strict steps reasoning reward (code judge), warm start = S1 (L2 SFT, highest reasoning score)
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
I=outputs/grpo_init/S1
[ -d $I ] || .venv/bin/python scripts/merge_lora.py outputs/sft/sft-qwen3.5-2b-L2-lora/final $I || { echo "EXIT=merge_fail"; exit 1; }
echo "=== v33 $(date +%H:%M)"
.venv/bin/python scripts/train_grpo.py --init $I --level 2 --reward gated --max-steps 100 --tag _v33 > logs/grpo_v33.log 2>&1
rc=$?; echo "### grpo_v33 train rc=$rc $(date +%H:%M)"; grep -a "^DONE" logs/grpo_v33.log | cut -c1-400
if [ $rc = 0 ]; then
  R=outputs/grpo/grpo-S1-L2-gated_v33; M=outputs/merged/G_v33
  .venv/bin/python scripts/merge_lora.py $R/final $M && \
  /opt/vllm-venv/bin/python scripts/eval.py --model $M --mode think --backend vllm --eager --no-prefix-cache \
     --split test_id,test_ood --n 64 2>&1 | grep -a -E '^\[|^\{"run|Traceback|Error|HSA' | cut -c1-700
  echo "### grpo_v33 eval rc=${PIPESTATUS[0]} $(date +%H:%M)"; rm -rf $M
fi
echo "EXIT=grpo_v33 $(date +%H:%M)"
