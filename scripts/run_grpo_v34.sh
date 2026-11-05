#!/bin/bash
# GRPO v3.4: v3.3 reward + curriculum easy->medium->hard, batch-std advantages, 3072-token budget reasoning reward (code judge), warm start = S1 (L2 SFT, highest reasoning score)
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
I=outputs/grpo_init/S1
[ -d $I ] || .venv/bin/python scripts/merge_lora.py outputs/sft/sft-qwen3.5-2b-L2-lora/final $I || { echo "EXIT=merge_fail"; exit 1; }
echo "=== v34 $(date +%H:%M)"
.venv/bin/python scripts/train_grpo.py --init $I --level 2 --reward gated --max-steps 100 --max-completion 3072 --scale-rewards batch --curriculum easy:40,easy+medium:40,easy+medium+hard:20 --tag _v34 > logs/grpo_v34.log 2>&1
rc=$?; echo "### grpo_v34 train rc=$rc $(date +%H:%M)"; grep -a "^DONE" logs/grpo_v34.log | cut -c1-400
if [ $rc = 0 ]; then
  R=outputs/grpo/grpo-S1-L2-gated_v34; M=outputs/merged/G_v34
  .venv/bin/python scripts/merge_lora.py $R/final $M && \
  /opt/vllm-venv/bin/python scripts/eval.py --model $M --mode think --backend vllm --eager --no-prefix-cache \
     --split test_id,test_ood --n 64 2>&1 | grep -a -E '^\[|^\{"run|Traceback|Error|HSA' | cut -c1-700
  echo "### grpo_v34 eval rc=${PIPESTATUS[0]} $(date +%H:%M)"; rm -rf $M
fi
echo "EXIT=grpo_v34 $(date +%H:%M)"
