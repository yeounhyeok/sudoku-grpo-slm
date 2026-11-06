#!/bin/bash
# Sudoku Zero (R1-Zero style): original 2B, no SFT, baseline prompt, outcome reward acc + 0.1*fmt,
# free-form reasoning, 16k generation cap. v3.5 optimisation (lr 5e-6, KL 0.04), vLLM colocate.
# 1) 2-step smoke (memory / step time)  2) 30-step easy diagnosis.
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
PY=/root/venvs/vllm/bin/python
COMMON="--init models/qwen3.5-2b --reward zero --vllm --vllm-mem 0.30 --prompts-per-step 8 --G 8 --micro 1 \
 --lr 5e-6 --beta 0.04 --scale-rewards batch --max-completion 16384"
echo "=== smoke $(date +%H:%M)"
$PY scripts/train_grpo.py $COMMON --max-steps 2 --curriculum easy:2 --save-steps 100 --tag _zero_smoke > logs/grpo_zero_smoke.log 2>&1
rc=$?; echo "### smoke rc=$rc $(date +%H:%M)"; grep -a "^DONE" logs/grpo_zero_smoke.log | cut -c1-300
[ $rc = 0 ] || { grep -a -E "Error|error" logs/grpo_zero_smoke.log | tail -3; echo "EXIT=grpo_zero smoke_fail"; exit 1; }
echo "=== diag $(date +%H:%M)"
$PY scripts/train_grpo.py $COMMON --max-steps 30 --curriculum easy:30 --save-steps 10 --tag _zero > logs/grpo_zero.log 2>&1
rc=$?; echo "### diag rc=$rc $(date +%H:%M)"; grep -a "^DONE" logs/grpo_zero.log | cut -c1-300
echo "EXIT=grpo_zero $(date +%H:%M)"
