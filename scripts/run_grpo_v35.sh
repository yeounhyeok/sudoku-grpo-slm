#!/bin/bash
# GRPO v3.5 diagnosis: v3.4 reward/curriculum, gentler optimisation (lr 5e-6, KL beta 0.04 to the S1 start),
# vLLM colocate rollouts with 16 puzzles x 8 per step. 1) 2-step smoke  2) 30-step easy-only diagnosis.
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
PY=/root/venvs/vllm/bin/python
I=outputs/grpo_init/S1
COMMON="--init $I --level 2 --reward gated --vllm --prompts-per-step 16 --G 8 --micro 8 --lr 5e-6 --beta 0.04 --scale-rewards batch --max-completion 3072"
echo "=== smoke $(date +%H:%M)"
$PY scripts/train_grpo.py $COMMON --max-steps 2 --curriculum easy:2 --save-steps 100 --tag _v35smoke > logs/grpo_v35smoke.log 2>&1
rc=$?; echo "### smoke rc=$rc $(date +%H:%M)"; grep -a "^DONE" logs/grpo_v35smoke.log | cut -c1-300
[ $rc = 0 ] || { echo "EXIT=grpo_v35 smoke_fail"; exit 1; }
echo "=== diag $(date +%H:%M)"
$PY scripts/train_grpo.py $COMMON --max-steps 30 --curriculum easy:30 --save-steps 10 --tag _v35 > logs/grpo_v35.log 2>&1
rc=$?; echo "### diag rc=$rc $(date +%H:%M)"; grep -a "^DONE" logs/grpo_v35.log | cut -c1-400
echo "EXIT=grpo_v35 $(date +%H:%M)"
