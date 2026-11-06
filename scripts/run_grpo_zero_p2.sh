#!/bin/bash
# Sudoku Zero phase 2 (easy+medium). Chained after phase 1 (run_grpo_zero.sh):
#  wait phase-1 EXIT -> merge final into outputs/grpo_init/Z30 -> GRPO easy+medium from Z30.
# Generation cap is read from phase2_cap (written after the medium signal probe), default 16384.
# Official eval of phase-1 final runs on vi005 instead of here (GPU stays on training).
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
export WANDB__SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=300
PY=/root/venvs/vllm/bin/python
until grep -a -q "EXIT=grpo_zero" grpo_zero.log; do sleep 60; done
R1=outputs/grpo/grpo-qwen3.5-2b-L2-zero_zero
[ -d $R1/final ] || { echo "EXIT=zero_p2 no_phase1_final"; exit 1; }
echo "=== merge phase1 final $(date +%H:%M)"
.venv/bin/python scripts/merge_lora.py $R1/final outputs/grpo_init/Z30 || { echo "EXIT=zero_p2 merge_fail"; exit 1; }
for f in tokenizer.json tokenizer_config.json chat_template.jinja; do [ -f outputs/grpo_init/Z30/$f ] || cp models/qwen3.5-2b/$f outputs/grpo_init/Z30/ 2>/dev/null; done
CAP=$(cat phase2_cap 2>/dev/null || echo 16384)
echo "=== phase2 cap=$CAP $(date +%H:%M)"
$PY scripts/train_grpo.py --init outputs/grpo_init/Z30 --reward zero --vllm --vllm-mem 0.30 --prompts-per-step 8 --G 8 --micro 1 \
  --lr 5e-6 --beta 0.04 --scale-rewards batch --max-completion $CAP --seed 1 \
  --max-steps 60 --curriculum easy+medium:60 --save-steps 10 --tag _zero_p2 > logs/grpo_zero_p2.log 2>&1
rc=$?; echo "### phase2 rc=$rc $(date +%H:%M)"; grep -a "^DONE" logs/grpo_zero_p2.log | cut -c1-300
echo "EXIT=zero_p2 $(date +%H:%M)"
