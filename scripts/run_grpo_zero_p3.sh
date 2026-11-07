#!/bin/bash
# Sudoku Zero phase 3 (11/7 decision after re-reading R1 v2 §2.2/§3.2): stronger update.
#  phase 2 plateaued (medium ~15-18% from step 11). R1-Zero: lr 3e-6 FULL fine-tune, KL 0.001 (+ref reset every
#  400 steps); ours was LoRA lr 5e-6 + KL 0.04 (40x). -> lr 2e-5, beta 0.005, everything else unchanged.
# Flow: wait phase-2 checkpoint-30 -> stop phase 2 -> merge onto Z30 = grpo_init/P2_30 -> GRPO from P2_30.
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
export WANDB__SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=300
PY=/root/venvs/vllm/bin/python
R2=outputs/grpo/grpo-Z30-L2-zero_zero_p2
until [ -f $R2/checkpoint-30/adapter_model.safetensors ]; do sleep 60; done
sleep 120   # let the save finish (trainer_state etc.)
P=$(pgrep -f "train_grpo.py .*--tag _zero_p2" | head -1); echo "=== stop phase2 pid=$P $(date +%H:%M)"
[ -n "$P" ] && { pkill -TERM -P $P; kill $P; }; for i in $(seq 1 60); do pgrep -f "train_grpo.py .*--tag _zero_p2" >/dev/null || break; sleep 5; done
pgrep -f "train_grpo.py .*--tag _zero_p2" && { echo "EXIT=zero_p3 phase2_still_alive"; exit 1; }
for i in $(seq 1 30); do v=$(rocm-smi --showmemuse 2>/dev/null | grep -o "VRAM%): [0-9]*" | grep -o "[0-9]*$"); [ "${v:-99}" -lt 5 ] && break; sleep 10; done
echo "VRAM% after stop: $v"; [ "${v:-99}" -lt 5 ] || { echo "EXIT=zero_p3 gpu_not_free"; exit 1; }
echo "=== merge P2_30 $(date +%H:%M)"
.venv/bin/python scripts/merge_lora.py $R2/checkpoint-30 outputs/grpo_init/P2_30 --base outputs/grpo_init/Z30 || { echo "EXIT=zero_p3 merge_fail"; exit 1; }
for f in tokenizer.json tokenizer_config.json chat_template.jinja; do [ -f outputs/grpo_init/P2_30/$f ] || cp models/qwen3.5-2b/$f outputs/grpo_init/P2_30/; done
echo "=== phase3 $(date +%H:%M)"
$PY scripts/train_grpo.py --init outputs/grpo_init/P2_30 --reward zero --vllm --vllm-mem 0.30 --prompts-per-step 8 --G 8 --micro 1 \
  --lr 2e-5 --beta 0.005 --scale-rewards batch --max-completion 24576 --seed 2 \
  --max-steps 60 --curriculum easy+medium:60 --save-steps 10 --tag _zero_p3 > logs/grpo_zero_p3.log 2>&1
rc=$?; echo "### phase3 rc=$rc $(date +%H:%M)"; grep -a "^DONE" logs/grpo_zero_p3.log | cut -c1-300
echo "EXIT=zero_p3 $(date +%H:%M)"
