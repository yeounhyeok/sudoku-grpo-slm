#!/bin/bash
# Sudoku Zero v3 (11/8, 운혁 님 승인 "ㄱ"): train under the exact eval decoding.
#  Why (GRPO 학습 계획서 §9-11): v2 (T 1.0 rollouts, lr 2e-5) beat the base only under its training decoding
#  (T 1.0, <=24k) and lost under the base's best decoding (T 0.6, pp 1.5: 15.4% vs 25.2%); KL kept climbing and v2b
#  (1:3) drifted further (easy 88% -> 44%, clipping back up). Low temperature already gives the base what T-1.0 RL
#  learned -> roll out at T 0.6 / top-p 0.95 / pp 1.5 (= eval), lr back to 5e-6 (no damage at that lr in phase 2).
#  top_k 512 ~ no top-k but keeps TRL's IS renormalisation valid with top_p < 1.
# Flow: wait v2b checkpoint-10 (kept for analysis) -> stop v2b -> start v3 from the ORIGINAL 2B.
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
export WANDB__SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=300
export PYTHONPATH=/root/cc1d_pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/root/venvs/vllm/bin/python
RB=outputs/grpo/grpo-V2_10-L2-zero_zero_v2b
until [ -f $RB/checkpoint-10/adapter_model.safetensors ]; do sleep 60; done
sleep 90
P=$(pgrep -f -- "--tag _zero_v2b$" | head -1); echo "=== stop v2b pid=$P $(date +%H:%M)"
[ -n "$P" ] && kill $P
for i in $(seq 1 60); do pgrep -f -- "--tag _zero_v2b$" >/dev/null || break; sleep 5; done
pgrep -f -- "--tag _zero_v2b$" && { echo "EXIT=zero_v3 v2b_still_alive"; exit 1; }
for i in $(seq 1 30); do v=$(rocm-smi --showmemuse 2>/dev/null | grep -o "VRAM%): [0-9]*" | grep -o "[0-9]*$"); [ "${v:-99}" -lt 5 ] && break; sleep 10; done
[ "${v:-99}" -lt 5 ] || { echo "EXIT=zero_v3 gpu_not_free"; exit 1; }
echo "=== v3 start $(date +%H:%M)"
$PY scripts/train_grpo.py --init models/qwen3.5-2b --reward zero --vllm --vllm-mem 0.30 --prompts-per-step 8 --G 8 --micro 1 \
  --lr 5e-6 --beta 0.005 --scale-rewards batch --max-completion 24576 --seed 5 \
  --temperature 0.6 --top-k 512 --presence-penalty 1.5 --is-mode token_truncate \
  --max-steps 60 --curriculum easy+medium+medium+medium:60 --save-steps 10 --tag _zero_v3 > logs/grpo_zero_v3.log 2>&1
rc=$?; echo "### v3 rc=$rc $(date +%H:%M)"
echo "EXIT=zero_v3 $(date +%H:%M)"
