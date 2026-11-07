#!/bin/bash
# Sudoku Zero v2 (11/7 decision): restart GRPO from the ORIGINAL 2B with rollouts matched to the eval preset.
#  Why: R1-style eval (k=8, T 0.6) showed presence penalty 1.5 lifts BOTH models (base 7.4 -> 25.2%, P2-step20
#  12.9 -> 26.2%), and at pp 1.5 the pp-0-trained model ties the base. The pp-0 RL gain was mostly "fewer
#  repetition loops", which the penalty already gives for free -> train under the same penalty.
#  Also: TRL's default sequence-level IS (sequence_mask) averaged ratio ~0.23 on 20k-token completions in phase 2
#  (per-token logp gap 0.006 x 20k tokens) -> most samples carried ~0 weight. Use token-level truncated IS.
#  Recipe: pp 1.5 rollouts, token_truncate IS, easy+medium from step 1, 24k cap, lr 2e-5, beta 0.005.
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
export WANDB__SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=300
# causal-conv1d HIP build (scripts/setup_kernels.sh; separate target dir, vLLM venv untouched): conv op 64 -> 3 ms per 20k-token seq
export PYTHONPATH=/root/cc1d_pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/root/venvs/vllm/bin/python
for tag in _zero_p3; do
  for P in $(pgrep -f "run_grpo_zero_p3.sh"); do kill $P; done
  P=$(pgrep -f "train_grpo.py .*--tag $tag" | head -1); [ -n "$P" ] && { pkill -TERM -P $P; kill $P; }
done
for i in $(seq 1 60); do pgrep -f "train_grpo.py" >/dev/null || break; sleep 5; done
pgrep -f "train_grpo.py" && { echo "EXIT=zero_v2 old_run_alive"; exit 1; }
for i in $(seq 1 30); do v=$(rocm-smi --showmemuse 2>/dev/null | grep -o "VRAM%): [0-9]*" | grep -o "[0-9]*$"); [ "${v:-99}" -lt 5 ] && break; sleep 10; done
echo "=== stopped old runs, VRAM% $v $(date +%H:%M)"; [ "${v:-99}" -lt 5 ] || { echo "EXIT=zero_v2 gpu_not_free"; exit 1; }
echo "=== v2 start $(date +%H:%M)"
$PY scripts/train_grpo.py --init models/qwen3.5-2b --reward zero --vllm --vllm-mem 0.30 --prompts-per-step 8 --G 8 --micro 1 \
  --lr 2e-5 --beta 0.005 --scale-rewards batch --max-completion 24576 --seed 3 \
  --presence-penalty 1.5 --is-mode token_truncate \
  --max-steps 60 --curriculum easy+medium:60 --save-steps 10 --tag _zero_v2 > logs/grpo_zero_v2.log 2>&1
rc=$?; echo "### v2 rc=$rc $(date +%H:%M)"; grep -a "^DONE" logs/grpo_zero_v2.log | cut -c1-300
echo "EXIT=zero_v2 $(date +%H:%M)"
