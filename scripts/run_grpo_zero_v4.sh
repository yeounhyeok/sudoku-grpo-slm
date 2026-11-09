#!/bin/bash
# Sudoku Zero v4 (11/9, 운혁 님 "R1에서 가져갈 거 + DAPO, 하드는 나중에(미디엄 충분히 오른 뒤 보너스)").
#  Observations from v3 (GRPO 학습 계획서 §9-13/9-14):
#   - 37% of GRPO groups had zero advantage (easy all-correct 50%, medium all-wrong 31%) -> wasted signal
#   - test: s10 -> s20 -> s30 rising inside budget, s30 ~ s40 plateau
#  Changes vs v3 (everything else identical: eval-matched decoding T0.6/top-p0.95/top-k512/pp1.5, lr 5e-6, beta 0.005,
#  token-truncated IS, cap 24k, G 8, 8 prompts/step):
#   1. data: medium only, pre-filtered with the starting checkpoint (4 samples @24k, same decoding) to puzzles with
#      0 < pass < 1 = offline stand-in for DAPO dynamic sampling (TRL 1.17 has no online resampling)
#   (clip-higher dropped 11/9: num_iterations=1 -> ratio==1, clip never fires, so it would be a no-op; DAPO element used = dynamic sampling only)
#   (token-level loss = TRL default loss_type "dapo", already used by v3)
# usage: bash scripts/run_grpo_zero_v4.sh <init_dir> <filtered_jsonl> [steps]
set -u
cd /root/sudoku-grpo-slm
INIT=$1; DATA=$2; STEPS=${3:-30}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
export WANDB__SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=300
export PYTHONPATH=/root/cc1d_pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/root/venvs/vllm/bin/python
test -f "$INIT/config.json" || { echo "EXIT=zero_v4 no_init"; exit 1; }
N=$(wc -l < "$DATA"); echo "=== v4 start $(date +%H:%M) init=$INIT data=$DATA n=$N steps=$STEPS"
$PY scripts/train_grpo.py --init "$INIT" --data "$DATA" --reward zero --vllm --vllm-mem 0.30 --prompts-per-step 8 --G 8 --micro 1 \
  --lr 5e-6 --beta 0.005 --scale-rewards batch --max-completion 24576 --seed 6 \
  --temperature 0.6 --top-k 512 --presence-penalty 1.5 --is-mode token_truncate \
  --max-steps $STEPS --curriculum medium:$STEPS --save-steps 10 --tag _zero_v4 > logs/grpo_zero_v4.log 2>&1
rc=$?; echo "### v4 rc=$rc $(date +%H:%M)"
echo "EXIT=zero_v4 $(date +%H:%M)"
