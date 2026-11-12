#!/bin/bash
# Sudoku Zero v4 (11/9, 운혁 님 "R1에서 가져갈 거 + DAPO, 하드는 나중에(미디엄 충분히 오른 뒤 보너스)").
#  Observations from v3 (GRPO 학습 계획서 §9-13/9-14):
#   - 37% of GRPO groups had zero advantage (easy all-correct 50%, medium all-wrong 31%) -> wasted signal
#   - test: s10 -> s20 -> s30 rising inside budget, s30 ~ s40 plateau
#  Changes vs v3 (everything else identical: eval-matched decoding T0.6/top-p0.95/top-k512/pp1.5, lr 5e-6, beta 0.005,
#  token-truncated IS, cap 24k, G 8, 8 prompts/step):
#   1. medium-only pool (rl_mpool, 320) + ONLINE DAPO dynamic sampling (--dynamic-sampling): groups whose 8 rollouts
#      are all-correct or all-wrong (incl. 24k-truncated) carry no advantage -> dropped and re-drawn from the pool in
#      the same step, up to DS_ROUNDS generation rounds per step (rest filled with dropped groups). Logged as dynamic/*.
#   (clip-higher dropped: num_iterations=1 -> ratio==1, no-op. token-level loss = TRL default "dapo", already in v3)
# usage: bash scripts/run_grpo_zero_v4.sh <init_dir> <pool_jsonl> [steps] [tag] [ds_rounds] [save_steps]
set -u
cd /root/sudoku-grpo-slm
INIT=$1; DATA=$2; STEPS=${3:-30}; TAG=${4:-_zero_v4}; DSR=${5:-3}; SAVE=${6:-10}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
export WANDB__SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=300
export PYTHONPATH=/root/cc1d_pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/root/venvs/vllm/bin/python
test -f "$INIT/config.json" || { echo "EXIT=zero_v4 no_init"; exit 1; }
N=$(wc -l < "$DATA"); echo "=== v4 start $(date +%H:%M) init=$INIT data=$DATA n=$N steps=$STEPS"
$PY scripts/train_grpo.py --init "$INIT" --data "$DATA" --reward zero --vllm --vllm-mem 0.30 --prompts-per-step 8 --G 8 --micro 1 \
  --lr 5e-6 --beta 0.005 --scale-rewards batch --max-completion 24576 --seed 8 \
  --temperature 0.6 --top-k 512 --presence-penalty 1.5 --is-mode token_truncate \
  --max-steps $STEPS --curriculum medium:$STEPS --save-steps $SAVE --dynamic-sampling --ds-max-rounds $DSR --tag $TAG > logs/grpo$TAG.log 2>&1
rc=$?; echo "### v4 rc=$rc $(date +%H:%M)"
echo "EXIT=zero_v4 $(date +%H:%M)"
