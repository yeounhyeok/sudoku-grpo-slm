#!/bin/bash
# Sudoku Zero v2b (11/8, delegated by 운혁 님 "10스텝에 알아서 판단해서 easy 빼던가 커리큘럼 진행"):
#  At v2 checkpoint-10 decide from the training batches of steps 6-10:
#   easy accuracy >= 0.85  -> easy is (nearly) solved; all-correct easy groups give zero GRPO advantage and waste half
#                             the batch -> continue from checkpoint-10 with easy:medium = 1:3 (keep some easy against
#                             forgetting), 50 more steps.
#   otherwise              -> keep v2 running unchanged.
cd /root/sudoku-grpo-slm
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface
export WANDB__SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=300
export PYTHONPATH=/root/cc1d_pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/root/venvs/vllm/bin/python
R=outputs/grpo/grpo-qwen3.5-2b-L2-zero_zero_v2
until [ -f $R/checkpoint-10/adapter_model.safetensors ] && [ -f $R/completions/completions_00010.parquet ]; do sleep 60; done
sleep 90
DEC=$($PY - <<'P'
import pandas as pd, glob, json, re
t = {json.loads(l)["puzzle"][:30]: json.loads(l)["tier"] for l in open("data/processed/rl_train.jsonl")}
def tier(p):
    rows = re.findall(r"(?m)^([1-9.](?: [1-9.]){8})$", p)
    return t.get("".join(r.replace(" ", "") for r in rows)[:30], "?")
fs = sorted(glob.glob("outputs/grpo/grpo-qwen3.5-2b-L2-zero_zero_v2/completions/*.parquet"))[5:10]
d = pd.concat([pd.read_parquet(f) for f in fs]); d["tier"] = d.prompt.map(tier)
e = d[d.tier == "easy"]; m = d[d.tier == "medium"]
allc = float((e.groupby("prompt").zero_accuracy.min() > 0).mean())
ea, ma = float(e.zero_accuracy.mean()), float(m.zero_accuracy.mean())
print(f"{'SWITCH' if ea >= 0.85 else 'KEEP'} easy={ea:.3f} medium={ma:.3f} easy_all_correct_groups={allc:.2f}")
P
)
echo "=== decision $(date +%H:%M): $DEC"
case "$DEC" in SWITCH*) ;; *) echo "EXIT=zero_v2b keep_v2"; exit 0;; esac
P=$(pgrep -f -- "--tag _zero_v2$" | head -1); echo "=== stop v2 pid=$P"
[ -n "$P" ] && kill $P
for i in $(seq 1 60); do pgrep -f -- "--tag _zero_v2$" >/dev/null || break; sleep 5; done
pgrep -f -- "--tag _zero_v2$" && { echo "EXIT=zero_v2b v2_still_alive"; exit 1; }
for i in $(seq 1 30); do v=$(rocm-smi --showmemuse 2>/dev/null | grep -o "VRAM%): [0-9]*" | grep -o "[0-9]*$"); [ "${v:-99}" -lt 5 ] && break; sleep 10; done
[ "${v:-99}" -lt 5 ] || { echo "EXIT=zero_v2b gpu_not_free"; exit 1; }
echo "=== merge V2_10 $(date +%H:%M)"
.venv/bin/python scripts/merge_lora.py $R/checkpoint-10 outputs/grpo_init/V2_10 --base models/qwen3.5-2b || { echo "EXIT=zero_v2b merge_fail"; exit 1; }
for f in tokenizer.json tokenizer_config.json chat_template.jinja; do [ -f outputs/grpo_init/V2_10/$f ] || cp models/qwen3.5-2b/$f outputs/grpo_init/V2_10/; done
echo "=== v2b start $(date +%H:%M)"
$PY scripts/train_grpo.py --init outputs/grpo_init/V2_10 --reward zero --vllm --vllm-mem 0.30 --prompts-per-step 8 --G 8 --micro 1 \
  --lr 2e-5 --beta 0.005 --scale-rewards batch --max-completion 24576 --seed 4 \
  --presence-penalty 1.5 --is-mode token_truncate \
  --max-steps 50 --curriculum easy+medium+medium+medium:50 --save-steps 10 --tag _zero_v2b > logs/grpo_zero_v2b.log 2>&1
rc=$?; echo "### v2b rc=$rc $(date +%H:%M)"
echo "EXIT=zero_v2b $(date +%H:%M)"
