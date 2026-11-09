#!/usr/bin/env python3
"""GRPO on top of an SFT warm-start (TRL GRPOTrainer, LoRA, rule-based rewards).

  python scripts/train_grpo.py --init outputs/grpo_init/D_1k --level 2 --max-steps 10 --tag _smoke
  python scripts/train_grpo.py --init outputs/grpo_init/D_1k --level 2 --reward process      # RQ2 treatment
  python scripts/train_grpo.py --init outputs/grpo_init/D_1k --level 2 --reward outcome      # RQ2 control

Prompts: data/processed/rl_train.jsonl rendered with the model's chat template exactly as in
evaluation (`enable_thinking=True`, generation prompt ends with "<think>\\n").
Rewards (src/sudoku_grpo/reward.py):
  gated    format GATE x (0.75 reason + 0.25 exact)   (v3.2 default; additive format was reward-hacked)
  reason   format 0.2 + reason 0.6 + exact 0.2   (G-RaR split; the LLM judge is replaced by a
           solution-free code judge of each deduction's logical validity)
  process  format 0.2 + step 0.5 + evidence 0.1 + exact 0.2   (v2, step progress checked against the key)
  outcome  format 0.2 + exact 0.8              (answer-only control)
Every run logs to W&B (team project sudoku-grpo-slm, run name = run id).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# reward order: format, step (answer-key progress), evidence, reason (solution-free code judge), exact, gated
# zero-weight terms are still computed and logged to W&B (rewards/<name>/mean).
WEIGHTS = {"gated": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],     # v3.2 main: fmt * (0.75 reason + 0.25 exact)
           "reason": [0.2, 0.0, 0.0, 0.6, 0.2, 0.0],    # v3.1 additive (format-hacked, see notes)
           "process": [0.2, 0.5, 0.1, 0.0, 0.2, 0.0],   # v2: answer-key step progress
           "outcome": [0.2, 0.0, 0.0, 0.0, 0.8, 0.0],   # answer only (RQ2 control)
           "zero": [0.1, 1.0]}                          # Sudoku Zero: [zero_format, zero_accuracy], free-form reasoning


def _pad_cat(pieces, pad_val):
    """right-pad dim-1 of [B_i, C_i] tensors to a common width and cat along dim 0"""
    import torch
    w = max(t.shape[1] for t in pieces)
    out = []
    for t in pieces:
        if t.shape[1] < w:
            t = torch.cat([t, t.new_full((t.shape[0], w - t.shape[1]), pad_val)], dim=1)
        out.append(t)
    return torch.cat(out, dim=0)


def merge_groups(rounds, selected, G, pad_id):
    """rounds: list of (out_dict, prompt_width, completion_width); selected: [(round_idx, group_idx)] in output order.
    Returns one TRL output dict with the selected groups (prompts left-padded, completions right-padded)."""
    import torch
    keys = [k for k in rounds[0][0] if k != "num_items_in_batch"]
    res = {}
    for k in keys:
        pieces = []
        for ri, gi in selected:
            out, pw, cw = rounds[ri]
            t = out[k]
            if t is None:
                continue
            pieces.append((t[gi * G:(gi + 1) * G], k, pw, cw))
        if not pieces:
            res[k] = None
            continue
        if pieces[0][0].dim() == 1:
            res[k] = torch.cat([x[0] for x in pieces], dim=0)
        elif k.startswith("prompt_"):        # left padded
            w = max(x[0].shape[1] for x in pieces)
            fill = pad_id if k == "prompt_ids" else 0
            res[k] = torch.cat([torch.cat([x[0].new_full((x[0].shape[0], w - x[0].shape[1]), fill), x[0]], dim=1)
                                for x in pieces], dim=0)
        elif pieces[0][0].shape[1] == pieces[0][3] or k.endswith("_ids") or k.endswith("_mask") or "logps" in k:
            fill = pad_id if k == "completion_ids" else (1.0 if k == "importance_sampling_ratio" else 0)
            res[k] = _pad_cat([x[0] for x in pieces], fill)
        else:                                 # e.g. sequence-level ratio [B, 1]
            res[k] = torch.cat([x[0] for x in pieces], dim=0)
    res["num_items_in_batch"] = res["completion_mask"].sum()
    return res


def make_dynamic_trainer(base_cls):
    class DynamicGRPOTrainer(base_cls):
        """GRPOTrainer + DAPO dynamic sampling. Informative = accuracy not identical within the group of G rollouts
        (all-correct and all-wrong, incl. 24k-truncated, groups carry no advantage and are replaced by fresh prompts)."""

        def __init__(self, *a, ds_rows=None, ds_max_rounds=3, ds_seed=0, **k):
            super().__init__(*a, **k)
            import random
            self._ds_rows, self._ds_max_rounds, self._ds_rng = ds_rows, ds_max_rounds, random.Random(ds_seed)

        def _generate_and_score_completions(self, inputs):
            if not self.model.training or self._ds_rows is None:
                return super()._generate_and_score_completions(inputs)
            G = self.num_generations
            P = len(inputs) // G
            rounds, kept, dropped, cur = [], [], [], inputs
            while True:
                out = super()._generate_and_score_completions(cur)
                n = len(cur)
                acc = [float(x) for x in list(self._logs["rewards"]["zero_accuracy"])[-n:]]
                ri = len(rounds)
                rounds.append((out, out["prompt_ids"].shape[1], out["completion_ids"].shape[1]))
                for g in range(n // G):
                    a = acc[g * G:(g + 1) * G]
                    (kept if max(a) - min(a) > 1e-6 else dropped).append((ri, g))
                if len(kept) >= P or len(rounds) >= self._ds_max_rounds:
                    break
                need = P - len(kept)
                cur = [dict(r) for r in self._ds_rng.sample(self._ds_rows, need) for _ in range(G)]
            sel = kept[:P]
            filled = 0
            if len(sel) < P:                  # budget exhausted: pad with the most recent dropped (zero-advantage) groups
                extra = dropped[::-1][:P - len(sel)]
                filled = len(extra)
                sel = sel + extra
            m = self._metrics["train"]
            m["dynamic/rounds"].append(float(len(rounds)))
            m["dynamic/kept_groups"].append(float(min(len(kept), P)))
            m["dynamic/dropped_groups"].append(float(len(dropped)))
            m["dynamic/filled_with_dropped"].append(float(filled))
            print(f"[dyn-sampling] rounds={len(rounds)} kept={len(kept)} dropped={len(dropped)} filled={filled}", flush=True)
            return merge_groups(rounds, sel, G, self._tokenizer.pad_token_id)

    return DynamicGRPOTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="merged SFT checkpoint dir (warm start)")
    ap.add_argument("--level", type=int, choices=[0, 1, 2, 3, 4], default=2, help="trace level checked by the step reward")
    ap.add_argument("--reward", choices=list(WEIGHTS), default="gated")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--G", type=int, default=8, help="completions per prompt (group size)")
    ap.add_argument("--micro", type=int, default=8, help="completions per forward/backward micro batch")
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coef (TRL default 0)")
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--curriculum", default="",
                    help='staged tiers, e.g. "easy:40,easy+medium:40,easy+medium+hard:20" (steps per stage); '
                         'prompts are laid out stage by stage and the trainer does not reshuffle them')
    ap.add_argument("--scale-rewards", default="group", choices=["group", "batch", "none"],
                    help="advantage scaling: group std (GRPO), batch std (less noise amplification), none (Dr. GRPO)")
    ap.add_argument("--vllm", action="store_true",
                    help="vLLM colocate rollouts (run with the vLLM env: /root/venvs/vllm/bin/python); eager on MI250")
    ap.add_argument("--vllm-mem", type=float, default=0.35, help="vLLM gpu_memory_utilization in colocate mode")
    ap.add_argument("--presence-penalty", type=float, default=0.0,
                    help="rollout presence penalty (match eval preset: 1.5); vLLM processed_logprobs keep IS consistent")
    ap.add_argument("--temperature", type=float, default=1.0, help="rollout temperature (eval preset: 0.6)")
    ap.add_argument("--top-k", type=int, default=20,
                    help="rollout top-k; with top_p<1 TRL's IS needs top_k>0 (use a large value, e.g. 512, for ~no top-k)")
    ap.add_argument("--data", default="data/processed/rl_train.jsonl",
                    help="RL prompt pool (jsonl with puzzle/solution/tier); v4 uses a difficulty-filtered medium pool")
    ap.add_argument("--epsilon-high", type=float, default=None,
                    help="DAPO clip-higher upper bound (e.g. 0.28); None = symmetric epsilon")
    ap.add_argument("--is-mode", default="sequence_mask",
                    choices=["sequence_mask", "sequence_truncate", "token_mask", "token_truncate"],
                    help="TRL vLLM importance-sampling mode; sequence-level ratios explode on 20k-token completions")
    ap.add_argument("--dynamic-sampling", action="store_true",
                    help="DAPO dynamic sampling (online): drop groups whose accuracy is all-equal (no learning signal) and "
                         "resample new prompts in the same step until prompts-per-step informative groups are collected")
    ap.add_argument("--ds-max-rounds", type=int, default=3,
                    help="max generation rounds per step (1 = no resampling); remaining slots are filled with dropped groups")
    ap.add_argument("--report-to", default=os.environ.get("SFT_REPORT_TO", "wandb"))
    args = ap.parse_args()

    if args.vllm:
        # single-process colocate: vLLM's external_launcher backend reads the torch.distributed env
        for k, v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1",
                     "MASTER_PORT": "29533"}.items():
            os.environ.setdefault(k, v)
        import trl.generation.vllm_generation as _vg
        _LLM = _vg.LLM

        def _eager_llm(*a, **k):          # HIP graphs crash on gfx90a (HSA_STATUS_ERROR_INVALID_PACKET_FORMAT)
            k.setdefault("enforce_eager", True)
            return _LLM(*a, **k)
        _vg.LLM = _eager_llm

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from sudoku_grpo import prompt as P
    from sudoku_grpo.reward import make_reward_funcs, make_zero_reward_funcs
    from train_sft import LM_TARGETS

    os.environ.setdefault("WANDB_PROJECT", "sudoku-grpo-slm")
    os.environ.setdefault("WANDB_ENTITY", "2091126-hansung-university")
    import subprocess
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=5).stdout.strip() or "nogit"
    except Exception:
        commit = "nogit"
    os.environ.setdefault("WANDB_TAGS", f"grpo,reward-{args.reward},L{args.level},init-{Path(args.init).name}")
    os.environ.setdefault("WANDB_NOTES", f"reward={args.reward} weights={WEIGHTS[args.reward]} init={args.init} "
                                         f"curriculum={args.curriculum or 'none'} scale_rewards={args.scale_rewards} "
                                         f"max_completion={args.max_completion} "
                                         f"code={commit}. Table column judge_view: per-line code-judge verdict.")
    run = f"grpo-{Path(args.init).name}-L{args.level}-{args.reward}{args.tag}"
    out = ROOT / "outputs/grpo" / run
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.init)

    rows = []
    for line in (ROOT / args.data).open():
        r = json.loads(line)
        p = tok.apply_chat_template(P.messages_think(r["puzzle"]), tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
        assert p.endswith("<think>\n"), p[-30:]
        rows.append({"prompt": p, "puzzle": r["puzzle"], "solution": r["solution"], "tier": r.get("tier", "")})
    pool_rows = list(rows)   # full pool: dynamic sampling redraws from here
    if args.curriculum:
        import random
        rng = random.Random(args.seed)
        pools = {}
        for r in rows:
            pools.setdefault(r["tier"], []).append(r)
        for v in pools.values():
            rng.shuffle(v)
        cursor = {k: 0 for k in pools}
        ordered = []
        for stage in args.curriculum.split(","):
            tiers, n_steps = stage.split(":")
            tiers = tiers.split("+")
            for i in range(int(n_steps) * args.prompts_per_step):   # round-robin over the stage's tiers
                t = tiers[i % len(tiers)]
                ordered.append(pools[t][cursor[t] % len(pools[t])])
                cursor[t] += 1
        rows = ordered
        ds = Dataset.from_list(rows)
        print("curriculum", args.curriculum, "prompts", len(rows), "used per tier", cursor, flush=True)
    else:
        ds = Dataset.from_list(rows).shuffle(seed=args.seed)
    print(f"run={run} prompts={len(ds)} init={args.init}", flush=True)

    accum = args.prompts_per_step * args.G // args.micro
    cfg = GRPOConfig(
        output_dir=str(out), run_name=run, seed=args.seed, bf16=True,
        learning_rate=args.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=5,
        max_steps=args.max_steps, per_device_train_batch_size=args.micro, gradient_accumulation_steps=accum,
        num_generations=args.G, max_completion_length=args.max_completion,
        temperature=args.temperature, top_p=0.95, top_k=args.top_k,   # eval preset: T 0.6, top-p 0.95 (+ pp 1.5)
        **({"generation_kwargs": {"presence_penalty": args.presence_penalty}} if args.presence_penalty else {}),
        vllm_importance_sampling_mode=args.is_mode,
        beta=args.beta, mask_truncated_completions=False, epsilon_high=args.epsilon_high,   # truncation is penalised by the reward, not hidden
        reward_weights=WEIGHTS[args.reward],
        scale_rewards=args.scale_rewards, shuffle_dataset=not args.curriculum,
        **({"use_vllm": True, "vllm_mode": "colocate", "vllm_gpu_memory_utilization": args.vllm_mem,
            "vllm_max_model_length": args.max_completion + 1024} if args.vllm else {}),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, log_completions=True, num_completions_to_print=1,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=4, save_only_model=True,
        report_to=args.report_to, model_init_kwargs={"dtype": torch.bfloat16},
    )
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                             target_modules=LM_TARGETS, task_type="CAUSAL_LM")
    extra = {}
    TrainerCls = GRPOTrainer
    if args.dynamic_sampling:
        assert args.reward == "zero", "dynamic sampling reads the zero_accuracy reward"
        TrainerCls = make_dynamic_trainer(GRPOTrainer)
        extra = dict(ds_rows=[dict(r) for r in pool_rows], ds_max_rounds=args.ds_max_rounds, ds_seed=args.seed + 1)
        print(f"dynamic sampling ON: max_rounds={args.ds_max_rounds} pool={len(pool_rows)}", flush=True)
    trainer = TrainerCls(model=args.init, args=cfg, train_dataset=ds, processing_class=tok,
                         reward_funcs=(make_zero_reward_funcs() if args.reward == "zero" else make_reward_funcs(args.level)),
                         peft_config=peft_config, **extra)
    n_tr = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"trainable={n_tr} completions/step={args.prompts_per_step * args.G} accum={accum}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = trainer.train()
    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    meta = {"run": run, "init": args.init, "level": args.level, "reward": args.reward, "weights": WEIGHTS[args.reward],
            "lr": args.lr, "max_steps": args.max_steps, "prompts_per_step": args.prompts_per_step, "G": args.G,
            "max_completion": args.max_completion, "beta": args.beta, "curriculum": args.curriculum,
            "scale_rewards": args.scale_rewards, "vllm": args.vllm,
            "presence_penalty": args.presence_penalty, "is_mode": args.is_mode,
            "temperature": args.temperature, "top_k": args.top_k, "data": args.data, "epsilon_high": args.epsilon_high, "dynamic_sampling": args.dynamic_sampling, "ds_max_rounds": args.ds_max_rounds, "lora_r": args.lora_r, "trainable_params": n_tr,
            "seconds": round(time.time() - t0), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "train_loss": res.training_loss, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "log_history": trainer.state.log_history}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE " + json.dumps({k: v for k, v in meta.items() if k != "log_history"}), flush=True)


if __name__ == "__main__":
    main()
