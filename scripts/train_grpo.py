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
           "outcome": [0.2, 0.0, 0.0, 0.0, 0.8, 0.0]}   # answer only (RQ2 control)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="merged SFT checkpoint dir (warm start)")
    ap.add_argument("--level", type=int, choices=[0, 1, 2, 3], default=2, help="trace level checked by the step reward")
    ap.add_argument("--reward", choices=list(WEIGHTS), default="gated")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--G", type=int, default=8, help="completions per prompt (group size)")
    ap.add_argument("--micro", type=int, default=8, help="completions per forward/backward micro batch")
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coef (TRL default 0)")
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--curriculum", default="",
                    help='staged tiers, e.g. "easy:40,easy+medium:40,easy+medium+hard:20" (steps per stage); '
                         'prompts are laid out stage by stage and the trainer does not reshuffle them')
    ap.add_argument("--scale-rewards", default="group", choices=["group", "batch", "none"],
                    help="advantage scaling: group std (GRPO), batch std (less noise amplification), none (Dr. GRPO)")
    ap.add_argument("--report-to", default=os.environ.get("SFT_REPORT_TO", "wandb"))
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from sudoku_grpo import prompt as P
    from sudoku_grpo.reward import make_reward_funcs
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
    for line in (ROOT / "data/processed/rl_train.jsonl").open():
        r = json.loads(line)
        p = tok.apply_chat_template(P.messages_think(r["puzzle"]), tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
        assert p.endswith("<think>\n"), p[-30:]
        rows.append({"prompt": p, "puzzle": r["puzzle"], "solution": r["solution"], "tier": r.get("tier", "")})
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
        temperature=1.0, top_p=0.95, top_k=20,          # eval preset minus presence penalty
        beta=args.beta, mask_truncated_completions=False,   # truncation is penalised by the reward, not hidden
        reward_weights=WEIGHTS[args.reward],
        scale_rewards=args.scale_rewards, shuffle_dataset=not args.curriculum,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, log_completions=True, num_completions_to_print=1,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=4, save_only_model=True,
        report_to=args.report_to, model_init_kwargs={"dtype": torch.bfloat16},
    )
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                             target_modules=LM_TARGETS, task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=args.init, args=cfg, train_dataset=ds, processing_class=tok,
                          reward_funcs=make_reward_funcs(args.level), peft_config=peft_config)
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
            "scale_rewards": args.scale_rewards, "lora_r": args.lora_r, "trainable_params": n_tr,
            "seconds": round(time.time() - t0), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "train_loss": res.training_loss, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "log_history": trainer.state.log_history}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE " + json.dumps({k: v for k, v in meta.items() if k != "log_history"}), flush=True)


if __name__ == "__main__":
    main()
