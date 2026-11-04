#!/usr/bin/env python3
"""SFT on solver traces (TRL SFTTrainer, prompt-completion, loss on the completion only).

  python scripts/train_sft.py --level 2 --model qwen3.5-2b --mode lora            # main run (S1)
  python scripts/train_sft.py --level 2 --max-steps 30 --tag _smoke               # smoke test

Data: data/sft/L{level}/{sft_train,val}.jsonl (scripts/build_sft.py). Each chat record is rendered
with the model's own chat template; the prompt is the generation prefix used at evaluation
(`add_generation_prompt=True, enable_thinking=True` -> ends with "<think>\\n"), and the completion is
the rest of the rendered assistant turn ("{trace}\\n</think>\\n\\n<answer>...</answer><|im_end|>").
The script asserts prompt + completion == full render, so train/eval formats cannot drift.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(tok, msgs: list[dict]) -> dict:
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=True)
    assert full.startswith(prompt), "chat template: prompt is not a prefix of the full render"
    completion = full[len(prompt):].rstrip("\n")
    assert completion.endswith("<|im_end|>"), completion[-40:]
    return {"prompt": prompt, "completion": completion}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--level", type=int, choices=[0, 1, 2], default=2)
    ap.add_argument("--mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--lr", type=float, default=None, help="default 2e-4 (lora) / 1e-5 (full)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=8, help="per-device micro batch")
    ap.add_argument("--accum", type=int, default=8, help="grad accumulation (effective = batch*accum)")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--n-train", type=int, default=0, help="subsample train rows (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--report-to", default="none")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    path = str(Path(os.environ.get("MODELS_DIR", ROOT / "models")) / args.model)
    run = f"sft-{args.model}-L{args.level}-{args.mode}{args.tag}"
    out = ROOT / "outputs/sft" / run
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(path)

    def load(split):
        recs = [json.loads(l) for l in (ROOT / f"data/sft/L{args.level}/{split}.jsonl").open()]
        if split == "sft_train" and args.n_train:
            recs = recs[:args.n_train]
        return Dataset.from_list([render(tok, r["messages"]) for r in recs])

    train, val = load("sft_train"), load("val")
    lens = sorted(len(tok(p + c, add_special_tokens=False).input_ids) for p, c in zip(train["prompt"][:500], train["completion"][:500]))
    print(f"run={run} train={len(train)} val={len(val)} tokens(first500) p50={lens[len(lens)//2]} max={lens[-1]} "
          f"eos={tok.eos_token!r} pad={tok.pad_token!r}", flush=True)
    assert lens[-1] <= args.max_len, f"sequence longer than --max-len ({lens[-1]})"

    peft_config = None
    if args.mode == "lora":
        from peft import LoraConfig
        peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                 target_modules="all-linear", task_type="CAUSAL_LM")
    lr = args.lr or (2e-4 if args.mode == "lora" else 1e-5)
    cfg = SFTConfig(
        output_dir=str(out), seed=args.seed, bf16=True,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=0.03,  # float <1 = ratio (transformers 5)
        weight_decay=0.0, optim="adamw_torch_fused", max_grad_norm=1.0,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len, packing=False, completion_only_loss=True,
        logging_steps=5, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2, save_only_model=True,
        report_to=args.report_to, dataloader_num_workers=2,
        model_init_kwargs={"dtype": torch.bfloat16},
    )
    trainer = SFTTrainer(model=path, args=cfg, train_dataset=train, eval_dataset=val,
                         processing_class=tok, peft_config=peft_config)

    # show exactly what the model is trained on (first example): masked prompt vs supervised completion
    ex = trainer.train_dataset[0]  # TRL 1.x stores masked labels (-100 = prompt, not trained)
    ids, lab = ex["input_ids"], ex["labels"]
    sup = [t for t, l in zip(ids, lab) if l != -100]
    first = next(i for i, l in enumerate(lab) if l != -100)
    print("SUPERVISED_HEAD:", repr(tok.decode(sup[:30])), flush=True)
    print("SUPERVISED_TAIL:", repr(tok.decode(sup[-20:])), flush=True)
    print("PROMPT_TAIL:", repr(tok.decode(ids[max(0, first - 10):first])), flush=True)
    n_tr = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable={n_tr} total={n_all} ({100*n_tr/n_all:.2f}%)", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = trainer.train()
    secs = time.time() - t0
    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    meta = {"run": run, "model": path, "level": args.level, "mode": args.mode, "lr": lr, "epochs": args.epochs,
            "max_steps": args.max_steps, "micro_batch": args.batch, "accum": args.accum,
            "effective_batch": args.batch * args.accum, "max_len": args.max_len, "optimizer": "adamw_torch_fused",
            "lora_r": args.lora_r if args.mode == "lora" else None, "trainable_params": n_tr, "total_params": n_all,
            "train_rows": len(train), "seconds": round(secs), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "train_loss": res.training_loss, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "log_history": trainer.state.log_history}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE " + json.dumps({k: v for k, v in meta.items() if k != "log_history"}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
