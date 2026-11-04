#!/usr/bin/env python3
"""Training-throughput micro-benchmark (fwd+bwd+optimizer) for the SFT setup.

Measures tokens/s, step time and peak GPU memory, and reports which Gated-DeltaNet /
causal-conv implementations transformers resolved (optimized kernel vs PyTorch fallback).
Also prints the loss of a fixed batch so kernel swaps can be checked for numerical agreement.

  python scripts/bench_train.py --model qwen3.5-2b --mode lora --batch 8 --seqlen 1536
  python scripts/bench_train.py --model qwen3.5-2b --mode full --batch 4 --seqlen 1536
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=1536)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--no-gc", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    path = str(Path(os.environ.get("MODELS_DIR", ROOT / "models")) / args.model)
    impl = {}
    try:
        import fla  # noqa: F401
        impl["fla"] = fla.__version__
    except Exception as e:  # noqa: BLE001
        impl["fla"] = f"absent ({type(e).__name__})"
    try:
        import causal_conv1d  # noqa: F401
        impl["causal_conv1d"] = getattr(causal_conv1d, "__version__", "present")
    except Exception as e:  # noqa: BLE001
        impl["causal_conv1d"] = f"absent ({type(e).__name__})"

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda")
    if not args.no_gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
        lr = 2e-4
    else:
        lr = 1e-5
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, fused=True)
    model.train()

    g = torch.Generator().manual_seed(0)
    vocab = 150000  # stay inside the text vocab
    batches = [torch.randint(1000, vocab, (args.batch, args.seqlen), generator=g) for _ in range(2)]
    fixed = batches[0].cuda()

    with torch.no_grad():  # numerical fingerprint of the forward pass (same weights, same batch)
        ref_loss = model(input_ids=fixed, labels=fixed).loss.item()

    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(args.warmup + args.steps):
        x = batches[i % 2].cuda()
        torch.cuda.synchronize(); t = time.time()
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= args.warmup:
            times.append(time.time() - t)
    step = sum(times) / len(times)
    res = {"tag": args.tag, "model": args.model, "mode": args.mode, "batch": args.batch, "seqlen": args.seqlen,
           "grad_ckpt": not args.no_gc, "trainable_params": trainable, "impl": impl,
           "step_s": round(step, 3), "tokens_per_s": round(args.batch * args.seqlen / step),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), "fwd_loss_fixed_batch": round(ref_loss, 5),
           "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    sys.exit(main())
