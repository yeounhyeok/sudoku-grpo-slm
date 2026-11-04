#!/usr/bin/env python3
"""Evaluate a model (base or fine-tuned) on the Sudoku splits with HF generate.

  python scripts/eval.py --model qwen3.5-0.8b --split test_id --n 200 --mode direct
  python scripts/eval.py --model qwen3.5-2b --split test_ood --n 64 --mode think --backend vllm  # native thinking,
        # Qwen3.5 model-card sampling (T=1.0, top_p=0.95, top_k=20, presence_penalty=1.5), 32768 tokens
  python scripts/eval.py --model outputs/sft-0.8b/final --split test_id      # a checkpoint dir

Metrics (all over the evaluated rows):
  exact      predicted grid == solution                     (the headline metric)
  valid      complete grid, no row/col/box conflicts, givens preserved
  format     a parsable <answer> block with exactly 81 cells
  cell_acc   mean fraction of originally blank cells that are correct (0 if unparsable)
Reports are broken down by tier (beta3) or rating bin (sapient). Writes
outputs/eval/<run>/{summary.json,samples.jsonl} (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sudoku_grpo import grid, prompt  # noqa: E402

# Qwen3.5 model card "Best Practices" (text tasks). presence_penalty is not native to HF
# generate, so it is implemented below (OpenAI/vLLM semantics: generated tokens only).
PRESETS = {
    "greedy":       dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=0.0),
    "qwen-think":   dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5),
    "qwen-nothink": dict(temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0),
}


def presence_penalty_processor(penalty: float, prompt_len: int):
    import torch
    from transformers import LogitsProcessor

    class PresencePenalty(LogitsProcessor):
        """Subtract `penalty` from logits of every token already generated (prompt excluded)."""
        def __init__(self):
            self.mask = None

        def __call__(self, input_ids, scores):
            if self.mask is None:
                self.mask = torch.zeros_like(scores, dtype=torch.bool)
            if input_ids.shape[1] > prompt_len:
                self.mask.scatter_(1, input_ids[:, -1:], True)
            return scores - penalty * self.mask.to(scores.dtype)

    return PresencePenalty()


def split_thinking(text: str) -> tuple[str, bool]:
    """Return (final part after the last </think>, think_closed). Without </think> the model
    never left its reasoning, so there is no final answer to score."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1], True
    return "", False


def resolve_model(arg: str) -> tuple[str, str]:
    """Return (path, run_name). `arg` is a key in configs/models.toml or a directory."""
    if Path(arg).is_dir():
        return arg, Path(arg).parent.name + "_" + Path(arg).name
    cfg = tomllib.loads((ROOT / "configs/models.toml").read_text())
    for m in cfg["model"]:
        if m["key"] == arg:
            local = Path(os.environ.get("MODELS_DIR", ROOT / "models")) / arg
            return (str(local) if local.is_dir() else m["repo"]), arg
    raise SystemExit(f"unknown model {arg}")


def score(rec: dict, text: str) -> dict:
    pred = grid.parse_answer(text)
    ok_fmt = pred is not None
    exact = ok_fmt and pred == rec["solution"]
    valid = ok_fmt and grid.is_complete_valid(pred) and grid.givens_preserved(rec["puzzle"], pred)
    cacc = grid.cell_accuracy(rec["puzzle"], pred, rec["solution"]) if ok_fmt else 0.0
    return {"format": ok_fmt, "exact": exact, "valid": valid, "cell_acc": cacc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--n", type=int, default=0, help="subsample size (0 = all), seeded")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mode", choices=["reason", "direct", "think"], default="reason",
                    help="reason: <reasoning>+<answer> prompt (thinking off); direct: grid only; "
                         "think: native Qwen3.5 thinking ON, answer scored after </think>")
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="default: 4096 (reason) / 400 (direct) / 32768 (think, model-card value)")
    ap.add_argument("--thinking", action="store_true", help="Qwen3.5 thinking mode (implied by --mode think)")
    ap.add_argument("--preset", choices=list(PRESETS), default=None,
                    help="decoding preset; default greedy, or qwen-think for --mode think")
    for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty"):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=None, help="override preset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                    help="hf: transformers generate; vllm: paged-KV engine (much faster for long thinking)")
    ap.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    ap.add_argument("--eager", action="store_true", help="vLLM enforce_eager (no HIP graphs)")
    ap.add_argument("--cg-mode", default=None, help="vLLM cudagraph_mode override, e.g. PIECEWISE")
    ap.add_argument("--no-prefix-cache", action="store_true", help="vLLM disable prefix caching")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.mode == "think":
        args.thinking = True
    if not args.max_new_tokens:
        args.max_new_tokens = {"direct": 400, "reason": 4096, "think": 32768}[args.mode]
    preset = args.preset or ("qwen-think" if args.mode == "think" else "greedy")
    dec = dict(PRESETS[preset])
    for k in dec:
        if getattr(args, k) is not None:
            dec[k] = getattr(args, k)
    dec["top_k"] = int(dec["top_k"])
    sample = dec["temperature"] > 0
    make_msgs = {"direct": prompt.messages_direct, "reason": prompt.messages,
                 "think": prompt.messages_think}[args.mode]
    path, name = resolve_model(args.model)
    splits = args.split.split(",")
    if args.backend == "hf" and len(splits) > 1:
        raise SystemExit("multiple splits in one call need --backend vllm")
    think_tag = "-think" if args.thinking and args.mode != "think" else ""
    samp_tag = "" if preset == "greedy" and not sample else f"-{preset}"
    be_tag = "-vllm" if args.backend == "vllm" else ""
    jobs = []  # one job per split: same subset/seed/run name as a single-split call
    for sp in splits:
        recs = [json.loads(l) for l in (ROOT / "data/processed" / f"{sp}.jsonl").open()]
        if args.n and args.n < len(recs):
            recs = random.Random(args.seed).sample(recs, args.n)
        run = f"{name}-{args.mode}{think_tag}{samp_tag}{be_tag}{args.tag}_{sp}_n{len(recs)}"
        out = ROOT / "outputs/eval" / run
        out.mkdir(parents=True, exist_ok=True)
        jobs.append({"split": sp, "recs": recs, "run": run, "out": out, "rows": [],
                     "f": (out / "samples.jsonl").open("w")})

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    for jb in jobs:
        jb["texts"] = [tok.apply_chat_template(make_msgs(r["puzzle"]), tokenize=False, add_generation_prompt=True,
                                               enable_thinking=args.thinking) for r in jb["recs"]]
    print(f"runs={[jb['run'] for jb in jobs]} backend={args.backend} decoding={'sample' if sample else 'greedy'} "
          f"{dec} max_new_tokens={args.max_new_tokens}", flush=True)
    t0 = time.time()

    def record(jb, b, gens):
        for r, (text, n_tok, trunc) in zip(b, gens):
            for sp in ("<|im_end|>", "<|endoftext|>"):
                text = text.replace(sp, "")
            final, closed = split_thinking(text) if args.thinking else (text, True)
            row = {"id": r["id"], "grp": r.get("tier") or r.get("bin"), "new_tokens": n_tok,
                   "truncated": trunc, "think_closed": closed, **score(r, final), "output": text}
            jb["rows"].append(row)
            jb["f"].write(json.dumps(row, ensure_ascii=False) + "\n")
        jb["f"].flush()
        rows = jb["rows"]
        ex = sum(x["exact"] for x in rows)
        print(f"[{jb['split']} {len(rows)}/{len(jb['recs'])}] exact={ex}/{len(rows)} elapsed={time.time()-t0:.0f}s",
              flush=True)

    if args.backend == "vllm":
        # vLLM: paged KV cache + continuous batching + Triton GDN kernels (no per-step KV copy).
        # On MI250 (gfx90a) HIP graphs crash (HSA_STATUS_ERROR_INVALID_PACKET_FORMAT) -> use --eager.
        from vllm import LLM, SamplingParams
        texts = [t for jb in jobs for t in jb["texts"]]
        plen = max(len(tok(t, add_special_tokens=False).input_ids) for t in texts)
        llm = LLM(model=path, dtype="bfloat16", seed=args.seed, max_model_len=plen + args.max_new_tokens + 16,
                  gpu_memory_utilization=args.gpu_mem, limit_mm_per_prompt={"image": 0, "video": 0},
                  enforce_eager=args.eager, enable_prefix_caching=not args.no_prefix_cache,
                  **({"compilation_config": {"cudagraph_mode": args.cg_mode}} if args.cg_mode else {}))
        print(f"vllm loaded in {time.time()-t0:.0f}s, device={torch.cuda.get_device_name(0)}", flush=True)
        t0 = time.time()
        sps = [SamplingParams(n=1, max_tokens=args.max_new_tokens, skip_special_tokens=False,
                              temperature=dec["temperature"], top_p=dec["top_p"] if sample else 1.0,
                              top_k=(dec["top_k"] or -1) if sample else -1, min_p=dec["min_p"],
                              presence_penalty=dec["presence_penalty"], seed=args.seed + j)
               for jb in jobs for j in range(len(jb["recs"]))]  # seed = per-split index
        res = llm.generate(texts, sps)
        k = 0
        for jb in jobs:
            m = len(jb["recs"])
            record(jb, jb["recs"], [(o.outputs[0].text, len(o.outputs[0].token_ids),
                                     o.outputs[0].finish_reason == "length") for o in res[k:k + m]])
            k += m
    else:
        jb = jobs[0]
        recs, texts = jb["recs"], jb["texts"]
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
        print(f"loaded {path} ({type(model).__name__}), {len(recs)} rows, device={torch.cuda.get_device_name(0)}", flush=True)
        for i in range(0, len(recs), args.batch_size):
            b = recs[i:i + args.batch_size]
            enc = tok(texts[i:i + args.batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gkw = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, do_sample=sample)
            if sample:
                torch.manual_seed(args.seed + i)
                gkw.update(temperature=dec["temperature"], top_p=dec["top_p"], top_k=dec["top_k"] or None,
                           min_p=dec["min_p"] or None)
            else:
                gkw.update(temperature=None, top_p=None, top_k=None)
            if dec["presence_penalty"]:
                gkw["logits_processor"] = [presence_penalty_processor(dec["presence_penalty"], enc.input_ids.shape[1])]
            with torch.no_grad():
                gen = model.generate(**enc, **gkw)
            gens = []
            for g in gen[:, enc.input_ids.shape[1]:]:
                n_tok = int((g != tok.pad_token_id).sum())
                gens.append((tok.decode(g[g != tok.pad_token_id], skip_special_tokens=False), n_tok,
                             n_tok >= args.max_new_tokens))
            record(jb, b, gens)
    seconds = round(time.time() - t0)

    def agg(rs):
        n = len(rs)
        return {"n": n, **{k: round(sum(float(x[k]) for x in rs) / n, 4)
                           for k in ("exact", "valid", "format", "cell_acc", "truncated", "think_closed")},
                "mean_new_tokens": round(sum(x["new_tokens"] for x in rs) / n, 1)}

    for jb in jobs:
        jb["f"].close()
        rows = jb["rows"]
        by = defaultdict(list)
        for x in rows:
            by[x["grp"]].append(x)
        # token-budget curve from the single max-budget run: solved AND finished within B tokens
        budgets = [b for b in (1024, 2048, 4096, 8192, 16384, 32768) if b <= args.max_new_tokens]
        curve = {str(b): round(sum(1 for x in rows if x["exact"] and x["new_tokens"] <= b) / len(rows), 4)
                 for b in budgets}
        summary = {"run": jb["run"], "model": path, "split": jb["split"], "mode": args.mode, "thinking": args.thinking,
                   "max_new_tokens": args.max_new_tokens, "decoding": ("sample" if sample else "greedy"),
                   "backend": args.backend, "preset": preset, "sampling": dec, "seed": args.seed,
                   "seconds": seconds, "seconds_note": "wall time of the whole call (all splits together)",
                   "overall": agg(rows), "exact_within_budget": curve,
                   "by_group": {k: agg(v) for k, v in sorted(by.items())}}
        (jb["out"] / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps({k: summary[k] for k in ("run", "overall", "exact_within_budget")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
