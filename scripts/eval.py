#!/usr/bin/env python3
"""Evaluate a model (base or fine-tuned) on the Sudoku splits with HF generate.

  python scripts/eval.py --model qwen3.5-0.8b --split test_id --n 200 --mode direct
  python scripts/eval.py --model qwen3.5-2b --split test_ood --n 64 --mode think   # native thinking,
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
    recs = [json.loads(l) for l in (ROOT / "data/processed" / f"{args.split}.jsonl").open()]
    if args.n and args.n < len(recs):
        recs = random.Random(args.seed).sample(recs, args.n)
    think_tag = "-think" if args.thinking and args.mode != "think" else ""
    samp_tag = "" if preset == "greedy" and not sample else f"-{preset}"
    run = f"{name}-{args.mode}{think_tag}{samp_tag}{args.tag}_{args.split}_n{len(recs)}"
    out = ROOT / "outputs/eval" / run
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    print(f"loaded {path} ({type(model).__name__}), {len(recs)} rows, device={torch.cuda.get_device_name(0)}", flush=True)

    texts = [tok.apply_chat_template(make_msgs(r["puzzle"]), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=args.thinking) for r in recs]
    print(f"run={run} decoding={'sample' if sample else 'greedy'} {dec} max_new_tokens={args.max_new_tokens}", flush=True)
    t0 = time.time()
    rows = []
    fsamp = (out / "samples.jsonl").open("w")  # written per batch so long runs keep partial results
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
        for r, g in zip(b, gen[:, enc.input_ids.shape[1]:]):
            n_tok = int((g != tok.pad_token_id).sum())
            text = tok.decode(g[g != tok.pad_token_id], skip_special_tokens=False)
            for sp in ("<|im_end|>", "<|endoftext|>"):
                text = text.replace(sp, "")
            final, closed = split_thinking(text) if args.thinking else (text, True)
            s = score(r, final)
            row = {"id": r["id"], "grp": r.get("tier") or r.get("bin"), "new_tokens": n_tok,
                   "truncated": n_tok >= args.max_new_tokens, "think_closed": closed, **s, "output": text}
            rows.append(row)
            fsamp.write(json.dumps(row, ensure_ascii=False) + "\n")
        fsamp.flush()
        done = len(rows)
        ex = sum(x["exact"] for x in rows)
        print(f"[{done}/{len(recs)}] exact={ex}/{done} elapsed={time.time()-t0:.0f}s", flush=True)
    fsamp.close()

    def agg(rs):
        n = len(rs)
        return {"n": n, **{k: round(sum(float(x[k]) for x in rs) / n, 4) for k in ("exact", "valid", "format", "cell_acc", "truncated", "think_closed")},
                "mean_new_tokens": round(sum(x["new_tokens"] for x in rs) / n, 1)}

    by = defaultdict(list)
    for x in rows:
        by[x["grp"]].append(x)
    summary = {"run": run, "model": path, "split": args.split, "mode": args.mode, "thinking": args.thinking,
               "max_new_tokens": args.max_new_tokens, "decoding": ("sample" if sample else "greedy"),
               "preset": preset, "sampling": dec, "seed": args.seed, "seconds": round(time.time() - t0),
               "overall": agg(rows), "by_group": {k: agg(v) for k, v in sorted(by.items())}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
