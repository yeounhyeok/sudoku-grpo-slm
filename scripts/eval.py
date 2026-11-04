#!/usr/bin/env python3
"""Evaluate a model (base or fine-tuned) on the Sudoku splits. Greedy decoding, HF generate.

  python scripts/eval.py --model qwen3.5-0.8b --split test_id --n 200
  python scripts/eval.py --model qwen3.5-2b --split test_ood --n 200 --thinking
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
    ap.add_argument("--mode", choices=["reason", "direct"], default="reason",
                    help="reason: <reasoning>+<answer> prompt; direct: grid only, no reasoning")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="default: 4096 (reason) / 400 (direct)")
    ap.add_argument("--thinking", action="store_true", help="Qwen3.5 thinking mode (default: off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.max_new_tokens:
        args.max_new_tokens = 400 if args.mode == "direct" else 4096
    make_msgs = prompt.messages_direct if args.mode == "direct" else prompt.messages
    path, name = resolve_model(args.model)
    recs = [json.loads(l) for l in (ROOT / "data/processed" / f"{args.split}.jsonl").open()]
    if args.n and args.n < len(recs):
        recs = random.Random(args.seed).sample(recs, args.n)
    run = f"{name}-{args.mode}{'-think' if args.thinking else ''}{args.tag}_{args.split}_n{len(recs)}"
    out = ROOT / "outputs/eval" / run
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    print(f"loaded {path} ({type(model).__name__}), {len(recs)} rows, device={torch.cuda.get_device_name(0)}", flush=True)

    texts = [tok.apply_chat_template(make_msgs(r["puzzle"]), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=args.thinking) for r in recs]
    t0 = time.time()
    rows = []
    for i in range(0, len(recs), args.batch_size):
        b = recs[i:i + args.batch_size]
        enc = tok(texts[i:i + args.batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for r, g in zip(b, gen[:, enc.input_ids.shape[1]:]):
            n_tok = int((g != tok.pad_token_id).sum())
            text = tok.decode(g, skip_special_tokens=True)
            s = score(r, text)
            rows.append({"id": r["id"], "grp": r.get("tier") or r.get("bin"), "new_tokens": n_tok,
                         "truncated": n_tok >= args.max_new_tokens, **s, "output": text})
        done = len(rows)
        ex = sum(x["exact"] for x in rows)
        print(f"[{done}/{len(recs)}] exact={ex}/{done} elapsed={time.time()-t0:.0f}s", flush=True)

    def agg(rs):
        n = len(rs)
        return {"n": n, **{k: round(sum(float(x[k]) for x in rs) / n, 4) for k in ("exact", "valid", "format", "cell_acc", "truncated")},
                "mean_new_tokens": round(sum(x["new_tokens"] for x in rs) / n, 1)}

    by = defaultdict(list)
    for x in rows:
        by[x["grp"]].append(x)
    summary = {"run": run, "model": path, "split": args.split, "mode": args.mode, "thinking": args.thinking,
               "max_new_tokens": args.max_new_tokens, "decoding": "greedy", "seconds": round(time.time() - t0),
               "overall": agg(rows), "by_group": {k: agg(v) for k, v in sorted(by.items())}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (out / "samples.jsonl").open("w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
