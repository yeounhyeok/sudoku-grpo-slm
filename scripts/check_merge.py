#!/usr/bin/env python3
"""Check a merged checkpoint against base+adapter (PEFT) on a few SFT prompts.

  python scripts/check_merge.py ADAPTER_DIR MERGED_DIR [--level 2]

Prints max |logit diff| and greedy next-token agreement over the completion span; exits 1 if they disagree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("merged")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    # same class the trainer built (Qwen3_5ForConditionalGeneration); a text-only class would not match adapter keys
    AutoModelForCausalLM = AutoModelForImageTextToText
    from train_sft import render
    base = json.loads((Path(a.adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(a.merged)
    recs = [json.loads(l) for _, l in zip(range(a.n), (ROOT / f"data/sft/L{a.level}/val.jsonl").open())]
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    peft = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, **kw), a.adapter).eval()
    merged = AutoModelForCausalLM.from_pretrained(a.merged, **kw).eval()
    plain = AutoModelForCausalLM.from_pretrained(base, **kw).eval()
    worst, agree, agree_base, tot = 0.0, 0, 0, 0
    acc = {"peft": 0, "merged": 0, "base": 0}
    with torch.no_grad():
        for r in recs:
            x = render(tok, r["messages"])
            p = tok(x["prompt"], add_special_tokens=False).input_ids
            ids = torch.tensor([p + tok(x["completion"], add_special_tokens=False).input_ids], device="cuda")
            s = slice(len(p) - 1, ids.shape[1] - 1)
            lp, lm, lb = (m(ids).logits[0, s].float() for m in (peft, merged, plain))
            worst = max(worst, (lp - lm).abs().max().item())
            agree += (lp.argmax(-1) == lm.argmax(-1)).sum().item()
            agree_base += (lp.argmax(-1) == lb.argmax(-1)).sum().item()
            tgt = ids[0, len(p):]
            for k, l in (("peft", lp), ("merged", lm), ("base", lb)):
                acc[k] += (l.argmax(-1) == tgt).sum().item()
            tot += lp.shape[0]
    res = {"max_abs_logit_diff": round(worst, 4), "argmax_agree_merged": round(agree / tot, 4),
           "argmax_agree_untrained_base": round(agree_base / tot, 4), "tokens": tot,
           "teacher_forced_acc": {k: round(v / tot, 4) for k, v in acc.items()}}
    print("MERGE_CHECK " + json.dumps(res), flush=True)
    sys.exit(0 if res["argmax_agree_merged"] >= 0.99 and res["argmax_agree_untrained_base"] < 0.99 else 1)


if __name__ == "__main__":
    main()
