#!/usr/bin/env python3
"""R1-style evaluation: k samples per problem, pass@1 = mean correctness (DeepSeek-R1 v2, Supp. D.1).

Same problem subset as scripts/eval.py (random.Random(seed).sample(split, n)) so results line up with the
earlier single-sample tables. Reports pass@1 per tier and per thinking budget (a sample counts as correct
within budget B if it finished with a correct grid using <= B new tokens), plus pass@k.
Sampling default follows R1: temperature 0.6, top-p 0.95, no top-k, no presence penalty.

  python scripts/eval_k.py --model MODEL --split test_id --n 64 --k 8 --max-tokens 32768
"""
import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sudoku_grpo import prompt  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from probe_zero import outcome  # noqa: E402

BUDGETS = (8192, 12288, 16384, 20480, 24576, 32768)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    recs = [json.loads(l) for l in (ROOT / f"data/processed/{a.split}.jsonl").open()]
    if a.n:
        recs = random.Random(a.seed).sample(recs, a.n)          # identical subset to eval.py
    tok = AutoTokenizer.from_pretrained(a.model)
    texts = [tok.apply_chat_template(prompt.messages_think(r["puzzle"]), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for r in recs]
    llm = LLM(model=a.model, enforce_eager=True, gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_tokens + 1024, seed=a.seed, enable_prefix_caching=False)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                        presence_penalty=a.presence_penalty, max_tokens=a.max_tokens, seed=a.seed)
    t0 = time.time()
    outs = llm.generate(texts, sp)
    secs = time.time() - t0

    name = f"{Path(a.model).name}_{a.split}_n{a.n}_k{a.k}_T{a.temperature}_pp{a.presence_penalty}{a.tag}"
    out = ROOT / "outputs/eval_k" / name
    out.mkdir(parents=True, exist_ok=True)
    per = []                      # (tier, [ (acc, tokens) * k ])
    with (out / "samples.jsonl").open("w") as f:
        for r, o in zip(recs, outs):
            g = []
            for c in o.outputs:
                fmt, acc, closed = outcome(r["puzzle"], c.text)
                g.append((acc, len(c.token_ids)))
                f.write(json.dumps({"id": r["id"], "tier": r.get("tier"), "acc": acc, "fmt": fmt, "closed": closed,
                                    "tokens": len(c.token_ids), "output": c.text}, ensure_ascii=False) + "\n")
            per.append((r.get("tier") or "all", g))

    def summ(rows):
        res = {"problems": len(rows)}
        for B in BUDGETS:
            p = [sum(acc > 0 and t <= B for acc, t in g) / len(g) for _, g in rows]   # per-problem pass@1
            m = sum(p) / len(p)
            se = math.sqrt(sum((x - m) ** 2 for x in p) / max(1, len(p) - 1) / len(p))
            res[f"pass@1_{B // 1024}k"] = [round(m, 4), round(se, 4)]
        res[f"pass@{a.k}_32k"] = round(sum(any(acc > 0 for acc, _ in g) for _, g in rows) / len(rows), 4)
        res["mean_tokens"] = round(sum(t for _, g in rows for _, t in g) / sum(len(g) for _, g in rows))
        return res

    tiers = collections.defaultdict(list)
    for t, g in per:
        tiers[t].append(g)
    res = {"run": name, "model": a.model, "sampling": {"T": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
                                                       "presence_penalty": a.presence_penalty, "k": a.k},
           "seconds": round(secs), "overall": summ(per),
           "tiers": {t: summ([(t, g) for g in gs]) for t, gs in sorted(tiers.items())}}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
