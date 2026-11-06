#!/usr/bin/env python3
"""Stage-0 signal probe for "Sudoku Zero" (R1-Zero style GRPO, no SFT).

Samples G completions per puzzle from the ORIGINAL model with the baseline prompt (structure only),
scores them with the outcome reward R = acc + w_fmt * fmt, and reports how many groups would carry a
GRPO learning signal (rewards not all equal). No training.

  /root/venvs/vllm/bin/python scripts/probe_zero.py --tiers easy,medium --per-tier 32 --G 8 --max-tokens 8192
"""
import argparse
import collections
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sudoku_grpo import grid, prompt  # noqa: E402


def outcome(puzzle: str, text: str):
    """R1-style rule reward parts. fmt: closed think + exactly one 81-cell answer, no blanks, givens kept.
    acc: that grid is a complete valid sudoku (unique solution => correct)."""
    if "</think>" not in text:
        return 0.0, 0.0, False
    rest = text.split("</think>", 1)[1]
    pred = grid.parse_answer(rest)
    fmt = (pred is not None and rest.count("<answer>") == 1 and "." not in pred
           and grid.givens_preserved(puzzle, pred))
    acc = fmt and grid.is_complete_valid(pred)
    return float(fmt), float(acc), True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/qwen3.5-2b"))
    ap.add_argument("--split", default="rl_train")
    ap.add_argument("--tiers", default="easy,medium")
    ap.add_argument("--per-tier", type=int, default=32)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--w-fmt", type=float, default=0.1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs/probe_zero"))
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in (ROOT / f"data/processed/{a.split}.jsonl").open()]
    rng = random.Random(a.seed)
    recs = []
    for t in a.tiers.split(","):
        pool = [r for r in rows if r.get("tier") == t]
        rng.shuffle(pool)
        recs += pool[:a.per_tier]
    tok = AutoTokenizer.from_pretrained(a.model)
    texts = [tok.apply_chat_template(prompt.messages_think(r["puzzle"]), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for r in recs]
    llm = LLM(model=a.model, enforce_eager=True, gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_tokens + 1024, seed=a.seed, enable_prefix_caching=False)
    sp = SamplingParams(n=a.G, temperature=1.0, top_p=0.95, top_k=20, max_tokens=a.max_tokens, seed=a.seed)
    t0 = time.time()
    outs = llm.generate(texts, sp)
    secs = time.time() - t0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    by = collections.defaultdict(list)
    ntok = 0
    with (out / f"samples_{a.max_tokens}.jsonl").open("w") as f:
        for r, o in zip(recs, outs):
            g = []
            for c in o.outputs:
                fmt, acc, closed = outcome(r["puzzle"], c.text)
                R = acc + a.w_fmt * fmt
                n = len(c.token_ids)
                ntok += n
                g.append({"R": R, "acc": acc, "fmt": fmt, "closed": closed, "tokens": n})
                f.write(json.dumps({"id": r["id"], "tier": r["tier"], "R": R, "acc": acc, "fmt": fmt,
                                    "closed": closed, "tokens": n, "output": c.text}, ensure_ascii=False) + "\n")
            by[r["tier"]].append(g)
    res = {"model": a.model, "max_tokens": a.max_tokens, "G": a.G, "w_fmt": a.w_fmt, "seconds": round(secs),
           "gen_tok_per_s": round(ntok / secs), "tiers": {}}
    for t, groups in by.items():
        flat = [x for g in groups for x in g]
        res["tiers"][t] = {
            "puzzles": len(groups),
            "pass@1": round(st.mean(x["acc"] for x in flat), 3),
            "pass@G": round(st.mean(float(any(x["acc"] for x in g)) for g in groups), 3),
            "fmt_rate": round(st.mean(x["fmt"] for x in flat), 3),
            "truncated": round(1 - st.mean(float(x["closed"]) for x in flat), 3),
            "groups_with_signal": round(st.mean(float(len({x["R"] for x in g}) > 1) for g in groups), 3),
            "groups_with_acc_signal": round(st.mean(float(len({x["acc"] for x in g}) > 1) for g in groups), 3),
            "mean_tokens": round(st.mean(x["tokens"] for x in flat)),
            "mean_tokens_correct": round(st.mean([x["tokens"] for x in flat if x["acc"]])) if any(x["acc"] for x in flat) else None,
        }
    (out / f"summary_{a.max_tokens}.json").write_text(json.dumps(res, indent=2))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
