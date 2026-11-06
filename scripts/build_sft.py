#!/usr/bin/env python3
"""Build SFT data (solver traces, formats L0/L1/L2) from data/processed/{sft_train,val}.jsonl.

  python scripts/build_sft.py            # -> data/sft/L{0,1,2}/{sft_train,val}.jsonl + data/sft/manifest.json

Each record is chat-format and model-agnostic:
  {"id", "tier", "level", "n_steps",
   "messages": [system, user, {"role": "assistant", "reasoning_content": <trace>, "content": <answer block>}]}
The Qwen3.5 chat template renders the assistant turn as
  <think>\\n{reasoning_content}\\n</think>\\n\\n{content}
which matches the inference prefix used by `eval.py --mode think`.

Only puzzles solvable with single-candidate techniques are kept (decision (A), 2026-11-04).
Every kept trace is replayed through the step verifier at its level and must be fully valid
and end at the given solution; otherwise the build fails.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sudoku_grpo import prompt, solver, verify  # noqa: E402
from sudoku_grpo.grid import format_grid  # noqa: E402

SPLITS = ("sft_train", "val")


def answer_block(solution81: str) -> str:
    return f"<answer>\n{format_grid(solution81)}\n</answer>"


def main() -> None:
    out_root = ROOT / "data/sft"
    manifest = {"source": "data/processed", "rule": "single-candidate solver (naked, hidden); unsolvable dropped",
                "levels": {}}
    for level in solver.LEVELS:
        (out_root / f"L{level}").mkdir(parents=True, exist_ok=True)
        manifest["levels"][f"L{level}"] = {}
    for split in SPLITS:
        recs = [json.loads(l) for l in (ROOT / "data/processed" / f"{split}.jsonl").open()]
        files = {lv: (out_root / f"L{lv}" / f"{split}.jsonl").open("w") for lv in solver.LEVELS}
        kept, dropped = collections.Counter(), collections.Counter()
        for r in recs:
            ok, steps, final = solver.solve_steps(r["puzzle"])
            grp = r.get("tier") or r.get("bin")
            if not ok:
                dropped[grp] += 1
                continue
            assert final == r["solution"], r["id"]
            kept[grp] += 1
            for lv in solver.LEVELS:
                tr = solver.trace_text(steps, lv, r["puzzle"])
                chk = verify.check_trace(r["puzzle"], tr, r["solution"])
                s = chk.summary(lv)
                assert s["all_ok"] and chk.final == r["solution"], (r["id"], lv, s)
                msgs = prompt.messages_think(r["puzzle"]) + [
                    {"role": "assistant", "reasoning_content": tr, "content": answer_block(r["solution"])}]
                files[lv].write(json.dumps({"id": r["id"], "tier": grp, "level": lv, "n_steps": len(steps),
                                            "messages": msgs}, ensure_ascii=False) + "\n")
        for lv, f in files.items():
            f.close()
            p = out_root / f"L{lv}" / f"{split}.jsonl"
            manifest["levels"][f"L{lv}"][split] = {
                "rows": sum(kept.values()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        manifest.setdefault("kept", {})[split] = dict(sorted(kept.items()))
        manifest.setdefault("dropped", {})[split] = dict(sorted(dropped.items()))
        print(f"[{split}] kept {sum(kept.values())}/{len(recs)}  kept={dict(kept)} dropped={dict(dropped)}")
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["levels"], indent=1))


if __name__ == "__main__":
    main()
