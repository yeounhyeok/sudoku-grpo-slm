#!/usr/bin/env python3
"""Structured-output check for SFT models (not sudoku accuracy).

For each eval sample: does the output follow the trained trace format, and is each written step
actually computed (forced by the board, evidence correct)? Uses the GRPO code judge (no answer key).

  python scripts/struct_check.py --level 3 outputs/eval/<run>/samples.jsonl [more.jsonl]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sudoku_grpo import reward, verify  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1] / "data/processed"
    recs = {}
    for f in root.glob("*.jsonl"):
        for line in f.open():
            r = json.loads(line)
            recs[r["id"]] = r
    for fn in a.files:
        S = [json.loads(l) for l in open(fn)]
        rows = []
        for s in S:
            r = recs[s["id"]]
            x = reward.score(r["puzzle"], r["solution"], s["output"], a.level)
            tr, _, closed = reward.split_completion(s["output"])
            snd = verify.check_trace(r["puzzle"], tr, None, apply="sound")
            x["first_ok"] = float(bool(snd.steps) and reward.counts(snd.steps[0], a.level))
            x["parsed_frac"] = (1 - x["n_unparsed"] / x["n_lines"]) if x["n_lines"] else 0.0
            x["cover"] = min(1.0, x["n_sound"] / x["n_empty"])
            rows.append(x)
        m = lambda k: st.mean(float(x[k]) for x in rows)  # noqa: E731
        out = {"file": Path(fn).parent.name, "n": len(rows),
               "structured_ok": round(m("format"), 3),          # closed think + one complete answer consistent with trace
               "closed": round(m("closed"), 3),
               "lines_parsed": round(m("parsed_frac"), 3),     # lines in the trained syntax
               "lines_forced": round(m("precision"), 3),       # lines that are board-forced with rule+evidence
               "first_line_forced": round(m("first_ok"), 3),
               "coverage": round(m("cover"), 3),
               "evidence_acc": round(st.mean(x["ev_mean"] for x in rows if x["ev_mean"] is not None), 3)
               if any(x["ev_mean"] is not None for x in rows) else None,
               "exact": round(m("exact"), 3)}
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
