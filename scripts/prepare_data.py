#!/usr/bin/env python3
"""Build train/val/test splits for the Sudoku project (stdlib only, deterministic).

Sources (configs/data.toml):
  * beta3/GridCorpus_9M_Sudoku_Puzzles_Enriched (cc-by-4.0): streams only the first
    `scan_rows` rows of the 5.7 GB CSV. Gives sft_train / rl_train / val / test_id,
    stratified by difficulty_tier. Splits are disjoint by SOLUTION GRID.
  * sapientinc/sudoku-extreme test.csv (79 MB): OOD hard test, stratified by tdoku rating.
    Its train.csv is never used (the dataset card says to evaluate on test only).

Output: data/processed/<split>.jsonl + data/manifest.json (row counts, sha256).
  record: {id, puzzle, solution, source, tier|rating, givens}   (81-char strings, '.' = blank)

  python scripts/prepare_data.py            # everything
  python scripts/prepare_data.py --only beta3
"""
from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import io
import json
import random
import sys
import tomllib
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sudoku_grpo import grid  # noqa: E402

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"
csv.field_size_limit(1 << 24)


def open_stream(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "sudoku-grpo-slm"})
    resp = urllib.request.urlopen(req, timeout=120)
    return io.TextIOWrapper(resp, encoding="utf-8", newline="")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_beta3(cfg, out: Path) -> dict[str, int]:
    b = cfg["beta3"]
    quota = {s: dict(q) for s, q in b["quota"].items()}
    need = {s: Counter(q) for s, q in quota.items()}
    rows: dict[str, list] = defaultdict(list)
    sol_owner: dict[str, str] = {}
    seen = Counter()
    n = 0
    reader = csv.DictReader(open_stream(HF.format(repo=b["repo"], file=b["file"])))
    for r in reader:
        n += 1
        if n > cfg["scan_rows"]:
            break
        tier = r["difficulty_tier"]
        seen[tier] += 1
        sol = grid.parse_grid_literal(r["solution_grid"])
        if sol_owner.get(sol):
            continue  # same solution already used (in any split) -> no leakage
        # first split that still needs this tier (priority = declaration order: test first is safest)
        for split in ("test_id", "val", "rl_train", "sft_train"):
            if need[split][tier] > 0:
                puz = grid.parse_grid_literal(r["puzzle_grid"])
                assert grid.givens_preserved(puz, sol) and grid.is_complete_valid(sol)
                need[split][tier] -= 1
                sol_owner[sol] = split
                rows[split].append({"id": f"beta3-{n}", "puzzle": puz, "solution": sol, "source": "beta3",
                                    "tier": tier, "givens": int(r["given_cells"])})
                break
        if all(sum(c.values()) == 0 for c in need.values()):
            break
    print(f"[beta3] scanned {n} rows; tier mix seen: {dict(seen)}")
    counts = {}
    for split, recs in rows.items():
        random.Random(cfg["seed"]).shuffle(recs)
        write_jsonl(out / f"{split}.jsonl", recs)
        counts[split] = len(recs)
        short = {t: c for t, c in need[split].items() if c > 0}
        print(f"[beta3] {split}: {len(recs)} rows" + (f"  (SHORT: {short})" if short else ""))
    return counts


def build_sapient(cfg, out: Path) -> dict[str, int]:
    s = cfg["sapient"]
    pools: dict[str, list] = {b["name"]: [] for b in s["bins"]}
    for i, r in enumerate(csv.DictReader(open_stream(HF.format(repo=s["repo"], file=s["file"])))):
        rating = int(r["rating"])
        for b in s["bins"]:
            if b["lo"] <= rating < b["hi"]:
                pools[b["name"]].append((i, r))
                break
    rng = random.Random(cfg["seed"])
    recs = []
    for b in s["bins"]:
        pool = pools[b["name"]]
        pick = rng.sample(pool, min(b["n"], len(pool)))
        print(f"[sapient] bin {b['name']}: pool {len(pool)} -> {len(pick)}")
        for i, r in pick:
            puz, sol = grid.normalize(r["question"]), grid.normalize(r["answer"])
            assert grid.givens_preserved(puz, sol) and grid.is_complete_valid(sol)
            recs.append({"id": f"sapient-{i}", "puzzle": puz, "solution": sol, "source": r["source"],
                         "rating": int(r["rating"]), "bin": b["name"],
                         "givens": sum(c != "." for c in puz)})
    rng.shuffle(recs)
    write_jsonl(out / "test_ood.jsonl", recs)
    return {"test_ood": len(recs)}


def write_jsonl(p: Path, recs: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["beta3", "sapient"])
    args = ap.parse_args()
    cfg = tomllib.loads((ROOT / "configs/data.toml").read_text())
    out = ROOT / "data" / "processed"
    counts: dict[str, int] = {}
    if args.only in (None, "beta3"):
        counts |= build_beta3(cfg, out)
    if args.only in (None, "sapient"):
        counts |= build_sapient(cfg, out)

    # cross-split leakage check by solution grid (beta3 splits are disjoint by construction)
    owner: dict[str, str] = {}
    for p in sorted(out.glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r["solution"] in owner and owner[r["solution"]] != p.stem and p.stem != "test_ood" and owner[r["solution"]] != "test_ood":
                raise SystemExit(f"LEAK: solution shared by {owner[r['solution']]} and {p.stem}")
            owner.setdefault(r["solution"], p.stem)
    man = {p.stem: {"rows": sum(1 for _ in p.open()), "sha256": sha256(p)} for p in sorted(out.glob("*.jsonl"))}
    (ROOT / "data" / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
