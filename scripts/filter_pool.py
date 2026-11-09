"""Offline stand-in for DAPO dynamic sampling: keep puzzles the current policy solves sometimes but not always.

usage: python scripts/filter_pool.py <eval_k samples.jsonl> <pool.jsonl> <out.jsonl> [--budget 24576]
Reads per-sample results from scripts/eval_k.py (same decoding as training), counts correct samples within the
budget per puzzle id, and writes the pool rows with 0 < correct < k (both all-correct and all-wrong groups give
zero GRPO advantage).
"""
import argparse, collections, json

ap = argparse.ArgumentParser()
ap.add_argument("samples"); ap.add_argument("pool"); ap.add_argument("out")
ap.add_argument("--budget", type=int, default=24576)
a = ap.parse_args()
ok, k = collections.Counter(), collections.Counter()
for line in open(a.samples):
    s = json.loads(line)
    k[s["id"]] += 1
    ok[s["id"]] += bool(s["acc"]) and s["tokens"] <= a.budget
rows = [json.loads(l) for l in open(a.pool)]
keep = [r for r in rows if r["id"] in k and 0 < ok[r["id"]] < k[r["id"]]]
hist = collections.Counter(ok[r["id"]] for r in rows if r["id"] in k)
with open(a.out, "w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\n")
print(json.dumps({"pool": len(rows), "evaluated": len(k), "kept": len(keep),
                  "correct_count_hist": dict(sorted(hist.items()))}))
