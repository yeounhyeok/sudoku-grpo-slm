import glob,re,json,os,collections
import pyarrow.parquet as pq
os.nice(19)
def grid(p):
    u=p.split("<|im_start|>user")[-1].replace("<|im_end|>","")
    rows=[re.sub(r"\s","",l) for l in u.split("\n") if re.fullmatch(r"(?:[1-9.] ){8}[1-9.]",l.strip())]
    return "".join(rows[:9]) if len(rows)>=9 else None
D="/root/sudoku-grpo-slm/data/processed/"
tests={}
for f in ("test_id","test_ood"):
    for l in open(D+f+".jsonl"):
        r=json.loads(l); tests[r["puzzle"]]=f
pools={}
for f in glob.glob(D+"*.jsonl"):
    b=os.path.basename(f)
    if b.startswith("test_"): continue
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        if isinstance(r,dict) and "puzzle" in r: pools.setdefault(r["puzzle"],set()).add(b)
print("test",len(tests),"pool puzzles",len(pools))
c=collections.Counter(); uniq={}; per_pz=collections.Counter(); srcs=collections.Counter()
for d in sorted(glob.glob("/root/sudoku-grpo-slm/outputs/grpo/*/completions")):
    for f in sorted(glob.glob(d+"/*.parquet")):
        if "zero_accuracy" not in pq.read_schema(f).names: continue
        for r in pq.read_table(f,columns=["prompt","completion","zero_accuracy"]).to_pylist():
            g=grid(r["prompt"]); c["rows"]+=1
            if g is None: c["nogrid"]+=1; continue
            if g in tests: c["IN_"+tests[g]]+=1
            if r["zero_accuracy"]==1.0 and "</think>" in r["completion"]:
                uniq[(g,r["completion"])]=1; per_pz[g]+=1
                for s in pools.get(g,{"NOPOOL"}): srcs[s]+=1
print(dict(c)); print("unique acc completions",len(uniq),"unique puzzles",len(per_pz))
v=sorted(per_pz.values()); print("per-puzzle completions: min",v[0],"median",v[len(v)//2],"max",v[-1],"mean",round(sum(v)/len(v),2))
print("puzzles with 1 completion",sum(1 for x in v if x==1),"<=3",sum(1 for x in v if x<=3),">=8",sum(1 for x in v if x>=8))
print("in pool files",srcs.most_common(8)); print("puzzles not in any non-test pool",sum(1 for g in per_pz if g not in pools))
