exec(open("/tmp/err2.py").read().split("random.seed(1)")[0])
def cov(th,p,S):
    ok=set(); w=0
    for m in COMMIT.finditer(th):
        ls=th.rfind("\n",0,m.start())+1
        if BAD.search(th[ls:m.start()]): continue
        r,c,d=cells(m)
        if p[r*9+c]!=".": continue
        if d==S[r*9+c]: ok.add((r,c))
        else: w+=1
    return len(ok)/p.count("."), w
d=sorted(glob.glob("/root/sudoku-grpo-slm/outputs/grpo/*v4ds_c3/completions"))[-1]
R=collections.defaultdict(list)
for f in sorted(glob.glob(d+"/*.parquet")):
    for r in pq.read_table(f,columns=["prompt","completion","zero_accuracy"]).to_pylist():
        p=grid(r["prompt"]); S=sol.get(p)
        if not S: continue
        comp=r["completion"]; k="correct" if r["zero_accuracy"]==1.0 else ("closed_wrong" if "</think>" in comp else "truncated")
        c,w=cov(comp.split("</think>")[0],p,S)
        R[k+("_clean" if w==0 else "_err")].append(c)
for k in sorted(R):
    v=R[k]; print(f"{k:22s} n={len(v):4d}  correctly-committed blanks: median {st.median(v):.0%}  p25 {sorted(v)[len(v)//4]:.0%}  >=90%: {sum(x>=.9 for x in v)/len(v):.0%}")
