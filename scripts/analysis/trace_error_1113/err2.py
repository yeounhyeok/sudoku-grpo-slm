exec(open("/tmp/err_an.py").read().split("def analyze")[0])
import random
# committed placement: "So/Thus/Therefore/Hence ... CELL = d" or bold "**CELL = d**", not followed by ' or'/','
COMMIT=re.compile(r"(?:\*\*\s*"+CELL+r"\s*=\s*([1-9])\s*\*\*|\b(?:So|so|Thus|thus|Therefore|therefore|Hence|hence)\b[^\n]{0,40}?"+CELL+r"\s*(?:=|is|must be)\s*([1-9])\b(?!\s*(?:or|,|/)))")
BAD=re.compile(r"\b(if|If|either|Either|OR|assume|Assume|suppose|Suppose|try|Try|Case|case|would|Possib|possib)\b|\?")
def cells(m):
    g=m.groups()
    if g[0] or g[2]: r,c,d=(g[0] or g[2]),(g[1] or g[3]),g[4]
    else: r,c,d=(g[5] or g[7]),(g[6] or g[8]),g[9]
    return int(r)-1,int(c)-1,d
def an(th,p,S):
    n=w=0; first=None; seen=set(); dup=0
    for m in COMMIT.finditer(th):
        ls=th.rfind("\n",0,m.start())+1
        if BAD.search(th[ls:m.start()]): continue
        r,c,d=cells(m); n+=1
        if (r,c) in seen: dup+=1
        seen.add((r,c))
        if d!=S[r*9+c]:
            w+=1
            if first is None: first=(m.start()/len(th), th[max(0,m.start()-200):m.end()+300])
    return n,w,first,dup
random.seed(1)
d=sorted(glob.glob("/root/sudoku-grpo-slm/outputs/grpo/*v4ds_c3/completions"))[-1]
G=collections.defaultdict(list); EX=collections.defaultdict(list)
for f in sorted(glob.glob(d+"/*.parquet")):
    for r in pq.read_table(f,columns=["prompt","completion","zero_accuracy"]).to_pylist():
        p=grid(r["prompt"]); S=sol.get(p)
        if not S: continue
        comp=r["completion"]
        k="correct" if r["zero_accuracy"]==1.0 else ("closed_wrong" if "</think>" in comp else "truncated")
        n,w,first,dup=an(comp.split("</think>")[0],p,S); G[k].append((n,w,dup,len(comp)))
        if first: EX[k].append(first)
for k,v in G.items():
    N=len(v); n=sum(x[0] for x in v); w=sum(x[1] for x in v); dup=sum(x[2] for x in v)
    print(f"== {k} n={N}: commits/trace {n/N:.1f} (re-commits of same cell {dup/N:.1f}), wrong commits/trace {w/N:.2f}, rate {w/max(1,n):.1%}, traces with >=1 wrong commit {sum(1 for x in v if x[1])/N:.0%}")
    fe=[e[0] for e in EX[k]]; 
    if fe: print(f"   first wrong commit at median {st.median(fe):.0%} of trace")
for k in ("correct","truncated"):
    for fr,t in random.sample(EX[k],min(3,len(EX[k]))):
        print("####",k,f"{fr:.0%}"); print(t.replace("\n"," | ")[:520])
