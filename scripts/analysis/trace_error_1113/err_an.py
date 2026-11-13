import glob,re,json,random,statistics as st,collections
import pyarrow.parquet as pq
D="/root/sudoku-grpo-slm/data/processed/"
sol={}
for f in glob.glob(D+"*.jsonl"):
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        if isinstance(r,dict) and "puzzle" in r and "solution" in r: sol[r["puzzle"]]=r["solution"]
def grid(p):
    u=p.split("<|im_start|>user")[-1].replace("<|im_end|>","")
    rows=[re.sub(r"\s","",l) for l in u.split("\n") if re.fullmatch(r"(?:[1-9.] ){8}[1-9.]",l.strip())]
    return "".join(rows[:9]) if len(rows)>=9 else None
CELL=r"(?:\b[Rr]([1-9])\s*[Cc]([1-9])\b|\(\s*([1-9])\s*,\s*([1-9])\s*\))"
ASSIGN=re.compile(CELL+r"\**\s*(?:=|:=|is|must be|->|→)\s*\**\s*([1-9])\b")
HYP=re.compile(r"\b(if|If|assume|Assume|suppose|Suppose|try|Try|would|could|cannot|can't|not|If)\b|\?")
WAITW=re.compile(r"\b(Wait|wait|mistake|Mistake|contradiction|Contradiction|wrong|Wrong|error|Error)\b")
BOARD=re.compile(r"^\s*(?:\**[Rr](?:ow)?\s*\d\**\s*[:|]?\s*)?\[?\s*(?:[1-9.][\s,|]+){8}[1-9.]\s*\]?\s*$")
def analyze(th,puz,S):
    a=dict(n_assign=0,wrong=0,given_misread=0,wrong_then_flag=0,wrong_corrected=0,first_err_frac=None)
    rnd_flag=0
    for m in ASSIGN.finditer(th):
        ls=th.rfind("\n",0,m.start())+1
        if HYP.search(th[ls:m.start()]): continue
        r=int(m.group(1) or m.group(3))-1; c=int(m.group(2) or m.group(4))-1; d=m.group(5)
        a["n_assign"]+=1
        if d!=S[r*9+c]:
            a["wrong"]+=1
            if puz[r*9+c]!=".": a["given_misread"]+=1
            if a["first_err_frac"] is None: a["first_err_frac"]=m.start()/max(1,len(th))
            if WAITW.search(th[m.end():m.end()+400]): a["wrong_then_flag"]+=1
            if re.search(r"(?:[Rr]%d\s*[Cc]%d|\(\s*%d\s*,\s*%d\s*\))\**\s*(?:=|is)\s*\**\s*%s\b"%(r+1,c+1,r+1,c+1,S[r*9+c]),th[m.end():]): a["wrong_corrected"]+=1
    # line categories by chars
    cat=collections.Counter()
    for line in th.split("\n"):
        L=len(line)+1
        if BOARD.match(line): cat["board"]+=L
        elif WAITW.search(line) or re.search(r"\b(double-check|Double-check|verify|Verify|re-check|Let me check|Let's check|check again)\b",line): cat["recheck"]+=L
        elif re.search(r"\{[1-9, ]*\}|[Mm]issing|[Cc]andidate|[Hh]as\b|[Cc]ontains",line): cat["cand_calc"]+=L
        else: cat["other"]+=L
    a["cat"]=cat; a["waits"]=len(re.findall(r"\bWait\b",th)); a["chars"]=len(th)
    # baseline: flag word within 400 chars after random position
    a["rand_flag"]=sum(bool(WAITW.search(th[p:p+400])) for p in [random.randrange(max(1,len(th)-400)) for _ in range(20)])/20
    return a
random.seed(0)
d=sorted(glob.glob("/root/sudoku-grpo-slm/outputs/grpo/*v4ds_c3/completions"))[-1]
G=collections.defaultdict(list)
for f in sorted(glob.glob(d+"/*.parquet")):
    for r in pq.read_table(f,columns=["prompt","completion","zero_accuracy"]).to_pylist():
        p=grid(r["prompt"]); S=sol.get(p)
        if not S: continue
        comp=r["completion"]; closed="</think>" in comp
        kind="correct" if r["zero_accuracy"]==1.0 else ("closed_wrong" if closed else "truncated")
        G[kind].append(analyze(comp.split("</think>")[0],p,S))
for k,v in G.items():
    n=len(v); A=lambda key: sum(x[key] for x in v)
    wr=A("wrong")
    cat=collections.Counter(); [cat.update(x["cat"]) for x in v]; tot=sum(cat.values())
    fe=[x["first_err_frac"] for x in v if x["first_err_frac"] is not None]
    print(f"== {k} n={n}  median chars {int(st.median(x['chars'] for x in v))}  median Wait {int(st.median(x['waits'] for x in v))}")
    print(f"  assignments/trace {A('n_assign')/n:.1f}  wrong/trace {wr/n:.2f}  traces with >=1 wrong {sum(1 for x in v if x['wrong'])/n:.0%}  wrong rate {wr/max(1,A('n_assign')):.2%}")
    print(f"  wrong on GIVEN cells {A('given_misread')}  wrong followed by flag(400ch) {A('wrong_then_flag')/max(1,wr):.0%} vs random pos {st.mean(x['rand_flag'] for x in v):.0%}  wrong later corrected {A('wrong_corrected')/max(1,wr):.0%}")
    print(f"  first error at median {st.median(fe):.0%} of trace" if fe else "  no errors")
    print("  char share:",{c:f"{cat[c]/tot:.0%}" for c in ("board","cand_calc","recheck","other")})
