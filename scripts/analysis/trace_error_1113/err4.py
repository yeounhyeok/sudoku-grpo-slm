exec(open("/tmp/err2.py").read().split("random.seed(1)")[0])
UNIT=re.compile(r"\b(Row|Col|Column|Box)\s*(\d)\b(?!\s*,?\s*[Cc]\d)")
MISS=re.compile(r"[Mm]issing\s*(?:numbers|digits|values)?\s*(?:is|are|:|=)?\s*\{?\s*((?:[1-9]\s*(?:,|and)?\s*)+)\}?")
def unit_cells(kind,n):
    n-=1
    if kind=="Row": return [n*9+c for c in range(9)]
    if kind in("Col","Column"): return [r*9+n for r in range(9)]
    br,bc=divmod(n,3); return [(br*3+i)*9+bc*3+j for i in range(3) for j in range(3)]
def an(th,p,S):
    # committed correct cells in order
    commits=[]
    for m in COMMIT.finditer(th):
        ls=th.rfind("\n",0,m.start())+1
        if BAD.search(th[ls:m.start()]): continue
        r,c,d=cells(m); commits.append((m.start(),r*9+c,d==S[r*9+c]))
    firstwrong=next((pos for pos,_,ok in commits if not ok),None)
    boards=[m.start() for m in re.finditer(r"(?m)^\s*(?:\**[Rr](?:ow)?\s*\d\**\s*[:|]?\s*)?\[?\s*(?:[1-9.][\s,|]+){8}[1-9.]\s*\]?\s*$",th)]
    # board render = 9 consecutive board lines (approx: board line whose next 8 also within 600 chars)
    renders=[boards[i] for i in range(len(boards)-8) if boards[i+8]-boards[i]<700]
    out=[]
    for m in MISS.finditer(th):
        back=th[max(0,m.start()-160):m.start()]
        us=list(UNIT.finditer(back))
        if not us: continue
        kind,n=us[-1].group(1),int(us[-1].group(2))
        if not 1<=n<=9: continue
        claim=set(re.findall(r"[1-9]",m.group(1)))
        cs=unit_cells(kind,n)
        filled={p[i] for i in cs if p[i]!="."}|{S[i] for pos,i,ok in commits if ok and pos<m.start() and i in cs}
        truth={str(d) for d in range(1,10)}-filled
        inc_given=bool(claim & {p[i] for i in cs if p[i]!="."})
        omit=bool(truth-claim)   # omits a digit not yet filled per our tracking (approx)
        last=max([b for b in renders if b<m.start()],default=None)
        dist=m.start()-last if last is not None else m.start()
        before_fw = firstwrong is not None and firstwrong-3000<m.start()<firstwrong
        out.append((inc_given,omit,dist,m.start()/len(th),before_fw))
    return out,firstwrong,len(renders)
d=sorted(glob.glob("/root/sudoku-grpo-slm/outputs/grpo/*v4ds_c3/completions"))[-1]
G=collections.defaultdict(list); RN=collections.defaultdict(list)
for f in sorted(glob.glob(d+"/*.parquet")):
    for r in pq.read_table(f,columns=["prompt","completion","zero_accuracy"]).to_pylist():
        p=grid(r["prompt"]); S=sol.get(p)
        if not S: continue
        comp=r["completion"]; k="correct" if r["zero_accuracy"]==1.0 else ("closed_wrong" if "</think>" in comp else "truncated")
        o,fw,nr=an(comp.split("</think>")[0],p,S); G[k]+=o; RN[k].append(nr)
for k,v in G.items():
    if len(v)<200: continue
    n=len(v); ig=sum(x[0] for x in v); om=sum(x[1] for x in v)
    print(f"== {k}: missing-claims {n}  claims incl. a GIVEN digit (definite misread) {ig/n:.1%}  claims omitting an unfilled digit (approx) {om/n:.1%}  median full-board renders/trace {st.median(RN[k])}")
    for lo,hi in ((0,2000),(2000,5000),(5000,10000),(10000,10**9)):
        s=[x for x in v if lo<=x[2]<hi]
        if s: print(f"   dist since last board render {lo}-{hi if hi<10**9 else 'inf'} chars: n={len(s)} misread {sum(x[0] for x in s)/len(s):.1%} omit {sum(x[1] for x in s)/len(s):.1%}")
    for lo,hi in ((0,.25),(.25,.5),(.5,.75),(.75,1.01)):
        s=[x for x in v if lo<=x[3]<hi]
        if s: print(f"   position {int(lo*100)}-{int(min(hi,1)*100)}%: misread {sum(x[0] for x in s)/len(s):.1%}")
    s=[x for x in v if x[4]]; s2=[x for x in v if not x[4]]
    if s: print(f"   within 3k chars BEFORE first wrong commit: misread {sum(x[0] for x in s)/len(s):.1%} (n={len(s)}) vs elsewhere {sum(x[0] for x in s2)/len(s2):.1%}")
