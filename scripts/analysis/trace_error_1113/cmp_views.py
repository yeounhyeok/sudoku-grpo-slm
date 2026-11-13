"""Compare v4s60 default prompt (_ctl, medium subset) vs views prompt (_ctl_views_med) on the same problems."""
import json,re,random,statistics as st,collections,sys
R="/home/guest/workspace/vi005/sudoku/repo/"
sol={json.loads(l)["id"]:json.loads(l) for l in open(R+"data/processed/test_id.jsonl")}
def load(p,tier="medium"):
    S=collections.defaultdict(list)
    for l in open(p):
        r=json.loads(l)
        if r["tier"]==tier: S[r["id"]].append(r)
    return S
A=load(R+"outputs/eval_k/v4s60_test_id_n64_k8_T0.6_pp1.5_ctl/samples.jsonl")
B=load(R+"outputs/eval_k/v4s60_test_id_n64_k8_T0.6_pp1.5_ctl_views_med/samples.jsonl")
ids=sorted(set(A)&set(B)); print("problems",len(ids),"samples/problem",len(A[ids[0]]),len(B[ids[0]]))
CELL=r"(?:\b[Rr]([1-9])\s*[Cc]([1-9])\b|\(\s*([1-9])\s*,\s*([1-9])\s*\))"
COMMIT=re.compile(r"(?:\*\*\s*"+CELL+r"\s*=\s*([1-9])\s*\*\*|\b(?:So|so|Thus|thus|Therefore|therefore|Hence|hence)\b[^\n]{0,40}?"+CELL+r"\s*(?:=|is|must be)\s*([1-9])\b(?!\s*(?:or|,|/)))")
BAD=re.compile(r"\b(if|If|either|Either|OR|assume|Assume|suppose|Suppose|try|Try|Case|case|would|Possib|possib)\b|\?")
UNIT=re.compile(r"\b(Row|Col|Column|Box|C|B)\s*(\d)\b(?!\s*,?\s*[Cc]\d)")
MISS=re.compile(r"[Mm]issing\s*(?:numbers|digits|values)?\s*(?:is|are|:|=)?\s*\{?\s*((?:[1-9]\s*(?:,|and)?\s*)+)\}?")
def unit_cells(k,n):
    n-=1
    if k=="Row": return [n*9+c for c in range(9)]
    if k in("Col","Column","C"): return [r*9+n for r in range(9)]
    br,bc=divmod(n,3); return [(br*3+i)*9+bc*3+j for i in range(3) for j in range(3)]
def metrics(r):
    p=sol[r["id"]]["puzzle"]; S=sol[r["id"]]["solution"]; th=r["output"].split("</think>")[0]
    n=w=0; seen=set(); rec=0
    for m in COMMIT.finditer(th):
        ls=th.rfind("\n",0,m.start())+1
        if BAD.search(th[ls:m.start()]): continue
        g=m.groups()
        rr,cc,d=((g[0] or g[2]),(g[1] or g[3]),g[4]) if (g[0] or g[2]) else ((g[5] or g[7]),(g[6] or g[8]),g[9])
        i=(int(rr)-1)*9+int(cc)-1; n+=1; w+= d!=S[i]; rec+= i in seen; seen.add(i)
    mis=collections.defaultdict(lambda:[0,0])
    for m in MISS.finditer(th):
        us=list(UNIT.finditer(th[max(0,m.start()-160):m.start()]))
        if not us: continue
        k,u=us[-1].group(1),int(us[-1].group(2))
        if not 1<=u<=9: continue
        kk={"Column":"Col","C":"Col","B":"Box"}.get(k,k)
        cs=unit_cells(k,u); claim=set(re.findall(r"[1-9]",m.group(1)))
        mis[kk][0]+=1; mis[kk][1]+=bool(claim & {p[c] for c in cs if p[c]!="."})
    vert=len(re.findall(r"(?m)(?:^\s*R\dC?\d?\s*[:=(]\s*[1-9.?][^\n]{0,6}\n){5,}",th))
    dump=sum(len(l)+1 for l in th.split("\n") if len(re.findall(r"(?<![0-9])[1-9.](?![0-9])",l))>=7 and not re.search(r"[Mm]issing|\{",l))
    return dict(commits=n,wrong=w,recommit=rec,mis=mis,vert=vert,dump_share=dump/max(1,len(th)),chars=len(th))
def summarize(D,name):
    acc24=[sum(s["acc"]>0 and s["tokens"]<=24576 for s in D[i])/len(D[i]) for i in ids]
    acc32=[sum(s["acc"]>0 for s in D[i])/len(D[i]) for i in ids]
    allS=[s for i in ids for s in D[i]]
    M=[metrics(s) for s in allS]
    mis=collections.defaultdict(lambda:[0,0])
    for m in M:
        for k,v in m["mis"].items(): mis[k][0]+=v[0]; mis[k][1]+=v[1]
    C=[m for m,s in zip(M,allS) if s["acc"]>0]
    print(f"== {name}: pass@1 24k {st.mean(acc24):.1%}  32k {st.mean(acc32):.1%}  closed {sum(s['closed'] for s in allS)/len(allS):.0%}  median tokens {int(st.median(s['tokens'] for s in allS))}  median tokens(correct) {int(st.median([s['tokens'] for s in allS if s['acc']>0] or [0]))}")
    print(f"   wrong-commit rate {sum(m['wrong'] for m in M)/max(1,sum(m['commits'] for m in M)):.1%}  recommits/trace {st.mean(m['recommit'] for m in M):.1f}  vertical blocks/trace {st.mean(m['vert'] for m in M):.1f}  unit-dump char share {st.mean(m['dump_share'] for m in M):.1%}")
    print("   misread by unit:",{k:(v[0],f"{v[1]/max(1,v[0]):.1%}") for k,v in mis.items()})
    return acc24,acc32
a24,a32=summarize(A,"default (_ctl)"); b24,b32=summarize(B,"views")
random.seed(0)
for nm,x,y in (("24k",a24,b24),("32k",a32,b32)):
    d=[yy-xx for xx,yy in zip(x,y)]; bs=sorted(st.mean(random.choice(d) for _ in d) for _ in range(5000))
    print(f"paired diff views-default @{nm}: {st.mean(d):+.1%}  95% CI [{bs[125]:+.1%}, {bs[4875]:+.1%}]")
