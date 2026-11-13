exec(open("/tmp/err4.py").read().split("d=sorted(")[0])
import random
random.seed(0)
d=sorted(glob.glob("/root/sudoku-grpo-slm/outputs/grpo/*v4ds_c3/completions"))[-1]
rows=[]
for f in sorted(glob.glob(d+"/*.parquet")): rows+=pq.read_table(f,columns=["prompt","completion","zero_accuracy"]).to_pylist()
M=collections.defaultdict(lambda:[0,0]); DUMP=collections.Counter(); VERT=collections.Counter()
for r in rows:
    if r["zero_accuracy"]!=1.0: continue
    p=grid(r["prompt"]); S=sol.get(p)
    if not S: continue
    th=r["completion"].split("</think>")[0]
    for m in MISS.finditer(th):
        us=list(UNIT.finditer(th[max(0,m.start()-160):m.start()]))
        if not us: continue
        kind,n=us[-1].group(1),int(us[-1].group(2)); kind="Col" if kind=="Column" else kind
        if not 1<=n<=9: continue
        claim=set(re.findall(r"[1-9]",m.group(1))); cs=unit_cells(kind,n)
        M[kind][0]+=1; M[kind][1]+=bool(claim & {p[i] for i in cs if p[i]!="."})
    # one-line unit dumps: "Col 3: ., ., 4, 9 ..." / "Row 5: ..."
    for m in re.finditer(r"(?m)^\W*(Row|Col|Column|Box)\s*\d\b[^\n]*$",th):
        line=m.group(0)
        if len(re.findall(r"(?<![0-9])[1-9.](?![0-9])",line))>=7: DUMP["Col" if m.group(1)=="Column" else m.group(1)]+=1
    # vertical listing: >=5 consecutive lines like "R1: ." / "R3: 9" / "R1C8=?" (one cell per line)
    for m in re.finditer(r"(?m)(?:^\s*R\dC?\d?\s*[:=(]\s*[1-9.?][^\n]{0,6}\n){5,}",th):
        blk=m.group(0); rs=re.findall(r"R(\d)(?:C(\d))?",blk)
        cols={c for _,c in rs if c}; 
        VERT["same-col (R#C# fixed C)" if len(cols)==1 else ("R#: lines (column readout)" if not cols else "mixed")]+=1
print("missing-claim misread by unit kind (correct traces):",{k:(v[0],f"{v[1]/v[0]:.1%}") for k,v in M.items()})
print("one-line unit dumps by kind:",dict(DUMP))
print("vertical one-cell-per-line blocks (>=5 lines):",dict(VERT), "per correct trace", round(sum(VERT.values())/sum(1 for r in rows if r['zero_accuracy']==1.0),1))
