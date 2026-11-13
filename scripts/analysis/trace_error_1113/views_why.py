exec(open("/home/guest/workspace/vi005/sudoku/cmp_views.py").read().split("def summarize")[0])
def why(D):
    use=0; tr=0; conf=[0,0]
    for i in ids:
        p=sol[i]["puzzle"]
        for s in D[i]:
            th=s["output"].split("</think>")[0]; tr+=1
            use+= bool(re.search(r"(?<![Rr\d])\b(C[1-9]|B[1-9])\s*:",th))
            for m in MISS.finditer(th):
                us=list(UNIT.finditer(th[max(0,m.start()-160):m.start()]))
                if not us or us[-1].group(1)!="Row": continue
                u=int(us[-1].group(2))
                if not 1<=u<=9: continue
                claim=set(re.findall(r"[1-9]",m.group(1)))
                rg={p[c] for c in unit_cells("Row",u) if p[c]!="."}
                if claim & rg:
                    conf[0]+=1
                    cg={p[c] for c in unit_cells("Col",u) if p[c]!="."}
                    conf[1]+= not (claim & cg)   # consistent with column u instead of row u
    return use/tr, conf
for nm,D in (("default",A),("views",B)):
    u,c=why(D); print(nm,f"traces using C#:/B#: labels {u:.0%}", f"row misreads {c[0]}, of which consistent with COLUMN of same index {c[1]} ({c[1]/max(1,c[0]):.0%})")
