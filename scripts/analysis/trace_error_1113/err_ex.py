exec(open("/tmp/err_an.py").read().split("random.seed(0)")[0])
import random
random.seed(3)
d=sorted(glob.glob("/root/sudoku-grpo-slm/outputs/grpo/*v4ds_c3/completions"))[-1]
fs=sorted(glob.glob(d+"/*.parquet"))
ex=[]
for f in random.sample(fs,6):
    for r in pq.read_table(f,columns=["prompt","completion","zero_accuracy"]).to_pylist():
        if r["zero_accuracy"]!=1.0: continue
        p=grid(r["prompt"]); S=sol.get(p); th=r["completion"].split("</think>")[0]
        for m in ASSIGN.finditer(th):
            ls=th.rfind("\n",0,m.start())+1
            if HYP.search(th[ls:m.start()]): continue
            rr=int(m.group(1) or m.group(3))-1; c=int(m.group(2) or m.group(4))-1
            if m.group(5)!=S[rr*9+c]: ex.append((S[rr*9+c], "given" if p[rr*9+c]!="." else "blank", th[max(0,m.start()-350):m.end()+250]))
random.shuffle(ex)
for s,g,t in ex[:9]:
    print("#### true",s,g); print(t.replace("\n"," | ")); print()
