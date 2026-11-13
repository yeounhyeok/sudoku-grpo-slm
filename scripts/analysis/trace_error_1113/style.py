import glob,re,json,statistics as st
import pyarrow.parquet as pq
def style(t):
    th=t.split("</think>")[0]
    return dict(chars=len(th), wait=len(re.findall(r"\bWait\b",th)), cand_sets=len(re.findall(r"\{[1-9, ]+\}",th)),
                rc=len(re.findall(r"r\d+c\d+",th,re.I)), rowcol_words=len(re.findall(r"[Rr]ow \d|[Cc]ol(?:umn)? \d",th)),
                board_lines=len(re.findall(r"(?m)^\s*(?:R\d+:?\s*)?\[?(?:[1-9.][ ,]+){8}[1-9.]\]?\s*$",th)),
                tuples=len(re.findall(r"\(\d,\s?\d\)",th)))
runs=sorted(glob.glob("/root/sudoku-grpo-slm/outputs/grpo/*v4*/completions"))
print(runs)
d=runs[-1]; fs=sorted(glob.glob(d+"/*.parquet"))[-10:]
ok=[]
for f in fs:
    if "zero_accuracy" not in pq.read_schema(f).names: continue
    for r in pq.read_table(f,columns=["completion","zero_accuracy"]).to_pylist():
        if r["zero_accuracy"]==1.0: ok.append(style(r["completion"]))
print("files",[f.split("/")[-1] for f in fs[:1]+fs[-1:]],"correct n",len(ok))
print({k:int(st.median(x[k] for x in ok)) for k in ok[0]})
