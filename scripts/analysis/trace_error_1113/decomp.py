import glob,re,json,random,collections,statistics as st
import pyarrow.parquet as pq
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("/root/sudoku-grpo-slm/models/qwen3.5-2b")
CELLR=r"(?:[Rr][1-9]\s*[Cc][1-9]|\(\s*[1-9]\s*,\s*[1-9]\s*\))"
def cat(line):
    s=line.strip()
    if not s: return "blank"
    if re.fullmatch(r"(?:\**[Rr](?:ow)?\s*\d\**\s*[:|]?\s*)?\[?\s*(?:[1-9.][\s,|]+){8}[1-9.]\s*\]?\.?",s): return "board_row"
    if re.search(r"\b(Wait|wait|mistake|contradiction|Contradiction|wrong|error|double-check|Double-check|verify|Verify|re-check|re-evaluate|Let me check|Let's check|check again|Let me re)",s): return "recheck"
    if len(re.findall(r"(?<![0-9])[1-9.](?![0-9])",s))>=7 and re.search(r"\b(Row|Col|Column|Box|R\d|C\d)\b|:",s) and not re.search(r"[Mm]issing|\{",s): return "unit_dump"
    if len(re.findall(CELLR+r"\s*=\s*[1-9]",s))>=3: return "state_dump"
    if re.search(r"\b(So|Thus|Therefore|Hence)\b.*"+CELLR+r"\s*(=|is|must be)\s*[1-9]",s) or re.fullmatch(r"\**\s*"+CELLR+r"\s*=\s*[1-9]\s*\**\.?",s): return "commit"
    if re.search(r"[Mm]issing|[Cc]andidate|\{[1-9, ]*\}|[Ii]ntersect|[Cc]an be|[Mm]ust be|[Cc]annot be|[Oo]ptions",s): return "cand_calc"
    if re.search(r"\b(Existing|Present|Values|Has|has|contains|Contains|Numbers present|Known)\b",s): return "unit_contents"
    if len(re.findall(CELLR,s))>=3 or re.match(r"(Cells|Empty|Empty cells)\b",s): return "cell_enum"
    if re.match(r"(#+|\*\*)",s) or re.search(r"\b(Let's look|Let's check|Look at|Next|Now|Update|Let's analyze|Let's try|Let's re|Focus)\b",s): return "nav_header"
    return "prose"
random.seed(0)
sol={}
for f in glob.glob("/root/sudoku-grpo-slm/data/processed/*.jsonl"):
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        if isinstance(r,dict) and "puzzle" in r: sol[r["puzzle"]]=1
d=sorted(glob.glob("/root/sudoku-grpo-slm/outputs/grpo/*v4ds_c3/completions"))[-1]
rows=[]
for f in sorted(glob.glob(d+"/*.parquet")):
    rows+=pq.read_table(f,columns=["completion","zero_accuracy"]).to_pylist()
for kind in ("correct","truncated"):
    R=[r for r in rows if (r["zero_accuracy"]==1.0)==(kind=="correct") and ((kind=="correct") or "</think>" not in r["completion"])]
    R=random.sample(R,min(300,len(R)))
    T=collections.Counter(); dup_tok=0; tot=0; ex=collections.defaultdict(list); tmpl=collections.Counter()
    for r in R:
        th=r["completion"].split("</think>")[0]
        parts=collections.defaultdict(list); seen=set()
        for line in th.split("\n"):
            c=cat(line); parts[c].append(line)
            key=line.strip()
            if key and key in seen and len(key)>15: parts["_dup"].append(line)
            seen.add(key)
            if c=="prose" and len(ex["prose"])<4000: ex["prose"].append(line.strip())
            if c in("prose","nav_header"): tmpl[re.sub(r"[1-9]","#",line.strip())[:60]]+=1
        for c,ls in parts.items():
            n=len(tok("\n".join(ls))["input_ids"]); T[c]+=n
        tot+=sum(v for k,v in T.items() if k!="_dup") and 0
    tot=sum(v for k,v in T.items() if k!="_dup")
    print(f"== {kind} (n={len(R)}) median tokens/trace ~{tot//len(R)}")
    for c,v in T.most_common():
        print(f"   {c:14s} {v/tot:6.1%}  ({v//len(R)} tok/trace)")
    print("   top prose/nav templates:",[ (k,v) for k,v in tmpl.most_common(18)])
    print("   random prose lines:",random.sample(ex["prose"],12))
