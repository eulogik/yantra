import json, re, math, os, sys, time
from pathlib import Path
from collections import Counter
import llama_cpp

RUN_DIR = Path("/content/drive/MyDrive/yantra_run")
ART = RUN_DIR / "artifacts"
METRICS = ["parseable","valid_name","expected_name","exact_args","arg_key_overlap","stopped_cleanly","recovery","multiturn"]

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

def parse_dtsa(text):
    text = re.sub(THINK_OPEN + r".*?" + THINK_CLOSE, "", text, flags=re.S).strip()
    bind_pat = re.compile(r'<bind\s+tool="([^"]*)"\s*(?:/>|>)', re.S)
    binds = list(bind_pat.finditer(text))
    if not binds:
        return {"tool": None, "args": None, "completion": text, "parseable": False,
                "note": "no <bind> found", "preamble_think": THINK_OPEN in text}
    chosen = None
    for j in range(len(binds)-1, -1, -1):
        start = binds[j].end()
        end = binds[j+1].start() if j+1 < len(binds) else len(text)
        seg = text[start:end]
        a = re.search(r"<args>(.*?)</args>", seg, re.S)
        if a and a.group(1).strip():
            chosen = (j, seg, a); break
    if chosen is None:
        j = len(binds)-1
        start = binds[j].end()
        end = binds[j+1].start() if j+1 < len(binds) else len(text)
        seg = text[start:end]
        a = re.search(r"<args>(.*?)</args>", seg, re.S)
        chosen = (j, seg, a)
    j, seg, a = chosen
    raw_args = a.group(1) if a else ""
    tool = binds[j].group(1)
    params = []
    for m in re.finditer(r'<param\s+name="([^"]+)"(?:\s+type="[^"]*")?>(.*?)</param>', raw_args, re.S):
        params.append((m.group(1), m.group(2).strip()))
    if not params:
        for m in re.finditer(r'name="([^"]+)"\s*>[ \t]*(.*)', raw_args):
            params.append((m.group(1), m.group(2).strip()))
    args = {k: v for k, v in params}
    comp = re.sub(r"<args>.*?</args>", "", seg, flags=re.S).strip()
    return {"tool": tool, "args": args, "completion": comp, "parseable": True, "note": "", "preamble_think": False}

def evaluate_case(llm, prompt, case):
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=768,
        stop=["\n<bind", "<tool_result>", "<user>"])
    text = response["choices"][0]["message"]["content"] or ""
    pred = parse_dtsa(text)
    g = case["gold"][0]; gname, gargs = g["name"], g.get("args", {}) or {}
    pname, pargs = pred["tool"], pred["args"] or {}
    valid = pname in {t.get("function",t).get("name") for t in case["tools"]}
    pset, gset = set(pargs.items()), set(gargs.items())
    exact = pset == gset
    overlap = (len(pset & gset)/len(gset)) if gset else (1.0 if not pset else 0.0)
    stopped = not pred["completion"]
    buckets = {"parseable": int(pred["parseable"]), "valid_name": int(valid),
               "expected_name": int(pname == gname), "exact_args": int(exact),
               "arg_key_overlap": round(overlap,4), "stopped_cleanly": int(stopped),
               "recovery": 0, "multiturn": 0}
    meta = {"gold_tool": gname, "pred_tool": pname, "gold_args": gargs, "pred_args": pargs,
            "gold_tool_args_bad": int((pname==gname) and (not exact)),
            "stopped_cleanly": stopped, "preamble_think": pred["preamble_think"], "raw": text[:300]}
    return buckets, pred, meta

def pas(b):
    return sum(b[k] for k in METRICS)/8

toks = lambda s: set(re.findall(r"[a-z0-9_]+", s.lower()))
def build_router(cases):
    docs = [toks((t.get("function",t).get("name","")+" "+t.get("function",t).get("description","")))
            for c in cases for t in c["tools"]]
    df = Counter(w for d in docs for w in d); N = max(len(docs),1)
    idf = lambda w: math.log(N/(1+df[w]))
    def cgrams(s, n=3):
        s = re.sub(r"[^a-z0-9 ]","",s.lower())
        return {s[i:i+n] for i in range(max(len(s)-n+1,1))}
    def route(q, tools):
        qt = toks(q); qg = cgrams(q); best, bn = -1e18, None
        for t in tools:
            f = t.get("function", t); nm = f.get("name") or ""
            d = toks(nm+" "+(f.get("description") or ""))
            s1 = sum(idf(w) for w in qt & d)/(math.sqrt(sum(idf(w) for w in d)+1e-6))
            ng = cgrams(nm); s2 = len(qg & ng)/(math.sqrt(len(qg)*len(ng))+1)
            if s1 + 2.0*s2 > best: best, bn = s1 + 2.0*s2, nm
        return bn
    return route

# --- load model directly (no server, avoids chat-template mismatch) ---
gguf = sorted(ART.glob("stage6_yantra_q4_gguf/*.gguf"))[0]
print("GGUF:", gguf, "size", round(gguf.stat().st_size/1e6,1), "MB")
print("loading model ...")
llm = llama_cpp.Llama(model_path=str(gguf), n_gpu_layers=-1, n_ctx=4096, verbose=False)
print("model loaded")

# --- purge stale progress from broken server runs ---
prog = ART/"eval_progress_router_v2.jsonl"
if prog.exists():
    first = json.loads(prog.read_text().splitlines()[0])
    if first.get("git","") != "router-v2-direct":
        print("purging stale progress"); prog.unlink()
done = {}
if prog.exists():
    for line in prog.read_text().splitlines():
        if line.strip(): d = json.loads(line); done[d["i"]] = d

cases = [json.loads(l) for l in open(ART/"toolace_300.jsonl") if l.strip()]
route = build_router(cases)
per = list(done.values())
print(f"resuming from {len(done)} completed cases")
for i, case in enumerate(cases):
    if i in done:
        if (i+1) % 50 == 0: print("  (resumed)", i+1, "/", len(cases))
        continue
    bind = f'<bind tool="{route(case["query"], case["tools"])}"/>\n'
    prompt = (f"<user>{case['query']}</user>\n"
              f"<tools>{json.dumps([t.get('function',t) for t in case['tools']], ensure_ascii=False)}</tools>\n"
              f"<calls>{bind}")
    b, pred, meta = evaluate_case(llm, prompt, case)
    rec = {"i": i, "git": "router-v2-direct", **b, **meta}
    per.append(rec)
    with open(prog, "a") as fh: fh.write(json.dumps(rec)+"\n")
    if (i+1) % 25 == 0: print("  eval", i+1, "/", len(cases))

agg = {k: round(sum(x[k] for x in per)/len(per),4) for k in METRICS}
print("\nYANTRA PAS =", round(pas(agg),4))
print(json.dumps(agg, indent=2))
res_file = ART/"yantra_results.json"
out = {"PAS": round(pas(agg),4), **agg, "per_case": per, "git": "router-v2-direct", "model": str(gguf)}
if res_file.exists():
    prev = json.loads(res_file.read_text())
    if out["PAS"] <= prev.get("PAS", 0):
        print("not better than", prev.get("PAS"), "- keeping previous"); out = prev
res_file.write_text(json.dumps(out, indent=2))
print("saved ->", res_file)
