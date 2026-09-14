import json, re, time, sys, math, requests, subprocess, signal
from pathlib import Path
from collections import Counter

ART = Path('/content/drive/MyDrive/yantra_run/artifacts')

# -- Find GGUF --
ggufs = sorted(ART.glob('stage6_yantra_q4_gguf/*.gguf'))
if not ggufs:
    print("ERROR: no GGUF found"); sys.exit(1)
GGUF = str(ggufs[0])
print(f"GGUF: {GGUF}")

# -- Find or install llama-server binary --
LLAMA = None

# 1) Check if Unsloth binary exists from a prior full run
for candidate in [
    Path.home() / '.unsloth/llama.cpp/llama-server',
    Path('/usr/local/bin/llama-server'),
    Path('/usr/bin/llama-server'),
]:
    if candidate.exists():
        LLAMA = str(candidate)
        break

# 2) Try pip install unsloth[cpp]
if not LLAMA:
    print("Trying pip install unsloth[cpp]...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'unsloth[cpp]'], timeout=300)
    for candidate in [
        Path.home() / '.unsloth/llama.cpp/llama-server',
        Path.home() / '.local/bin/llama-server',
    ]:
        if candidate.exists():
            LLAMA = str(candidate)
            break

# 3) Try building llama-cpp-python from source (fast on Colab T4)
if not LLAMA:
    print("Trying pip install llama-cpp-python[server]...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'llama-cpp-python[server]', '--force-reinstall', '--no-cache-dir'],
                   timeout=600, env={**dict(__import__('os').environ),
                                     'CMAKE_ARGS': '-DGGML_CUDA=on'})
    # llama-cpp-python installs a Python module, not a binary. Check for its server.
    result = subprocess.run([sys.executable, '-c',
                             'import llama_cpp.server; print(llama_cpp.server.__file__)'],
                            capture_output=True, text=True)
    # We can still use it as a library, but for chat.completions API we need the server binary.
    # Try the llama_cpp llama_server path
    for candidate in [
        Path.home() / '.local/bin/llama-server',
        Path.home() / '.local/lib/python3.13/site-packages/llama_cpp/bin/llama-server',
    ]:
        if candidate.exists():
            LLAMA = str(candidate)
            break

if not LLAMA:
    # 4) Last resort: use llama-cpp-python as a library (OpenAI-compatible server)
    print("No standalone binary found. Starting llama-cpp-python OpenAI server...")
    # llama_cpp.server provides an OpenAI-compatible endpoint
    server_proc = subprocess.Popen(
        [sys.executable, '-m', 'llama_cpp.server',
         '--model', GGUF, '--port', '8299', '--host', '127.0.0.1',
         '--n_gpu_layers', '999', '--n_ctx', '4096'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    PORT = 8299
    server_started = False
    for _ in range(60):
        time.sleep(1)
        try:
            r = requests.get(f'http://127.0.0.1:{PORT}/v1/models', timeout=3)
            if r.status_code == 200:
                server_started = True
                break
        except Exception:
            pass
    if not server_started:
        print("ERROR: all server methods failed"); sys.exit(1)
    print(f"Server started via llama-cpp-python on port {PORT}")
    LLAMA = "__LLAMA_CPP_PYTHON_SERVER__"

else:
    PORT = 8299
    print(f"Binary: {LLAMA}")
    # Kill any stale server
    subprocess.run(['pkill', '-f', f'llama-server.*{PORT}'], capture_output=True)
    time.sleep(1)
    subprocess.Popen([LLAMA, '-m', GGUF, '--port', str(PORT),
                      '--host', '127.0.0.1', '--ctx-size', '4096',
                      '--n-gpu-layers', '999'],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        time.sleep(1)
        try:
            r = requests.get(f'http://127.0.0.1:{PORT}/v1/models', timeout=3)
            if r.status_code == 200: break
        except Exception: pass
    else:
        print("ERROR: server failed to start"); sys.exit(1)
    print(f"Server ready on port {PORT}\n")

# -- Load eval set --
cases = []
with open(ART / 'toolace_300.jsonl') as f:
    for line in f: cases.append(json.loads(line))
print(f"Loaded {len(cases)} cases")

# -- Build tool docs for Router v2 --
tool_docs = {}
for case in cases:
    for t in case['tools']:
        fn = t.get('function', t)
        name = fn['name']
        if name not in tool_docs:
            desc = fn.get('description', '')
            params = list(fn.get('parameters', {}).get('properties', {}).keys())
            tool_docs[name] = (desc, params)

# -- Router v2: IDF + char-3gram --
N = len(tool_docs)
df = Counter()
for desc, _ in tool_docs.values():
    for w in set(re.findall(r'\w+', desc.lower())):
        df[w] += 1

def idf_score(query, desc):
    qtoks = set(re.findall(r'\w+', query.lower()))
    dtoks = set(re.findall(r'\w+', desc.lower()))
    score = 0.0
    for w in qtoks & dtoks:
        score += math.log((N + 1) / (df.get(w, 0) + 1))
    return score

def char_ngrams(s, n=3):
    s = re.sub(r'\s+', ' ', s.lower().strip())
    return Counter(s[i:i+n] for i in range(len(s) - n + 1))

CGRAM_CACHE = {name: char_ngrams(name) for name in tool_docs}

def char3gram_score(query, name):
    qg = char_ngrams(query)
    ng = CGRAM_CACHE[name]
    overlap = sum((qg & ng).values())
    total = sum(ng.values()) or 1
    return overlap / total

def route_v2(query):
    best_name, best_score = None, -1
    for name, (desc, _) in tool_docs.items():
        score = idf_score(query, desc) + 2.0 * char3gram_score(query, name)
        if score > best_score:
            best_name, best_score = name, score
    return best_name

# -- Parser: last-bind-with-args --
def parse_dtsa(text):
    binds = list(re.finditer(r'<bind\s+tool="([^"]+)"', text))
    args = list(re.finditer(r'<args>(.*?)</end>', text, re.DOTALL))
    if not binds:
        m = re.search(r'<param\s+name="(\w[\w\s]*?)"\s*>\s*(.+?)\s*</param>', text)
        if m: return '_tagless', {m.group(1).strip(): m.group(2).strip()}, text
        return None, {}, text
    if args:
        last_a = args[-1]
        preceding = [b for b in binds if b.end() <= last_a.start()]
        if preceding:
            name = preceding[-1].group(1)
            a_text = last_a.group(1).strip()
            if a_text:
                parsed = {}
                for p in re.finditer(r'(\w[\w\s]*?)\s*=\s*["\'](.+?)["\']', a_text):
                    parsed[p.group(1).strip()] = p.group(2)
                return name, parsed, text
    name = binds[-1].group(1)
    return name, {}, text

# -- PAS scoring --
halt_re = re.compile(r'</(output|call|calls|bind|args|end)>', re.IGNORECASE)

def score_case(pred_name, pred_args, gold_name, gold_args, stopped):
    parseable = 1.0
    valid_name = 1.0 if pred_name else 0.0
    expected_name = 1.0 if pred_name == gold_name else 0.0
    exact_args = 1.0 if pred_args == gold_args else 0.0
    if gold_args:
        ga = set(k.lower().strip() for k in gold_args)
        pa = set(k.lower().strip() for k in pred_args)
        arg_key_overlap = len(ga & pa) / len(ga) if ga else 1.0
    else:
        arg_key_overlap = 1.0
    stopped_cleanly = 1.0 if stopped else 0.0
    return {
        'parseable': parseable, 'valid_name': valid_name,
        'expected_name': expected_name, 'exact_args': exact_args,
        'arg_key_overlap': arg_key_overlap, 'stopped_cleanly': stopped_cleanly,
        'recovery': 0.0, 'multiturn': 0.0,
    }

# -- Run eval --
PRED = ART / 'eval_progress_router_v2_official.jsonl'
correct = 0
metrics_acc = {k: 0.0 for k in ['parseable','valid_name','expected_name','exact_args','arg_key_overlap','stopped_cleanly','recovery','multiturn']}
preds = []
stop_reasons = Counter()

for idx, case in enumerate(cases):
    query = case['query']
    gold_name = case['gold'][0]['name']
    gold_args = case['gold'][0].get('args', {})
    tools_json = json.dumps([t.get('function', t) for t in case['tools']], ensure_ascii=False)

    tool_name = route_v2(query)
    prefix = f'<tools>{tools_json}</tools>\n<calls><bind tool="{tool_name}"/>\n'

    msgs = [
        {"role": "system", "content": "You are a precise API caller. Output exactly one <bind> with <args>...</end> and stop."},
        {"role": "user",   "content": prefix + f"\nQuery: {query}"},
    ]

    try:
        resp = requests.post(f'http://127.0.0.1:{PORT}/v1/chat/completions',
                             json={"model": "minicpm", "messages": msgs,
                                   "max_tokens": 512, "temperature": 0.0,
                                   "stop": ["</calls>"]},
                             timeout=120)
        content = resp.json()['choices'][0]['message']['content']
        finish = resp.json()['choices'][0].get('finish_reason', '')
    except Exception as e:
        content = ""
        finish = "error"

    stopped = bool(halt_re.search(content)) or finish == 'stop'
    if stopped and '</calls>' not in content:
        content += '</calls>'
    stop_reasons[finish] += 1

    pred_name, pred_args, _ = parse_dtsa(content)
    scores = score_case(pred_name, pred_args, gold_name, gold_args, stopped)

    is_correct = (pred_name == gold_name) and (pred_args == gold_args)
    if is_correct: correct += 1

    for k in metrics_acc: metrics_acc[k] += scores[k]
    preds.append({
        'idx': idx, 'query': query[:80], 'pred_name': pred_name,
        'pred_args': pred_args, 'gold_name': gold_name, 'gold_args': gold_args,
        'is_correct': is_correct, 'stopped': stopped, 'finish_reason': finish,
    })

    if (idx + 1) % 50 == 0 or idx == len(cases) - 1:
        n = idx + 1
        pas = sum(scores.values()) / 8.0 if n > 0 else 0
        print(f"[{n}/{len(cases)}] exact={correct/n:.4f} pas={pas:.4f} stop={stop_reasons.get('stop',0)/n:.3f}")

# -- Summary --
n = len(cases)
print(f"\n=== RESULTS ===")
print(f"Exact match: {correct}/{n} = {correct/n:.4f}")
print(f"Stop rate:   {stop_reasons.get('stop',0)}/{n} = {stop_reasons.get('stop',0)/n:.4f}")
for k, v in sorted(metrics_acc.items()):
    print(f"  {k}: {v/n:.4f}")

# -- Save --
with open(PRED, 'w') as f:
    for p in preds:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')
print(f"\nSaved {len(preds)} predictions to {PRED}")

# -- Show worst cases --
wrongs = [p for p in preds if not p['is_correct']]
print(f"\n=== WORST CASES (first 5) ===")
for p in wrongs[:5]:
    print(f"  [{p['idx']}] gold={p['gold_name']} pred={p['pred_name']} args={p['pred_args']}")
    print(f"       query: {p['query']}")
