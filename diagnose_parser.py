# === DIAGNOSTIC: See what the model actually generates ===
# Run this in Colab after the server is up via Unsloth binary.
# It shows raw output and how first-bind vs last-bind parsers handle it.

import json, re, requests, time
from pathlib import Path

ART = Path('/content/drive/MyDrive/yantra_run/artifacts')
LLAMA = str(Path.home() / '.unsloth/llama.cpp/llama-server')
GGUF = str(ART / 'stage6_yantra_q4_gguf/base_model.Q4_K_M.gguf')
PORT = 8299

# --- Start server if needed ---
def ensure_server():
    import subprocess, signal
    try:
        r = requests.get(f'http://127.0.0.1:{PORT}/v1/models', timeout=3)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    subprocess.Popen([LLAMA, '-m', GGUF, '--port', str(PORT),
                      '--host', '127.0.0.1', '--ctx-size', '4096',
                      '--n-gpu-layers', '999'],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        time.sleep(1)
        try:
            r = requests.get(f'http://127.0.0.1:{PORT}/v1/models', timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False

# --- Parse functions ---
def first_bind(text):
    """Old parser: first <bind tool="X"/>"""
    for m in re.finditer(r'<bind\s+tool="([^"]+)"', text):
        return m.group(1)
    return None

def last_bind_with_args(text):
    """New parser: last <bind> that has a following <args>...<end/> block."""
    binds = list(re.finditer(r'<bind\s+tool="([^"]+)"', text))
    args = list(re.finditer(r'<args>(.*?)</end>', text, re.DOTALL))
    if not binds:
        return None, {}, text
    if args:
        last_a = args[-1]
        preceding_binds = [b for b in binds if b.end() <= last_a.start()]
        if preceding_binds:
            name = preceding_binds[-1].group(1)
            a_text = last_a.group(1).strip()
            if a_text:
                parsed = {}
                for p in re.finditer(r'(\w[\w\s]*?)\s*=\s*["\'](.+?)["\']', a_text):
                    parsed[p.group(1).strip()] = p.group(2)
                return name, parsed, text
    name = binds[-1].group(1)
    return name, {}, text

# --- Load 5 cases ---
cases = []
with open(ART / 'toolace_300.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        cases.append(json.loads(line))

if not ensure_server():
    print("ERROR: server failed")
    exit(1)

print("Server ready. Running 5 diagnostic cases...\n")

for idx, case in enumerate(cases):
    tools_json = json.dumps([t.get('function', t) for t in case['tools']], ensure_ascii=False)
    # Replicate eval prompt exactly
    msgs = [
        {"role": "user", "content": f"<tools>{tools_json}</tools>\n{case['query']}"},
    ]
    resp = requests.post(f'http://127.0.0.1:{PORT}/v1/chat/completions',
                         json={"model": "minicpm", "messages": msgs,
                               "max_tokens": 512, "temperature": 0.0},
                         timeout=120)
    raw = resp.json()['choices'][0]['message']['content']

    gold_name = case['gold'][0]['name']
    gold_args = case['gold'][0].get('args', {})

    fb_name = first_bind(raw)
    fb_correct = fb_name == gold_name

    lb_name, lb_args, _ = last_bind_with_args(raw)
    lb_correct = lb_name == gold_name

    print(f"--- Case {idx+1} ---")
    print(f"Gold: {gold_name}({gold_args})")
    print(f"Raw output (first 400 chars):")
    print(raw[:400])
    print(f"\nFirst-bind:  {fb_name}  correct={fb_correct}")
    print(f"Last-bind:   {lb_name}  args={lb_args}  correct={lb_correct}")
    print(f"Bind count:  {len(re.findall(r'<bind', raw))}")
    print(f"Args count:  {len(re.findall(r'<args>', raw))}")
    print()
