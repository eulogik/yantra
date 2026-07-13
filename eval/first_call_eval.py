#!/usr/bin/env python3
"""First-call evaluator — reproduces the community baseline's 300-case metrics
and measures Yantra against them.

Metrics (identical definitions to the source model card):
  parseable        : one complete call block extracted
  valid_name       : bound/emitted tool is in the available tool set
  expected_name    : tool == gold[0].name
  exact_args       : generated args dict == gold args dict (exact)
  arg_key_overlap  : |keys_gen ∩ keys_gold| / |keys_gold|
  no_repetition    : no repeated identical call blocks
  stopped_cleanly  : complete call then natural termination (EOS) at boundary

Usage:
  # baseline (legacy minicpm5 XML, no router binding):
  python first_call_eval.py --eval-set eval/data/toolace_300.jsonl \
      --base-url http://localhost:8000/v1 --model <baseline> --mode legacy

  # Yantra (DTSA, optional router pre-bind):
  python first_call_eval.py --eval-set eval/data/toolace_300.jsonl \
      --base-url http://localhost:8000/v1 --model yantra-1b --mode dtsa

  # no model/server available (pipeline self-test):
  python first_call_eval.py --eval-set eval/data/toolace_300.jsonl --mock
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------- parsing ----------
DTSA_BIND = re.compile(r'<bind\s+tool="([^"]+)"\s*/>', re.S)
DTSA_ARGS = re.compile(r"<args>(.*?)</args>", re.S)
DTSA_PARAM = re.compile(r'<param\s+name="([^"]+)"\s*>(.*?)</param>', re.S)
LEGACY_FN = re.compile(r"<function\s+name=\"([^\"]+)\"\s*>(.*?)</function>", re.S)
LEGACY_PARAM = re.compile(r'<param\s+name="([^"]+)"\s*>(.*?)</param>', re.S)


def _unescape(v: str) -> str:
    return v.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


def parse_dtsa(text: str):
    # Target the LAST complete call block (so multi-turn / recovery traces parse
    # the corrected call, not the earlier failed one). For single-call outputs
    # the last block is the only block, so eval behavior is unchanged.
    binds = list(DTSA_BIND.finditer(text))
    if not binds:
        return None
    last_bind = binds[-1]
    tool = last_bind.group(1)
    am = DTSA_ARGS.search(text[last_bind.end():])
    args = {}
    if am:
        for pm in DTSA_PARAM.finditer(am.group(1)):
            args[pm.group(1)] = _unescape(pm.group(2).strip())
    ends = list(re.finditer(r"<action_end/>", text))
    stopped_clean = bool(ends) and text[ends[-1].end():].strip() == ""
    return {"tool": tool, "args": args, "stopped_clean": stopped_clean}


def parse_legacy(text: str):
    m = LEGACY_FN.search(text)
    if not m:
        return None
    tool = m.group(1)
    args = {}
    for pm in LEGACY_PARAM.finditer(m.group(2)):
        args[pm.group(1)] = _unescape(pm.group(2).strip())
    tail = text[text.rfind("</function>") + len("</function>"):].strip()
    return {"tool": tool, "args": args, "stopped_clean": tail == ""}


def parse_call(text: str, mode: str):
    if mode == "dtsa":
        return parse_dtsa(text) or parse_legacy(text)
    return parse_legacy(text) or parse_dtsa(text)


def _norm_value(a) -> object:
    """Normalize LLM-emitted string args against gold typed values so that
    '2' == 2, '3.0' == 3.0, 'true' == True, etc."""
    if isinstance(a, str):
        s = a.strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            if "." in s:
                return float(s)
            return int(s)
        except ValueError:
            return s
    return a


def _vals_equal(gold_v, gen_v) -> bool:
    return _norm_value(gold_v) == _norm_value(gen_v)


def available_tool_names(tools: list) -> set:
    out = set()
    for t in tools:
        fn = t.get("function", t)
        if "name" in fn:
            out.add(fn["name"])
    return out


# ---------- generation ----------
def generate(client, model: str, prompt: str, temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=256,
    )
    return resp.choices[0].message.content or ""


def build_prompt(mode: str, query: str, tools: list) -> str:
    tool_json = json.dumps(
        [t.get("function", t) for t in tools], ensure_ascii=False, indent=2
    )
    if mode == "dtsa":
        return f"<user>{query}</user>\n<tools>{tool_json}</tools>\n<calls>"
    return f"<user>{query}</user>\n<tools>{tool_json}</tools>\n<calls>"


def mock_generate(query: str, gold: list, tools: list) -> str:
    """Deterministic pseudo-correct response for pipeline self-test."""
    g = gold[0]
    params = "".join(
        f'<param name="{k}">{v}</param>' for k, v in g["arguments"].items()
    )
    return f'<function name="{g["name"]}">{params}</function>'


# ---------- metrics ----------
def evaluate_case(text: str, case: dict, mode: str) -> dict:
    tools = case["tools"]
    gold = case["gold"]
    avail = available_tool_names(tools)
    gold0 = gold[0]

    parsed = parse_call(text, mode)
    parseable = parsed is not None
    if not parsed:
        return {
            "parseable": 0, "valid_name": 0, "expected_name": 0,
            "exact_args": 0, "arg_key_overlap": 0, "no_repetition": 1,
            "stopped_cleanly": 0,
        }

    valid = parsed["tool"] in avail
    expected = parsed["tool"] == gold0["name"]
    exact = parsed["args"] == gold0["arguments"] or (
        set(parsed["args"].keys()) == set(gold0["arguments"].keys())
        and all(_vals_equal(gold0["arguments"][k], parsed["args"][k])
                for k in gold0["arguments"])
    )
    gk, pk = set(gold0["arguments"].keys()), set(parsed["args"].keys())
    overlap = len(gk & pk) / len(gk) if gk else 1.0
    # repetition: more than one complete call block => repetition
    n_blocks = len(LEGACY_FN.findall(text)) + len(DTSA_BIND.findall(text))
    no_rep = 1 if n_blocks <= 1 else 0
    return {
        "parseable": 1, "valid_name": int(valid), "expected_name": int(expected),
        "exact_args": int(exact), "arg_key_overlap": round(overlap, 4),
        "no_repetition": no_rep, "stopped_cleanly": int(parsed["stopped_clean"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--mode", choices=["legacy", "dtsa"], default="legacy")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="eval/results/latest.json")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.eval_set) if l.strip()]

    client = None
    if not args.mock:
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url, api_key="sk-noauth")

    agg = {}
    per_case = []
    for i, case in enumerate(cases):
        prompt = build_prompt(args.mode, case["query"], case["tools"])
        if args.mock:
            text = mock_generate(case["query"], case["gold"], case["tools"])
        else:
            text = generate(client, args.model, prompt, args.temperature)
        m = evaluate_case(text, case, args.mode)
        per_case.append({"id": case.get("id", i), **m})
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v

    n = len(cases)
    summary = {k: round(v / n, 4) for k, v in agg.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_case": per_case}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
