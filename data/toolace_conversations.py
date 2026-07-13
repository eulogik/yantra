#!/usr/bin/env python3
"""Convert the live Team-ACE/ToolACE ShareGPT format into the shape the rest of
Yantra expects: {query, tools:[{function:...}], gold:[{name, arguments}]}.

The live dataset (Team-ACE/ToolACE on the Hub) stores each example as:
  - `system`  : instruction + the tool JSONSchema array, ending right before
                "Should you decide to return the function call(s)."
  - `conversations`: [user, assistant(call), tool(result), ...]

Assistant calls are bracketed: `[Func Name(arg="v", n=1), Other(x=[1,2])]`.
This is a heuristic parser robust enough for the single-turn / single-call
examples the eval set selects.
"""
from __future__ import annotations

import json
import re
from typing import Any

TOOLS_MARKER = "Should you decide to return the function call"


def extract_tools(system: str) -> list[dict]:
    if not system:
        return []
    m = re.search(r"(\[\s*\{.*\}\])\s*[.\s]*" + re.escape(TOOLS_MARKER), system, re.S)
    if not m:
        return []
    arr_text = m.group(1).strip()
    try:
        return json.loads(arr_text)
    except json.JSONDecodeError:
        return []


def _split_top(s: str) -> list[str]:
    """Split on commas that are not inside quotes or []/{} brackets."""
    out, depth, buf = [], 0, ""
    quote = None
    for ch in s:
        if quote:
            buf += ch
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf += ch
        elif ch in "[{":
            depth += 1
            buf += ch
        elif ch in "]}":
            depth -= 1
            buf += ch
        elif ch == "," and depth == 0:
            out.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf)
    return out


def _parse_value(v: str) -> Any:
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (
        v.startswith("'") and v.endswith("'")
    ):
        return v[1:-1]
    low = v.lower()
    if low in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[low]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def parse_kwargs(s: str) -> dict:
    args: dict[str, Any] = {}
    if not s.strip():
        return args
    for part in _split_top(s):
        if "=" not in part:
            continue
        k, val = part.split("=", 1)
        args[k.strip()] = _parse_value(val)
    return args


def parse_calls(text: str) -> list[dict]:
    """Parse `[Func(a="x"), Func2(b=1)]`-style assistant messages."""
    res: list[dict] = []
    for seg in re.findall(r"\[([^\[\]]*)\]", text):
        for m in re.finditer(r"([A-Za-z0-9_ ]+?)\s*\(([^\(\)]*)\)", seg):
            name = m.group(1).strip()
            if not name:
                continue
            res.append({"name": name, "arguments": parse_kwargs(m.group(2))})
    return res


def extract_example(ex: dict) -> dict | None:
    """Return {query, tools, gold} or None if not a clean single-call example."""
    conv = ex.get("conversations") or []
    tools = extract_tools(ex.get("system", ""))
    if not tools:
        return None
    user_turn = next((c for c in conv if c.get("from") == "user"), None)
    if not user_turn:
        return None
    query = user_turn["value"]
    gold: list[dict] = []
    for c in conv:
        if c.get("from") == "assistant":
            calls = parse_calls(c["value"])
            if calls:
                gold = calls
                break
    if not gold:
        return None
    return {
        "query": query,
        "tools": [{"function": t} for t in tools],
        "gold": gold,
    }


if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("Team-ACE/ToolACE", split="train")
    ok = 0
    for ex in ds:
        if extract_example(ex):
            ok += 1
    print(f"parseable examples: {ok}/{len(ds)}")
