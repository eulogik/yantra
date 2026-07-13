#!/usr/bin/env python3
"""Convert Team-ACE/ToolACE examples into Yantra DTSA training format.

DTSA (Decoupled Tool Selection / Argument generation) format:

    <bind tool="weather.get_forecast"/>
    <args>
      <param name="city">Paris</param>
      <param name="days">3</param>
    </args>
    <action_end/>

The tool id is bound structurally (by the runtime router), so the LM only
learns to emit well-formed <args>...</args><action_end/> blocks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_answers(answers: Any) -> list[dict]:
    """ToolACE `answers` may be a JSON string or an already-parsed list."""
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except json.JSONDecodeError:
            return []
    if not isinstance(answers, list):
        return []
    out = []
    for a in answers:
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except json.JSONDecodeError:
                continue
        if isinstance(a, dict) and "name" in a:
            out.append({"name": a["name"], "arguments": a.get("arguments", {}) or {}})
    return out


def to_dtsa(name: str, arguments: dict) -> str:
    lines = ["<bind tool=\"%s\"/>" % name, "<args>"]
    for k, v in arguments.items():
        v = "" if v is None else str(v)
        # escape XML special chars in values
        v = v.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append('  <param name="%s">%s</param>' % (k, v))
    lines.append("</args>")
    lines.append("<action_end/>")
    return "\n".join(lines)


def build_user_prompt(query: str, tools: list[dict]) -> str:
    """Render a minicpm5-style tool-aware prompt (compatible with the base)."""
    tool_json = json.dumps(
        [t.get("function", t) for t in tools], ensure_ascii=False, indent=2
    )
    return (
        "<user>%s</user>\n"
        "<tools>%s</tools>\n"
        "<calls>" % (query, tool_json)
    )


def convert_example(ex: dict) -> dict | None:
    queries = ex.get("queries") or []
    tools = ex.get("tools") or []
    if not queries or not tools:
        return None
    query = queries[0]
    calls = parse_answers(ex.get("answers"))
    if not calls:
        return None
    # For SFT we train one assistant turn per gold call (kept simple/flat).
    dtsa_blocks = "\n".join(to_dtsa(c["name"], c["arguments"]) for c in calls)
    return {
        "query": query,
        "tools": tools,
        "gold": calls,
        "prompt": build_user_prompt(query, tools),
        "completion": dtsa_blocks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Team-ACE/ToolACE", help="HF dataset or local jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    records: list[dict] = []
    if Path(args.src).exists():
        with open(args.src) as f:
            raw = [json.loads(l) for l in f if l.strip()]
    else:
        from datasets import load_dataset

        ds = load_dataset(args.src, split=args.split)
        raw = list(ds)

    for ex in raw:
        rec = convert_example(ex)
        if rec:
            records.append(rec)
        if args.limit and len(records) >= args.limit:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} DTSA records -> {args.out}")


if __name__ == "__main__":
    main()
