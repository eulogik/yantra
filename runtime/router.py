#!/usr/bin/env python3
"""DTSA Tool Router — selects the tool id so the LM never has to recall it.

At inference the runtime calls `route(query, tools)` to get the top-k tool id(s),
binds the best one via `<bind tool="..."/>`, and only then asks the LM for args.

Dependency-tolerant: uses sentence-transformers if present, else a fast
BM25/keyword fallback so the runtime works with zero extra installs.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Tool:
    name: str
    description: str
    schema: dict

    @staticmethod
    def from_def(t: dict) -> "Tool":
        fn = t.get("function", t)
        return Tool(fn.get("name", ""), fn.get("description", ""), fn.get("parameters", {}))


def _tokenize(s: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", s.lower())


class ToolRouter:
    def __init__(self, model_name: Optional[str] = None):
        self.embedder = None
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer

                self.embedder = SentenceTransformer(model_name)
            except Exception:
                self.embedder = None

    def _score(self, query: str, tool: Tool) -> float:
        if self.embedder is not None:
            q = self.embedder.encode([query])[0]
            d = self.embedder.encode([tool.name.replace(".", " ") + " " + tool.description])[0]
            return float(q @ d) / (math.norm(q) * math.norm(d) + 1e-9)
        # fallback: BM25-ish lexical overlap
        q_tokens = set(_tokenize(query))
        d_tokens = set(_tokenize(tool.name + " " + tool.description))
        if not q_tokens or not d_tokens:
            return 0.0
        overlap = len(q_tokens & d_tokens)
        return overlap / math.sqrt(len(q_tokens) * len(d_tokens))

    def route(self, query: str, tools: list[dict], top_k: int = 1) -> list[str]:
        scored = [(self._score(query, Tool.from_def(t)), Tool.from_def(t).name) for t in tools]
        scored.sort(reverse=True)
        return [name for _, name in scored[:top_k]]

    def bind_prefix(self, query: str, tools: list[dict]) -> str:
        best = self.route(query, tools, top_k=1)[0]
        return f'<bind tool="{best}"/>\n'


if __name__ == "__main__":
    tools = [
        {"function": {"name": "weather.get_forecast", "description": "get weather forecast for a city", "parameters": {}}},
        {"function": {"name": "calc.add", "description": "add two numbers", "parameters": {}}},
    ]
    r = ToolRouter()
    print(r.bind_prefix("What is the weather in Paris?", tools))
