#!/usr/bin/env python3
"""DTSA Tool Router V2 — IDF + char-3gram lexical scorer with optional embedding blend.

This is the canonical router shipped with Yantra. It matches the winning
configuration from the Colab eval notebooks:

- Lexical V2: IDF-weighted token overlap over ``name + description`` plus
  char-3gram name similarity (weight 2.0). Benchmarked at 244/300 on ToolACE-300.
- Embedding blend (optional): cosine similarity from ``all-MiniLM-L6-v2``
  (tool text = ``name — description``), per-query min-max normalized and blended
  as ``(1 - beta) * lexical + beta * embedding`` with ``beta = 0.5``.
  Benchmarked at 247/300 on ToolACE-300.

At inference the runtime calls ``route(query, tools)`` to pick the tool,
binds it via ``<bind tool="..."/>``, and only then asks the LM for ``<args>``.
See ``Yantra_EmbRouter_Eval.ipynb`` (Cell 3) for the benchmark harness.

Dependency-tolerant: works with zero extra installs (lexical only). If
``sentence-transformers`` is installed and ``model_name`` is given, the
embedding blend is enabled automatically.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


def _toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (s or "").lower()))


def _cgrams(s: str, n: int = 3) -> set[str]:
    s = re.sub(r"[^a-z0-9 ]", "", (s or "").lower())
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


@dataclass
class Tool:
    name: str
    description: str
    schema: dict = field(default_factory=dict)

    @staticmethod
    def from_def(t: dict) -> "Tool":
        fn = t.get("function", t)
        return Tool(
            fn.get("name", ""),
            fn.get("description", "") or "",
            fn.get("parameters", {}) or {},
        )


class ToolRouter:
    """Router V2 with optional embedding blend.

    Args:
        model_name: sentence-transformers model id (e.g.
            ``"sentence-transformers/all-MiniLM-L6-v2"``) or None for
            lexical-only routing.
        beta: blend weight for embeddings in [0, 1]. Ignored when no
            embedder is available. Winning value from the sweep: 0.5.
    """

    def __init__(self, model_name: Optional[str] = None, beta: float = 0.5):
        self.beta = beta
        self.embedder = None
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer

                self.embedder = SentenceTransformer(model_name)
            except Exception:
                self.embedder = None
        # Fitted corpus statistics (rebuilt per `route` call if tools change).
        self._tool_names: list[str] = []
        self._descs: dict[str, str] = {}
        self._docs: list[set[str]] = []
        self._df: Counter = Counter()
        self._n: int = 1
        self._tool_vectors = None  # normalized embedding matrix, aligned with _tool_names

    # -- corpus fitting -------------------------------------------------
    def fit_corpus(self, tools: list[dict]) -> None:
        """Pre-fit IDF (+ embedding cache) on a full tool registry.

        When the router is pre-fitted, per-query ``route`` calls over subsets
        reuse the global document frequencies instead of refitting on the
        small per-query list (matches the eval notebook, where IDF is built
        over all unique tools in ToolACE-300).
        """
        self._tool_names = []
        self._fit(tools, force=True)

    def _fit(self, tools: list[dict], force: bool = False) -> None:
        names = [Tool.from_def(t).name for t in tools]
        if not force and names == self._tool_names and self._tool_names:
            return
        if (
            not force
            and self._tool_names
            and all(n in self._tool_names for n in names)
        ):
            # Already fitted on a superset registry: keep global DF.
            return
        self._tool_names = names
        self._descs = {Tool.from_def(t).name: Tool.from_def(t).description for t in tools}
        self._docs = [_toks(n + " " + self._descs.get(n, "")) for n in names]
        self._df = Counter(w for d in self._docs for w in d)
        self._n = max(len(self._docs), 1)
        self._tool_vectors = None  # invalidate embedding cache
        if self.embedder is not None and names:
            try:
                import numpy as np

                texts = [n + " — " + self._descs.get(n, "") for n in names]
                vecs = np.asarray(
                    self.embedder.encode(
                        texts, show_progress_bar=False, normalize_embeddings=True
                    ),
                    dtype=float,
                )
                self._tool_vectors = vecs
            except Exception:
                self._tool_vectors = None

    def _idf(self, w: str) -> float:
        return math.log(self._n / (1 + self._df.get(w, 0)))

    def _lex_scores(self, query: str, names: list[str]) -> list[float]:
        qt = _toks(query)
        qg = _cgrams(query)
        out: list[float] = []
        for nm in names:
            d = _toks(nm + " " + self._descs.get(nm, ""))
            denom = math.sqrt(sum(self._idf(w) for w in d) + 1e-6)
            s1 = sum(self._idf(w) for w in qt & d) / denom if denom else 0.0
            ng = _cgrams(nm)
            s2 = len(qg & ng) / (math.sqrt(len(qg) * len(ng)) + 1) if qg and ng else 0.0
            out.append(s1 + 2.0 * s2)
        return out

    @staticmethod
    def _minmax(xs: list[float]) -> list[float]:
        if not xs:
            return xs
        lo, hi = min(xs), max(xs)
        if hi - lo < 1e-12:
            return [0.0 for _ in xs]
        return [(x - lo) / (hi - lo) for x in xs]

    # -- public API -----------------------------------------------------
    def route(self, query: str, tools: list[dict], top_k: int = 1) -> list[str]:
        """Return the top-k tool names for *query* (best first)."""
        if not tools:
            return []
        self._fit(tools)
        names = [Tool.from_def(t).name for t in tools]
        lex = self._lex_scores(query, names)

        emb: Optional[list[float]] = None
        if self.embedder is not None and self._tool_vectors is not None:
            try:
                import numpy as np

                q = np.asarray(
                    self.embedder.encode(
                        [query], show_progress_bar=False, normalize_embeddings=True
                    ),
                    dtype=float,
                )[0]
                name2col = {n: j for j, n in enumerate(self._tool_names)}
                cols = [name2col[n] for n in names]
                emb = [float(q @ self._tool_vectors[j]) for j in cols]
            except Exception:
                emb = None

        if emb is None or self.beta <= 0:
            scored = sorted(zip(lex, names), reverse=True)
        else:
            lex_n = self._minmax(lex)
            emb_n = self._minmax(emb)
            blended = [
                (1 - self.beta) * l + self.beta * e
                for l, e in zip(lex_n, emb_n)
            ]
            scored = sorted(zip(blended, names), reverse=True)
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
