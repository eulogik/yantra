#!/usr/bin/env python3
"""LM server wrapper + DTSA request builder for standalone inference.

Supports llama.cpp (`llama-server`) and vLLM. The runtime:
  1. routes to a tool id (router.py)
  2. prepends `<bind tool="..."/>`
  3. streams the LM, stopping via boundary_decoder.py (no SGLang parser)
  4. validates the call (verifier.py) and executes externally
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from router import ToolRouter
from verifier import ToolVerifier
from boundary_decoder import should_stop


class NanoAgentRuntime:
    def __init__(self, base_url: str, model: str, router: ToolRouter, verifier: ToolVerifier):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="sk-noauth")
        self.model = model
        self.router = router
        self.verifier = verifier

    def act(self, query: str, tools: list[dict], max_tokens: int = 256) -> dict:
        bind = self.router.bind_prefix(query, tools)
        prompt = f"<user>{query}</user>\n<tools>{tools}</tools>\n<calls>{bind}"
        # stream and stop via boundary decoder
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            stream=True,
        )
        text = ""
        required = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            text += delta
            if should_stop(text, required):
                break
        # parse + verify
        from first_call_eval import parse_dtsa  # reuse parser

        parsed = parse_dtsa(text) or {}
        tool_name = parsed.get("tool")
        args = parsed.get("args", {})
        # find schema
        schema = {}
        for t in tools:
            fn = t.get("function", t)
            if fn.get("name") == tool_name:
                schema = fn.get("parameters", {})
                required = schema.get("required", [])
                break
        reward, info = self.verifier.reward(tool_name, args, schema)
        return {"text": text, "tool": tool_name, "args": args,
                "reward": reward, "info": info}

    @staticmethod
    def launch_llamacpp(gguf: str, port: int = 8000) -> subprocess.Popen:
        cmd = ["llama-server", "-m", gguf, "--port", str(port)]
        return subprocess.Popen(cmd)


if __name__ == "__main__":
    print("NanoAgentRuntime ready. Launch a server and call .act().")
