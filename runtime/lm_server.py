#!/usr/bin/env python3
"""LM server wrapper + DTSA request builder for standalone inference.

Supports llama.cpp (`llama-server`) and vLLM. The runtime:
  1. routes to a tool id (router.py)
  2. prepends `<bind tool="..."/>` (first turn only)
  3. streams the LM, stopping via boundary_decoder.py:
       - grammar-complete + explicit <action_end/>, OR
       - the LM's own P(<action_end/>) logprob exceeding `tau` (improvement #2:
         replaces the unusable hidden-state "boundary head" — llama.cpp's OpenAI
         API does not expose last-hidden-states, but it does expose token
         logprobs)
  4. validates the call (verifier.py) and executes externally

Improvement #4: `run_turn` adds a multi-turn recovery loop. On a failed /
low-reward call it feeds `<tool_error>` + `<reflect/>` back into the prompt so
the LM can emit a corrected call — exactly the RTE training format.
"""
from __future__ import annotations

import json
import subprocess

from router import ToolRouter
from verifier import ToolVerifier
from boundary_decoder import should_stop, end_token_prob


class YantraRuntime:
    def __init__(
        self,
        base_url: str,
        model: str,
        router: ToolRouter,
        verifier: ToolVerifier,
        tau: float = 0.9,
        top_logprobs: int = 10,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="sk-noauth")
        self.model = model
        self.router = router
        self.verifier = verifier
        self.tau = tau
        self.top_logprobs = top_logprobs

    # ---- prompt building ----
    def _build_prompt(
        self, query: str, tools: list[dict], context: str = "", bind: bool = True
    ) -> str:
        tool_json = json.dumps(
            [t.get("function", t) for t in tools], ensure_ascii=False, indent=2
        )
        prompt = f"<user>{query}</user>\n<tools>{tool_json}</tools>\n<calls>"
        if bind:
            prompt += self.router.bind_prefix(query, tools)
        return prompt + context

    # ---- schema lookup ----
    def _schema_for(self, tool_name: str, tools: list[dict]) -> dict:
        for t in tools:
            fn = t.get("function", t)
            if fn.get("name") == tool_name:
                return fn.get("parameters", {})
        return {}

    # ---- single action ----
    def act(
        self,
        query: str,
        tools: list[dict],
        context: str = "",
        bind: bool = True,
        max_tokens: int = 256,
    ) -> dict:
        prompt = self._build_prompt(query, tools, context=context, bind=bind)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            stream=True,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )
        text = ""
        boundary_prob = 0.0
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            text += delta
            lp = getattr(chunk.choices[0], "logprobs", None)
            if lp is not None and lp.top_logprobs:
                boundary_prob = max(boundary_prob, end_token_prob(lp.top_logprobs[0]))
            if should_stop(text, [], boundary_prob, self.tau):
                break
        # parse + verify (last complete DTSA block)
        from first_call_eval import parse_dtsa

        parsed = parse_dtsa(text) or {}
        tool_name = parsed.get("tool")
        args = parsed.get("args", {})
        schema = self._schema_for(tool_name, tools)
        reward, info = self.verifier.reward(tool_name, args, schema)
        return {
            "text": text,
            "tool": tool_name,
            "args": args,
            "reward": reward,
            "info": info,
        }

    # ---- improvement #4: multi-turn recovery loop ----
    def run_turn(self, query: str, tools: list[dict], max_turns: int = 3) -> dict:
        ctx = ""
        out = None
        for turn in range(max_turns):
            out = self.act(query, tools, context=ctx, bind=(turn == 0))
            if out["reward"] >= 1.0:
                return out  # success
            # feed the failure back so the LM can self-correct (RTE format)
            ctx = out["text"] + f'\n<tool_error>{out["info"]}</tool_error>\n<reflect/>\n'
        return out  # exhausted recovery attempts; return last attempt

    @staticmethod
    def launch_llamacpp(gguf: str, port: int = 8000) -> subprocess.Popen:
        cmd = ["llama-server", "-m", gguf, "--port", str(port)]
        return subprocess.Popen(cmd)


if __name__ == "__main__":
    print("YantraRuntime ready. Launch a server and call .act() / .run_turn().")
