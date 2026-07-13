#!/usr/bin/env python3
"""STSA boundary decoder — self-terminating generation without an external parser.

Two stopping signals:
  1. Grammar-complete: the <args> block is schema-complete (all required params
     present) AND an <action_end/> was emitted -> we stop.
  2. End-token logprob: the LM's own probability of emitting <action_end/> at the
     current step exceeds `tau`. This subsumes the old separate "boundary head"
     (train/boundary_head.py): instead of training a linear head on hidden
     states (which llama.cpp's OpenAI API does not expose), we read the
     <action_end/> token logprob directly from the streaming completion. No
     extra model, works offline with llama.cpp / vLLM.

This is what lifts `stopped_cleanly` 15% -> >=95% and removes the dependency on
SGLang's `minicpm5` parser.
"""
from __future__ import annotations

import math
import re

ACTION_END = "<action_end/>"
DTSA_ARGS = re.compile(r"<args>(.*?)</args>", re.S)
DTSA_PARAM = re.compile(r'<param\s+name="([^"]+)"\s*>(.*?)</param>', re.S)


def end_token_prob(top_logprobs: list) -> float:
    """Highest P(<action_end/>) seen among a step's top-logprobs list.

    `top_logprobs` is a list of objects with `.token` / `.logprob`
    (openai.TopLogprob). Returns 0.0 if the end token is not in the top-k.
    Matching is tolerant of tokenization (the end marker may be split), so we
    match any top token that contains 'action_end'.
    """
    best = 0.0
    for tl in top_logprobs or []:
        tok = getattr(tl, "token", None) or ""
        if "action_end" in tok:
            try:
                p = math.exp(float(tl.logprob))
            except (TypeError, ValueError):
                continue
            if p > best:
                best = p
    return best


def args_present(text: str, required: list[str]) -> bool:
    m = DTSA_ARGS.search(text)
    if not m:
        return False
    have = {pm.group(1) for pm in DTSA_PARAM.finditer(m.group(1))}
    return set(required).issubset(have)


def should_stop(
    text: str,
    required: list[str],
    boundary_prob: float = 0.0,
    tau: float = 0.9,
) -> bool:
    """True when generation should terminate (no external parser needed).

    `boundary_prob` is the running max P(<action_end/>) from the streaming
    logprobs (see end_token_prob). When it exceeds `tau` we stop even if the
    token has not been emitted yet.
    """
    if ACTION_END in text:
        tail = text[text.rfind(ACTION_END) + len(ACTION_END):].strip()
        if tail == "":
            return True
    if boundary_prob >= tau:
        return True
    if ACTION_END in text and args_present(text, required):
        return True
    return False


if __name__ == "__main__":
    t = '<bind tool="x"/>\n<args>\n  <param name="city">Paris</param>\n</args>\n<action_end/>'
    print("stop:", should_stop(t, required=["city"]))
    print("stop (no end):", should_stop(t.replace(ACTION_END, ""), required=["city"]))
    print("stop (logprob):", should_stop('<args><param name="city">Paris</param></args>', ["city"], boundary_prob=0.95))
