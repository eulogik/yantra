#!/usr/bin/env python3
"""Stage 4 — Boundary-head training (STSA).

Trains a linear head on the LM's last hidden state to predict P(end-of-action)
so generation can self-terminate (lifting stopped_cleanly 15% -> >=95%). Labels
are 1 at the position of `<action_end/>` and 0 elsewhere.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn


class BoundaryHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden).sigmoid().squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="DTSA jsonl with <action_end/> labels")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="train/outputs/boundary_head.pt")
    args = ap.parse_args()
    # TODO: freeze LM, attach BoundaryHead to last hidden, BCE loss on positions
    # where label==1 at <action_end/>, 0 otherwise. Save head.
    raise NotImplementedError("wire boundary-head training here")


if __name__ == "__main__":
    main()
