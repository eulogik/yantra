#!/usr/bin/env python3
"""Stage 2 — EG-OPD: Execution-Grounded On-Policy Distillation.

Extends On-Policy Distillation (Thinking Machines Lab) by making the
distillation advantage an *executed* tool-outcome from the verifier.

For each student-generated call we run runtime/verifier.ToolVerifier and get a
reward r in {0, 0.5, 1}. The loss weights the reverse-KL between student and
teacher top-k logits by r, so failed executions are up-weighted and the teacher
logits act as the correction target.

    L = sum_t  r_t * KL( student_topk || teacher_topk )_t
"""
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F


def eg_opd_loss(
    student_logits: torch.Tensor,   # (T, V)
    teacher_logits: torch.Tensor,   # (T, V)
    reward: torch.Tensor,           # (T,) in {0, .5, 1}
    k: int = 50,
) -> torch.Tensor:
    """Reverse-KL on the union of top-k student/teacher tokens, reward-weighted."""
    T = student_logits.shape[0]
    loss = torch.zeros((), requires_grad=True)
    for t in range(T):
        s_topk = torch.topk(student_logits[t], k).indices
        t_topk = torch.topk(teacher_logits[t], k).indices
        union = torch.unique(torch.cat([s_topk, t_topk]))
        ps = F.softmax(student_logits[t][union], dim=-1)
        pt = F.softmax(teacher_logits[t][union], dim=-1)
        # reverse KL: sum p_teacher * log(p_teacher / p_student)
        kl = (pt * (pt.clamp_min(1e-8).log() - ps.clamp_min(1e-8).log())).sum()
        loss = loss + reward[t] * kl
    return loss / T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True, help="strong teacher (e.g. Qwen2.5-7B)")
    ap.add_argument("--data", required=True, help="DTSA SFT jsonl")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    # TODO: load student (LoRA) + teacher (frozen); for each batch, generate,
    # run ToolVerifier to get r, compute eg_opd_loss, backward. Resume from
    # Stage-1 LoRA.
    raise NotImplementedError("wire EG-OPD training loop here")


if __name__ == "__main__":
    main()
