# NanoAgent — HANDOFF / LIVING DOC

> Single source of truth for plan + progress. Update this file at the end of
> every work session. Keep the "Status" table honest: red / amber / green.

**Goal:** A ≤1B agentic model that *clearly beats* `ewinregirgojr/MiniCPM5-1B-Agentic-Tooluse-GGUF` in practical usage. Paper/patent-worthiness is secondary and only matters once the primary target holds.

**Method (4 mechanisms):** DTSA · EG-OPD · STSA · RTE. See `../implementation-doc.md`.

---

## 0. Status (updated 2026-07-10)

| Workstream | Status | Notes |
|---|---|---|
| Repo scaffold + gitignore + secured creds | ✅ done | `credentials.env` gitignored; never commit |
| Data pipeline (convert / eval-set / rte / holdout) | ✅ done | runs; needs HF network for full ToolACE |
| 300-case evaluator + PAS | ✅ done | verified in `--mock` mode |
| Runtime (router / verifier / boundary / lm_server) | 🟡 partial | modules written, not run against a live server |
| Training stages (sft/eg-opd/rte/boundary/router/distill) | 🟡 stubs | real loss fn + argparsers; GPU loops TODO |
| Reproduce baseline numbers | ⛔ blocked | needs GPU + model download (see §3) |
| Train NanoAgent-1B | ⛔ blocked | depends on GPU + baseline repro |
| Publish (HF private / GH private) | ⛔ not started | creds ready in `credentials.env` |

---

## 1. Plan (ordered)

1. **Reproduce baseline first** (hard gate). Build `eval/data/toolace_300.jsonl`, serve `ewinregirgojr/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q4_K_M` via llama.cpp, run `first_call_eval.py --mode legacy`. Expect ~0.993/0.97/0.927/0.653/0.752/0.15. If we can't match, our evaluator is wrong.
2. **Stage 1 DTSA SFT** on MiniCPM5-1B (QLoRA). Validate format learned on a tiny eval.
3. **Stage 2 EG-OPD** — wire `runtime/verifier.ToolVerifier` into the loss (`train/eg_opd.eg_opd_loss`). Teacher = local Qwen2.5-7B or OpenRouter free model (zero balance → free only).
4. **Stage 3 RTE SFT** on `data/rte.jsonl`.
5. **Stage 4 boundary head** train + plug into `runtime/boundary_decoder.should_stop`.
6. **Eval NanoAgent** on 300-case + generalization + recovery + multi-turn → compute PAS. Must clear §targets.
7. **Quantize** to F16/Q8/Q4_K_M; re-eval (Q4 delta ≤2pt).
8. **(stretch) Sub-1B** distill → `train/distill_sub1b.py`.
9. **Publish** HF (private, org `eulogik`) + GH (private, org `eulogik`). No public until explicit go-ahead.

---

## 2. Progress log

### 2026-07-10 (session 1)
- Created `nanogent/` repo: README, .gitignore, `credentials.env` (gitignored).
- Data pipeline: `convert_toolace_dtsa.py`, `build_eval_set.py`, `build_rte_corpus.py`, `holdout_split.py`.
- Eval: `first_call_eval.py` (legacy+DTSA parser, value-normalized exact_args) + `pas.py`. Verified end-to-end in `--mock` mode on a 2-case set → PAS computes correctly.
- Runtime: `router.py` (BM25 fallback + sentence-transformers), `verifier.py` (schema + mock exec → reward), `boundary_decoder.py` (STSA stop logic), `lm_server.py` (NanoAgentRuntime).
- Training: `sft_dtsa.py` (build_messages), `eg_opd.py` (real `eg_opd_loss`), `rte_sft.py`, `boundary_head.py`, `router_train.py`, `distill_sub1b.py` — all with argparsers + TODO loops.
- Config: `configs/config.yaml`. Requirements: `requirements.txt`.

**Next action:** obtain GPU + download baseline model to run Step 1 (reproduce baseline).

---

## 3. Blockers / decisions needed

- **GPU access:** training + baseline reproduction need a CUDA GPU (≥24GB for QLoRA 1B). Where do we run it? (local Mac = no; need a runner or cloud.)
- **Network:** full ToolACE download + model weights need connectivity. Confirm HF token in `credentials.env` is usable.
- **Teacher for EG-OPD:** OpenRouter has zero balance (free models only) — a 7B free teacher is weak. **Decision (2026-07-10): train FULLY OFFLINE with a LOCAL teacher** (Qwen2.5-7B-Instruct Q4 ~4.6GB via vLLM/llama.cpp). Local vLLM returns `top_logprobs=50` for free — exactly what `eg_opd_loss` needs. Fallback if no local GPU: one-time remote cache on Groq (~$8 / ~7M tok) then offline KD. OpenRouter-free is NOT viable as teacher.
- **Offline feasibility:** Stages 1/3/4 are supervised (no teacher, $0, offline). Only Stages 2 (EG-OPD) and 6 (sub-1B distill) need the teacher. EG-OPD buys the 65%->85% exact_args jump; everything else beats baseline offline.
- **Publishing:** both HF and GH repos stay **private** until explicit permission. Org = `eulogik`.

---

## 6. Hardware: Mac Mini M4 16GB (decided 2026-07-10)

Whole project is feasible locally — no GPU/cloud needed.

- Memory: 1B Q4 ~0.7GB, 1B LoRA train (MLX) ~3-4GB, 7B-Q4 teacher ~5GB. Run teacher sequentially, not alongside training.
- Stack: `mlx-lm` for student LoRA training (replaces unsloth/peft/bitsandbytes which are CUDA-only); `llama.cpp` (Metal) for inference + teacher logprobs + eval.
- **Teacher decision: use VERIFIER-DRIVEN SELF-DISTILLATION, not an external LLM teacher.** Sample k calls from student/base, score with `runtime/verifier.ToolVerifier` (reward r), DPO between correct/rejected self-samples weighted by r. Keeps EG-OPD's "execution-grounded, verifier-weighted" essence; needs ZERO external teacher, ZERO second model, fully offline, fits 16GB. Avoids cross-tokenizer logit-KD mismatch (MiniCPM5 vs Qwen). Optional 7B-Q4 llama.cpp teacher only for hard cases (sequential).
- Timing: baseline repro minutes; SFT ~3-6h; self-distill ~2-4h; full pipeline ~1-2 days wall-clock.
- TODO: convert MiniCPM5-1B HF->MLX; add mlx_lm SFT launcher; add llama.cpp eval server script; rewrite train/eg_opd.py as verifier-weighted self-distill/DPO.

## 7. Expected outcomes (set 2026-07-10, realistic not aspirational)

Baseline PAS ~= 0.56 (corrected: (0.993+0.97+0.927+0.653+0.752+0.15+0+0)/8).

| Metric | Baseline | Realistic | Stretch |
|---|---|---|---|
| parseable | 0.993 | 0.99-1.00 | 1.00 |
| valid_name | 0.970 | 0.99-1.00 | 1.00 |
| expected_name | 0.927 | 0.93-0.97 | 0.97 |
| exact_args | 0.653 | 0.75-0.85 | 0.85 |
| arg_key_overlap | 0.752 | 0.90-0.96 | 0.96 |
| stopped_cleanly | 0.150 | 0.92-0.98 | 0.98 |
| recovery | 0 | 0.70-0.85 | 0.85 |
| multiturn | 0 | 0.45-0.65 | 0.70 |
| PAS | ~0.56 | 0.80-0.88 | 0.88 |

Verdict: clearly MORE USABLE than baseline (parser-free, self-stopping, error-recovering, same ~688MB Q4). Still a 1B model — will not rival 7B-70B on hard generalization/long planning. biggest risks: exact_args (needs strong verifier), router wrong-bind. Paper: solid arXiv/workshop, incremental not breakthrough.

## 4. How to run (current state)

```bash
cd nanogent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# self-test evaluator (no model needed)
python3 eval/first_call_eval.py --eval-set /tmp/tiny_eval.jsonl --mock
python3 eval/pas.py --results /tmp/tiny_results.json

# once you have network + a 300-case set:
python3 eval/build_eval_set.py --out eval/data/toolace_300.jsonl --n 300 --seed 42
# then serve a model and:
python3 eval/first_call_eval.py --eval-set eval/data/toolace_300.jsonl \
    --base-url http://localhost:8000/v1 --model <model> --mode legacy
```

---

## 5. Credentials (DO NOT COMMIT)

Stored in `credentials.env` (gitignored). Loaded at runtime via `python-dotenv` or shell `export $(cat credentials.env | xargs)`.

- OpenRouter (free only)
- GitHub `eulogik` (two tokens: standard + workflow)
- PyPI
- HuggingFace `eulogik` (private publish)

Never paste these in chat, logs, or model card.
