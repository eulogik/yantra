<div align="center">

# 🧠 Yantra

### A Sub-1B Agentic Tool-Calling Model That Actually Works

**3.52× baseline accuracy** on ToolACE-300 · **100% parseable output** · **Zero training-time API calls** · **Fully reproducible on a free Colab T4**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-yantra--1b--agent-blue)](https://huggingface.co/eulogik/yantra-1b-agent)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eulogik/yantra/blob/main/Yantra_T4_pipeline.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](#)

<br/>

```
  ╔══════════════════════════════════════════════════════════════╗
  ║  Yantra: A Reproducible Pipeline for Training              ║
  ║  Sub-1B Agentic Tool-Calling Models                        ║
  ║                                                            ║
  ║  Base: MiniCPM5-1B · QLoRA · DTSA · EG-OPD · RTE          ║
  ║  Eval: ToolACE-300 · PAS Metric · Q4_K_M GGUF             ║
  ║  Result: 0.6775 PAS (3.52× baseline 0.1922)                ║
  ╚══════════════════════════════════════════════════════════════╝
```

</div>

---

## 📊 Results

| Metric | Baseline | Yantra | Δ |
|---|---|---|---|
| Parseable | 0.0000 | **1.0000** | +100% |
| Valid Tool Name | 0.0000 | **1.0000** | +100% |
| Expected Tool (w/ router) | 0.5567 | **0.8233** | +48% |
| Exact Args | 0.0000 | **0.6500** | +65% |
| Arg Key Overlap | 0.0000 | **0.9467** | +95% |
| Stopped Cleanly | 0.0000 | **1.0000** | +100% |
| **PAS (Primary)** | **0.1922** | **0.6775** | **+253%** |

> **PAS** = average of 8 sub-metrics: parseable, valid_name, expected_name, exact_args, arg_key_overlap, stopped_cleanly, recovery, multiturn. See [Evaluation](#-evaluation) for details.

### Routing Accuracy

| Router | Accuracy | Method |
|---|---|---|
| Lexical (IDF + char-3gram) | 244/300 (81.3%) | BM25-inspired |
| Pure Embedding (MiniLM) | 242/300 (80.7%) | cosine similarity |
| **Blend (β=0.5)** | **247/300 (82.3%)** | lexical + embedding |

---

## 🚀 Quick Start

### Option 1: Run the Full Pipeline (Colab T4)

```python
# In Google Colab (Runtime → Run all, ~2-3 hours total):
!git clone https://github.com/eulogik/yantra.git
%cd yantra
!pip install -r requirements.txt

# Or open the notebook directly:
# https://colab.research.google.com/github/eulogik/yantra/blob/main/Yantra_T4_pipeline.ipynb
```

### Option 2: Use the Trained Model (2 minutes)

```python
from llama_cpp import Llama

llm = Llama(model_path="base_model.Q4_K_M.gguf", n_ctx=4096, n_gpu_layers=-1)

prompt = """<user>Find me a VR game for Oculus Quest</user>
<tools>[{"name": "getVRGame", "description": "Search VR games", "parameters": {"properties": {"platform": {"type": "string"}, "genre": {"type": "string"}}}}]
<calls><bind tool="getVRGame"/>
"""

output = llm(prompt, max_tokens=512, stop=["</calls>"])
print(output["choices"][0]["text"])
# → <args><param name="platform">Oculus Quest</param><param name="genre">adventure</param></args><action_end/>
```

### Option 3: Evaluate the Model

```bash
# Run the embedding-router evaluation notebook:
# Yantra_EmbRouter_Eval.ipynb (Colab, ~10 min)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Yantra Pipeline                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  Stage 1  │───▶│  Stage 3  │───▶│  Stage 4  │         │
│  │   Data    │    │  DTSA     │    │  EG-OPD   │         │
│  │  Builder  │    │   SFT     │    │   DPO     │         │
│  └──────────┘    └──────────┘    └──────────┘         │
│       │               │               │                 │
│       ▼               ▼               ▼                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │ ToolACE   │    │  QLoRA   │    │  Self-   │         │
│  │ 1988+     │    │  r=64    │    │  Distill │         │
│  │ pairs     │    │  α=128   │    │  + Exec  │         │
│  └──────────┘    └──────────┘    └──────────┘         │
│       │               │               │                 │
│       ▼               ▼               ▼                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  Stage 5  │───▶│  Stage 6  │───▶│   Eval   │         │
│  │   RTE     │    │  GGUF    │    │  300     │         │
│  │   SFT     │    │  Export  │    │  Cases   │         │
│  └──────────┘    └──────────┘    └──────────┘         │
│                                                         │
│  ┌─────────────────────────────────────────────┐       │
│  │           Router V2 (IDF + Embed)           │       │
│  │  User Query → Route → Bind → Args → Stop    │       │
│  └─────────────────────────────────────────────┘       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### DTSA: Decoupled Tool Selection / Argument Generation

The key insight: **separate tool routing from argument generation**.

1. **Router** selects the tool (IDF + char-3gram + embedding blend)
2. **Model** generates only the arguments (not the tool name)
3. **Stop tokens** ensure clean termination (no hallucinated follow-ups)

This decoupling means:
- Router accuracy (82.3%) is independent of model quality
- Model only needs to learn argument formatting, not tool selection
- Easy to swap routers (lexical → embedding → LLM-based)

---

## 📁 Repository Structure

```
yantra/
├── Yantra_T4_pipeline.ipynb          # Main pipeline (Colab T4)
├── Yantra_EmbRouter_Eval.ipynb      # Embedding router eval
├── Yantra_SFT2_Replay_Export_Eval.ipynb  # SFT replay experiment
├── Yantra_NextStep_StopsFixed_DPO2.ipynb # DPO experiment
├── Yantra_DPO2_Train_Export_Eval.ipynb   # DPO training
├── Yantra_Eval_RouterV2_Standalone.ipynb # Standalone eval
│
├── data/
│   ├── toolace_conversations.py     # ToolACE extractor
│   ├── convert_toolace_dtsa.py      # DTSA converter
│   ├── build_rte_corpus.py          # RTE data builder
│   └── holdout_split.py             # Train/val split
│
├── train/
│   ├── sft_dtsa.py                  # Stage 3: DTSA SFT
│   ├── eg_opd.py                    # Stage 4: EG-OPD DPO
│   ├── rte_sft.py                   # Stage 5: RTE SFT
│   └── ...
│
├── eval/
│   ├── build_eval_set.py            # 300-case builder
│   ├── first_call_eval.py           # Evaluator
│   └── pas.py                       # PAS metric
│
├── runtime/
│   ├── router.py                    # Router V2
│   ├── verifier.py                  # Schema verifier
│   └── boundary_decoder.py          # Stop logic
│
├── configs/
│   └── config.yaml                  # Training config
│
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🔧 Evaluation

### PAS (Primary Aggregated Score)

PAS averages 8 sub-metrics equally:

| Metric | Description |
|---|---|
| `parseable` | Output contains valid DTSA structure |
| `valid_name` | Predicted tool name exists in the tool list |
| `expected_name` | Predicted tool matches gold standard |
| `exact_args` | All argument keys AND values match exactly |
| `arg_key_overlap` | Fraction of gold argument keys present |
| `stopped_cleanly` | Output ends with `<action_end/>` |
| `recovery` | Corrected output after tool error (future) |
| `multiturn` | Multi-turn conversation support (future) |

### Eval Set

- **300 cases** from ToolACE (deterministic seed 42)
- Single-call only (no multi-tool scenarios)
- Balanced across tool categories

---

## 🧪 Experiments

### Experiment 1: DPO on 66 Pairs (FAILED)

**Hypothesis:** Contrastive learning on 66 arg-failures improves argument accuracy.

**Result:** DPO collapsed. Model learned to emit one blob param keyed by tool name instead of per-key params. `exact_args` dropped 0.65 → 0.14.

**Lesson:** 66 pairs is too few for DPO at `lr=5e-5` × 3 epochs. Small-data contrastive learning collapses.

### Experiment 2: SFT Replay (SUCCESS)

**Hypothesis:** Supervised fine-tuning on 66 fixes + 200 replay rows improves args without collapse.

**Result:** +0.0028 PAS (0.6747 → 0.6775). All metrics improved. No collapse.

**Lesson:** SFT-only is safe for small datasets. Replay prevents forgetting.

### Experiment 3: Embedding Router (SUCCESS)

**Hypothesis:** Blending IDF+char-3gram with MiniLM embeddings improves routing accuracy.

**Result:** +3 routing accuracy (244 → 247/300). +0.0028 PAS.

**Lesson:** Embeddings catch semantic gaps (e.g., "VR game" → `getVRGame`) that lexical matching misses.

---

## 📦 Model Details

| Property | Value |
|---|---|
| **Base Model** | [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) |
| **Training** | QLoRA (r=64, α=128, dropout=0.05) |
| **Quantization** | Q4_K_M (688 MB) |
| **Context Length** | 4096 tokens |
| **Max Output** | 768 tokens |
| **License** | MIT |
| **HF Repo** | [eulogik/yantra-1b-agent](https://huggingface.co/eulogik/yantra-1b-agent) |

### Training Pipeline

| Stage | Method | Data | Epochs | LR | Time |
|---|---|---|---|---|---|
| 1 | Data Builder | ToolACE | — | — | 5 min |
| 3 | DTSA SFT | 1988 pairs | 2 | 1e-4 | 30 min |
| 4 | EG-OPD DPO | 7795 pairs | 1 | 5e-5 | 2 hr |
| 5 | RTE SFT | error corrections | 1 | 1e-4 | 15 min |
| 6 | GGUF Export | Q4_K_M | — | — | 10 min |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Citation

```bibtex
@software{yantra2026,
  author = {Eulogik},
  title = {Yantra: A Sub-1B Agentic Tool-Calling Model},
  year = {2026},
  url = {https://github.com/eulogik/yantra},
  license = {MIT}
}
```

---

## 🙏 Acknowledgments

- [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) by OpenBMB
- [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) by Team ACE
- [Unsloth](https://github.com/unslothai/unsloth) for QLoRA training
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for inference
- [sentence-transformers](https://www.sbert.net/) for embeddings

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Eulogik](https://github.com/eulogik)**

⭐ Star this repo if you find it useful!

</div>
