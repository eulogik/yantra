#!/usr/bin/env python3
"""Generate README/HF-card assets (PNG + SVG) with matplotlib only.

Outputs to assets/:
  pas_bars.{png,svg}            — baseline vs Yantra across PAS sub-metrics
  router_comparison.{png,svg}   — routing accuracy by router variant
  architecture.{png,svg}        — DTSA inference + training pipeline diagram

Numbers are the official reported results (Colab T4, ToolACE-300, seed 42):
  baseline PAS 0.1922, Yantra PAS 0.6775, routing 244 -> 247/300.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = Path(__file__).resolve().parent.parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

BASE = "#9aa0a6"  # grey for baseline
YANTRA = "#1a73e8"  # blue for Yantra
ACCENT = "#34a853"  # green for winner highlight


def _save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", format=ext, bbox_inches="tight")
    print("wrote", stem)


# -- 1. PAS grouped bars ---------------------------------------------
def pas_bars():
    labels = [
        "Parseable",
        "Valid\nname",
        "Expected\nname",
        "Exact\nargs",
        "Arg key\noverlap",
        "Stopped\ncleanly",
        "PAS",
    ]
    baseline = [0.0, 0.0, 0.5567, 0.0, 0.0, 0.0, 0.1922]
    yantra = [1.0, 1.0, 0.8233, 0.65, 0.9467, 1.0, 0.6775]
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.6))
    b1 = ax.bar([i - w / 2 for i in x], baseline, w, label="Baseline (llama.cpp)", color=BASE)
    b2 = ax.bar([i + w / 2 for i in x], yantra, w, label="Yantra (Router V2 + SFT2)", color=YANTRA)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("Yantra vs baseline on ToolACE-300 (PAS sub-metrics)")
    for b in list(b1) + list(b2):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{b.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "pas_bars")
    plt.close(fig)


# -- 2. Router comparison ---------------------------------------------
def router_chart():
    labels = ["Lexical\n(IDF+3gram)", "Embedding\n(MiniLM)", "Blend\n(β=0.5)"]
    correct = [244, 242, 247]
    total = 300
    acc = [c / total for c in correct]
    colors = [BASE, BASE, ACCENT]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, acc, color=colors, width=0.55)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("routing accuracy")
    ax.set_title("Tool routing accuracy on ToolACE-300 (n=300)")
    for b, c in zip(bars, correct):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.015,
            f"{c}/300 = {c/300:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if c == max(correct) else "normal",
        )
    fig.tight_layout()
    _save(fig, "router_comparison")
    plt.close(fig)


# -- 3. Architecture diagram -------------------------------------------
def architecture():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def box(xy, w, h, text, sub=None, fc="white", ec="#1a73e8", lw=1.6):
        x, y = xy
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.08",
                facecolor=fc, edgecolor=ec, linewidth=lw,
            )
        )
        ax.text(x + w / 2, y + h / 2 + (0.12 if sub else 0), text,
                ha="center", va="center", fontsize=10, fontweight="bold")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.28, sub,
                    ha="center", va="center", fontsize=8, color="#5f6368")

    def arrow(x0, y0, x1, y1, label=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#5f6368", lw=1.6))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label,
                    ha="center", va="bottom", fontsize=8, color="#5f6368")

    ax.text(6, 5.05, "Yantra — DTSA inference (decoupled routing + args) + training pipeline",
            ha="center", fontsize=12, fontweight="bold")

    # Inference row (5 boxes across 12 units)
    box((0.2, 3.1), 1.8, 1.1, "Query", "user text")
    box((2.4, 3.1), 2.0, 1.1, "Router V2", "IDF + 3gram + MiniLM")
    box((4.8, 3.1), 1.8, 1.1, "Bind", '<bind tool="…"/>')
    box((7.0, 3.1), 2.0, 1.1, "MiniCPM5-1B", "args only (QLoRA)")
    box((9.4, 3.1), 1.8, 1.1, "Stop", "<action_end/>", fc="#e8f0fe")
    arrow(2.0, 3.65, 2.4, 3.65)
    arrow(4.4, 3.65, 4.8, 3.65)
    arrow(6.6, 3.65, 7.0, 3.65)
    arrow(9.0, 3.65, 9.4, 3.65)

    # Training row
    box((0.2, 0.9), 2.2, 1.1, "Stage 1+3", "DTSA SFT · 1988 pairs")
    box((2.9, 0.9), 2.2, 1.1, "Stage 4", "EG-OPD DPO · 7795 pairs")
    box((5.6, 0.9), 2.2, 1.1, "Stage 5+8", "RTE + replay SFT")
    box((8.3, 0.9), 2.2, 1.1, "Stage 6", "Q4_K_M GGUF · 688MB", fc="#e8f0fe")
    arrow(2.4, 1.45, 2.9, 1.45)
    arrow(5.1, 1.45, 5.6, 1.45)
    arrow(7.8, 1.45, 8.3, 1.45)

    ax.text(0.2, 2.55, "Inference (runtime):", fontsize=9, color="#5f6368", fontweight="bold")
    ax.text(0.2, 0.45, "Training (Colab T4, QLoRA r=64):", fontsize=9, color="#5f6368", fontweight="bold")
    # link between rows
    ax.annotate("", xy=(10.3, 3.1), xytext=(9.4, 2.0),
                arrowprops=dict(arrowstyle="->", color="#9aa0a6", lw=1.2, linestyle="dashed"))
    ax.text(9.55, 2.55, "export", fontsize=8, color="#9aa0a6")

    fig.tight_layout()
    _save(fig, "architecture")
    plt.close(fig)


if __name__ == "__main__":
    pas_bars()
    router_chart()
    architecture()
    print("assets in", OUT)
