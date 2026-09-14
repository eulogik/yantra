#!/usr/bin/env python3
"""Paper figures for paper/figs/.

- Copies architecture.png and router_comparison.png from assets/.
- Generates pas_yantra.png: Yantra sub-metric bars with the baseline PAS
  (0.1922) as a reference line. Baseline sub-metric bars are deliberately
  omitted: the reference model's outputs under llama.cpp serving are
  parser-sensitive (schema echoes), so only its headline PAS from the
  controlled baseline run is reported (see paper Sec. 4.3).
"""
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

for f in ("architecture.png", "router_comparison.png"):
    shutil.copy2(ROOT / "assets" / f, FIGS / f)
    print("copied", f)

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

labels = ["Parseable", "Valid\nname", "Expected\nname", "Exact\nargs",
          "Arg key\noverlap", "Stopped\ncleanly", "PAS"]
yantra = [1.0, 1.0, 0.8233, 0.65, 0.9467, 1.0, 0.6775]
BASE_PAS = 0.1922

fig, ax = plt.subplots(figsize=(9.5, 3.8))
bars = ax.bar(labels, yantra, color="#1a73e8", width=0.55)
ax.axhline(BASE_PAS, color="#9aa0a6", linestyle="--", linewidth=1.4)
ax.text(6.4, BASE_PAS + 0.03, f"baseline PAS {BASE_PAS}",
        ha="right", va="bottom", fontsize=9, color="#5f6368")
ax.set_ylim(0, 1.15)
ax.set_ylabel("score")
ax.set_title("Yantra on ToolACE-300 (n=300): PAS sub-metrics")
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
            f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "pas_yantra.png", bbox_inches="tight")
print("wrote pas_yantra.png")
