"""Generates docs/architecture.png — a clean architecture diagram."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

def box(ax, x, y, w, h, label, color="#e3f2fd"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.2, edgecolor="#0d47a1", facecolor=color))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=10)

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=14, color="#37474f"))

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

box(ax, 0.02, 0.78, 0.18, 0.15, "News (CSV / NewsAPI / RSS)")
box(ax, 0.02, 0.58, 0.18, 0.15, "Tweets / X (CSV)")
box(ax, 0.02, 0.38, 0.18, 0.15, "Prices (yfinance cached)")
box(ax, 0.02, 0.18, 0.18, 0.15, "Synthetic portfolio")

box(ax, 0.26, 0.60, 0.20, 0.30, "Preprocess\n+ Dedup / Novelty", color="#fff3e0")
box(ax, 0.52, 0.55, 0.22, 0.40,
    "AI/NLP Risk Engine\n\nFinBERT Sentiment\nEvent Classifier\nImpact Score (1-10)\nEntity extraction",
    color="#e8f5e9")
box(ax, 0.80, 0.70, 0.18, 0.20, "FastAPI\n/signals, /analyze")
box(ax, 0.80, 0.45, 0.18, 0.20, "signals.csv / .json")
box(ax, 0.80, 0.20, 0.18, 0.20, "Streamlit Dashboard")

box(ax, 0.52, 0.20, 0.22, 0.22,
    "Downstream Modules\n\nA) Sentiment Rebalancer\nB) Stress Tester",
    color="#fce4ec")

arrow(ax, 0.20, 0.855, 0.26, 0.80)
arrow(ax, 0.20, 0.655, 0.26, 0.72)
arrow(ax, 0.20, 0.455, 0.26, 0.68)
arrow(ax, 0.20, 0.255, 0.52, 0.35)
arrow(ax, 0.46, 0.75, 0.52, 0.75)
arrow(ax, 0.74, 0.80, 0.80, 0.80)
arrow(ax, 0.74, 0.65, 0.80, 0.55)
arrow(ax, 0.74, 0.55, 0.52, 0.35)
arrow(ax, 0.89, 0.45, 0.89, 0.40)
arrow(ax, 0.80, 0.30, 0.74, 0.30)

plt.tight_layout()
out = Path(__file__).parent / "architecture.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Wrote {out}")
