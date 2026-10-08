"""Shared figure style so every committed figure looks like one course."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# Same palette as the website: blue, teal, purple, orange, then neutrals.
BLUE, TEAL, PURPLE, ORANGE = "#2563eb", "#0d9488", "#7c3aed", "#ea580c"
GRAY, INK = "#a1a1aa", "#27272a"
PALETTE = [BLUE, TEAL, PURPLE, ORANGE]

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": INK,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e4e7",
    "grid.linewidth": 0.6,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
})


def save(fig, name: str) -> Path:
    """Save to guide/figures/<name>.png and close the figure."""
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"saved {path.relative_to(FIGURES.parent)}")
    return path
