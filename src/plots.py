"""Shared plot style and the plotting functions used in the notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Colorblind-safe palette (validated): blue, orange, aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

FIG_DIR = Path("/content/figures")


def set_style():
    """One consistent look for every plot in the notebook."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "sans-serif", "font.size": 10, "text.color": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 12, "axes.labelsize": 10, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 3, "ytick.major.size": 0,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def save(fig, name):
    """Save a figure as a high-resolution PNG in /content/figures."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def plot_hba1c_distribution(hba1c, prediabetes=5.7, diabetes=6.5, upper=9.0):
    """Histogram of HbA1c with the two ADA thresholds marked."""
    values = hba1c.clip(upper=upper)            # values above 9% go into the last bar
    # HbA1c is reported to one decimal, so center each bar on a value (edges at x.x5)
    bins = np.arange(np.floor(values.min() * 10) / 10 - 0.05, upper + 0.1, 0.1)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    counts, edges, patches = ax.hist(values, bins=bins, edgecolor="white", linewidth=0.8)
    for patch, left in zip(patches, edges[:-1]):
        center = left + 0.05
        patch.set_facecolor(ORANGE if center >= prediabetes - 1e-6 else BLUE)

    # Threshold lines sit on the bar edge where each range starts; labels sit above the plot
    top = ax.get_xaxis_transform()
    ax.axvline(prediabetes - 0.05, color=INK_2, linestyle="--", linewidth=1)
    ax.axvline(diabetes - 0.05, color=INK_2, linestyle="--", linewidth=1)
    ax.text(prediabetes - 0.1, 1.01, f"{prediabetes}% prediabetes", transform=top,
            ha="right", va="bottom", color=INK_2, fontsize=9)
    ax.text(diabetes, 1.01, f"{diabetes}% diabetes", transform=top,
            ha="left", va="bottom", color=INK_2, fontsize=9)

    n_elevated = int((hba1c >= prediabetes).sum())
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, [f"Normal (< {prediabetes}%): {len(hba1c) - n_elevated:,}",
                        f"Elevated (≥ {prediabetes}%): {n_elevated:,}"], loc="upper right")

    ax.set_title("HbA1c in the training set", pad=22)
    ax.set_xlabel(f"HbA1c (%)   ·   values above {upper}% are shown in the last bar")
    ax.set_ylabel("Participants")
    return fig
