"""Shared plot style and the plotting functions used in the notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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



def plot_elevated_by_group(age, bmi, elevated):
    """Share of participants with elevated HbA1c by age group and by BMI group."""
    age_groups = pd.cut(age, [18, 30, 40, 50, 60, 70, np.inf], right=False,
                        labels=["18–29", "30–39", "40–49", "50–59", "60–69", "70+"])
    bmi_groups = pd.cut(bmi, [0, 18.5, 25, 30, 35, 40, np.inf], right=False,
                        labels=["<18.5", "18.5–24.9", "25–29.9", "30–34.9", "35–39.9", "40+"])
    overall = elevated.mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, groups, title, xlabel in [(axes[0], age_groups, "By age", "Age (years)"),
                                      (axes[1], bmi_groups, "By BMI", "BMI (kg/m²)")]:
        stats = elevated.groupby(groups, observed=False).agg(["mean", "size"])
        share = stats["mean"] * 100
        x = np.arange(len(stats))
        ax.bar(x, share, color=ORANGE, width=0.7, zorder=3)   # bars in front of the line
        for xi, value in zip(x, share):
            # Label inside the top of the bar (white), or above it if the bar is too short
            inside = value >= 8
            ax.text(xi, value - 1.2 if inside else value + 1.2, f"{value:.0f}%", ha="center",
                    va="top" if inside else "bottom", fontsize=9,
                    color="white" if inside else INK_2, fontweight="bold" if inside else "normal",
                    zorder=4)
        ax.axhline(overall, color=INK_2, linestyle="--", linewidth=1,
                   label=f"All participants: {overall:.0f}%")
        ax.set_xticks(x, [f"{g}\n(n = {n:,})" for g, n in zip(stats.index, stats["size"])],
                      fontsize=8.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Participants with elevated HbA1c (%)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Elevated HbA1c (≥ 5.7%) by age and BMI, training set",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig
