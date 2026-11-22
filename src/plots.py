"""Shared plot style and the plotting functions used in the notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Colorblind-safe palette (validated): blue, orange, aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

FIG_DIR = Path("/content/figures")

# Readable names for the features, used in all plots
FEATURE_LABELS = {
    "age": "Age", "female": "Female", "income_ratio": "Income-to-poverty ratio",
    "bmi": "BMI", "waist": "Waist circumference", "waist_height_ratio": "Waist-to-height ratio",
    "systolic_bp": "Systolic BP", "diastolic_bp": "Diastolic BP", "told_high_bp": "Told high BP",
    "cvd": "Cardiovascular disease", "activity_log": "Physical activity (log)",
    "alcohol_freq": "Drinking frequency", "alcohol_drinks": "Drinks per drinking day",
    "sleep_hours": "Sleep hours", "phq9": "PHQ-9 score", "insured": "Health insurance",
    "health_rating": "Self-rated health (1 = excellent)", "hba1c": "HbA1c",
}

# Diverging scale for correlations: blue (negative) - gray (0) - red (positive)
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#184f95", "#86b6ef", "#f0efec", "#f2a09f", "#b8302f"])


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



def plot_correlation_matrix(df, min_label=0.3):
    """Lower-triangle Spearman correlation matrix; values shown where |r| >= min_label."""
    corr = df.corr(method="spearman").iloc[1:, :-1]     # drop the empty first row / last column
    n_rows, n_cols = corr.shape
    mask = np.triu(np.ones((n_rows, n_cols), dtype=bool), k=1)
    values = np.ma.masked_array(corr.values, mask)
    cmap = DIVERGING.copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n_cols), [FEATURE_LABELS.get(c, c) for c in corr.columns],
                  rotation=90, fontsize=8.5)
    ax.set_yticks(range(n_rows), [FEATURE_LABELS.get(c, c) for c in corr.index], fontsize=8.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for i in range(n_rows):
        for j in range(n_cols):
            r = corr.iat[i, j]
            if not mask[i, j] and abs(r) >= min_label:
                ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(r) >= 0.6 else INK)

    bar = fig.colorbar(image, ax=ax, shrink=0.6, pad=0.02)
    bar.set_label(f"Spearman correlation (values shown where |r| ≥ {min_label})", color=INK_2)
    bar.outline.set_visible(False)
    ax.set_title("Correlations between the features and HbA1c, training set")
    return fig


def plot_activity_transform(activity_log):
    """Weekly physical activity before and after the log transform."""
    after = activity_log.dropna()
    before = np.expm1(after)                    # back to minutes per week
    cut = before.quantile(0.99)
    zero_share = (before == 0).mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(before.clip(upper=cut), bins=40, color=BLUE, edgecolor="white", linewidth=0.6)
    axes[0].set_title(f"Before: minutes per week  ·  skewness {before.skew():.1f}", fontsize=11)
    axes[0].set_xlabel(f"Minutes per week (values above {cut:,.0f} shown in the last bar)")
    axes[0].set_ylabel("Participants")

    axes[1].hist(after, bins=40, color=BLUE, edgecolor="white", linewidth=0.6)
    axes[1].set_title(f"After: log(minutes + 1)  ·  skewness {after.skew():.1f}", fontsize=11)
    axes[1].set_xlabel("log(minutes per week + 1)")
    axes[1].text(0.98, 0.95, f"{zero_share:.0%} report no activity (value 0)",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9, color=INK_2)

    fig.suptitle("Physical activity (moderate-equivalent minutes) before and after the log transform,"
                 " training set", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig



def plot_body_measures(X, elevated):
    """BMI, waist and waist-to-height ratio by HbA1c group, with each measure's AUC on its own."""
    from sklearn.metrics import roc_auc_score

    measures = [("bmi", "BMI (kg/m²)"), ("waist", "Waist circumference (cm)"),
                ("waist_height_ratio", "Waist-to-height ratio")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, (col, label) in zip(axes, measures):
        present = X[col].notna()
        values, target = X.loc[present, col], elevated[present]
        auc = roc_auc_score(target, values)
        box = ax.boxplot([values[target == 0], values[target == 1]], patch_artist=True,
                         widths=0.55, medianprops=dict(color="white", linewidth=2),
                         whiskerprops=dict(color=AXIS), capprops=dict(color=AXIS),
                         flierprops=dict(marker="o", markersize=2, markerfacecolor=MUTED,
                                         markeredgecolor="none", alpha=0.6))
        for patch, color in zip(box["boxes"], [BLUE, ORANGE]):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
        ax.set_xticks([1, 2], ["Normal\n(< 5.7%)", "Elevated\n(≥ 5.7%)"])
        ax.set_title(f"{label}\nAUC on its own: {auc:.2f}", fontsize=10.5)
    fig.suptitle("Body measures by HbA1c group, training set",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig



def plot_roc_pr(y, scores, operating_points):
    """ROC and precision-recall curves for each approach, with its screening threshold marked.

    scores: dict name -> predicted risk on the test set (at most three approaches).
    operating_points: dict name -> threshold used in the screening table.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    y = np.asarray(y)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for (name, s), color in zip(scores.items(), [BLUE, ORANGE, AQUA]):
        s = np.asarray(s)
        fpr, tpr, _ = roc_curve(y, s)
        precision, recall, _ = precision_recall_curve(y, s)
        ax_roc.plot(fpr, tpr, color=color, label=f"{name} (ROC-AUC {roc_auc_score(y, s):.3f})")
        ax_pr.plot(recall, precision, color=color, label=f"{name} (PR-AUC {average_precision_score(y, s):.3f})")

        # Mark the screening threshold from the screening table
        flagged = s >= operating_points[name]
        sens = (flagged & (y == 1)).sum() / (y == 1).sum()
        spec = (~flagged & (y == 0)).sum() / (y == 0).sum()
        prec = (flagged & (y == 1)).sum() / flagged.sum()
        for ax, x, yy in [(ax_roc, 1 - spec, sens), (ax_pr, sens, prec)]:
            ax.plot(x, yy, "o", markersize=8, color=color, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=5)

    ax_roc.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Chance")
    ax_pr.axhline(y.mean(), color=MUTED, linestyle="--", linewidth=1,
                  label=f"Chance (share elevated = {y.mean():.3f})")
    ax_roc.set(xlabel="1 − specificity (false positive rate)", ylabel="Sensitivity (true positive rate)",
               xlim=(0, 1), ylim=(0, 1.01))
    ax_pr.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.set_title("ROC curve", fontsize=11)
    ax_pr.set_title("Precision-recall curve", fontsize=11)
    for ax in (ax_roc, ax_pr):
        ax.grid(True, axis="both")
        ax.legend(loc="lower right" if ax is ax_roc else "upper right", fontsize=8.5)
    fig.suptitle("Discrimination on the test set  ·  dots mark the screening thresholds from Table S",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig
