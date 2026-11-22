"""Evaluation on the test set: discrimination, bootstrap confidence intervals, and comparisons."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def bootstrap_discrimination(y, scores, reference, n_boot=1000, random_state=42):
    """ROC-AUC and PR-AUC with 95% bootstrap confidence intervals.

    scores: dict of name -> predicted risk (or risk score) for the test set.
    reference: the name in scores to compare against (the ADA risk score).
    All scores are resampled with the same participants in each round, so the
    differences to the reference are paired.
    """
    y = np.asarray(y)
    scores = {name: np.asarray(s) for name, s in scores.items()}
    rng = np.random.default_rng(random_state)

    def metrics(idx):
        return {name: (roc_auc_score(y[idx], s[idx]), average_precision_score(y[idx], s[idx]))
                for name, s in scores.items()}

    point = metrics(np.arange(len(y)))
    rounds = []
    while len(rounds) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():          # both classes must be present
            rounds.append(metrics(idx))

    def fmt(estimate, samples):
        low, high = np.percentile(samples, [2.5, 97.5])
        return f"{estimate:.3f} ({low:.3f} to {high:.3f})"

    rows = []
    for name in scores:
        roc = [r[name][0] for r in rounds]
        pr = [r[name][1] for r in rounds]
        row = {"Model": name, "ROC-AUC (95% CI)": fmt(point[name][0], roc),
               "PR-AUC (95% CI)": fmt(point[name][1], pr)}
        if name != reference:
            diff = [r[name][0] - r[reference][0] for r in rounds]
            row[f"ROC-AUC difference vs. {reference} (95% CI)"] = fmt(
                point[name][0] - point[reference][0], diff)
        else:
            row[f"ROC-AUC difference vs. {reference} (95% CI)"] = "reference"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")
