"""Evaluation on the test set: discrimination, bootstrap confidence intervals, and comparisons."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict


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



def oof_predictions(model, X, y, random_state):
    """Out-of-fold predicted risk on the training set (each participant predicted by a model
    that did not see them), using the same 5 folds as tuning."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def threshold_for_sensitivity(y, scores, target=0.80):
    """Highest threshold whose sensitivity reaches the target (fewest people flagged)."""
    fpr, tpr, thresholds = roc_curve(y, scores)
    return thresholds[np.argmax(tpr >= target)]


def screening_metrics(y, scores, threshold):
    """Screening performance when everyone with a score at or above the threshold is flagged."""
    y = np.asarray(y)
    flagged = np.asarray(scores) >= threshold
    tp = int((flagged & (y == 1)).sum())
    fn = int((~flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    tn = int((~flagged & (y == 0)).sum())
    return {
        "Threshold": round(float(threshold), 3),
        "Sensitivity": round(tp / (tp + fn), 3),
        "Specificity": round(tn / (tn + fp), 3),
        "Precision (PPV)": round(tp / (tp + fp), 3),
        "NPV": round(tn / (tn + fn), 3),
        "Accuracy": round((tp + tn) / len(y), 3),
        "Flagged (%)": round(float(flagged.mean()) * 100, 1),
        "Found / missed": f"{tp} / {fn}",
    }



def calibration_summary(y, probabilities, reference_rate):
    """Brier score, Brier skill score, calibration-in-the-large and calibration slope.

    reference_rate: share of elevated HbA1c in the training set. Predicting this
    rate for everyone is the reference that the Brier skill score compares against.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    y = np.asarray(y)
    reference = float(np.mean((y - reference_rate) ** 2))
    rows = []
    for name, p in probabilities.items():
        p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        slope = LogisticRegression(C=1e6).fit(logit, y).coef_[0, 0]   # 1 = ideal
        brier = brier_score_loss(y, p)
        rows.append({"Model": name, "Brier score": round(brier, 3),
                     "Brier skill score": round(1 - brier / reference, 3),
                     "Mean predicted risk": round(float(p.mean()), 3),
                     "Observed rate": round(float(y.mean()), 3),
                     "Calibration slope": round(float(slope), 2)})
    rows.append({"Model": f"Reference (predict {reference_rate:.3f} for everyone)",
                 "Brier score": round(reference, 3), "Brier skill score": 0.0,
                 "Mean predicted risk": round(reference_rate, 3),
                 "Observed rate": round(float(y.mean()), 3), "Calibration slope": None})
    return pd.DataFrame(rows).set_index("Model")



# Features grouped by what they measure; a missing indicator goes with its feature
FEATURE_GROUPS = {
    "Age": ["age"],
    "Body size (BMI, waist, waist-to-height)": ["bmi", "waist", "waist_height_ratio"],
    "Blood pressure": ["systolic_bp", "diastolic_bp", "told_high_bp", "bp_missing"],
    "Cardiovascular disease": ["cvd", "cvd_missing"],
    "Sex": ["female"],
    "Race/ethnicity": ["race_ethnicity"],
    "Income-to-poverty ratio": ["income_ratio", "income_missing"],
    "Health insurance": ["insured"],
    "Smoking": ["smoking_status"],
    "Physical activity": ["activity_log"],
    "Alcohol": ["alcohol_freq", "alcohol_drinks"],
    "Sleep": ["sleep_hours"],
    "Depression (PHQ-9)": ["phq9"],
    "Self-rated health": ["health_rating"],
    "Alcohol/PHQ-9 not answered": ["alcohol_phq9_missing"],
}


def grouped_permutation_importance(model, X, y, groups=FEATURE_GROUPS, n_repeats=20,
                                   random_state=42):
    """Drop in ROC-AUC when the columns of one feature group are shuffled together.

    Shuffling a whole group at once (e.g. BMI, waist and waist-to-height ratio) avoids
    underestimating features that are strongly correlated with each other.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    baseline = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, columns in groups.items():
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[columns] = X[columns].iloc[order].to_numpy()
            drops.append(baseline - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        rows.append({"Feature group": name, "Mean drop in ROC-AUC": float(np.mean(drops)),
                     "SD": float(np.std(drops))})
    return pd.DataFrame(rows).set_index("Feature group")


def cv_auc_by_feature_set(make_model, params, X, y, feature_sets, reference, random_state):
    """Cross-validated ROC-AUC on the training set for different sets of input features.

    make_model: make_random_forest or make_neural_network. params: the tuned settings.
    feature_sets: dict label -> list of columns. reference: label to compare against.
    All feature sets use the same 5 folds, so the differences are paired by fold.
    """
    from sklearn.model_selection import cross_val_score
    from .preprocessing import CATEGORICAL

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    fold_scores = {}
    for label, columns in feature_sets.items():
        categorical = [c for c in columns if c in CATEGORICAL]
        numeric = [c for c in columns if c not in CATEGORICAL]
        model = make_model(numeric, random_state, categorical_columns=categorical).set_params(**params)
        fold_scores[label] = cross_val_score(model, X[columns], y, cv=cv, scoring="roc_auc")

    rows = []
    for label, scores in fold_scores.items():
        diff = scores - fold_scores[reference]
        rows.append({"Feature set": label, "Features": len(feature_sets[label]),
                     "CV ROC-AUC": round(scores.mean(), 3), "SD": round(scores.std(), 3),
                     f"Difference vs. {reference}": round(diff.mean(), 3),
                     "SD of difference": round(diff.std(), 3)})
    return pd.DataFrame(rows).set_index("Feature set")
