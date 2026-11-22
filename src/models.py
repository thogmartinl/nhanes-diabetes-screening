"""Model pipelines, tuning grids, and cross-validated tuning."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .preprocessing import make_preprocessor

# Random forest settings tried during tuning (3 x 4 x 3 = 36 combinations)
RF_PARAM_GRID = {
    "model__max_depth": [None, 12, 8],             # how deep each tree may grow
    "model__min_samples_leaf": [1, 5, 10, 20],     # smallest group allowed at a leaf
    "model__max_features": ["sqrt", 0.3, 0.5],     # share of features tried at each split
}


def make_random_forest(numeric_columns, random_state):
    """Imputation + one-hot encoding, followed by a random forest (no scaling needed)."""
    return Pipeline([
        ("prep", make_preprocessor(numeric_columns, scale=False)),
        ("model", RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)),
    ])


def tune(pipeline, param_grid, X, y, random_state, scoring="roc_auc"):
    """Grid search with stratified 5-fold cross-validation on the training set.

    The whole pipeline, including imputation, is refitted in every fold, so the
    validation fold never influences the imputed values. The best settings are
    refitted on the full training set at the end.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv, refit=True)
    search.fit(X, y)
    return search
