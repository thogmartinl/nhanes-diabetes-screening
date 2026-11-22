"""Model pipelines, tuning grids, and cross-validated tuning."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import make_preprocessor

# Random forest settings tried during tuning (3 x 4 x 3 = 36 combinations)
RF_PARAM_GRID = {
    "model__max_depth": [None, 12, 8],             # how deep each tree may grow
    "model__min_samples_leaf": [1, 5, 10, 20],     # smallest group allowed at a leaf
    "model__max_features": ["sqrt", 0.3, 0.5],     # share of features tried at each split
}


# Neural network settings tried during tuning (6 x 4 x 2 x 2 = 96 combinations)
NN_PARAM_GRID = {
    "model__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (64, 64, 32)],
    "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],      # L2 penalty (weight decay)
    "model__learning_rate_init": [1e-3, 3e-3],      # step size of the Adam optimizer
    "model__batch_size": [32, 128],                 # participants per training step
}


def make_random_forest(numeric_columns, random_state):
    """Imputation + one-hot encoding, followed by a random forest (no scaling needed)."""
    return Pipeline([
        ("prep", make_preprocessor(numeric_columns, scale=False)),
        ("model", RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)),
    ])


def make_neural_network(numeric_columns, random_state):
    """Imputation + one-hot encoding + standardization, followed by a feed-forward neural network.

    ReLU hidden layers and the Adam optimizer. Early stopping holds back 10% of the
    training data and stops when the validation score has not improved for 20 epochs.
    """
    return Pipeline([
        ("prep", make_preprocessor(numeric_columns, scale=True)),
        ("model", MLPClassifier(activation="relu", solver="adam", early_stopping=True,
                                validation_fraction=0.1, n_iter_no_change=20, max_iter=500,
                                random_state=random_state)),
    ])


def tune(pipeline, param_grid, X, y, random_state, scoring="roc_auc"):
    """Grid search with stratified 5-fold cross-validation on the training set.

    The whole pipeline, including imputation, is refitted in every fold, so the
    validation fold never influences the imputed values. The best settings are
    refitted on the full training set at the end.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv, refit=True, n_jobs=-1)
    search.fit(X, y)
    return search
