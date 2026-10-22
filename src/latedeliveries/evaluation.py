"""Compare models fairly and count real QuickBite voucher dollars."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

from latedeliveries.pipeline import SEED, build_pipeline, training_folds

VOUCHER_COST = 4
MISSED_LATE_COST = 15
BREAK_EVEN_THRESHOLD = VOUCHER_COST / MISSED_LATE_COST


def decision_cost(actual, flags):
    """Pay $4 per flag and $15 for every missed late order."""
    truth = np.asarray(actual, dtype=int)
    flagged = np.asarray(flags, dtype=bool)
    if truth.shape != flagged.shape:
        raise ValueError("actual and flags must have the same shape")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("actual must be 0 or 1")
    return int(
        VOUCHER_COST * np.count_nonzero(flagged)
        + MISSED_LATE_COST * np.count_nonzero((truth == 1) & ~flagged)
    )


def candidates():
    """Compare a dummy and four real learners, including a balanced one."""
    return {
        "Flag-nobody baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Balanced logistic": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED
        ),
        "Decision tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=15, random_state=SEED
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=80, min_samples_leaf=5, random_state=SEED, n_jobs=1
        ),
    }


def negative_cost_per_order(fitted_pipeline, features, actual):
    """A bigger scikit-learn score means fewer dollars spent."""
    probabilities = fitted_pipeline.predict_proba(features)[:, 1]
    flags = probabilities >= BREAK_EVEN_THRESHOLD
    return -decision_cost(actual, flags) / len(actual)


def compare_models(train_x, train_y):
    """Try all models with the SAME five training folds and pipeline."""
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "negative_cost": negative_cost_per_order,
    }
    results = []
    for name, model in candidates().items():
        scores = cross_validate(
            build_pipeline(model),
            train_x,
            train_y,
            cv=training_folds(),
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        row = {"model": name}
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
        row["cost_per_order_mean"] = float(-scores["test_negative_cost"].mean())
        row["cost_per_order_std"] = float(scores["test_negative_cost"].std())
        results.append(row)
    return (
        pd.DataFrame(results)
        .sort_values("cost_per_order_mean", kind="stable")
        .reset_index(drop=True)
    )


def choose_model(cv_table):
    """Choose the lowest-cost REAL model using only the training CV."""
    real_models = cv_table[cv_table["model"] != "Flag-nobody baseline"]
    return str(real_models.iloc[0]["model"])


def cost_by_threshold(actual, probabilities):
    """Try all 19 choices, using only predictions from training folds."""
    chances = np.asarray(probabilities, dtype=float)
    truth = np.asarray(actual, dtype=int)
    if chances.shape != truth.shape:
        raise ValueError("actual and probabilities must have the same shape")
    if not np.isfinite(chances).all() or not ((0 <= chances) & (chances <= 1)).all():
        raise ValueError("probabilities must be numbers between zero and one")
    thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "training_cost": [
                decision_cost(truth, chances >= cut) for cut in thresholds
            ],
            "flagged_orders": [
                int(np.count_nonzero(chances >= cut)) for cut in thresholds
            ],
        }
    )


def choose_threshold(actual, probabilities):
    """Return the cheapest threshold; in ties, choose the smallest one."""
    table = cost_by_threshold(actual, probabilities)
    row = table.loc[table["training_cost"].idxmin()]
    return float(row["threshold"])
