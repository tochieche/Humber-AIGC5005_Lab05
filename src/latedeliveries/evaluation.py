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
