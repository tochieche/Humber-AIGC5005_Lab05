"""The finished reports must agree with the saved model settings."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from latedeliveries.pipeline import FEATURE_COLUMNS, SEED

ROOT = Path(__file__).resolve().parents[1]


def test_reports_exist_and_have_expected_columns():
    table = pd.read_csv(ROOT / "reports/cv_results.csv")
    assert len(table) >= 4
    assert "Flag-nobody baseline" in table.model.values
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert f"{name}_mean" in table.columns
        assert f"{name}_std" in table.columns


def test_model_metadata_and_final_scores_agree(training_data):
    info = json.loads((ROOT / "models/model_info.json").read_text())
    scores = json.loads((ROOT / "reports/metrics.json").read_text())
    assert info["seed"] == SEED
    assert info["feature_columns"] == FEATURE_COLUMNS
    assert info["threshold"] == scores["threshold"]
    assert info["test_results"] == scores
    assert scores["test_orders"] == 750
    assert np.asarray(scores["confusion_matrix"]).sum() == 750
    pipeline = joblib.load(ROOT / "models/late_orders.joblib")
    chance = pipeline.predict_proba(training_data[0].head(1))[0, 1]
    assert isinstance(chance, np.float64)
