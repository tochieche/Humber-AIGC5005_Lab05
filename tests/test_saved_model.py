"""A reloaded pipeline must give exactly the same chances as before."""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline


def test_saved_pipeline_can_be_reloaded(tmp_path, training_data):
    train_x, _, train_y, _ = training_data
    pipeline = build_pipeline(LogisticRegression(max_iter=2000))
    pipeline.fit(train_x, train_y)
    a_few_orders = train_x.head(8)
    before = pipeline.predict_proba(a_few_orders)
    saved_file = tmp_path / "my_saved_model.joblib"
    joblib.dump(pipeline, saved_file)
    after = joblib.load(saved_file).predict_proba(a_few_orders)
    assert np.allclose(before, after)


def test_pipeline_does_not_need_future_information(tmp_path, training_data):
    train_x, _, train_y, _ = training_data
    pipeline = build_pipeline(LogisticRegression(max_iter=2000))
    pipeline.fit(train_x, train_y)
    fresh_order = train_x.head(1).copy()
    fresh_order["prep_minutes"] = float("nan")
    fresh_order["customer_rating"] = float("nan")
    fresh_order["courier_months"] = float("nan")
    fresh_order["vehicle"] = "unknown"
    expected = pipeline.predict_proba(train_x.head(1))
    result = pipeline.predict_proba(fresh_order)
    assert np.allclose(result, expected)
