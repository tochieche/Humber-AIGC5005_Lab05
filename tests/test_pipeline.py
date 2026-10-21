"""Check that we use safe features and protect the final test rows."""

import numpy as np
import pytest

from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    split_orders,
)


@pytest.mark.parametrize("forbidden", EXCLUDED_FEATURES + ["late"])
def test_not_allowed_at_checkout(forbidden):
    assert forbidden not in FEATURE_COLUMNS


def test_feature_lists_do_not_overlap():
    assert not set(NUMERIC_FEATURES).intersection(CATEGORICAL_FEATURES)


def test_split_is_75_25(training_data):
    train_x, test_x, train_y, test_y = training_data
    assert len(train_x) == 2250
    assert len(test_x) == 750
    assert len(train_y) + len(test_y) == 3000
    assert set(train_x.index).isdisjoint(test_x.index)
    assert list(train_x.columns) == FEATURE_COLUMNS


def test_split_is_stratified(orders, training_data):
    _, _, train_y, test_y = training_data
    assert abs(train_y.mean() - orders.late.mean()) < 0.01
    assert abs(test_y.mean() - orders.late.mean()) < 0.01


def test_split_is_repeatable(orders, training_data):
    x_train, x_test, y_train, y_test = training_data
    x_again, test_again, y_again, test_y_again = split_orders(orders)
    assert np.array_equal(x_train.index, x_again.index)
    assert np.array_equal(x_test.index, test_again.index)
    assert y_train.equals(y_again)
    assert y_test.equals(test_y_again)


def test_pipeline_handles_missing_number_and_new_weather(training_data):
    from sklearn.linear_model import LogisticRegression

    from latedeliveries.pipeline import build_pipeline

    train_x, _, train_y, _ = training_data
    machine = build_pipeline(LogisticRegression(max_iter=2000))
    machine.fit(train_x, train_y)
    unknown_order = train_x.iloc[[0]].copy()
    unknown_order["weather"] = "hail"
    unknown_order["distance_km"] = np.nan
    unknown_order["prep_minutes"] = np.nan  # This future column must be ignored.
    chances = machine.predict_proba(unknown_order)
    assert chances.shape == (1, 2)
    assert np.isfinite(chances).all()
    assert 0 <= chances[0, 1] <= 1


def test_predicted_probabilities_add_to_one(training_data):
    from sklearn.linear_model import LogisticRegression

    from latedeliveries.pipeline import build_pipeline

    train_x, _, train_y, _ = training_data
    machine = build_pipeline(LogisticRegression(max_iter=2000))
    machine.fit(train_x, train_y)
    chances = machine.predict_proba(train_x.head(20))
    assert np.all((chances >= 0) & (chances <= 1))
    assert np.allclose(chances.sum(axis=1), 1)
