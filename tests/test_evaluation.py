"""Tiny examples whose dollar answers can be checked with a pencil."""

import numpy as np
import pytest

from latedeliveries.evaluation import (
    choose_threshold,
    cost_by_threshold,
    decision_cost,
)


def test_hand_counted_voucher_cost():
    # Flag first TWO ($8), miss the third late order ($15) = $23.
    truth = [1, 0, 1, 0]
    flags = [True, True, False, False]
    assert decision_cost(truth, flags) == 23
    assert decision_cost(truth, [False] * 4) == 30
    assert decision_cost(truth, [True] * 4) == 16


def test_hand_counted_best_threshold():
    truth = [1, 0, 1, 0]
    chances = [0.90, 0.55, 0.30, 0.10]
    # At 0.15, exactly 3 vouchers cost $12 and no late order is missed.
    # At 0.05, four vouchers cost $16; at 0.35, two vouchers plus a
    # missed late order cost $8 + $15 = $23.
    assert choose_threshold(truth, chances) == pytest.approx(0.15)
    table = cost_by_threshold(truth, chances)
    assert len(table) == 19
    assert int(table.training_cost.min()) == 12
    assert np.isclose(table.iloc[0].threshold, 0.05)
    assert np.isclose(table.iloc[-1].threshold, 0.95)


@pytest.mark.parametrize("actual, flags", [([0], [1, 0]), ([1, 0], [True])])
def test_cost_rejects_different_shapes(actual, flags):
    with pytest.raises(ValueError, match="same shape"):
        decision_cost(actual, flags)


def test_threshold_rejects_bad_probabilities():
    with pytest.raises(ValueError, match="between zero and one"):
        choose_threshold([1], [1.1])
