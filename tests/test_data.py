"""Make sure the teacher's sample stays the same for everyone."""

import numpy as np

from latedeliveries.data import make_orders


def test_data_has_3000_orders_and_expected_columns(orders):
    assert orders.shape == (3000, 11)
    assert orders.late.isin([0, 1]).all()
    assert int(orders.order_id.iloc[0]) == 100000
    assert orders.prep_minutes.isna().any()
    assert orders.customer_rating.isna().any()


def test_teacher_data_is_repeatable():
    a = make_orders(n=10, seed=7)
    b = make_orders(n=10, seed=7)
    assert a.equals(b)
    assert np.array_equal(a.order_id, np.arange(100000, 100010))
