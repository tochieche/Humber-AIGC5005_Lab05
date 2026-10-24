"""Reusable small test data, so we don't create it over and over."""

import pytest

from latedeliveries.data import make_orders
from latedeliveries.pipeline import SEED, split_orders


@pytest.fixture(scope="session")
def orders():
    return make_orders(seed=SEED)


@pytest.fixture(scope="session")
def training_data(orders):
    return split_orders(orders)
