"""Make sure model searches are real and training-only."""

import pytest
from sklearn.model_selection import ParameterGrid

from latedeliveries.train import make_grid_search


@pytest.mark.parametrize(
    "name",
    ["Logistic regression", "Balanced logistic", "Decision tree", "Random forest"],
)