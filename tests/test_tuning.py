"""Make sure model searches are real and training-only."""

import pytest
from sklearn.model_selection import ParameterGrid

from latedeliveries.train import make_grid_search


@pytest.mark.parametrize(
    "name",
    ["Logistic regression", "Balanced logistic", "Decision tree", "Random forest"],
)
def test_search_has_at_least_six_combinations_and_two_settings(name):
    search = make_grid_search(name)
    assert len(search.param_grid) >= 2
    assert len(ParameterGrid(search.param_grid)) >= 6
    assert all(key.startswith("model__") for key in search.param_grid)
    assert search.scoring == "roc_auc"


def test_dummy_is_not_tuned():
    with pytest.raises(ValueError, match="dummy"):
        make_grid_search("Flag-nobody baseline")
