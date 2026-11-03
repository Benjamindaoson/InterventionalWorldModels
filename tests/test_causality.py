import numpy as np
import pytest

from iwm.causality import branch_diagnostics


def test_counterfactual_diagnostic_exact_and_reversed():
    actual = np.asarray([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.]])
    actions = actual.copy()
    exact = branch_diagnostics(actual, actual, actions)
    assert exact["counterfactual_delta_mse"] == 0
    assert exact["direction_accuracy"] == 1
    reversed_result = branch_diagnostics(actual, -actual, actions)
    assert reversed_result["direction_accuracy"] == 0
    assert reversed_result["x_outcome_ranking_accuracy"] == 0


def test_counterfactual_rejects_unpaired_shapes():
    with pytest.raises(ValueError):
        branch_diagnostics(np.zeros((2, 3)), np.zeros((3, 3)), np.zeros((2, 3)))
