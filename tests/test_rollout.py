import pytest

torch = pytest.importorskip("torch")
from iwm.models import ActionPredictor
from iwm.rollout import predict


def test_anchor_changes_rollout():
    model = ActionPredictor(hidden=8)
    initial = torch.zeros(1, 3)
    actions = torch.zeros(1, 3, 3)
    unanchored = predict(model, initial, actions)
    anchored = predict(model, initial, actions, {1: torch.ones(1, 3)})
    assert anchored.shape == (1, 3, 3)
    assert not torch.allclose(unanchored[:, 1:], anchored[:, 1:])
