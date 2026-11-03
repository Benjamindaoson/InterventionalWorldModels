import json

import pytest

torch = pytest.importorskip("torch")
from iwm.data import validate
from iwm.models import RSSM, ActionPredictor


@pytest.mark.parametrize("model", [RSSM(hidden=16, latent=4), ActionPredictor(hidden=16)])
def test_model_forward_backward(model):
    state = torch.randn(2, 5, 3)
    action = torch.randn(2, 4, 3)
    loss = model.loss(state, action)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model(state, action, teacher_forcing=False)["prediction"].shape == (2, 4, 3)


def test_dataset_rejects_tamper(tmp_path):
    record = {"scene_id": "s", "split": "train", "initial_state_sha256": "same"}
    (tmp_path / "cube-0.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="artifact hash"):
        validate(tmp_path)


def test_rssm_evaluation_is_deterministic():
    model = RSSM(hidden=8, latent=4).eval()
    state = torch.randn(2, 4, 3)
    action = torch.randn(2, 3, 3)
    with torch.no_grad():
        first = model(state, action)["prediction"]
        second = model(state, action)["prediction"]
    assert torch.equal(first, second)
