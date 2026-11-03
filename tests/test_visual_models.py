import pytest

torch = pytest.importorskip("torch")
from iwm.visual_models import JointWorldAction, VideoFlow, VisualPredictor


@pytest.mark.parametrize("model", [VisualPredictor(8), VideoFlow(8), JointWorldAction(8)])
def test_visual_backward(model):
    frames = torch.rand(1, 3, 3, 32, 32)
    actions = torch.rand(1, 2, 3)
    loss = model.loss(frames, actions)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_video_generate():
    model = VideoFlow(8)
    video = model.generate(torch.rand(1, 3, 32, 32), torch.zeros(1, 2, 3), steps=2)
    assert video.shape == (1, 3, 3, 32, 32)
