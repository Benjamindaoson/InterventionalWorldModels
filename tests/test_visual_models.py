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


def test_video_teacher_force_no_future_frame_leakage():
    model = VideoFlow(8).eval()
    frames = torch.rand(1, 3, 3, 32, 32)
    changed = frames.clone()
    changed[:, 1:] = 0
    actions = torch.zeros(1, 2, 3)
    torch.manual_seed(5)
    first = model.teacher_force_generate(frames, actions, steps=2)
    torch.manual_seed(5)
    second = model.teacher_force_generate(changed, actions, steps=2)
    assert torch.equal(first[:, 1], second[:, 1])


def test_joint_inverse_action_has_no_action_input():
    model = JointWorldAction(8).eval()
    frames = torch.rand(1, 3, 3, 32, 32)
    actions = torch.zeros(1, 2, 3)
    predicted_action = model.predict_actions(frames)
    assert predicted_action.shape == actions.shape
    world = model.predict_world(frames[:, :-1], actions)
    changed = frames.clone()
    changed[:, -1] = 0
    assert torch.equal(world, model.predict_world(changed[:, :-1], actions))


def test_joint_independent_modes_do_not_update_unused_heads():
    frames = torch.rand(1, 3, 3, 32, 32)
    actions = torch.rand(1, 2, 3)
    for world_weight, action_weight, unused in ((1., 0., "action_head"),
                                                (0., 1., "world_head")):
        model = JointWorldAction(8)
        model.loss(frames, actions, world_weight, action_weight).backward()
        assert all(parameter.grad is None for parameter in getattr(model, unused).parameters())
