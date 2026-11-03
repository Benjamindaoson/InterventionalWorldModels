import pytest

torch = pytest.importorskip("torch")
from iwm.geometry import inverse_pose, project, transform_points, unproject


def test_projection_round_trip_gradient():
    points = torch.tensor([[[0.2, 0.3, 2.0]]], requires_grad=True)
    intrinsic = torch.tensor([[[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]]])
    pixels = project(points, intrinsic)
    restored = unproject(pixels, points[..., 2], intrinsic)
    assert torch.allclose(restored, points, atol=1e-5)
    pixels.sum().backward()
    assert torch.isfinite(points.grad).all()


def test_inverse_pose():
    rotation = torch.eye(3).unsqueeze(0)
    translation = torch.tensor([[1., 2., 3.]])
    inverse_rotation, inverse_translation = inverse_pose(rotation, translation)
    point = torch.tensor([[[0., 0., 1.]]])
    assert torch.allclose(transform_points(transform_points(point, rotation, translation), inverse_rotation, inverse_translation), point)
