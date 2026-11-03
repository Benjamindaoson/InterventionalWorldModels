"""Differentiable batched camera geometry in a right-handed world frame."""
from __future__ import annotations

import torch


def transform_points(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Apply x' = R x + t to row-vector tensors."""
    if points.shape[-1] != 3 or rotation.shape[-2:] != (3, 3) or translation.shape[-1] != 3:
        raise ValueError("invalid SE(3) dimensions")
    return points @ rotation.transpose(-1, -2) + translation.unsqueeze(-2)


def inverse_pose(rotation: torch.Tensor, translation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_rotation = rotation.transpose(-1, -2)
    inverse_translation = -(inverse_rotation @ translation.unsqueeze(-1)).squeeze(-1)
    return inverse_rotation, inverse_translation


def project(points_camera: torch.Tensor, intrinsics: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if points_camera.shape[-1] != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("invalid projection dimensions")
    if torch.any(points_camera[..., 2] <= eps):
        raise ValueError("points must be in front of camera")
    homogeneous = points_camera @ intrinsics.transpose(-1, -2)
    return homogeneous[..., :2] / homogeneous[..., 2:3]


def unproject(pixels: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    if pixels.shape[-1] != 2 or pixels.shape[:-1] != depth.shape:
        raise ValueError("pixel/depth mismatch")
    homogeneous = torch.cat((pixels, torch.ones_like(depth.unsqueeze(-1))), -1)
    rays = homogeneous @ torch.linalg.inv(intrinsics).transpose(-1, -2)
    return rays * depth.unsqueeze(-1)


def reprojection_loss(points_world: torch.Tensor, observed_pixels: torch.Tensor,
                      rotation_world_to_camera: torch.Tensor, translation: torch.Tensor,
                      intrinsics: torch.Tensor) -> torch.Tensor:
    predicted = project(transform_points(points_world, rotation_world_to_camera, translation), intrinsics)
    return torch.nn.functional.smooth_l1_loss(predicted, observed_pixels)


def temporal_consistency_loss(positions: torch.Tensor, velocities: torch.Tensor, dt: float) -> torch.Tensor:
    if dt <= 0 or positions.shape[1] != velocities.shape[1] + 1:
        raise ValueError("expected positions [B,T+1,3], velocities [B,T,3]")
    return torch.nn.functional.mse_loss(positions[:, 1:] - positions[:, :-1], velocities * dt)
