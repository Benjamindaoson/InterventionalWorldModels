"""Velocity-aware, camera-calibrated MuJoCo scenes with frozen evaluation families."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from .data import validate
from .mujoco_paired_backend import MuJoCoCubeBackend

ACTIONS = ((0., 0., 0.), (5., 0., 0.), (-5., 0., 0.),
           (10., 0., 0.), (-10., 0., 0.), (0., 10., 0.), (0., -10., 0.))
FAMILIES = ("train", "iid_val", "iid_test", "physics_ood", "geometry_ood", "confirmatory")
SPECS = {
    "train": ((0.5, 0.3, 0.05), (1.0, 0.5, 0.05), (1.5, 0.7, 0.05), (0.8, 0.4, 0.05)),
    "iid_val": ((0.5, 0.3, 0.05), (1.0, 0.5, 0.05), (1.5, 0.7, 0.05), (0.8, 0.4, 0.05)),
    "iid_test": ((0.5, 0.3, 0.05), (1.0, 0.5, 0.05), (1.5, 0.7, 0.05), (0.8, 0.4, 0.05)),
    "physics_ood": ((2.0, 0.9, 0.05), (0.65, 0.2, 0.05)),
    "geometry_ood": ((1.0, 0.5, 0.035), (1.0, 0.5, 0.07)),
    "confirmatory": ((1.3, 0.8, 0.06), (0.9, 0.25, 0.045)),
}


def camera_calibration(backend: MuJoCoCubeBackend, width: int = 32,
                       height: int = 32) -> dict:
    camera_id = backend.mujoco.mj_name2id(backend.model,
                                         backend.mujoco.mjtObj.mjOBJ_CAMERA, "track")
    center = backend.data.cam_xpos[camera_id].copy()
    camera_to_world = backend.data.cam_xmat[camera_id].reshape(3, 3).copy()
    cv_flip = np.diag([1., -1., -1.])
    world_to_camera = cv_flip @ camera_to_world.T
    translation = -world_to_camera @ center
    fy = height / (2 * math.tan(math.radians(float(backend.model.cam_fovy[camera_id])) / 2))
    fx = fy * width / height
    return {"name": "track", "image_width": width, "image_height": height,
            "intrinsics": [[fx, 0., width / 2], [0., fy, height / 2], [0., 0., 1.]],
            "camera_center_world": center.tolist(),
            "camera_to_world_rotation": camera_to_world.tolist(),
            "world_to_camera_cv_rotation": world_to_camera.tolist(),
            "world_to_camera_cv_translation": translation.tolist(),
            "projection_convention": "CV x-right y-down z-forward"}


def generate_v2(root: Path, scenes_per_family: dict[str, int], horizon: int = 64) -> Path:
    if set(scenes_per_family) != set(FAMILIES) or any(v < 2 for v in scenes_per_family.values()) or horizon < 4:
        raise ValueError("all six families require at least two scenes; horizon >=4")
    root.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    total = 0
    for family_id, family in enumerate(FAMILIES):
        split = "train" if family == "train" else ("val" if family == "iid_val" else "test")
        for i in range(scenes_per_family[family]):
            mass, friction, half_size = SPECS[family][i % len(SPECS[family])]
            backend = MuJoCoCubeBackend(mass=mass, friction=friction, box_half_size=half_size)
            renderer = mujoco.Renderer(backend.model, height=32, width=32)
            seed = 10000 * (family_id + 1) + i
            backend.reset(seed)
            backend.data.qpos[0] = -0.15 + 0.025 * (i % 13) + 0.0005 * family_id
            backend.data.qpos[1] = -0.08 + 0.025 * (i // 13) + 0.0007 * family_id
            backend.data.qpos[2] = (half_size, half_size + 0.005, 0.35)[i % 3]
            backend.data.qvel[0] = (-0.18, 0., 0.18)[i % 3]
            backend.data.qvel[1] = (0., 0.1, -0.1)[i % 3]
            mujoco.mj_forward(backend.model, backend.data)
            initial = backend.snapshot()
            initial_hash = hashlib.sha256(initial).hexdigest()
            calibration = camera_calibration(backend)
            scene = f"{family}-{i:04d}"
            for branch, action in enumerate(ACTIONS):
                backend.restore(initial)
                if backend.snapshot() != initial:
                    raise RuntimeError("initial physics restore mismatch")
                start_position = backend.data.xpos[backend.body_id].copy().tolist()
                start_velocity = backend.data.cvel[backend.body_id, 3:].copy().tolist()
                trajectory = backend.trajectory(action, horizon, renderer, camera="track")
                state = [start_position] + trajectory["position"]
                velocity = [start_velocity] + trajectory["velocity"]
                dynamics_state = [p + v for p, v in zip(state, velocity, strict=True)]
                rgb_file = f"cube-{scene}-b{branch}.npz"
                rgb_path = root / rgb_file
                np.savez_compressed(rgb_path, rgb=np.asarray(trajectory.pop("rgb"), dtype=np.uint8))
                record = {"schema": "worlddynamics-v2/v1", "scene_id": scene,
                          "task_id": "cube-force-velocity", "split": split,
                          "evaluation_family": family, "seed": seed, "branch": branch,
                          "initial_state_hex": initial.hex(), "initial_state_sha256": initial_hash,
                          "physics": {"mass": mass, "friction": friction, "box_half_size": half_size,
                                      "timestep": 0.005},
                          "action": list(action), "action_sequence": [list(action)] * horizon,
                          "state": state, "velocity": velocity[1:],
                          "initial_velocity": start_velocity, "dynamics_state": dynamics_state,
                          "contact": trajectory["contact"], "outcome": state[-1],
                          "camera": calibration, "rgb_file": rgb_file,
                          "rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                          "mujoco_version": mujoco.__version__,
                          "engine_version": mujoco.mj_versionString(),
                          "source_revision": revision}
                raw = json.dumps(record, sort_keys=True, allow_nan=False).encode()
                record["artifact_sha256"] = hashlib.sha256(raw).hexdigest()
                (root / f"cube-{scene}-b{branch}.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8")
                total += 1
            renderer.close()
    manifest = {"schema": "worlddynamics-v2/v1", "source_revision": revision,
                "scenes_per_family": scenes_per_family, "branches_per_scene": len(ACTIONS),
                "horizon": horizon, "records": total, "specs": SPECS}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate(root)
    return root
