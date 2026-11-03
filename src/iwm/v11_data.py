"""V1.1 scene/physics-disjoint paired MuJoCo data generator."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from .data import validate
from .mujoco_paired_backend import MuJoCoCubeBackend

PHYSICS = {
    "train": ((0.5, 0.3), (1.0, 0.5), (1.5, 0.4), (0.8, 0.7)),
    "val": ((1.2, 0.2), (1.8, 0.6)),
    "test": ((2.0, 0.8), (0.6, 0.9)),
}
ACTIONS = ((0., 0., 0.), (2., 0., 0.), (-2., 0., 0.),
           (5., 0., 0.), (-5., 0., 0.), (0., 5., 0.), (0., -5., 0.))


def generate_v11(root: Path, train_scenes: int = 20, val_scenes: int = 6,
                 test_scenes: int = 6, horizon: int = 32) -> Path:
    if min(train_scenes, val_scenes, test_scenes) < 3 or horizon < 4:
        raise ValueError("need >=3 scenes per split and horizon >=4")
    root.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    count = 0
    for split, scenes in (("train", train_scenes), ("val", val_scenes), ("test", test_scenes)):
        for i in range(scenes):
            mass, friction = PHYSICS[split][i % len(PHYSICS[split])]
            backend = MuJoCoCubeBackend(friction=friction, mass=mass)
            renderer = mujoco.Renderer(backend.model, height=32, width=32)
            seed = 1000 * (1 + ("train", "val", "test").index(split)) + i
            backend.reset(seed)
            backend.data.qpos[0] = -0.1 + 0.02 * (i % 11)
            backend.data.qpos[1] = -0.08 + 0.02 * (i // 11) + 0.001 * ("train", "val", "test").index(split)
            backend.data.qpos[2] = (0.05, 0.055, 0.4)[i % 3]
            mujoco.mj_forward(backend.model, backend.data)
            initial = backend.snapshot()
            initial_hash = hashlib.sha256(initial).hexdigest()
            scene = f"{split}-scene-{i:04d}"
            for branch, action in enumerate(ACTIONS):
                backend.restore(initial)
                if backend.snapshot() != initial:
                    raise RuntimeError("physics state restore failed")
                start = backend.data.xpos[backend.body_id].copy().tolist()
                trajectory = backend.trajectory(action, horizon, renderer)
                rgb_file = f"cube-{scene}-b{branch}.npz"
                rgb_path = root / rgb_file
                np.savez_compressed(rgb_path, rgb=np.asarray(trajectory.pop("rgb"), dtype=np.uint8))
                state = [start] + trajectory["position"]
                record = {"schema": "worlddynamics-v11/v1", "scene_id": scene,
                          "task_id": "cube-force", "split": split, "seed": seed,
                          "branch": branch, "initial_state_hex": initial.hex(),
                          "initial_state_sha256": initial_hash, "action": list(action),
                          "action_sequence": [list(action)] * horizon,
                          "physics": {"mass": mass, "friction": friction, "timestep": 0.005},
                          "mujoco_version": mujoco.__version__,
                          "engine_version": mujoco.mj_versionString(),
                          "source_revision": revision, "state": state,
                          "velocity": trajectory["velocity"],
                          "contact": trajectory["contact"], "outcome": state[-1],
                          "rgb_file": rgb_file,
                          "rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest()}
                raw = json.dumps(record, sort_keys=True, allow_nan=False).encode()
                record["artifact_sha256"] = hashlib.sha256(raw).hexdigest()
                (root / f"cube-{scene}-b{branch}.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8")
                count += 1
            renderer.close()
    manifest = {"schema": "worlddynamics-v11/v1", "backend": "MUJOCO",
                "source_revision": revision, "train_scenes": train_scenes,
                "val_scenes": val_scenes, "test_scenes": test_scenes,
                "horizon": horizon, "branches_per_scene": len(ACTIONS), "records": count,
                "physics_by_split": PHYSICS}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate(root)
    return root
