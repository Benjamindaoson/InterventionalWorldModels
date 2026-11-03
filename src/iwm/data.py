"""Frozen, provenance checked MuJoCo cube trajectories."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .mujoco_paired_backend import MuJoCoCubeBackend
from .paired_replay import validate_scene_isolation


def _revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def generate(root: Path, scenes: int = 12, horizon: int = 16) -> Path:
    import mujoco

    if scenes < 3 or horizon < 2:
        raise ValueError("need at least 3 scenes and 2 steps")
    root.mkdir(parents=True, exist_ok=False)
    records = []
    actions = ((0., 0., 0.), (5., 0., 0.), (-5., 0., 0.))
    for i in range(scenes):
        friction = 0.3 + (i % 4) * 0.1
        backend = MuJoCoCubeBackend(friction=friction)
        renderer = mujoco.Renderer(backend.model, height=32, width=32)
        backend.reset(i)
        initial = backend.snapshot()
        initial_hash = hashlib.sha256(initial).hexdigest()
        split = "train" if i % 5 < 3 else ("val" if i % 5 == 3 else "test")
        scene_id = f"cube-{i:04d}"
        for branch, action in enumerate(actions):
            backend.restore(initial)
            if backend.snapshot() != initial:
                raise RuntimeError("MuJoCo restore mismatch")
            start = backend.data.xpos[backend.body_id].copy().tolist()
            trajectory = backend.trajectory(action, horizon, renderer)
            rgb_path = root / f"{scene_id}-b{branch}.npz"
            np.savez_compressed(rgb_path, rgb=np.asarray(trajectory.pop("rgb"), dtype=np.uint8))
            rgb_hash = hashlib.sha256(rgb_path.read_bytes()).hexdigest()
            state = [start] + trajectory["position"]
            payload = {"scene_id": scene_id, "task_id": "cube-force", "split": split,
                       "seed": i, "branch": branch, "initial_state_sha256": initial_hash,
                       "initial_state_hex": initial.hex(), "action": list(action),
                       "action_sequence": [list(action)] * horizon,
                       "physics": {"friction": friction, "timestep": 0.005, "mass": 1.0},
                       "mujoco_version": mujoco.__version__,
                       "engine_version": mujoco.mj_versionString(), "source_revision": _revision(),
                       "state": state, "velocity": trajectory["velocity"],
                       "contact": trajectory["contact"], "outcome": state[-1],
                       "rgb_file": rgb_path.name, "rgb_sha256": rgb_hash}
            raw = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
            payload["artifact_sha256"] = hashlib.sha256(raw).hexdigest()
            path = root / f"{scene_id}-b{branch}.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            records.append(payload)
        renderer.close()
    validate(root)
    (root / "manifest.json").write_text(json.dumps({"backend": "MUJOCO", "scenes": scenes,
        "horizon": horizon, "records": len(records), "source_revision": _revision()}, indent=2), encoding="utf-8")
    return root


def validate(root: Path) -> dict:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("cube-*.json"))]
    if not records:
        raise ValueError("no records")
    validate_scene_isolation(records)
    groups: dict[str, list[dict]] = {}
    for record in records:
        artifact_hash = record.pop("artifact_sha256", None)
        if not artifact_hash:
            raise ValueError("artifact hash missing")
        raw = json.dumps(record, sort_keys=True, allow_nan=False).encode()
        if hashlib.sha256(raw).hexdigest() != artifact_hash:
            raise ValueError("artifact hash mismatch")
        if "rgb_file" in record:
            rgb_path = root / record["rgb_file"]
            if not rgb_path.is_file() or hashlib.sha256(rgb_path.read_bytes()).hexdigest() != record["rgb_sha256"]:
                raise ValueError("RGB artifact hash mismatch")
        record["artifact_sha256"] = artifact_hash
        groups.setdefault(record["scene_id"], []).append(record)
    for scene, branches in groups.items():
        if len(branches) < 2 or len({b["initial_state_sha256"] for b in branches}) != 1:
            raise ValueError(f"invalid paired state: {scene}")
        if len({b["split"] for b in branches}) != 1:
            raise ValueError(f"split leakage: {scene}")
    initial_by_split: dict[str, str] = {}
    for branches in groups.values():
        record = branches[0]
        key = record["initial_state_sha256"]
        if key in initial_by_split and initial_by_split[key] != record["split"]:
            raise ValueError("identical physical initial state across frozen splits")
        initial_by_split[key] = record["split"]
    return {"records": len(records), "scenes": len(groups), "splits": validate_scene_isolation(records)}


def load_arrays(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    validate(root)
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("cube-*.json"))]
    selected = [r for r in records if r["split"] == split]
    if not selected:
        raise ValueError(f"no {split} records")
    return (np.asarray([r["state"] for r in selected], dtype=np.float32),
            np.asarray([r["action_sequence"] for r in selected], dtype=np.float32))
