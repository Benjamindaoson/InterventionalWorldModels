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
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
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
        if hashlib.sha256(bytes.fromhex(record["initial_state_hex"])).hexdigest() != record["initial_state_sha256"]:
            raise ValueError("initial state hash mismatch")
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
        if any(b["state"][0] != branches[0]["state"][0] for b in branches):
            raise ValueError(f"initial observable state mismatch: {scene}")
        if len({json.dumps(b["physics"], sort_keys=True) for b in branches}) != 1:
            raise ValueError(f"physics configuration mismatch: {scene}")
        if manifest.get("schema") in {"worlddynamics-v11/v1", "worlddynamics-v2/v1"} and len(branches) != manifest["branches_per_scene"]:
            raise ValueError(f"incomplete branch group: {scene}")
    initial_by_split: dict[str, str] = {}
    for branches in groups.values():
        record = branches[0]
        key = record["initial_state_sha256"]
        if key in initial_by_split and initial_by_split[key] != record["split"]:
            raise ValueError("identical physical initial state across frozen splits")
        initial_by_split[key] = record["split"]
    if manifest.get("schema") == "worlddynamics-v11/v1":
        physics_splits: dict[str, str] = {}
        for branches in groups.values():
            record = branches[0]
            signature = json.dumps(record["physics"], sort_keys=True)
            prior = physics_splits.get(signature)
            if prior is not None and prior != record["split"]:
                raise ValueError("physics configuration leakage across splits")
            physics_splits[signature] = record["split"]
        if len(records) != manifest["records"]:
            raise ValueError("manifest record count mismatch")
    if manifest.get("schema") == "worlddynamics-v2/v1":
        families: dict[str, set[str]] = {}
        for branches in groups.values():
            record = branches[0]
            if len({b["evaluation_family"] for b in branches}) != 1:
                raise ValueError("family mismatch inside scene")
            signature = json.dumps(record["physics"], sort_keys=True)
            families.setdefault(signature, set()).add(record["evaluation_family"])
            if len(record["dynamics_state"]) != len(record["state"]) or any(len(s) != 6 for s in record["dynamics_state"]):
                raise ValueError("invalid position/velocity state")
            if len(record["contact"]) != len(record["action_sequence"]):
                raise ValueError("contact/action time mismatch")
        for members in families.values():
            if members - {"train", "iid_val", "iid_test"} and len(members) > 1:
                raise ValueError("OOD or confirmatory physics configuration leakage")
        if len(records) != manifest["records"]:
            raise ValueError("manifest record count mismatch")
    return {"records": len(records), "scenes": len(groups), "splits": validate_scene_isolation(records)}


def load_arrays(root: Path, split: str, family: str | None = None,
                observation_key: str = "state") -> tuple[np.ndarray, np.ndarray]:
    validate(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8")) if (root / "manifest.json").exists() else {}
    if manifest.get("schema") == "worlddynamics-v2/v1":
        if split == "test" and family is None:
            raise ValueError("V2 test requires explicit evaluation family")
        family = family or ("train" if split == "train" else "iid_val")
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("cube-*.json"))]
    selected = [r for r in records if r["split"] == split and
                (family is None or r.get("evaluation_family") == family)]
    if not selected:
        raise ValueError(f"no {split} records")
    return (np.asarray([r[observation_key] for r in selected], dtype=np.float32),
            np.asarray([r["action_sequence"] for r in selected], dtype=np.float32))
