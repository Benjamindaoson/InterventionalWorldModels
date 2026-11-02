"""Auditable same-initial-state action branches for world-model research.

The built-in backend is a *toy fixture*, not a physics simulator.
This module establishes a replay-integrity contract; it is not evidence of
neural training, improved physics prediction, or downstream policy gains.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Protocol, Sequence


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: bytes) -> str:
    return sha256(value).hexdigest()


class ReplayBackend(Protocol):
    """Backend must capture *all* state that affects deterministic execution."""

    def reset(self, seed: int) -> None: ...
    def snapshot(self) -> bytes: ...
    def restore(self, state: bytes) -> None: ...
    def execute(self, action: tuple[float, ...], horizon: int) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ReplaySpec:
    task_id: str
    scene_id: str
    seed: int
    horizon: int
    split: str
    backend_id: str
    source_revision: str
    evidence_tier: str

    def validate(self) -> None:
        if not self.task_id.strip() or not self.scene_id.strip() or not self.backend_id.strip():
            raise ValueError("task_id, scene_id, backend_id must be set")
        if not self.source_revision.strip() or not self.evidence_tier.strip():
            raise ValueError("source_revision and evidence_tier must be set")
        if self.horizon < 1 or self.seed < 0:
            raise ValueError("horizon must be positive and seed nonnegative")
        if self.split not in {"train", "val", "test", "dev"}:
            raise ValueError("invalid dataset split")


def _validate_outcome(outcome: Mapping[str, Any]) -> dict[str, Any]:
    required = {"displacement", "event", "final_state", "steps"}
    if not required.issubset(outcome):
        raise ValueError(f"missing outcome keys: {sorted(required - set(outcome))}")
    displacement = float(outcome["displacement"])
    if not math.isfinite(displacement):
        raise ValueError("nonfinite displacement")
    if type(outcome["event"]) is not bool:
        raise ValueError("event must be an observed boolean")
    steps = outcome["steps"]
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    final_state = [float(v) for v in outcome["final_state"]]
    if not final_state or any(not math.isfinite(v) for v in final_state):
        raise ValueError("nonfinite or empty final_state")
    return {"displacement": displacement, "event": outcome["event"],
            "final_state": final_state, "steps": steps}


def paired_replay(backend: ReplayBackend, spec: ReplaySpec,
                  actions: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Replay actions from the *same state*, refusing weak provenance.

    A new reset occurs once, followed by exact state restoration per branch.
    These checks prove a backend contract, not a physical-system guarantee.
    """
    spec.validate()
    parsed: list[tuple[float, ...]] = []
    for action in actions:
        a = tuple(float(v) for v in action)
        if not a or not all(math.isfinite(v) for v in a):
            raise ValueError("empty or nonfinite action")
        parsed.append(a)
    if len(parsed) < 2 or len(set(parsed)) != len(parsed):
        raise ValueError("need at least two distinct actions")
    if len({len(a) for a in parsed}) != 1:
        raise ValueError("action dimensions disagree")

    backend.reset(spec.seed)
    initial = backend.snapshot()
    if not isinstance(initial, bytes) or not initial:
        raise ValueError("backend snapshot must return nonempty bytes")
    initial_sha = digest(initial)
    branches: list[dict[str, Any]] = []
    for index, action in enumerate(parsed):
        backend.restore(initial)
        restored = backend.snapshot()
        if restored != initial:
            raise RuntimeError(f"initial-state restoration failed on branch {index}")
        outcome = _validate_outcome(backend.execute(action, spec.horizon))
        branches.append({"branch_id": index, "action": list(action),
                         "initial_state_sha256": initial_sha, "outcome": outcome})

    comparisons = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            comparisons.append({"branch_a": i, "branch_b": j,
                                "displacement_delta": branches[i]["outcome"]["displacement"]
                                                      - branches[j]["outcome"]["displacement"],
                                "event_disagrees": branches[i]["outcome"]["event"]
                                                   != branches[j]["outcome"]["event"]})
    receipt: dict[str, Any] = {
        "schema_version": "rs-g0-paired-replay/v1",
        "research_status": "CONTRACT_ONLY_NOT_MODEL_EVIDENCE",
        "task_id": spec.task_id, "scene_id": spec.scene_id,
        "seed": spec.seed, "horizon": spec.horizon, "split": spec.split,
        "backend_id": spec.backend_id,
        "source_revision": spec.source_revision,
        "evidence_tier": spec.evidence_tier,
        "initial_state_sha256": initial_sha,
        "branches": branches, "comparisons": comparisons,
    }
    receipt["receipt_sha256"] = digest(canonical_bytes(receipt))
    return receipt


def validate_scene_isolation(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Prevent scene-level leakage between train/val/test; dev is also isolated."""
    scene_splits: dict[str, str] = {}
    for r in records:
        scene, split = str(r["scene_id"]), str(r["split"])
        if not scene:
            raise ValueError("empty scene id")
        if split not in {"train", "val", "test", "dev"}:
            raise ValueError("unknown split")
        if scene in scene_splits and scene_splits[scene] != split:
            raise ValueError(f"scene leakage: {scene}: {scene_splits[scene]} vs {split}")
        scene_splits[scene] = split
    return {key: sum(split == key for split in scene_splits.values())
            for key in ("train", "val", "test", "dev")}


class ToyBackend:
    """Deterministic API-only fixture, NOT MuJoCo and NOT physical evidence."""
    def __init__(self) -> None:
        self.x = 0.0

    def reset(self, seed: int) -> None:
        self.x = (seed % 7) / 10.0

    def snapshot(self) -> bytes:
        return canonical_bytes({"toy_state_x": self.x})

    def restore(self, state: bytes) -> None:
        self.x = float(json.loads(state)["toy_state_x"])

    def execute(self, action: tuple[float, ...], horizon: int) -> Mapping[str, Any]:
        delta = (action[0] - action[-1] * 0.2) * horizon / 10.0
        self.x += delta
        return {"displacement": delta, "event": delta > 0.05,
                "final_state": [self.x], "steps": horizon}
