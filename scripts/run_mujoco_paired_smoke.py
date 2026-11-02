"""G0 physical-engine smoke: requires MuJoCo, NOT a full physics data audit.

Records a provenance-bound candidate receipt; W1 still requires full trajectories,
independent labels, exact engine/model hashes, many seeds and scene-level splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from iwm.mujoco_paired_backend import MuJoCoCubeBackend
from iwm.paired_replay import ReplaySpec, paired_replay, canonical_bytes, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Exact source Git commit; never an unverified label")
    parser.add_argument("--output", required=True, type=Path, help="New JSON manifest path; existing file not overwritten")
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing evidence file")
    import mujoco
    backend = MuJoCoCubeBackend(friction=args.friction)
    receipt = paired_replay(backend,
        ReplaySpec(task_id="cube-push-g0", scene_id=f"cube-friction-{args.friction:.5f}",
                   seed=args.seed, horizon=args.horizon, split="dev",
                   backend_id="mujoco-cube-g0", source_revision=args.revision,
                   evidence_tier="SIMULATION_CANDIDATE_UNVERIFIED"),
        actions=[[0.0,0.0,0.0], [5.0,0.0,0.0], [-5.0,0.0,0.0]])
    manifest = {"status": "PHYSICS_ENGINE_SMOKE_NOT_MODEL_OR_W1_EVIDENCE",
                "mujoco_python_version": mujoco.__version__,
                "mujoco_engine_version": mujoco.mj_versionString(),
                "friction": args.friction, "horizon": args.horizon,
                "receipt": receipt}
    manifest["manifest_sha256"] = digest(canonical_bytes(manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"output":str(args.output),"status":manifest["status"],
                      "manifest_sha256":manifest["manifest_sha256"]}, indent=2))

if __name__ == "__main__":
    main()
