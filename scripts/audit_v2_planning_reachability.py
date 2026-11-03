"""Audit goal reachability under dense force oracle and constant-force plans."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

if __package__ == "scripts":
    from scripts.evaluate_v2_planning import initial_scene, oracle_candidate_scores
else:
    from evaluate_v2_planning import initial_scene, oracle_candidate_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenes", type=int, default=8)
    args = parser.parse_args()
    candidates = np.column_stack((np.arange(-10., 11., 1.),
                                  np.zeros(21), np.zeros(21)))
    rows = []
    for index in range(args.scenes):
        backend, goal, metadata = initial_scene(index)
        initial = backend.snapshot()
        constant_scores = oracle_candidate_scores(backend, initial, candidates, goal, 32)
        best_constant = int(np.argmin(constant_scores))
        for _ in range(8):
            scores = oracle_candidate_scores(backend, backend.snapshot(), candidates, goal, 4)
            action = candidates[int(np.argmin(scores))]
            backend.trajectory(tuple(action.tolist()), 4)
        greedy_distance = float(np.linalg.norm(backend.data.xpos[backend.body_id, :2] - goal))
        rows.append({"scene_id": f"v2-planning-{index:03d}",
                     "initial_snapshot_sha256": hashlib.sha256(initial).hexdigest(),
                     "goal_xy": goal.tolist(), "mass": metadata["mass"],
                     "friction": metadata["friction"],
                     "best_constant_force_newtons": candidates[best_constant].tolist(),
                     "best_constant_final_distance_m": float(constant_scores[best_constant]),
                     "dense_greedy_final_distance_m": greedy_distance,
                     "constant_success": bool(constant_scores[best_constant] <= 0.015),
                     "dense_greedy_success": greedy_distance <= 0.015})
    result = {"protocol": "v2-planning-reachability-audit/v1",
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "force_grid_newtons": "x=-10..10 in 1 N increments; y=z=0",
              "success_radius_m": 0.015, "scenes": rows,
              "constant_successes": sum(r["constant_success"] for r in rows),
              "dense_greedy_successes": sum(r["dense_greedy_success"] for r in rows),
              "note": "neither candidate class proves global reachability"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"constant_successes": result["constant_successes"],
                      "dense_greedy_successes": result["dense_greedy_successes"]}))


if __name__ == "__main__":
    main()
