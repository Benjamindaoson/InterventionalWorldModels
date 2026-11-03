"""Aggregate three-seed planning runs with independent scene as bootstrap unit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("blind_vel", "conditioned_vel", "rssm_vel")
SEEDS = (17, 29, 43)
POLICIES = ("mpc", "cem")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = args.results / f"v2-planning-{arm}-s{seed}-{args.suffix}.json"
            runs[(arm, seed)] = (path, json.loads(path.read_text(encoding="utf-8")))
    reference = runs[(ARMS[0], SEEDS[0])][1]
    reference_scenes = {(r["scene_id"], r["policy"]):
                        (r["initial_snapshot_hex"], r["goal_xy"])
                        for r in reference["episodes"]}
    for _, run in runs.values():
        scenes = {(r["scene_id"], r["policy"]):
                  (r["initial_snapshot_hex"], r["goal_xy"]) for r in run["episodes"]}
        if scenes != reference_scenes:
            raise ValueError("planning runs do not share exact initial scene and goal")
    scene_ids = sorted({scene for scene, _ in reference_scenes})
    summary = {}
    scene_values = {}
    for arm in ARMS:
        summary[arm] = {}
        for policy in POLICIES:
            selected = [runs[(arm, seed)][1]["summary"][policy] for seed in SEEDS]
            summary[arm][policy] = {
                "seed_successes": [s["successes"] for s in selected],
                "success_rate_mean": float(np.mean([s["success_rate"] for s in selected])),
                "success_rate_std_across_seeds": float(np.std([s["success_rate"] for s in selected], ddof=1)),
                "final_distance_m_mean": float(np.mean([s["mean_final_goal_distance_m"] for s in selected])),
                "regret_vs_greedy_oracle_m_mean": float(np.mean([s["mean_regret_vs_oracle_m"] for s in selected])),
                "prediction_error_m_mean": float(np.mean([s["mean_position_prediction_error_m"] for s in selected])),
                "latency_ms_mean": float(np.mean([s["mean_planning_latency_ms"] for s in selected])),
                "exploitation_failures_total": int(sum(s["model_exploitation_failures"] for s in selected)),
            }
            scene_values[(arm, policy)] = np.asarray([
                np.mean([next(r["success"] for r in runs[(arm, seed)][1]["episodes"]
                              if r["scene_id"] == scene and r["policy"] == policy)
                         for seed in SEEDS]) for scene in scene_ids], dtype=float)
    rng = np.random.default_rng(123)
    differences = {}
    for policy in POLICIES:
        values = scene_values[("conditioned_vel", policy)] - scene_values[("blind_vel", policy)]
        draws = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(2000)]
        differences[policy] = {"conditioned_minus_blind_success_mean": float(values.mean()),
                               "scene_bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
                               "per_scene": dict(zip(scene_ids, values.tolist(), strict=True))}
    oracle = reference["summary"]["oracle"]
    random = reference["summary"]["random"]
    result = {"protocol": "v2-held-out-closed-loop/v1", "scenes": len(scene_ids),
              "seeds": SEEDS, "run_files": [str(path) for path, _ in runs.values()],
              "greedy_oracle_successes": oracle["successes"],
              "random_successes": random["successes"], "summary": summary,
              "conditioned_minus_blind": differences,
              "note": "oracle is a one-step privileged greedy reference, not a global optimum"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"summary": summary, "conditioned_minus_blind": differences}, indent=2))


if __name__ == "__main__":
    main()
