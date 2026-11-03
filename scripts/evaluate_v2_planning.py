"""Held-out MuJoCo closed-loop cube control with model-only candidate scoring."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from iwm.models import make_model
from iwm.mujoco_paired_backend import MuJoCoCubeBackend

DISCRETE_ACTIONS = np.asarray(((0., 0., 0.), (5., 0., 0.), (-5., 0., 0.),
                               (10., 0., 0.), (-10., 0., 0.),
                               (0., 10., 0.), (0., -10., 0.)), dtype=np.float32)


def model_candidate_scores(model: torch.nn.Module, current_state: np.ndarray,
                           candidates: np.ndarray, goal_xy: np.ndarray,
                           horizon: int, action_blind: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Score candidate forces without a simulator or any future observation."""
    initial = torch.as_tensor(current_state, dtype=torch.float32).expand(len(candidates), -1)
    controls = torch.as_tensor(candidates, dtype=torch.float32)[:, None].expand(-1, horizon, -1)
    if action_blind:
        controls = torch.zeros_like(controls)
    placeholder = initial[:, None].expand(-1, horizon + 1, -1)
    with torch.no_grad():
        prediction = model(placeholder, controls, teacher_forcing=False)["prediction"]
    future = prediction[:, -1, :2].numpy()
    return np.linalg.norm(future - goal_xy, axis=1), prediction[:, -1, :3].numpy()


def oracle_candidate_scores(backend: MuJoCoCubeBackend, snapshot: bytes,
                            candidates: np.ndarray, goal_xy: np.ndarray,
                            horizon: int) -> np.ndarray:
    """Privileged simulator ranking, used only by the Oracle policy."""
    scores = []
    for candidate in candidates:
        backend.restore(snapshot)
        future = backend.trajectory(tuple(float(x) for x in candidate), horizon)
        scores.append(float(np.linalg.norm(np.asarray(future["position"][-1])[:2] - goal_xy)))
    backend.restore(snapshot)
    return np.asarray(scores)


def initial_scene(index: int) -> tuple[MuJoCoCubeBackend, np.ndarray, dict]:
    mass, friction = ((0.6, 0.25), (1.3, 0.75), (1.8, 0.6), (1., 0.45))[index % 4]
    backend = MuJoCoCubeBackend(mass=mass, friction=friction)
    backend.reset(70000 + index)
    backend.data.qpos[0] = -0.12 + 0.025 * index
    backend.data.qpos[1] = -0.05 + 0.02 * (index % 4)
    backend.data.qpos[2] = 0.05
    backend.data.qvel[0] = (-0.12, 0., 0.12)[index % 3]
    backend.data.qvel[1] = 0.
    backend.mujoco.mj_forward(backend.model, backend.data)
    start = backend.data.xpos[backend.body_id].copy()
    goal = start[:2] + np.asarray((0.025 if index % 2 == 0 else -0.025, 0.))
    metadata = {"mass": mass, "friction": friction, "initial_xyz": start.tolist(),
                "initial_velocity_xyz": backend.data.cvel[backend.body_id, 3:].tolist(),
                "initial_snapshot_hex": backend.snapshot().hex()}
    return backend, goal, metadata


def run_episode(scene: int, policy: str, model: torch.nn.Module | None,
                rng: np.random.Generator, action_blind: bool = False) -> dict:
    backend, goal, metadata = initial_scene(scene)
    actions, latencies, errors, exploitation = [], [], [], 0
    cycles, steps_per_cycle = 8, 4
    for _ in range(cycles):
        position = backend.data.xpos[backend.body_id].copy()
        velocity = backend.data.cvel[backend.body_id, 3:].copy()
        current = np.concatenate((position, velocity))
        distance_before = float(np.linalg.norm(position[:2] - goal))
        start = time.perf_counter()
        selected_prediction = None
        if policy == "random":
            action = DISCRETE_ACTIONS[int(rng.integers(len(DISCRETE_ACTIONS)))]
        elif policy == "oracle":
            scores = oracle_candidate_scores(backend, backend.snapshot(),
                                             DISCRETE_ACTIONS, goal, steps_per_cycle)
            action = DISCRETE_ACTIONS[int(np.argmin(scores))]
        elif policy == "mpc":
            if model is None:
                raise ValueError("MPC requires a learned model")
            scores, predictions = model_candidate_scores(model, current, DISCRETE_ACTIONS,
                                                          goal, steps_per_cycle, action_blind)
            best = int(np.argmin(scores))
            action, selected_prediction = DISCRETE_ACTIONS[best], predictions[best]
        elif policy == "cem":
            if model is None:
                raise ValueError("CEM requires a learned model")
            mean, scale = np.zeros(2), np.full(2, 5.)
            for _ in range(3):
                sampled = np.clip(rng.normal(mean, scale, (24, 2)), -10, 10)
                candidates = np.column_stack((sampled, np.zeros(len(sampled)))).astype(np.float32)
                scores, predictions = model_candidate_scores(model, current, candidates,
                                                              goal, steps_per_cycle, action_blind)
                elite = sampled[np.argsort(scores)[:6]]
                mean, scale = elite.mean(0), np.maximum(elite.std(0), 0.3)
            best = int(np.argmin(scores))
            action, selected_prediction = candidates[best], predictions[best]
        else:
            raise ValueError(policy)
        latencies.append(time.perf_counter() - start)
        backend.trajectory(tuple(float(x) for x in action), steps_per_cycle)
        actual = backend.data.xpos[backend.body_id].copy()
        distance_after = float(np.linalg.norm(actual[:2] - goal))
        actions.append(action.tolist())
        if selected_prediction is not None:
            errors.append(float(np.linalg.norm(selected_prediction - actual)))
            exploitation += int(np.linalg.norm(selected_prediction[:2] - goal) < distance_before
                                and distance_after > distance_before)
    final = float(np.linalg.norm(backend.data.xpos[backend.body_id, :2] - goal))
    return {"scene_id": f"v2-planning-{scene:03d}", "policy": policy,
            "goal_xy": goal.tolist(), **metadata, "actions": actions,
            "final_goal_distance_m": final, "success": final <= 0.015,
            "mean_planning_latency_ms": float(np.mean(latencies) * 1000),
            "mean_position_prediction_error_m": float(np.mean(errors)) if errors else None,
            "model_exploitation_failures": exploitation}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=8)
    args = parser.parse_args()
    if args.scenes < 4:
        parser.error("at least four held-out scenes required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if int(config.get("obs_dim", 3)) != 6:
        parser.error("V2 planning requires a six-dimensional position/velocity model")
    model = make_model(config["model"], int(config["hidden"]), 6).eval()
    model.load_state_dict(checkpoint["model"])
    torch.set_num_threads(2)
    rows = []
    for scene in range(args.scenes):
        for policy in ("random", "oracle", "mpc", "cem"):
            rows.append(run_episode(scene, policy, model if policy in {"mpc", "cem"} else None,
                                    np.random.default_rng(90000 + scene),
                                    bool(config.get("action_blind", False))))
    summary = {}
    oracle = {r["scene_id"]: r["final_goal_distance_m"] for r in rows if r["policy"] == "oracle"}
    for policy in ("random", "oracle", "mpc", "cem"):
        subset = [r for r in rows if r["policy"] == policy]
        summary[policy] = {"successes": sum(r["success"] for r in subset), "episodes": len(subset),
                           "success_rate": float(np.mean([r["success"] for r in subset])),
                           "mean_final_goal_distance_m": float(np.mean([r["final_goal_distance_m"] for r in subset])),
                           "mean_planning_latency_ms": float(np.mean([r["mean_planning_latency_ms"] for r in subset])),
                           "mean_position_prediction_error_m": (float(np.mean([r["mean_position_prediction_error_m"]
                               for r in subset])) if policy in {"mpc", "cem"} else None),
                           "model_exploitation_failures": sum(r["model_exploitation_failures"] for r in subset),
                           "mean_regret_vs_oracle_m": float(np.mean([
                               r["final_goal_distance_m"] - oracle[r["scene_id"]] for r in subset]))}
    result = {"protocol": "v2-held-out-closed-loop/v1", "checkpoint": str(args.checkpoint),
              "checkpoint_source_revision": checkpoint["source_revision"],
              "evaluation_source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "observation": "current cube xyz and linear velocity only",
              "model_candidate_information": "current observation, proposed action, checkpoint weights",
              "action_blind": bool(config.get("action_blind", False)),
              "oracle_candidate_information": "full MuJoCo snapshot and exact candidate rollout",
              "control_cycles": 8, "steps_per_cycle": 4, "success_radius_m": 0.015,
              "summary": summary, "episodes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
