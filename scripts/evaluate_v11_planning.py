"""Independent closed-loop MuJoCo action selection; model policies never query futures."""
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

ACTIONS = np.asarray(((0., 0., 0.), (5., 0., 0.), (-5., 0., 0.),
                      (0., 5., 0.), (0., -5., 0.)), dtype=np.float32)


def model_scores(model, position: np.ndarray, actions: np.ndarray,
                 goal: np.ndarray, horizon: int) -> np.ndarray:
    initial = torch.tensor(position, dtype=torch.float32).repeat(len(actions), 1)
    sequence = torch.tensor(actions, dtype=torch.float32)[:, None].expand(-1, horizon, -1)
    placeholder = initial[:, None].expand(-1, horizon + 1, -1)
    with torch.no_grad():
        future = model(placeholder, sequence, teacher_forcing=False)["prediction"][:, -1, :2].numpy()
    return np.linalg.norm(future - goal, axis=1)


def oracle_scores(backend, snapshot: bytes, actions: np.ndarray,
                  goal: np.ndarray, horizon: int) -> np.ndarray:
    values = []
    for action in actions:
        backend.restore(snapshot)
        outcome = backend.trajectory(tuple(map(float, action)), horizon)
        values.append(float(np.linalg.norm(np.asarray(outcome["position"][-1][:2]) - goal)))
    backend.restore(snapshot)
    return np.asarray(values)


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = make_model(config["model"], int(config["hidden"])).eval()
    model.load_state_dict(checkpoint["model"])
    return model


def run_episode(scene: int, policy: str, model, rng: np.random.Generator) -> dict:
    mass, friction = ((1.7, 0.85), (0.7, 0.25), (1.3, 0.95))[scene % 3]
    backend = MuJoCoCubeBackend(mass=mass, friction=friction)
    backend.reset(5000 + scene)
    backend.data.qpos[0] = -0.07 + 0.028 * scene
    backend.data.qpos[1] = 0.04 * ((scene % 3) - 1)
    backend.data.qpos[2] = (0.05, 0.055, 0.4)[scene % 3]
    backend.mujoco.mj_forward(backend.model, backend.data)
    initial = backend.data.xpos[backend.body_id].copy()
    goal = initial[:2] + np.asarray([0.04 if scene % 2 == 0 else -0.04, 0.0])
    latencies, exploitation, prediction_error, actions_taken = [], 0, [], []
    for _ in range(8):
        position = backend.data.xpos[backend.body_id].copy()
        snapshot = backend.snapshot()
        start = time.perf_counter()
        if policy == "random":
            action = ACTIONS[int(rng.integers(len(ACTIONS)))]
            predicted_distance = None
        elif policy == "oracle":
            scores = oracle_scores(backend, snapshot, ACTIONS, goal, horizon=16)
            action = ACTIONS[int(scores.argmin())]
            predicted_distance = None
        elif policy == "mpc":
            scores = model_scores(model, position, ACTIONS, goal, horizon=16)
            action = ACTIONS[int(scores.argmin())]
            predicted_distance = float(model_scores(model, position, action[None], goal, 4)[0])
        elif policy == "cem":
            mean = np.zeros(2)
            std = np.full(2, 3.0)
            action = ACTIONS[0]
            for _ in range(2):
                samples = np.clip(rng.normal(mean, std, size=(12, 2)), -5, 5)
                candidates = np.column_stack((samples, np.zeros(len(samples)))).astype(np.float32)
                scores = model_scores(model, position, candidates, goal, horizon=16)
                elite = samples[np.argsort(scores)[:4]]
                mean, std = elite.mean(0), elite.std(0).clip(0.2)
                action = candidates[int(scores.argmin())]
            predicted_distance = float(model_scores(model, position, action[None], goal, 4)[0])
        else:
            raise ValueError(policy)
        latencies.append(time.perf_counter() - start)
        before = float(np.linalg.norm(position[:2] - goal))
        backend.trajectory(tuple(map(float, action)), 4)
        after_position = backend.data.xpos[backend.body_id].copy()
        after = float(np.linalg.norm(after_position[:2] - goal))
        actions_taken.append(action.tolist())
        if predicted_distance is not None:
            prediction_error.append(abs(predicted_distance - after))
            exploitation += int(predicted_distance < before and after > before)
    final_distance = float(np.linalg.norm(backend.data.xpos[backend.body_id, :2] - goal))
    return {"scene_id": f"planning-{scene:03d}", "policy": policy,
            "mass": mass, "friction": friction, "initial_xyz": initial.tolist(),
            "goal_xy": goal.tolist(), "final_goal_distance_m": final_distance,
            "success": final_distance <= 0.02,
            "mean_planning_latency_ms": float(np.mean(latencies) * 1000),
            "mean_prediction_distance_error_m": float(np.mean(prediction_error)) if prediction_error else None,
            "model_exploitation_failures": exploitation,
            "actions": actions_taken}


p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--scenes", type=int, default=6)
a = p.parse_args()
if a.scenes < 3:
    p.error("need at least 3 new scenes")
torch.set_num_threads(2)
model = load_model(a.checkpoint)
rows = []
for scene in range(a.scenes):
    for policy in ("random", "oracle", "mpc", "cem"):
        rows.append(run_episode(scene, policy, model, np.random.default_rng(9000 + scene)))
summary = {}
for policy in ("random", "oracle", "mpc", "cem"):
    selected = [r for r in rows if r["policy"] == policy]
    summary[policy] = {"success_rate": float(np.mean([r["success"] for r in selected])),
                       "mean_final_goal_distance_m": float(np.mean([r["final_goal_distance_m"] for r in selected])),
                       "mean_planning_latency_ms": float(np.mean([r["mean_planning_latency_ms"] for r in selected])),
                       "mean_prediction_distance_error_m": (float(np.mean([r["mean_prediction_distance_error_m"]
                            for r in selected])) if policy in ("mpc", "cem") else None),
                       "model_exploitation_failures": sum(r["model_exploitation_failures"] for r in selected)}
oracle = {r["scene_id"]: r["final_goal_distance_m"] for r in rows if r["policy"] == "oracle"}
for policy in ("random", "mpc", "cem"):
    summary[policy]["mean_regret_vs_oracle_m"] = float(np.mean([
        r["final_goal_distance_m"] - oracle[r["scene_id"]] for r in rows if r["policy"] == policy]))
result = {"status": "CLOSED_LOOP_MUJOCO_PILOT", "checkpoint": str(a.checkpoint),
          "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "policy_observation": "current cube center xyz only; model proposals use no physics simulator",
          "oracle_access": "full simulator snapshot and candidate future rollouts",
          "control_cycles": 8, "steps_per_cycle": 4, "planning_horizon": 16,
          "success_radius_m": 0.02, "summary": summary, "episodes": rows}
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("x", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(json.dumps(summary, indent=2))
