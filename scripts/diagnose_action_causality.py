"""Evaluate same-state counterfactual response and physical strata."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from iwm.causality import branch_diagnostics
from iwm.data import validate
from iwm.models import make_model
from iwm.mujoco_paired_backend import MuJoCoCubeBackend

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, type=Path)
p.add_argument("--runs", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
validate(a.dataset)
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2)
groups = defaultdict(list)
for path in sorted(a.dataset.glob("cube-*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["split"] == "test":
        groups[record["scene_id"]].append(record)


def load(run: Path):
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = make_model(config["model"], int(config["hidden"])).eval()
    model.load_state_dict(checkpoint["model"])
    return model, config


models = {name: load(a.runs / f"{name}-s17") for name in
          ("blind", "conditioned", "paired", "shuffled_pair", "rssm")}
rows = []
for scene, records in groups.items():
    state = np.asarray([r["state"] for r in records], dtype=np.float32)
    actions = np.asarray([r["action_sequence"] for r in records], dtype=np.float32)
    true_final = state[:, -1]
    action_values = actions[:, 0]
    for name in ("identity", "constant_velocity", *models):
        if name in {"identity", "constant_velocity"}:
            prediction = np.repeat(state[:, :1], actions.shape[1], axis=1)
            if name == "constant_velocity":
                physics = records[0]["physics"]
                backend = MuJoCoCubeBackend(mass=physics["mass"], friction=physics["friction"])
                backend.restore(bytes.fromhex(records[0]["initial_state_hex"]))
                velocity = backend.data.qvel[:3].copy()
                prediction += np.arange(1, actions.shape[1] + 1)[None, :, None] * 0.005 * velocity
            jacobian = 0.0
        else:
            model, config = models[name]
            inputs = torch.from_numpy(state)
            control = torch.from_numpy(actions)
            if config.get("action_blind", False):
                control = torch.zeros_like(control)
            with torch.no_grad():
                prediction = model(inputs, control, teacher_forcing=False)["prediction"].numpy()
            if config.get("action_blind", False):
                jacobian = 0.0
            else:
                force = torch.zeros(1, 3, requires_grad=True)
                repeated = force[:, None].expand(1, actions.shape[1], 3)
                output = model(inputs[:1], repeated, teacher_forcing=False)["prediction"][0, -1, 0]
                jacobian = float(torch.autograd.grad(output, force)[0][0, 0])
        metrics = branch_diagnostics(true_final, prediction[:, -1], action_values)
        per_branch = []
        for branch, record in enumerate(records):
            displacement = float(np.linalg.norm(state[branch, -1] - state[branch, 0]))
            category = "contact" if any(record["contact"]) else "non_contact"
            per_branch.append({"branch": record["branch"], "action_n": record["action"],
                               "contact_category": category,
                               "near_contact": abs(record["state"][0][2] - 0.055) < 1e-4,
                               "action_magnitude_n": float(np.linalg.norm(action_values[branch])),
                               "dynamic": displacement > 0.01,
                               "trajectory_mse": float(np.mean((prediction[branch] - state[branch, 1:]) ** 2))})
        rows.append({"scene_id": scene, "model": name,
                     "physics": records[0]["physics"], "initial_z": records[0]["state"][0][2],
                     "jacobian_dx_dfx_m_per_n": jacobian, **metrics,
                     "branches": per_branch})
with (a.output / "per_scene.jsonl").open("x", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
summary = {}
for name in ("identity", "constant_velocity", *models):
    subset = [r for r in rows if r["model"] == name]
    means = {}
    for key in ("counterfactual_delta_mse", "true_delta_norm_m", "predicted_delta_norm_m",
                "response_magnitude_error_m", "direction_accuracy", "x_outcome_ranking_accuracy",
                "true_action_sensitivity_m_per_n", "predicted_action_sensitivity_m_per_n",
                "jacobian_dx_dfx_m_per_n"):
        values = [r[key] for r in subset if r[key] is not None]
        means[key] = float(np.mean(values)) if values else None
    strata = {}
    branches = [b for r in subset for b in r["branches"]]
    selectors = {"contact": lambda b: b["contact_category"] == "contact",
                 "non_contact": lambda b: b["contact_category"] == "non_contact",
                 "near_contact": lambda b: b["near_contact"],
                 "low_action": lambda b: b["action_magnitude_n"] <= 2,
                 "high_action": lambda b: b["action_magnitude_n"] > 2,
                 "static": lambda b: not b["dynamic"],
                 "dynamic": lambda b: b["dynamic"]}
    for label, selector in selectors.items():
        selected = [b["trajectory_mse"] for b in branches if selector(b)]
        strata[label] = {"branches": len(selected), "mse": float(np.mean(selected)) if selected else None}
    summary[name] = {**means, "strata": strata}
    summary[name]["by_mass"] = {str(mass): float(np.mean([
        b["trajectory_mse"] for r in subset if r["physics"]["mass"] == mass
        for b in r["branches"]])) for mass in sorted({r["physics"]["mass"] for r in subset})}
    summary[name]["by_friction"] = {str(friction): float(np.mean([
        b["trajectory_mse"] for r in subset if r["physics"]["friction"] == friction
        for b in r["branches"]])) for friction in sorted({r["physics"]["friction"] for r in subset})}
(a.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
