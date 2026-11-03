"""Frozen V2 IID/OOD/confirmation evaluation with scene-level counterfactual metrics."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from iwm.causality import branch_diagnostics
from iwm.data import validate
from iwm.evaluation import evaluate, save_evaluation
from iwm.models import make_model

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, type=Path)
p.add_argument("--runs", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--families", nargs="+", choices=["iid_test", "physics_ood", "geometry_ood", "confirmatory"],
               required=True)
a = p.parse_args()
evidence = validate(a.dataset)
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2)
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
rows = []
scene_rows = []


def family_records(family: str) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for path in sorted(a.dataset.glob(f"cube-{family}-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["evaluation_family"] == family:
            groups[record["scene_id"]].append(record)
    return groups


for family in a.families:
    groups = family_records(family)
    for run in sorted(a.runs.iterdir()):
        if not run.is_dir():
            continue
        checkpoint_path = run / "last.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = make_model(config["model"], int(config["hidden"]),
                           int(config.get("obs_dim", 3))).eval()
        model.load_state_dict(checkpoint["model"])
        metrics = evaluate(checkpoint_path, a.dataset, family=family)
        save_evaluation(metrics, a.output / f"{run.name}-{family}.json")
        label, seed_text = run.name.rsplit("-s", 1)
        seed = int(seed_text)
        deltas = []
        directions = []
        for scene, records in groups.items():
            observations = np.asarray([r[config.get("observation_key", "state")] for r in records], dtype=np.float32)
            actions = np.asarray([r["action_sequence"] for r in records], dtype=np.float32)
            control = torch.from_numpy(actions)
            if config.get("action_blind", False):
                control = torch.zeros_like(control)
            with torch.no_grad():
                predicted = model(torch.from_numpy(observations), control,
                                  teacher_forcing=False)["prediction"][:, -1, :3].numpy()
            true = np.asarray([r["state"][-1] for r in records], dtype=np.float32)
            diagnostic = branch_diagnostics(true, predicted, actions[:, 0])
            deltas.append(diagnostic["counterfactual_delta_mse"])
            directions.append(diagnostic["direction_accuracy"])
            scene_rows.append({"family": family, "condition": label, "seed": seed,
                               "scene_id": scene, "physics": records[0]["physics"],
                               "counterfactual_delta_mse": diagnostic["counterfactual_delta_mse"],
                               "direction_accuracy": diagnostic["direction_accuracy"]})
        rows.append({"family": family, "condition": label, "seed": seed,
                     "checkpoint": str(checkpoint_path), "parameter_count": sum(p.numel() for p in model.parameters()),
                     "free_position_mse": metrics["free_running_mse"],
                     "one_step_position_mse": metrics["one_step_mse"],
                     "velocity_mse": metrics["velocity_mse"],
                     "contact_f1": metrics["contact_f1"],
                     "counterfactual_delta_mse": float(np.mean(deltas)),
                     "direction_accuracy": float(np.mean([x for x in directions if x is not None]))})
    for label in ("identity", "constant_velocity"):
        values = []
        for scene, records in groups.items():
            true = np.asarray([r["state"] for r in records], dtype=np.float32)
            initial = true[:, :1]
            if label == "identity":
                predicted = np.repeat(initial, true.shape[1] - 1, axis=1)
            else:
                initial_velocity = np.asarray([r["initial_velocity"] for r in records], dtype=np.float32)
                times = np.arange(1, true.shape[1], dtype=np.float32)[None, :, None] * 0.005
                predicted = initial + initial_velocity[:, None] * times
            actions = np.asarray([r["action"] for r in records], dtype=np.float32)
            diagnostic = branch_diagnostics(true[:, -1], predicted[:, -1], actions)
            values.append((float(np.mean((predicted - true[:, 1:]) ** 2)),
                           diagnostic["counterfactual_delta_mse"]))
            scene_rows.append({"family": family, "condition": label, "seed": None,
                               "scene_id": scene, "physics": records[0]["physics"],
                               "counterfactual_delta_mse": diagnostic["counterfactual_delta_mse"],
                               "direction_accuracy": diagnostic["direction_accuracy"]})
        rows.append({"family": family, "condition": label, "seed": None,
                     "checkpoint": None, "parameter_count": 0,
                     "free_position_mse": float(np.mean([v[0] for v in values])),
                     "counterfactual_delta_mse": float(np.mean([v[1] for v in values]))})

summary = {}
for family in a.families:
    summary[family] = {}
    for label in sorted({r["condition"] for r in rows if r["family"] == family}):
        selected = [r for r in rows if r["family"] == family and r["condition"] == label]
        summary[family][label] = {"seeds": [r["seed"] for r in selected],
            "free_position_mse_mean": float(np.mean([r["free_position_mse"] for r in selected])),
            "counterfactual_delta_mse_mean": float(np.mean([r["counterfactual_delta_mse"] for r in selected]))}
    if family == "confirmatory":
        names = sorted(family_records(family))
        differences = []
        for scene in names:
            blind = np.mean([r["counterfactual_delta_mse"] for r in scene_rows
                             if r["family"] == family and r["scene_id"] == scene and r["condition"] == "blind_vel"])
            conditioned = np.mean([r["counterfactual_delta_mse"] for r in scene_rows
                                   if r["family"] == family and r["scene_id"] == scene and r["condition"] == "conditioned_vel"])
            differences.append(blind - conditioned)
        rng = np.random.default_rng(123)
        bootstrap = [float(np.mean(rng.choice(differences, size=len(differences), replace=True))) for _ in range(2000)]
        summary[family]["primary_blind_minus_conditioned_delta_mse"] = {
            "mean": float(np.mean(differences)),
            "scene_bootstrap_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "per_scene": dict(zip(names, differences, strict=True))}
result = {"schema": "worlddynamics-v2-matrix/v1", "source_revision": revision,
          "dataset": evidence, "families": a.families, "summary": summary,
          "runs": rows, "per_scene": scene_rows}
(a.output / "matrix.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
