"""Frozen split, independent rollout evaluation of state prediction."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import load_arrays, validate
from .models import make_model


def evaluate(checkpoint_path: Path, dataset: Path, split: str = "test") -> dict:
    if split not in {"val", "test"}:
        raise ValueError("frozen evaluation requires val or test split")
    evidence = validate(dataset)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = make_model(config["model"], int(config.get("hidden", 64)))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    states, actions = load_arrays(dataset, split)
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(dataset.glob("cube-*.json"))]
    records = [record for record in records if record["split"] == split]
    if len(records) != len(states):
        raise ValueError("record/array alignment mismatch")
    with torch.no_grad():
        target = torch.from_numpy(states)
        control = torch.from_numpy(actions)
        if config.get("action_blind", False):
            control = torch.zeros_like(control)
        forced = model(target, control, teacher_forcing=True)["prediction"]
        free = model(target, control, teacher_forcing=False)["prediction"]
        actual = target[:, 1:]
        forced_error = (forced - actual) ** 2
        free_error = (free - actual) ** 2
        errors = free_error.mean(dim=(0, 2)).numpy().tolist()
        horizons = {str(h): errors[h - 1] for h in (1, 4, 8, 16, 32) if h <= len(errors)}
        scene_indices: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            scene_indices[record["scene_id"]].append(index)
        per_scene = []
        branch_errors = []
        for scene_id, indices in scene_indices.items():
            predicted_end = free[indices, -1].numpy()
            true_end = actual[indices, -1].numpy()
            branch_errors.extend(np.linalg.norm(
                (predicted_end[1:] - predicted_end[:1]) -
                (true_end[1:] - true_end[:1]), axis=-1).tolist())
            per_scene.append({"scene_id": scene_id, "physics": records[indices[0]]["physics"],
                              "branches": len(indices),
                              "one_step_mse": float(forced_error[indices].mean()),
                              "free_running_mse": float(free_error[indices].mean()),
                              "horizon_mse": {str(h): float(free_error[indices, h - 1].mean())
                                              for h in (1, 4, 8, 16, 32) if h <= len(errors)}})
        contact_true = np.asarray([r["contact"] for r in records], dtype=bool)
        contact_pred = (free[..., 2].numpy() <= 0.052)
        tp = int(np.logical_and(contact_pred, contact_true).sum())
        fp = int(np.logical_and(contact_pred, ~contact_true).sum())
        fn = int(np.logical_and(~contact_pred, contact_true).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        contact_f1 = (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None
        result = {"split": split, "dataset": evidence, "checkpoint": str(checkpoint_path),
                  "samples": len(states), "scenes": len(scene_indices),
                  "one_step_mse": float(forced_error.mean()),
                  "one_step_mae": float((forced - actual).abs().mean()),
                  "first_step_mse": float(forced_error[:, 0].mean()),
                  "teacher_forcing_mse": float(((forced - actual) ** 2).mean()),
                  "free_running_mse": float(((free - actual) ** 2).mean()),
                  "drift_mse_by_horizon": errors, "horizon_mse": horizons,
                  "per_scene": per_scene,
                  "action_branch_difference_error_meters": float(np.mean(branch_errors)),
                  "contact_f1": contact_f1, "contact_precision": precision,
                  "contact_recall": recall, "contact_rule": "predicted cube center z <= 0.052 m",
                  "calibration": {"status": "NOT_AVAILABLE", "reason": "no calibrated predictive probability"},
                  "psnr": {"status": "NOT_AVAILABLE", "reason": "state-only model"},
                  "planning_success": {"status": "NOT_AVAILABLE", "reason": "planning not evaluated"}}
    return result


def save_evaluation(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
