"""Frozen split, independent rollout evaluation of state prediction."""
from __future__ import annotations

import json
from pathlib import Path

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
    with torch.no_grad():
        target = torch.from_numpy(states)
        control = torch.from_numpy(actions)
        if config.get("action_blind", False):
            control = torch.zeros_like(control)
        forced = model(target, control, teacher_forcing=True)["prediction"]
        free = model(target, control, teacher_forcing=False)["prediction"]
        actual = target[:, 1:]
        errors = ((free - actual) ** 2).mean(dim=(0, 2)).numpy().tolist()
        result = {"split": split, "dataset": evidence, "checkpoint": str(checkpoint_path),
                  "samples": len(states), "one_step_mse": float(((forced[:, 0] - actual[:, 0]) ** 2).mean()),
                  "one_step_mae": float((forced[:, 0] - actual[:, 0]).abs().mean()),
                  "teacher_forcing_mse": float(((forced - actual) ** 2).mean()),
                  "free_running_mse": float(((free - actual) ** 2).mean()),
                  "drift_mse_by_horizon": errors,
                  "contact_f1": {"status": "NOT_AVAILABLE", "reason": "model has no contact head"},
                  "psnr": {"status": "NOT_AVAILABLE", "reason": "state-only model"},
                  "planning_success": {"status": "NOT_AVAILABLE", "reason": "planning not evaluated"}}
    return result


def save_evaluation(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
