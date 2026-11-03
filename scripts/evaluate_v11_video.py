"""Independent held-out RGB evaluation with aligned framewise metrics."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from iwm.data import validate
from iwm.visual_models import VideoFlow


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """11x11 local SSIM on [0,1] RGB, averaged over channel and pixels."""
    mu_x = F.avg_pool2d(x, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(x * x, 11, stride=1, padding=5) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, 11, stride=1, padding=5) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, 11, stride=1, padding=5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) /
            ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))).mean(dim=(1, 2, 3))


def frame_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict:
    # The initial context frame is excluded from predictive metrics.
    p, y = predicted[:, 1:].clamp(0, 1), target[:, 1:]
    mse = ((p - y) ** 2).mean(dim=(0, 2, 3, 4)).numpy()
    frame_ssim = ssim(p.reshape(-1, 3, 32, 32), y.reshape(-1, 3, 32, 32))
    frame_ssim = frame_ssim.reshape(len(p), p.shape[1]).mean(0).numpy()
    return {"mse": float(mse.mean()), "psnr_db": float(-10 * math.log10(max(float(mse.mean()), 1e-12))),
            "ssim": float(frame_ssim.mean()), "mse_by_frame": mse.tolist(),
            "psnr_db_by_frame": (-10 * np.log10(np.maximum(mse, 1e-12))).tolist(),
            "ssim_by_frame": frame_ssim.tolist()}


p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True, type=Path)
p.add_argument("--dataset", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--family", choices=["test", "iid_test", "physics_ood", "geometry_ood"], default="test")
a = p.parse_args()
evidence = validate(a.dataset)
dataset_schema = json.loads((a.dataset / "manifest.json").read_text(encoding="utf-8")).get("schema")
if dataset_schema == "worlddynamics-v2/v1" and a.family == "test":
    raise ValueError("V2 requires an explicit IID or OOD evaluation family")
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2)
checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
model = VideoFlow().eval()
model.load_state_dict(checkpoint["model"])
groups = defaultdict(list)
paths = (a.dataset.glob(f"cube-{a.family}-*.json")
         if a.family != "test" else a.dataset.glob("cube-*.json"))
for path in sorted(paths):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("evaluation_family", record["split"]) == a.family:
        groups[record["scene_id"]].append(record)
if not groups:
    raise ValueError(f"No records for evaluation family {a.family}")
scene_rows = []
with torch.no_grad():
    for scene, records in groups.items():
        rgb = np.stack([np.load(a.dataset / r["rgb_file"])["rgb"] for r in records])
        frames = torch.from_numpy(rgb).float().permute(0, 1, 4, 2, 3) / 255
        actions = torch.tensor([r["action_sequence"] for r in records], dtype=torch.float32)
        if checkpoint.get("action_blind", False):
            actions = torch.zeros_like(actions)
        reconstructed = model.codec.decode(model.codec.encode(frames))
        torch.manual_seed(700)
        forced = model.teacher_force_generate(frames, actions, steps=a.steps)
        torch.manual_seed(700)
        free = model.generate(frames[:, 0], actions, steps=a.steps)
        row = {"scene_id": scene, "branches": len(records),
               "reconstruction": frame_metrics(reconstructed, frames),
               "teacher_forced": frame_metrics(forced, frames),
               "free_running": frame_metrics(free, frames),
               "visual_branch_difference_error": float(
                   ((free[1:] - free[:1]) - (frames[1:] - frames[:1])).abs().mean())}
        scene_rows.append(row)
        with (a.output / f"{scene}.npz").open("xb") as f:
            np.savez_compressed(f, ground_truth=rgb,
                reconstructed=(reconstructed.permute(0, 1, 3, 4, 2).numpy() * 255).clip(0, 255).astype(np.uint8),
                teacher_forced=(forced.permute(0, 1, 3, 4, 2).numpy() * 255).clip(0, 255).astype(np.uint8),
                free_running=(free.permute(0, 1, 3, 4, 2).numpy() * 255).clip(0, 255).astype(np.uint8))
result = {"split": a.family, "dataset": evidence, "checkpoint": str(a.checkpoint),
          "action_blind": checkpoint.get("action_blind", False),
          "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "scenes": len(scene_rows), "branches": sum(r["branches"] for r in scene_rows),
          "flow_euler_steps": a.steps,
          "reference_alignment": f"predicted[t] versus RGB[t], t=1..{len(frames[0]) - 1}; context frame excluded",
          "metrics": {key: {metric: float(np.mean([r[key][metric] for r in scene_rows]))
                             for metric in ("mse", "psnr_db", "ssim")}
                      for key in ("reconstruction", "teacher_forced", "free_running")},
          "visual_branch_difference_error": float(np.mean(
              [r["visual_branch_difference_error"] for r in scene_rows])),
          "physical_state_error": {"status": "NOT_AVAILABLE", "reason": "video model has no state head"},
          "per_scene": scene_rows}
(a.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"scenes": result["scenes"], "branches": result["branches"],
                  "metrics": result["metrics"]}, indent=2))
