"""Evaluate trained world/action heads on exploratory V2 RGB families."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from iwm.data import validate
from iwm.visual_models import JointWorldAction

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, type=Path)
p.add_argument("--checkpoint", required=True, type=Path)
p.add_argument("--family", required=True, choices=["iid_test", "physics_ood", "geometry_ood"])
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
validate(a.dataset)
torch.set_num_threads(2)
checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
model = JointWorldAction().eval()
model.load_state_dict(checkpoint["model"])
records = [json.loads(path.read_text(encoding="utf-8"))
           for path in sorted(a.dataset.glob(f"cube-{a.family}-*.json"))]
scene_rows = []
for scene in sorted({r["scene_id"] for r in records}):
    selected = [r for r in records if r["scene_id"] == scene]
    rgb = np.stack([np.load(a.dataset / r["rgb_file"])["rgb"] for r in selected])
    frames = torch.from_numpy(rgb).float().permute(0, 1, 4, 2, 3) / 255
    actions = torch.tensor([r["action_sequence"] for r in selected], dtype=torch.float32)
    row = {"scene_id": scene, "branches": len(selected)}
    with torch.no_grad():
        if checkpoint["mode"] in {"world", "joint"}:
            predicted = model.predict_world(frames[:, :-1], actions)
            actual = model._latent(frames[:, 1:])
            decoded = model.visual.decode(predicted)
            row["world_latent_mse"] = float(F.mse_loss(predicted, actual))
            row["decoded_next_frame_mse"] = float(F.mse_loss(decoded, frames[:, 1:]))
            predicted_difference = decoded[1:] - decoded[:1]
            real_difference = frames[1:, 1:] - frames[:1, 1:]
            row["branch_difference_mse"] = float(F.mse_loss(predicted_difference, real_difference))
            row["predicted_branch_difference_abs"] = float(predicted_difference.abs().mean())
            row["real_branch_difference_abs"] = float(real_difference.abs().mean())
        if checkpoint["mode"] in {"action", "joint"}:
            predicted_action = model.predict_actions(frames)
            row["action_mae_newtons"] = float((predicted_action - actions).abs().mean())
            row["zero_action_mae_newtons"] = float(actions.abs().mean())
    scene_rows.append(row)
metrics = {key: float(np.mean([r[key] for r in scene_rows]))
           for key in ("world_latent_mse", "decoded_next_frame_mse", "branch_difference_mse",
                       "predicted_branch_difference_abs", "real_branch_difference_abs",
                       "action_mae_newtons", "zero_action_mae_newtons")
           if key in scene_rows[0]}
result = {"mode": checkpoint["mode"], "family": a.family,
          "source_revision": checkpoint["source_revision"],
          "dataset_manifest_sha256": checkpoint["dataset_manifest_sha256"],
          "checkpoint": str(a.checkpoint), "scenes": len(scene_rows),
          "metrics": metrics, "per_scene": scene_rows,
          "note": "one-step teacher-forced visual token prediction; no free video rollout claim"}
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("x", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(json.dumps({"mode": result["mode"], "family": a.family, "metrics": metrics}))
