"""Generate and export a future RGB rollout from a trained compact flow model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from iwm.data import validate
from iwm.visual_models import VideoFlow

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.manual_seed(17)
validate(a.dataset)
checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
if checkpoint["model_name"] != "video_flow":
    p.error("checkpoint is not a video flow model")
model = VideoFlow()
model.load_state_dict(checkpoint["model"])
model.eval()
records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(a.dataset.glob("cube-*.json"))]
record = next(r for r in records if r["split"] == "test")
rgb = np.load(a.dataset / record["rgb_file"])["rgb"]
initial = torch.from_numpy(rgb[0]).float().permute(2, 0, 1).unsqueeze(0) / 255
actions = torch.tensor(record["action_sequence"], dtype=torch.float32).unsqueeze(0)
generated = model.generate(initial, actions)
video = (generated.squeeze(0).permute(0, 2, 3, 1).clamp(0, 1).numpy() * 255).astype(np.uint8)
mse = float(np.mean((video.astype(np.float32) / 255 - rgb.astype(np.float32) / 255) ** 2))
psnr = float(10 * np.log10(1 / mse)) if mse > 0 else float("inf")
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("xb") as f:
    np.savez_compressed(f, generated_rgb=video, target_rgb=rgb,
                        scene_id=record["scene_id"], branch=record["branch"])
print(json.dumps({"output": str(a.output), "frames": len(video), "scene": record["scene_id"],
                  "seed": 17, "mse": mse, "psnr": psnr}))
