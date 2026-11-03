"""Train world-only, inverse-action-only, or joint compact visual models."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from iwm.data import validate
from iwm.visual_models import JointWorldAction

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--mode", required=True, choices=["world", "action", "joint"])
p.add_argument("--seed", type=int, default=17)
p.add_argument("--epochs", type=int, default=2)
a = p.parse_args()
evidence = validate(a.dataset)
a.output.mkdir(parents=True, exist_ok=False)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
random.seed(a.seed)
torch.set_num_threads(2)
records = [json.loads(path.read_text(encoding="utf-8"))
           for family in ("train", "iid_val")
           for path in sorted(a.dataset.glob(f"cube-{family}-*.json"))]


def loader(family: str, shuffle: bool) -> DataLoader:
    selected = [r for r in records if r["evaluation_family"] == family]
    images = np.stack([np.load(a.dataset / r["rgb_file"])["rgb"] for r in selected])
    frames = torch.from_numpy(images).permute(0, 1, 4, 2, 3)
    actions = torch.tensor([r["action_sequence"] for r in selected], dtype=torch.float32)
    return DataLoader(TensorDataset(frames, actions), batch_size=2, shuffle=shuffle)


train_loader = loader("train", True)
val_loader = loader("iid_val", False)
model = JointWorldAction()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
world_weight, action_weight = ((1., 0.) if a.mode == "world" else
                               (0., 1.) if a.mode == "action" else (1., 1.))
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
dataset_hash = hashlib.sha256((a.dataset / "manifest.json").read_bytes()).hexdigest()
best = float("inf")
steps = 0
begin = time.perf_counter()
active_parameters = 0
for epoch in range(a.epochs):
    model.train()
    losses = []
    for frames, actions in train_loader:
        frames = frames.float() / 255
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(frames, actions, world_weight, action_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite WAM loss")
        loss.backward()
        active_parameters = sum(p.numel() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        steps += 1
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        val_loss = float(np.mean([float(model.loss(frames.float() / 255, actions,
                                                   world_weight, action_weight))
                                  for frames, actions in val_loader]))
    scheduler.step()
    row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
           "val_loss": val_loss, "optimizer_steps": steps,
           "elapsed_seconds": time.perf_counter() - begin}
    with (a.output / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict(), "mode": a.mode, "epoch": epoch,
                  "seed": a.seed, "source_revision": revision,
                  "dataset_manifest_sha256": dataset_hash,
                  "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                  "python_rng": random.getstate()}
    torch.save(checkpoint, a.output / "last.pt")
    if val_loss < best:
        best = val_loss
        torch.save(checkpoint, a.output / "best.pt")
manifest = {"model": "JointWorldAction", "mode": a.mode, "seed": a.seed,
            "dataset": str(a.dataset), "dataset_manifest_sha256": dataset_hash,
            "source_revision": revision, "data": evidence, "epochs": a.epochs,
            "optimizer_steps": steps, "batch_size": 2,
            "parameter_count_total": sum(p.numel() for p in model.parameters()),
            "parameter_count_with_gradient": active_parameters,
            "best_validation_loss": best, "cpu_wall_seconds": time.perf_counter() - begin}
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest))
