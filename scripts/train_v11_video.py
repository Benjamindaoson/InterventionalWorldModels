"""Validated CPU-friendly training of the compact latent video flow baseline."""
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
from iwm.visual_models import VideoFlow

p = argparse.ArgumentParser()
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch-size", type=int, default=2)
a = p.parse_args()
evidence = validate(a.dataset)
a.output.mkdir(parents=True, exist_ok=False)
random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.set_num_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
records = [json.loads(path.read_text(encoding="utf-8"))
           for path in sorted(a.dataset.glob("cube-*.json"))]


def make_loader(split: str, shuffle: bool) -> DataLoader:
    selected = [r for r in records if r["split"] == split]
    rgb = np.stack([np.load(a.dataset / r["rgb_file"])["rgb"] for r in selected])
    frames = torch.from_numpy(rgb).permute(0, 1, 4, 2, 3)
    actions = torch.tensor([r["action_sequence"] for r in selected], dtype=torch.float32)
    return DataLoader(TensorDataset(frames, actions), batch_size=a.batch_size, shuffle=shuffle)


train_loader = make_loader("train", True)
val_loader = make_loader("val", False)
model = VideoFlow().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
best = float("inf")
begin = time.perf_counter()
steps = 0
manifest_hash = hashlib.sha256((a.dataset / "manifest.json").read_bytes()).hexdigest()
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
for epoch in range(a.epochs):
    model.train()
    losses = []
    for frames, actions in train_loader:
        frames = frames.to(device).float() / 255
        actions = actions.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(frames, actions)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite video loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        steps += 1
        losses.append(float(loss.detach()))
    model.eval()
    values = []
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(a.seed + 10000)
        for frames, actions in val_loader:
            values.append(float(model.loss(frames.to(device).float() / 255,
                                           actions.to(device))))
    val_loss = float(np.mean(values))
    scheduler.step()
    row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
           "val_loss": val_loss, "optimizer_steps": steps,
           "elapsed_seconds": time.perf_counter() - begin}
    with (a.output / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict(), "epoch": epoch,
                  "seed": a.seed, "dataset_manifest_sha256": manifest_hash,
                  "source_revision": revision,
                  "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                  "python_rng": random.getstate()}
    torch.save(checkpoint, a.output / "last.pt")
    if val_loss < best:
        best = val_loss
        torch.save(checkpoint, a.output / "best.pt")
manifest = {"model": "video_flow", "dataset": str(a.dataset),
            "dataset_manifest_sha256": manifest_hash, "data": evidence,
            "source_revision": revision,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "seed": a.seed, "epochs": a.epochs, "batch_size": a.batch_size,
            "optimizer_steps": steps, "best_val_loss": best,
            "wall_seconds": time.perf_counter() - begin, "device": str(device)}
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest))
