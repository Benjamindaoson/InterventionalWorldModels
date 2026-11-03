"""Train actual RGB predictor, latent video flow, or joint world-action baseline."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from iwm.data import validate
from iwm.visual_models import JointWorldAction, VideoFlow, VisualPredictor

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["visual", "video_flow", "joint_wam"], required=True)
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--seed", type=int, default=17)
a = p.parse_args()
if a.output.exists():
    p.error("output exists")
evidence = validate(a.dataset)
random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = {"visual": VisualPredictor, "video_flow": VideoFlow, "joint_wam": JointWorldAction}[a.model]().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
a.output.mkdir(parents=True)
records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(a.dataset.glob("cube-*.json"))]
for epoch in range(a.epochs):
    random.shuffle(records)
    losses = []
    for record in records:
        if record["split"] != "train":
            continue
        frames = np.load(a.dataset / record["rgb_file"])["rgb"]
        frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0).to(device) / 255
        actions = torch.tensor(record["action_sequence"], dtype=torch.float32, device=device).unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(frames, actions)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite visual loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    scheduler.step()
    with (a.output / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses))}) + "\n")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(), "model_name": a.model,
                "dataset": evidence}, a.output / "last.pt")
print(json.dumps({"model": a.model, "epochs": a.epochs, "last_loss": float(np.mean(losses)),
                  "checkpoint": str(a.output / "last.pt"), "device": str(device)}))
