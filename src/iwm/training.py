"""Config-driven CPU/CUDA training with resumable optimizer and RNG state."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import load_arrays, validate
from .models import make_model


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: dict) -> dict:
    dataset = Path(config["dataset"])
    evidence = validate(dataset)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_all(int(config.get("seed", 17)))
    if config.get("cpu_threads"):
        torch.set_num_threads(min(int(config["cpu_threads"]), os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    model = make_model(config["model"], int(config.get("hidden", 64))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 1e-3)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config.get("epochs", 5)))
    start_epoch = 0
    best = float("inf")
    patience = 0
    if config.get("resume"):
        checkpoint = torch.load(config["resume"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        start_epoch = checkpoint["epoch"] + 1
        best = float(checkpoint.get("best_validation_loss", float("inf")))
        patience = int(checkpoint.get("patience", 0))
    if start_epoch >= int(config.get("epochs", 5)):
        raise ValueError("resume checkpoint already reached requested epochs")
    train_state, train_action = load_arrays(dataset, "train")
    val_state, val_action = load_arrays(dataset, "val")
    pair_weight = float(config.get("pair_weight", 0.0))
    if pair_weight:
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        branches_per_scene = int(manifest.get("branches_per_scene", 3))
        if len(train_state) % branches_per_scene or not np.allclose(
            train_state.reshape(-1, branches_per_scene, *train_state.shape[1:])[:, :, 0],
            train_state.reshape(-1, branches_per_scene, *train_state.shape[1:])[:, :1, 0]):
            raise ValueError("paired training requires complete same-state branch scenes")
        train_state = train_state.reshape(-1, branches_per_scene, *train_state.shape[1:])
        train_action = train_action.reshape(-1, branches_per_scene, *train_action.shape[1:])
    else:
        branches_per_scene = 1
    loader = DataLoader(TensorDataset(torch.from_numpy(train_state), torch.from_numpy(train_action)),
                        batch_size=max(1, int(config.get("batch_size", 8)) // branches_per_scene), shuffle=True)
    val_state = torch.from_numpy(val_state).to(device)
    val_action = torch.from_numpy(val_action).to(device)
    if config.get("action_blind", False):
        val_action = torch.zeros_like(val_action)
    log_path = run_dir / "metrics.jsonl"
    begin = time.perf_counter()
    optimizer_steps = 0
    for epoch in range(start_epoch, int(config.get("epochs", 5))):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        accumulation = int(config.get("accumulation", 1))
        for index, (state, action) in enumerate(loader):
            if pair_weight:
                scene_count, branches = state.shape[:2]
                state = state.flatten(0, 1)
                action = action.flatten(0, 1)
            if config.get("shuffle_actions", False):
                action = action[torch.randperm(len(action))]
            if config.get("action_blind", False):
                action = torch.zeros_like(action)
            state = state.to(device)
            action = action.to(device)
            output = model(state, action)
            loss = F.mse_loss(output["prediction"], state[:, 1:]) + 0.01 * output["kl"]
            if pair_weight:
                prediction = output["prediction"][:, -1].reshape(scene_count, branches, -1)
                actual = state[:, -1].reshape(scene_count, branches, -1)
                if config.get("shuffle_pair_targets", False):
                    actual = actual[:, torch.randperm(branches)]
                loss = loss + pair_weight * torch.nn.functional.mse_loss(
                    prediction[:, 1:] - prediction[:, :1], actual[:, 1:] - actual[:, :1])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            (loss / accumulation).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
                optimizer.step()
                optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation = float(model.loss(val_state, val_action))
        if not np.isfinite(validation):
            raise FloatingPointError("nonfinite validation")
        scheduler.step()
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": validation,
               "elapsed_seconds": time.perf_counter() - begin}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if validation < best:
            best, patience = validation, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            patience += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(), "config": config, "dataset": evidence,
                    "best_validation_loss": best, "patience": patience}, run_dir / "last.pt")
        if patience >= int(config.get("patience", 5)):
            break
    manifest = {"model": config["model"], "device": str(device), "data": evidence,
                "best_validation_loss": best, "completed_epochs": epoch + 1,
                "checkpoint": str(run_dir / "last.pt"), "optimizer_steps": optimizer_steps,
                "train_samples": len(train_state) * branches_per_scene,
                "batch_size_branches": int(config.get("batch_size", 8)),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "wall_seconds": time.perf_counter() - begin}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
