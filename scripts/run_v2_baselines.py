"""Predeclared V2 position/velocity model matrix; validation only during training."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from iwm.evaluation import evaluate, save_evaluation
from iwm.training import train

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
p.add_argument("--epochs", type=int, default=10)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
conditions = {
    "blind_pos": {"action_blind": True},
    "conditioned_pos": {},
    "paired_pos": {"pair_weight": 0.1},
    "blind_vel": {"action_blind": True, "obs_dim": 6, "observation_key": "dynamics_state"},
    "conditioned_vel": {"obs_dim": 6, "observation_key": "dynamics_state"},
    "paired_vel": {"obs_dim": 6, "observation_key": "dynamics_state", "pair_weight": 0.1},
    "shuffled_vel": {"obs_dim": 6, "observation_key": "dynamics_state",
                     "pair_weight": 0.1, "shuffle_pair_targets": True},
    "rssm_vel": {"obs_dim": 6, "observation_key": "dynamics_state"},
}
for seed in a.seeds:
    for name, overrides in conditions.items():
        run_dir = a.output / f"{name}-s{seed}"
        config = {"model": "rssm" if name == "rssm_vel" else "action_predictor",
                  "hidden": 32 if name == "rssm_vel" else 96,
                  "dataset": str(a.dataset), "run_dir": str(run_dir),
                  "seed": seed, "epochs": a.epochs, "batch_size": 14,
                  "lr": 0.001, "patience": a.epochs + 1,
                  "cpu_threads": 2, "source_revision": revision, **overrides}
        manifest = train(config)
        metrics = evaluate(run_dir / "last.pt", a.dataset, split="val", family="iid_val")
        save_evaluation(metrics, run_dir / "iid_val.json")
        row = {"condition": name, "seed": seed, "manifest": manifest,
               "iid_val_position_mse": metrics["free_running_mse"]}
        with (a.output / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"condition": name, "seed": seed,
                          "iid_val_position_mse": metrics["free_running_mse"]}), flush=True)
