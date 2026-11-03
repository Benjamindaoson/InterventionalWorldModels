"""Matched-budget action shuffle control using the same frozen MuJoCo dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iwm.evaluation import evaluate, save_evaluation
from iwm.training import train

p = argparse.ArgumentParser()
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
for label in ("conditioned", "shuffled", "blind", "paired"):
    run_dir = a.output / label
    config = {"model": "action_predictor", "dataset": str(a.dataset), "run_dir": str(run_dir),
              "seed": 17, "hidden": 32, "epochs": 5, "batch_size": 6, "lr": 0.001,
              "shuffle_actions": label == "shuffled", "action_blind": label == "blind",
              "pair_weight": 0.1 if label == "paired" else 0.0}
    train(config)
    result = evaluate(run_dir / "last.pt", a.dataset)
    save_evaluation(result, a.output / f"{label}-test.json")
    print(json.dumps({"condition": label, "free_running_mse": result["free_running_mse"]}))
