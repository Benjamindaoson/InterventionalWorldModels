"""Five controls, three seeds, fixed data and optimizer budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iwm.evaluation import evaluate, save_evaluation
from iwm.training import train

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
conditions = {
    "blind": {"action_blind": True},
    "conditioned": {},
    "paired": {"pair_weight": 0.1},
    "shuffled_pair": {"pair_weight": 0.1, "shuffle_pair_targets": True},
    "rssm": {},
}
rows = []
for seed in args.seeds:
    for label, overrides in conditions.items():
        run_dir = args.output / f"{label}-s{seed}"
        config = {"model": "rssm" if label == "rssm" else "action_predictor",
                  "hidden": 32 if label == "rssm" else 96,
                  "dataset": str(args.dataset), "run_dir": str(run_dir),
                  "seed": seed, "epochs": args.epochs, "batch_size": 14,
                  "lr": 0.001, "patience": args.epochs + 1, "cpu_threads": 2,
                  **overrides}
        manifest = train(config)
        metrics = evaluate(run_dir / "last.pt", args.dataset)
        save_evaluation(metrics, run_dir / "test.json")
        row = {"condition": label, "seed": seed, **manifest,
               "one_step_mse": metrics["one_step_mse"],
               "free_running_mse": metrics["free_running_mse"]}
        rows.append(row)
        with (args.output / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"condition": label, "seed": seed,
                          "steps": manifest["optimizer_steps"],
                          "free_running_mse": metrics["free_running_mse"]}), flush=True)
