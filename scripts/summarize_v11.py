"""Re-evaluate frozen runs and bootstrap independent scenes, preserving raw files."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from iwm.evaluation import evaluate, save_evaluation

p = argparse.ArgumentParser()
p.add_argument("--runs", required=True, type=Path)
p.add_argument("--dataset", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
rows = []
scene_rows = []
for run in sorted(a.runs.iterdir()):
    if not run.is_dir():
        continue
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    metrics = evaluate(run / "last.pt", a.dataset)
    save_evaluation(metrics, a.output / f"{run.name}-test.json")
    condition, seed_text = run.name.rsplit("-s", 1)
    row = {"condition": condition, "seed": int(seed_text),
           "parameter_count": manifest["parameter_count"],
           "optimizer_steps": manifest["optimizer_steps"],
           "train_samples": manifest["train_samples"],
           "wall_seconds": manifest["wall_seconds"],
           "one_step_mse": metrics["one_step_mse"],
           "free_running_mse": metrics["free_running_mse"],
           "action_branch_difference_error_meters": metrics["action_branch_difference_error_meters"],
           "contact_f1": metrics["contact_f1"],
           "horizon_mse": metrics["horizon_mse"]}
    rows.append(row)
    for scene in metrics["per_scene"]:
        scene_rows.append({"condition": condition, "seed": int(seed_text), **scene})
with (a.output / "per_scene.jsonl").open("x", encoding="utf-8") as f:
    for row in scene_rows:
        f.write(json.dumps(row) + "\n")
by_condition = defaultdict(list)
for row in rows:
    by_condition[row["condition"]].append(row)
rng = np.random.default_rng(123)
summary = {}
for condition, condition_rows in by_condition.items():
    scene_names = sorted({r["scene_id"] for r in scene_rows if r["condition"] == condition})
    scene_values = np.asarray([
        np.mean([r["free_running_mse"] for r in scene_rows
                 if r["condition"] == condition and r["scene_id"] == scene])
        for scene in scene_names])
    bootstrap = np.asarray([rng.choice(scene_values, size=len(scene_values), replace=True).mean()
                            for _ in range(2000)])
    seed_values = np.asarray([r["free_running_mse"] for r in condition_rows])
    summary[condition] = {"seeds": [r["seed"] for r in condition_rows],
        "parameter_count": condition_rows[0]["parameter_count"],
        "optimizer_steps_each": condition_rows[0]["optimizer_steps"],
        "train_samples_each": condition_rows[0]["train_samples"],
        "mean_cpu_wall_seconds": float(np.mean([r["wall_seconds"] for r in condition_rows])),
        "one_step_mse_mean": float(np.mean([r["one_step_mse"] for r in condition_rows])),
        "free_running_mse_mean": float(seed_values.mean()),
        "free_running_mse_seed_sd": float(seed_values.std(ddof=1)),
        "free_running_mse_scene_bootstrap_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "action_branch_difference_error_meters_mean": float(np.mean(
            [r["action_branch_difference_error_meters"] for r in condition_rows])),
        "contact_f1_mean": float(np.mean([r["contact_f1"] for r in condition_rows])),
        "horizon_mse_mean": {h: float(np.mean([r["horizon_mse"][h] for r in condition_rows]))
                             for h in condition_rows[0]["horizon_mse"]}}
(a.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
with (a.output / "horizon.csv").open("x", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["condition", "horizon", "mse"])
    for condition, values in summary.items():
        for horizon, mse in values["horizon_mse_mean"].items():
            writer.writerow([condition, horizon, mse])
colors = {"blind": "#2563eb", "conditioned": "#dc2626", "paired": "#16a34a",
          "shuffled_pair": "#9333ea", "rssm": "#ea580c"}
all_values = [value for item in summary.values() for value in item["horizon_mse_mean"].values()]
lo, hi = math.floor(math.log10(min(all_values))), math.ceil(math.log10(max(all_values)))
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="460" viewBox="0 0 760 460">',
         '<rect width="760" height="460" fill="white"/>',
         '<text x="80" y="28" font-size="18">Free-running state MSE vs horizon (scene-held-out)</text>',
         '<path d="M80 40 V390 H570" fill="none" stroke="black"/>']
for exponent in range(lo, hi + 1):
    y = 390 - (exponent - lo) / max(hi - lo, 1) * 340
    parts.append(f'<path d="M80 {y:.1f} H570" stroke="#ddd"/>')
    parts.append(f'<text x="28" y="{y+4:.1f}" font-size="12">1e{exponent}</text>')
for horizon in (1, 4, 8, 16, 32):
    x = 80 + (horizon - 1) / 31 * 490
    parts.append(f'<text x="{x-8:.1f}" y="410" font-size="12">{horizon}</text>')
for index, (condition, item) in enumerate(summary.items()):
    points = []
    for horizon, value in item["horizon_mse_mean"].items():
        x = 80 + (int(horizon) - 1) / 31 * 490
        y = 390 - (math.log10(value) - lo) / max(hi - lo, 1) * 340
        points.append(f"{x:.1f},{y:.1f}")
    color = colors[condition]
    parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    parts.append(f'<text x="600" y="{80+index*28}" fill="{color}" font-size="14">{condition}</text>')
parts.append('<text x="300" y="445" font-size="13">Prediction horizon (steps)</text></svg>')
(a.output / "horizon.svg").write_text("\n".join(parts), encoding="utf-8")
print(json.dumps(summary, indent=2))
