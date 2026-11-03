"""Offline fixed-candidate planning check on held-out MuJoCo branch outcomes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from iwm.data import validate
from iwm.models import make_model

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
validate(a.dataset)
checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
config = checkpoint["config"]
model = make_model(config["model"], int(config.get("hidden", 64))).eval()
model.load_state_dict(checkpoint["model"])
groups = defaultdict(list)
for path in sorted(a.dataset.glob("cube-*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["split"] == "test":
        groups[record["scene_id"]].append(record)
rows = []
with torch.no_grad():
    for scene, branches in groups.items():
        initial = torch.tensor([b["state"][0] for b in branches], dtype=torch.float32)
        actions = torch.tensor([b["action_sequence"] for b in branches], dtype=torch.float32)
        if config.get("action_blind", False):
            actions = torch.zeros_like(actions)
        placeholder = torch.cat((initial[:, None], initial[:, None].expand(-1, actions.shape[1], -1)), 1)
        prediction = model(placeholder, actions, teacher_forcing=False)["prediction"][:, -1, 0]
        choice = int(prediction.argmax())
        true_x = [float(b["outcome"][0]) for b in branches]
        rows.append({"scene": scene, "chosen_branch": branches[choice]["branch"],
                     "chosen_outcome_x": true_x[choice], "oracle_outcome_x": max(true_x),
                     "regret_x_meters": max(true_x) - true_x[choice]})
result = {"status": "OFFLINE_FIXED_CANDIDATE_CHECK_NOT_CLOSED_LOOP",
          "checkpoint": str(a.checkpoint), "test_scenes": len(rows),
          "mean_regret_x_meters": sum(row["regret_x_meters"] for row in rows) / len(rows),
          "decisions": rows}
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("x", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
