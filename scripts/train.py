from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from iwm.training import train

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, required=True)
p.add_argument("--dataset", type=Path)
p.add_argument("--run-dir", type=Path)
a = p.parse_args()
config = yaml.safe_load(a.config.read_text(encoding="utf-8"))
if a.dataset:
    config["dataset"] = str(a.dataset)
if a.run_dir:
    config["run_dir"] = str(a.run_dir)
if config["model"] in {"visual", "video_flow", "joint_wam"}:
    subprocess.run([sys.executable, str(Path(__file__).with_name("train_visual.py")),
                    "--model", config["model"], "--dataset", config["dataset"],
                    "--output", config["run_dir"], "--epochs", str(config.get("epochs", 2)),
                    "--seed", str(config.get("seed", 17))], check=True)
else:
    print(json.dumps(train(config), indent=2))
