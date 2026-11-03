"""Hash immutable local V1.1 evidence without copying or changing artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, type=Path)
p.add_argument("--runs", required=True, type=Path)
p.add_argument("--analysis", required=True, type=Path)
p.add_argument("--video-run", required=True, type=Path)
p.add_argument("--video-test", required=True, type=Path)
p.add_argument("--planning", required=True, type=Path, nargs="+")
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
paths = [a.dataset / "manifest.json", a.analysis / "summary.json",
         a.analysis / "per_scene.jsonl", a.analysis / "horizon.csv",
         a.analysis / "horizon.svg", a.video_run / "best.pt",
         a.video_run / "last.pt", a.video_run / "metrics.jsonl",
         a.video_test / "metrics.json", *a.planning]
for run in sorted(a.runs.iterdir()):
    if run.is_dir():
        paths.extend((run / "last.pt", run / "metrics.jsonl",
                      run / "manifest.json", a.analysis / f"{run.name}-test.json"))
ledger = {"code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("x", encoding="utf-8") as f:
    json.dump(ledger, f, indent=2)
print(json.dumps({"output": str(a.output), "files": len(paths),
                  "code_revision": ledger["code_revision"]}))
