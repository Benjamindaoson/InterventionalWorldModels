from __future__ import annotations

import argparse
import json
from pathlib import Path

from iwm.evaluation import evaluate, save_evaluation

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
result = evaluate(a.checkpoint, a.dataset)
save_evaluation(result, a.output)
print(json.dumps(result, indent=2))
