from __future__ import annotations

import argparse
from pathlib import Path

from iwm.v2_data import FAMILIES, generate_v2

p = argparse.ArgumentParser()
p.add_argument("--output", required=True, type=Path)
p.add_argument("--train-scenes", type=int, default=32)
p.add_argument("--eval-scenes", type=int, default=8)
p.add_argument("--horizon", type=int, default=64)
a = p.parse_args()
counts = {family: a.train_scenes if family == "train" else a.eval_scenes for family in FAMILIES}
print(generate_v2(a.output, counts, a.horizon))
