from __future__ import annotations

import argparse
from pathlib import Path

from iwm.data import generate

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--scenes", type=int, default=12)
p.add_argument("--horizon", type=int, default=16)
a = p.parse_args()
print(generate(a.output, a.scenes, a.horizon))
