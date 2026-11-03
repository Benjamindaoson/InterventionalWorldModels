from __future__ import annotations

import argparse
from pathlib import Path

from iwm.v11_data import generate_v11

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--train-scenes", type=int, default=20)
parser.add_argument("--val-scenes", type=int, default=6)
parser.add_argument("--test-scenes", type=int, default=6)
parser.add_argument("--horizon", type=int, default=32)
args = parser.parse_args()
print(generate_v11(args.output, args.train_scenes, args.val_scenes,
                   args.test_scenes, args.horizon))
