from __future__ import annotations

import argparse
import json
from pathlib import Path

from iwm.data import validate

p = argparse.ArgumentParser()
p.add_argument("dataset", type=Path)
print(json.dumps(validate(p.parse_args().dataset), indent=2))
