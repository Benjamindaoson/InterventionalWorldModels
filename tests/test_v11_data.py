import json

import pytest

pytest.importorskip("mujoco")

from iwm.data import validate
from iwm.v11_data import generate_v11


def test_v11_split_and_branch_integrity(tmp_path):
    root = generate_v11(tmp_path / "v11", 3, 3, 3, 4)
    result = validate(root)
    assert result["records"] == 63
    records = [json.loads(p.read_text()) for p in root.glob("cube-*.json")]
    config_splits = {}
    for record in records:
        key = tuple(sorted(record["physics"].items()))
        config_splits.setdefault(key, set()).add(record["split"])
    assert all(len(splits) == 1 for splits in config_splits.values())
    assert any(sum(r["contact"]) == 0 for r in records)
    assert any(sum(r["contact"]) > 0 for r in records)
