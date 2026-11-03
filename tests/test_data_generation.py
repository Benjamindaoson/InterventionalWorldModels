import json

import pytest

mujoco = pytest.importorskip("mujoco")
from iwm.data import generate, validate


def test_mujoco_paired_dataset_and_rgb_hash(tmp_path):
    root = tmp_path / "dataset"
    generate(root, scenes=3, horizon=2)
    assert validate(root)["records"] == 9
    records = [json.loads(p.read_text()) for p in root.glob("cube-*.json")]
    for scene in {r["scene_id"] for r in records}:
        branches = [r for r in records if r["scene_id"] == scene]
        assert len({r["initial_state_sha256"] for r in branches}) == 1
    assert len({r["initial_state_sha256"] for r in records}) == 3
    rgb = next(root.glob("*.npz"))
    with rgb.open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="RGB artifact hash"):
        validate(root)
