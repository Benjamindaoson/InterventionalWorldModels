import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from iwm.data import load_arrays, validate
from iwm.v2_data import FAMILIES, generate_v2


def test_v2_family_isolation_velocity_camera(tmp_path):
    root = generate_v2(tmp_path / "v2", {family: 2 for family in FAMILIES}, horizon=4)
    assert validate(root)["records"] == 84
    with pytest.raises(ValueError, match="explicit evaluation family"):
        load_arrays(root, "test")
    states, actions = load_arrays(root, "test", "confirmatory", "dynamics_state")
    assert states.shape == (14, 5, 6)
    assert actions.shape == (14, 4, 3)
    records = [json.loads(path.read_text()) for path in root.glob("cube-*.json")]
    assert any(abs(r["initial_velocity"][0]) > 0 for r in records)
    record = records[0]
    camera = record["camera"]
    point = np.asarray(record["state"][0])
    rotation = np.asarray(camera["world_to_camera_cv_rotation"])
    translation = np.asarray(camera["world_to_camera_cv_translation"])
    projected = rotation @ point + translation
    assert projected[2] > 0
    pixel = np.asarray(camera["intrinsics"]) @ projected
    pixel = pixel[:2] / pixel[2]
    assert (pixel > 0).all() and (pixel < 32).all()
