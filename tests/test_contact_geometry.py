"""Contact proxy must follow each scene's physical box dimensions."""
import numpy as np
import pytest

from iwm.evaluation import infer_floor_contact


def test_contact_proxy_uses_scene_geometry():
    heights = np.array([[0.04, 0.06], [0.04, 0.06]], dtype=np.float32)
    predicted = infer_floor_contact(heights, np.array([0.035, 0.07]))
    assert predicted.tolist() == [[False, False], [True, True]]
    with pytest.raises(ValueError):
        infer_floor_contact(heights, np.array([0.05]))
