import numpy as np
import pytest
from iwm.graph_theory import certificate
from iwm.experiment_protocol import Experiment,validate_scene_splits
def test_graph_certificate():
    rng=np.random.default_rng(3);edges=[(0,1),(1,2),(2,3),(3,4)]
    for _ in range(100):
        e=rng.normal(size=5)
        assert abs(e[4])<=certificate(e,edges,0,4)+1e-9
def test_disconnected():
    assert np.isinf(certificate([1,2,3],[(0,1)],0,2))
def test_scene_leakage():
    assert validate_scene_splits({"train":["a","b"],"test":["c"]})==3
    with pytest.raises(ValueError):validate_scene_splits({"train":["a"],"test":["a"]})
def test_manifest():
    e=Experiment("E0","replay","v1","pilot","physics",0,0)
    assert e.manifest()["fingerprint"]==e.manifest()["fingerprint"]
