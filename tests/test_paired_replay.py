"""G0: no GPU, no neural result claims, no physics claims from ToyBackend."""
import json
import math
import pytest

from iwm.paired_replay import (
    ReplaySpec, ToyBackend, paired_replay, validate_scene_isolation,
)


def spec(**changes):
    params = dict(task_id="cube-push", scene_id="scene-A", seed=11,
                  horizon=8, split="dev", backend_id="toy-contract-v1",
                  source_revision="local-test-commit", evidence_tier="TOY_CONTRACT_TEST")
    params.update(changes)
    return ReplaySpec(**params)


def test_every_branch_from_identical_initial_state():
    result = paired_replay(ToyBackend(), spec(), [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert len(result["branches"]) == 3
    assert all(b["initial_state_sha256"] == result["initial_state_sha256"]
               for b in result["branches"])
    assert len(result["comparisons"]) == 3
    assert result["research_status"] == "CONTRACT_ONLY_NOT_MODEL_EVIDENCE"


def test_deterministic_full_receipt():
    one = paired_replay(ToyBackend(), spec(), [[0.0, 0.0], [1.0, 0.0]])
    two = paired_replay(ToyBackend(), spec(), [[0.0, 0.0], [1.0, 0.0]])
    assert one == two
    assert json.dumps(one, allow_nan=False)


def test_branch_order_does_not_change_action_outcomes():
    first = paired_replay(ToyBackend(), spec(), [[1.,0.], [0.,1.]])
    other = paired_replay(ToyBackend(), spec(), [[0.,1.], [1.,0.]])
    outcomes_a = {tuple(b["action"]): b["outcome"] for b in first["branches"]}
    outcomes_b = {tuple(b["action"]): b["outcome"] for b in other["branches"]}
    assert outcomes_a == outcomes_b


@pytest.mark.parametrize("actions", [[], [[1.0, 0.0]], [[1.0, 0.0],[1.0,0.0]],
                                       [[math.nan, 0.0],[0.0,0.0]],
                                       [[1.0],[0.0,1.0]], [[math.inf],[0.0]]])
def test_malformed_action_matrix_is_blocked(actions):
    with pytest.raises(ValueError):
        paired_replay(ToyBackend(), spec(), actions)


def test_corrupted_restore_is_blocked():
    class BrokenRestore(ToyBackend):
        def restore(self, state):
            super().restore(state)
            self.x += 1
    with pytest.raises(RuntimeError, match="restoration failed"):
        paired_replay(BrokenRestore(), spec(), [[1.0], [0.0]])


def test_nonfinite_simulation_output_is_blocked():
    class Nonfinite(ToyBackend):
        def execute(self, action, horizon):
            out = dict(super().execute(action, horizon))
            out["displacement"] = float("nan")
            return out
    with pytest.raises(ValueError, match="nonfinite displacement"):
        paired_replay(Nonfinite(), spec(), [[1.0], [0.0]])


def test_nonboolean_event_is_blocked():
    class InvalidEvent(ToyBackend):
        def execute(self, action, horizon):
            out = dict(super().execute(action, horizon))
            out["event"] = 1
            return out
    with pytest.raises(ValueError, match="observed boolean"):
        paired_replay(InvalidEvent(), spec(), [[1.0], [0.0]])


def test_scene_split_leakage_is_blocked():
    a = paired_replay(ToyBackend(), spec(split="train"), [[1.], [0.]])
    b = paired_replay(ToyBackend(), spec(split="test", seed=12), [[1.], [0.]])
    with pytest.raises(ValueError, match="scene leakage"):
        validate_scene_isolation([a,b])
    b["scene_id"] = "scene-B"
    assert validate_scene_isolation([a,b]) == {"train":1,"val":0,"test":1,"dev":0}


def test_receipt_changes_if_seed_changes():
    a = paired_replay(ToyBackend(), spec(seed=11), [[1.], [0.]])
    b = paired_replay(ToyBackend(), spec(seed=12), [[1.], [0.]])
    assert a["receipt_sha256"] != b["receipt_sha256"]


def test_unknown_split_blocked():
    with pytest.raises(ValueError, match="split"):
        paired_replay(ToyBackend(), spec(split="unknown"), [[1.], [0.]])


def test_mujoco_requires_dependency_or_real_physics():
    from iwm.mujoco_paired_backend import MuJoCoCubeBackend
    try:
        import mujoco
    except ImportError:
        with pytest.raises(ImportError):
            MuJoCoCubeBackend()
    else:
        b = MuJoCoCubeBackend()
        r = paired_replay(b, spec(backend_id="mujoco-cube", evidence_tier="SIMULATION_CANDIDATE_UNVERIFIED"),
                          [[0.,0.,0.],[5.,0.,0.]])
        assert len(r["branches"]) == 2
