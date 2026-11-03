"""Planning information boundary and privileged oracle state restoration."""
import numpy as np
import torch

from iwm.mujoco_paired_backend import MuJoCoCubeBackend
from scripts.evaluate_v2_planning import model_candidate_scores, oracle_candidate_scores


class SpyModel(torch.nn.Module):
    def forward(self, observations, actions, teacher_forcing=True):
        assert not teacher_forcing
        assert torch.equal(observations, observations[:, :1].expand_as(observations))
        predicted = observations[:, 1:].clone()
        predicted[..., :2] += actions[..., :2] * 0.001
        return {"prediction": predicted}


def test_model_scores_use_only_current_state_and_candidate_actions():
    state = np.array([0., 0., 0.05, 0., 0., 0.], dtype=np.float32)
    actions = np.array([[0., 0., 0.], [5., 0., 0.]], dtype=np.float32)
    scores, prediction = model_candidate_scores(SpyModel(), state, actions,
                                                 np.array([0.005, 0.]), 4)
    assert scores[1] < 1e-7
    assert prediction.shape == (2, 3)
    blind_scores, _ = model_candidate_scores(SpyModel(), state, actions,
                                               np.array([0.005, 0.]), 4, action_blind=True)
    assert blind_scores[0] == blind_scores[1]


def test_oracle_restores_exact_physics_snapshot():
    backend = MuJoCoCubeBackend()
    backend.reset(733)
    before = backend.snapshot()
    candidates = np.asarray([[0., 0., 0.], [5., 0., 0.]])
    scores = oracle_candidate_scores(backend, before, candidates, np.array([0.02, 0.]), 4)
    assert scores.shape == (2,)
    assert backend.snapshot() == before
