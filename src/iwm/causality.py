"""Counterfactual action diagnostics on exact same-state branch groups."""
from __future__ import annotations

import numpy as np


def branch_diagnostics(true_final: np.ndarray, predicted_final: np.ndarray,
                       actions: np.ndarray, eps: float = 1e-8) -> dict[str, float | None]:
    """Branch zero is the zero-action reference; all branches share an initial state."""
    if true_final.shape != predicted_final.shape or true_final.ndim != 2:
        raise ValueError("final states must have matching [branches, state] shapes")
    if actions.shape[0] != len(true_final) or len(true_final) < 2:
        raise ValueError("action/branch mismatch")
    true_delta = true_final[1:] - true_final[:1]
    predicted_delta = predicted_final[1:] - predicted_final[:1]
    true_norm = np.linalg.norm(true_delta, axis=1)
    predicted_norm = np.linalg.norm(predicted_delta, axis=1)
    valid = true_norm > eps
    direction = float(np.mean(np.sum(true_delta[valid] * predicted_delta[valid], axis=1) > 0)) if valid.any() else None
    ranks = []
    for i in range(len(true_final)):
        for j in range(i + 1, len(true_final)):
            actual = true_final[i, 0] - true_final[j, 0]
            predicted = predicted_final[i, 0] - predicted_final[j, 0]
            if abs(actual) > eps:
                ranks.append(actual * predicted > 0)
    action_norm = np.linalg.norm(actions[1:] - actions[:1], axis=1)
    return {"counterfactual_delta_mse": float(np.mean((predicted_delta - true_delta) ** 2)),
            "true_delta_norm_m": float(true_norm.mean()),
            "predicted_delta_norm_m": float(predicted_norm.mean()),
            "response_magnitude_error_m": float(np.mean(np.abs(predicted_norm - true_norm))),
            "direction_accuracy": direction,
            "x_outcome_ranking_accuracy": float(np.mean(ranks)) if ranks else None,
            "true_action_sensitivity_m_per_n": float(np.mean(true_norm / np.maximum(action_norm, eps))),
            "predicted_action_sensitivity_m_per_n": float(np.mean(predicted_norm / np.maximum(action_norm, eps)))}
