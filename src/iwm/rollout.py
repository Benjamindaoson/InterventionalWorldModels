"""State-space rollout and observation re-anchoring for drift studies."""
from __future__ import annotations

import torch


@torch.no_grad()
def predict(model, initial_state: torch.Tensor, actions: torch.Tensor,
            anchors: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
    """Roll forward; an anchor at step t replaces the state observed before action t."""
    if initial_state.ndim != 2 or actions.ndim != 3 or len(initial_state) != len(actions):
        raise ValueError("initial_state [B,D], actions [B,T,A]")
    if anchors and any(t < 0 or t >= actions.shape[1] for t in anchors):
        raise ValueError("anchor index outside rollout")
    state = initial_state
    predicted = []
    for t in range(actions.shape[1]):
        if anchors and t in anchors:
            if anchors[t].shape != state.shape:
                raise ValueError("anchor shape mismatch")
            state = anchors[t]
        # The public model contract accepts an initial observation plus one action.
        dummy = torch.stack((state, state), 1)
        state = model(dummy, actions[:, t:t + 1], teacher_forcing=False)["prediction"][:, 0]
        predicted.append(state)
    return torch.stack(predicted, 1)
