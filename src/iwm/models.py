"""Small trainable action-conditioned state models; no pretrained-model claims."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RSSM(nn.Module):
    def __init__(self, obs_dim: int = 3, action_dim: int = 3, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.hidden, self.latent = hidden, latent
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ELU())
        self.recurrent = nn.GRUCell(latent + action_dim, hidden)
        self.prior = nn.Linear(hidden, latent * 2)
        self.posterior = nn.Linear(hidden * 2, latent * 2)
        self.decoder = nn.Sequential(nn.Linear(hidden + latent, hidden), nn.ELU(), nn.Linear(hidden, obs_dim))
        self.reward = nn.Linear(hidden + latent, 1)
        self.termination = nn.Linear(hidden + latent, 1)

    @staticmethod
    def distribution(parameters: torch.Tensor) -> torch.distributions.Normal:
        mean, raw_scale = parameters.chunk(2, -1)
        return torch.distributions.Normal(mean, F.softplus(raw_scale) + 0.1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor,
                teacher_forcing: bool = True) -> dict[str, torch.Tensor]:
        if observations.ndim != 3 or actions.ndim != 3 or observations.shape[:2] != (actions.shape[0], actions.shape[1] + 1):
            raise ValueError("observations [B,T+1,D], actions [B,T,A]")
        batch, steps, _ = actions.shape
        h = self.encoder(observations[:, 0])
        z = observations.new_zeros(batch, self.latent)
        predicted, divergences = [], []
        for t in range(steps):
            h = self.recurrent(torch.cat((z, actions[:, t]), -1), h)
            prior = self.distribution(self.prior(h))
            if teacher_forcing:
                posterior = self.distribution(self.posterior(torch.cat((h, self.encoder(observations[:, t + 1])), -1)))
                z = posterior.rsample()
                divergences.append(torch.distributions.kl_divergence(posterior, prior).sum(-1))
            else:
                z = prior.mean
            predicted.append(self.decoder(torch.cat((h, z), -1)))
        return {"prediction": torch.stack(predicted, 1),
                "kl": torch.stack(divergences, 1).mean() if divergences else observations.new_zeros(())}

    def loss(self, observations: torch.Tensor, actions: torch.Tensor, kl_weight: float = 0.01) -> torch.Tensor:
        output = self(observations, actions)
        return F.mse_loss(output["prediction"], observations[:, 1:]) + kl_weight * output["kl"]


class ActionPredictor(nn.Module):
    """Independent deterministic action-conditioned state predictor baseline."""
    def __init__(self, obs_dim: int = 3, action_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, obs_dim))

    def forward(self, observations: torch.Tensor, actions: torch.Tensor,
                teacher_forcing: bool = True) -> dict[str, torch.Tensor]:
        if observations.shape[:2] != (actions.shape[0], actions.shape[1] + 1):
            raise ValueError("time dimension mismatch")
        state = observations[:, 0]
        predictions = []
        for t in range(actions.shape[1]):
            if teacher_forcing:
                state = observations[:, t]
            state = state + self.net(torch.cat((state, actions[:, t]), -1))
            predictions.append(state)
        return {"prediction": torch.stack(predictions, 1), "kl": observations.new_zeros(())}

    def loss(self, observations: torch.Tensor, actions: torch.Tensor, kl_weight: float = 0.) -> torch.Tensor:
        return F.mse_loss(self(observations, actions)["prediction"], observations[:, 1:])


def make_model(name: str, hidden: int = 64) -> nn.Module:
    if name == "rssm":
        return RSSM(hidden=hidden)
    if name == "action_predictor":
        return ActionPredictor(hidden=hidden)
    raise ValueError(f"unknown model: {name}")
