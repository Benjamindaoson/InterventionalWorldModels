"""Small RGB baselines trained from scratch on recorded MuJoCo frames."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrameAutoencoder(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.SiLU(),
                                     nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(),
                                     nn.Flatten(), nn.Linear(32 * 8 * 8, width))
        self.decoder = nn.Sequential(nn.Linear(width, 32 * 8 * 8), nn.Unflatten(1, (32, 8, 8)),
                                     nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(),
                                     nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[-3:] != (3, 32, 32):
            raise ValueError("frames must be 32x32 RGB")
        return self.encoder(frames.reshape(-1, 3, 32, 32)).reshape(*frames.shape[:-3], -1)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents.reshape(-1, latents.shape[-1])).reshape(*latents.shape[:-1], 3, 32, 32)


class VisualPredictor(nn.Module):
    """Self implemented visual representation baseline, not V-JEPA 2-AC."""
    def __init__(self, width: int = 32):
        super().__init__()
        self.visual = FrameAutoencoder(width)
        self.action = nn.Linear(3, width)
        self.temporal = nn.GRUCell(width * 2, width)
        self.predictor = nn.Linear(width, width)

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.visual.encode(frames)
        state = target[:, 0]
        predictions = []
        for t in range(actions.shape[1]):
            state = self.temporal(torch.cat((state, self.action(actions[:, t])), -1), state)
            predictions.append(self.predictor(state))
        return torch.stack(predictions, 1), target

    def loss(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        predicted, target = self(frames, actions)
        return F.mse_loss(predicted, target[:, 1:].detach()) + 0.1 * F.mse_loss(self.visual.decode(target), frames)


class VideoFlow(nn.Module):
    """Conditional latent flow matching with a learned frame codec and temporal attention."""
    def __init__(self, width: int = 32):
        super().__init__()
        self.codec = FrameAutoencoder(width)
        self.action = nn.Linear(3, width)
        self.time = nn.Linear(1, width)
        self.position = nn.Parameter(torch.randn(1, 64, width) * 0.01)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1)
        self.velocity = nn.Linear(width, width)

    def velocity_field(self, noisy: torch.Tensor, context: torch.Tensor,
                       actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        length = noisy.shape[1]
        if length > self.position.shape[1]:
            raise ValueError("sequence longer than positional encoding")
        tokens = noisy + context + self.action(actions) + self.time(time) + self.position[:, :length]
        causal = torch.triu(torch.ones(length, length, device=noisy.device, dtype=torch.bool), diagonal=1)
        return self.velocity(self.transformer(tokens, mask=causal))

    def loss(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        latent = self.codec.encode(frames)
        target = latent[:, 1:]
        context = latent[:, :-1]
        noise = torch.randn_like(target)
        time = torch.rand(*target.shape[:2], 1, device=target.device)
        interpolated = (1 - time) * noise + time * target
        prediction = self.velocity_field(interpolated, context, actions, time)
        flow = F.mse_loss(prediction, target - noise)
        reconstruction = F.mse_loss(self.codec.decode(latent), frames)
        return flow + reconstruction

    @torch.no_grad()
    def generate(self, initial: torch.Tensor, actions: torch.Tensor, steps: int = 8) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        context = self.codec.encode(initial)
        frames = [initial]
        past_targets: list[torch.Tensor] = []
        past_contexts: list[torch.Tensor] = []
        for t in range(actions.shape[1]):
            noisy = torch.randn_like(context).unsqueeze(1)
            for k in range(steps):
                current_time = noisy.new_full((len(noisy), 1, 1), k / steps)
                if past_targets:
                    input_seq = torch.cat((*past_targets, noisy), dim=1)
                    context_seq = torch.cat((*past_contexts, context.unsqueeze(1)), dim=1)
                    time_seq = torch.cat((noisy.new_ones(len(noisy), t, 1), current_time), dim=1)
                else:
                    input_seq = noisy
                    context_seq = context.unsqueeze(1)
                    time_seq = current_time
                velocity = self.velocity_field(input_seq, context_seq, actions[:, :t + 1], time_seq)[:, -1:]
                noisy = noisy + velocity / steps
            past_contexts.append(context.unsqueeze(1))
            past_targets.append(noisy)
            context = noisy[:, 0]
            frames.append(self.codec.decode(context))
        return torch.stack(frames, 1)

    @torch.no_grad()
    def teacher_force_generate(self, frames: torch.Tensor, actions: torch.Tensor,
                               steps: int = 8) -> torch.Tensor:
        """At t, condition only on observed frames through t and actions through t."""
        if frames.shape[1] != actions.shape[1] + 1 or steps < 1:
            raise ValueError("frames/actions time mismatch or invalid steps")
        observed_latent = self.codec.encode(frames)
        generated = [frames[:, 0]]
        for t in range(actions.shape[1]):
            current = torch.randn_like(observed_latent[:, t:t + 1])
            contexts = observed_latent[:, :t + 1]
            for k in range(steps):
                noisy_sequence = torch.cat((observed_latent[:, 1:t + 1], current), 1)
                times = current.new_ones(len(current), t + 1, 1)
                times[:, -1] = k / steps
                velocity = self.velocity_field(noisy_sequence, contexts,
                                               actions[:, :t + 1], times)[:, -1:]
                current = current + velocity / steps
            generated.append(self.codec.decode(current[:, 0]))
        return torch.stack(generated, 1)


class JointWorldAction(nn.Module):
    """Shared visual context; inverse action never observes the target action."""
    def __init__(self, width: int = 32):
        super().__init__()
        self.visual = FrameAutoencoder(width)
        self.action_encoder = nn.Linear(3, width)
        self.joint = nn.GRU(width, width, batch_first=True)
        self.world_head = nn.Linear(width * 2, width)
        self.action_head = nn.Linear(width * 2, 3)

    def loss(self, frames: torch.Tensor, actions: torch.Tensor,
             world_weight: float = 1., action_weight: float = 1.) -> torch.Tensor:
        latent = self.visual.encode(frames)
        current = latent[:, :-1]
        future = latent[:, 1:]
        hidden, _ = self.joint(current)
        loss = 0.1 * F.mse_loss(self.visual.decode(latent), frames)
        if world_weight:
            predicted_world = self.world_head(torch.cat((hidden, self.action_encoder(actions / 10)), -1))
            loss = loss + world_weight * F.mse_loss(predicted_world, future.detach())
        if action_weight:
            predicted_action = self.action_head(torch.cat((hidden, future), -1))
            loss = loss + action_weight * F.mse_loss(predicted_action, actions / 10)
        return loss

    def predict_world(self, observed_frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Teacher-forced next latent, using only frames through the current step."""
        if observed_frames.shape[1] != actions.shape[1]:
            raise ValueError("current frame/action length mismatch")
        current = self.visual.encode(observed_frames)
        hidden, _ = self.joint(current)
        return self.world_head(torch.cat((hidden, self.action_encoder(actions / 10)), -1))

    def predict_actions(self, frames: torch.Tensor) -> torch.Tensor:
        """Inverse dynamics from adjacent frames; no action input is accepted."""
        latent = self.visual.encode(frames)
        hidden, _ = self.joint(latent[:, :-1])
        return self.action_head(torch.cat((hidden, latent[:, 1:]), -1)) * 10
