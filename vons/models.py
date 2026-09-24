"""Optional PyTorch Direct and Diffusion model implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 256
    latent_size: int = 32
    diffusion_steps: int = 32
    inference_steps: int = 4
    abstain_threshold: float = 0.55


def require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("training requires optional dependencies: pip install -e '.[train]'") from exc
    return torch, nn


def build_model(backend: str, *, option_count: int, config: ModelConfig | None = None) -> Any:
    """Build the decision head while keeping torch optional for data-only installs."""
    _torch, nn = require_torch()
    if not 2 <= option_count <= 32:
        raise ValueError("option_count must be between 2 and 32")
    settings = config or ModelConfig()

    class DirectDecision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scorer = nn.Sequential(nn.Linear(settings.hidden_size, settings.hidden_size // 2), nn.GELU(), nn.Linear(settings.hidden_size // 2, 1))
            self.answerability = nn.Linear(settings.hidden_size, 1)

        def forward(self, option_embeddings: Any, pooled: Any, option_mask: Any | None = None) -> dict[str, Any]:
            logits = self.scorer(option_embeddings).squeeze(-1)
            if option_mask is not None:
                logits = logits.masked_fill(~option_mask.bool(), float("-inf"))
            return {"logits": logits, "answerability": self.answerability(pooled).squeeze(-1)}

    class DiffusionDecision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.condition = nn.Linear(settings.hidden_size, settings.latent_size)
            self.input = nn.Linear(option_count, settings.latent_size)
            self.time = nn.Embedding(settings.diffusion_steps, settings.latent_size)
            self.denoiser = nn.Sequential(nn.Linear(settings.latent_size, settings.hidden_size // 2), nn.SiLU(), nn.Linear(settings.hidden_size // 2, option_count))
            self.answerability = nn.Linear(settings.hidden_size, 1)

        def forward(self, noisy_scores: Any, pooled: Any, timestep: Any) -> dict[str, Any]:
            latent = self.input(noisy_scores) + self.condition(pooled) + self.time(timestep)
            return {"noise": self.denoiser(latent), "answerability": self.answerability(pooled).squeeze(-1)}

    return DirectDecision() if backend == "direct" else DiffusionDecision()


def linear_beta_schedule(steps: int) -> Any:
    torch, _ = require_torch()
    if steps < 2:
        raise ValueError("diffusion schedule requires at least two steps")
    return torch.linspace(1e-4, 0.02, steps)


def ddim_step(sample: Any, predicted_noise: Any, alpha: Any, alpha_previous: Any) -> Any:
    """Deterministic DDIM update for a score vector."""
    torch, _ = require_torch()
    if sample.shape != predicted_noise.shape:
        raise ValueError("sample and predicted_noise must have the same shape")
    predicted_x0 = (sample - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
    return torch.sqrt(alpha_previous) * predicted_x0 + torch.sqrt(1 - alpha_previous) * predicted_noise


def ddim_sample(model: Any, pooled: Any, *, option_count: int, diffusion_steps: int = 32,
                inference_steps: int = 4, seed: int | None = None) -> Any:
    """Sample candidate scores with deterministic DDIM updates.

    The returned scores are converted to probabilities by the caller so that
    candidate masking and abstention remain visible at the contract boundary.
    """
    torch, _ = require_torch()
    if not 2 <= inference_steps <= diffusion_steps:
        raise ValueError("inference_steps must be between 2 and diffusion_steps")
    betas = linear_beta_schedule(diffusion_steps).to(pooled.device)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pooled.device)
        generator.manual_seed(seed)
    sample = torch.randn((pooled.shape[0], option_count), device=pooled.device, generator=generator)
    timesteps = torch.linspace(diffusion_steps - 1, 0, inference_steps, device=pooled.device).round().long()
    for index, timestep in enumerate(timesteps):
        current_alpha = alpha_bars[timestep]
        previous_alpha = alpha_bars[timesteps[index + 1]] if index + 1 < len(timesteps) else pooled.new_tensor(1.0)
        timestep_batch = torch.full((pooled.shape[0],), int(timestep), dtype=torch.long, device=pooled.device)
        predicted_noise = model(sample, pooled, timestep_batch)["noise"]
        sample = ddim_step(sample, predicted_noise, current_alpha, previous_alpha)
    return sample
