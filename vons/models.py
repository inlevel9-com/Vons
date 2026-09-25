"""Optional PyTorch Direct and Diffusion model implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 256
    latent_size: int = 32
    diffusion_steps: int = 64
    inference_steps: int = 8
    abstain_threshold: float = 0.55
    diffusion_conditioning: str = "token_cross_attention"
    attention_heads: int = 4


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

    class PooledDiffusionDecision(nn.Module):
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

    class TokenConditionedDiffusionDecision(nn.Module):
        """Denoise each option while attending to its complete token sequence."""

        def __init__(self) -> None:
            super().__init__()
            if settings.latent_size % settings.attention_heads:
                raise ValueError("latent_size must be divisible by attention_heads")
            self.score_projection = nn.Linear(1, settings.latent_size)
            self.token_projection = nn.Linear(settings.hidden_size, settings.latent_size)
            self.time = nn.Embedding(settings.diffusion_steps, settings.latent_size)
            self.cross_attention = nn.MultiheadAttention(
                settings.latent_size,
                settings.attention_heads,
                batch_first=True,
            )
            self.query_norm = nn.LayerNorm(settings.latent_size)
            self.denoiser = nn.Sequential(
                nn.Linear(settings.latent_size, settings.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(settings.hidden_size // 2, 1),
            )
            self.answerability = nn.Linear(settings.hidden_size, 1)

        def forward(
            self,
            noisy_scores: Any,
            sequence_hidden: Any,
            attention_mask: Any,
            timestep: Any,
            option_mask: Any | None = None,
        ) -> dict[str, Any]:
            if sequence_hidden.ndim != 4 or attention_mask.ndim != 3:
                raise ValueError(
                    "token-conditioned diffusion expects [batch, options, sequence, hidden] states"
                )
            batch, options, sequence, hidden = sequence_hidden.shape
            if hidden != settings.hidden_size:
                raise ValueError("sequence hidden size does not match the model configuration")
            flat_tokens = self.token_projection(
                sequence_hidden.reshape(batch * options, sequence, hidden)
            )
            flat_attention = attention_mask.reshape(batch * options, sequence).bool()
            time_embedding = self.time(timestep).unsqueeze(1)
            queries = self.score_projection(noisy_scores.unsqueeze(-1)) + time_embedding
            flat_queries = queries.reshape(batch * options, 1, settings.latent_size)
            attended, _weights = self.cross_attention(
                flat_queries,
                flat_tokens,
                flat_tokens,
                key_padding_mask=~flat_attention,
                need_weights=False,
            )
            option_logits = self.denoiser(
                self.query_norm(flat_queries + attended).squeeze(1)
            ).reshape(batch, options)
            if option_mask is not None:
                option_logits = option_logits.masked_fill(~option_mask.bool(), float("-inf"))

            token_weights = attention_mask.to(sequence_hidden.dtype).unsqueeze(-1)
            candidate_embeddings = (sequence_hidden * token_weights).sum(dim=2) / token_weights.sum(
                dim=2
            ).clamp_min(1)
            if option_mask is None:
                pooled = candidate_embeddings.mean(dim=1)
            else:
                option_weights = option_mask.to(sequence_hidden.dtype).unsqueeze(-1)
                pooled = (candidate_embeddings * option_weights).sum(dim=1) / option_weights.sum(
                    dim=1
                ).clamp_min(1)
            return {
                "logits": option_logits,
                "answerability": self.answerability(pooled).squeeze(-1),
            }

    if backend == "direct":
        return DirectDecision()
    if backend != "diffusion":
        raise ValueError(f"unsupported backend {backend!r}")
    if settings.diffusion_conditioning == "pooled":
        return PooledDiffusionDecision()
    if settings.diffusion_conditioning == "token_cross_attention":
        return TokenConditionedDiffusionDecision()
    raise ValueError(
        f"unsupported diffusion_conditioning {settings.diffusion_conditioning!r}"
    )


def linear_beta_schedule(steps: int) -> Any:
    torch, _ = require_torch()
    if steps < 2:
        raise ValueError("diffusion schedule requires at least two steps")
    return torch.linspace(1e-4, 0.02, steps)


def cosine_beta_schedule(steps: int, *, offset: float = 0.008) -> Any:
    """Cosine cumulative-noise schedule from Nichol and Dhariwal."""
    torch, _ = require_torch()
    if steps < 2:
        raise ValueError("diffusion schedule requires at least two steps")
    if not 0.0 < offset < 1.0:
        raise ValueError("cosine offset must be between zero and one")
    positions = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bars = torch.cos(((positions / steps + offset) / (1 + offset)) * torch.pi / 2) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
    return betas.clamp(1e-5, 0.999).to(torch.float32)


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


def ddim_sample_token_conditioned(
    model: Any,
    sequence_hidden: Any,
    attention_mask: Any,
    option_mask: Any,
    *,
    diffusion_steps: int = 64,
    inference_steps: int = 8,
    seed: int | None = None,
) -> Any:
    """Sample class logits from the token-conditioned x0 predictor."""
    torch, _ = require_torch()
    if not 2 <= inference_steps <= diffusion_steps:
        raise ValueError("inference_steps must be between 2 and diffusion_steps")
    batch, option_count = option_mask.shape
    betas = cosine_beta_schedule(diffusion_steps).to(sequence_hidden.device)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=sequence_hidden.device)
        generator.manual_seed(seed)
    sample = torch.randn(
        (batch, option_count),
        device=sequence_hidden.device,
        generator=generator,
    )
    timesteps = torch.linspace(
        diffusion_steps - 1,
        0,
        inference_steps,
        device=sequence_hidden.device,
    ).round().long()
    final_logits = sample
    for index, timestep in enumerate(timesteps):
        timestep_batch = torch.full(
            (batch,), int(timestep), dtype=torch.long, device=sequence_hidden.device
        )
        output = model(
            sample,
            sequence_hidden,
            attention_mask,
            timestep_batch,
            option_mask,
        )
        final_logits = output["logits"]
        predicted_x0 = torch.softmax(final_logits, dim=-1)
        current_alpha = alpha_bars[timestep]
        predicted_noise = (
            sample - torch.sqrt(current_alpha) * predicted_x0
        ) / torch.sqrt(1 - current_alpha).clamp_min(1e-6)
        previous_alpha = (
            alpha_bars[timesteps[index + 1]]
            if index + 1 < len(timesteps)
            else sample.new_tensor(1.0)
        )
        sample = ddim_step(sample, predicted_noise, current_alpha, previous_alpha)
    return final_logits.masked_fill(~option_mask.bool(), float("-inf"))
