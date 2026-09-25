from itertools import pairwise

import pytest

torch = pytest.importorskip("torch")

from vons.models import ModelConfig, build_model, cosine_beta_schedule
from vons.training import soft_diffusion_targets


def _model(hidden_size=16):
    return build_model(
        "diffusion",
        option_count=32,
        config=ModelConfig(
            hidden_size=hidden_size,
            latent_size=8,
            diffusion_steps=16,
            inference_steps=4,
            attention_heads=2,
        ),
    )


def test_cosine_schedule_reaches_high_terminal_noise():
    betas = cosine_beta_schedule(16)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)

    assert betas.shape == (16,)
    assert torch.all(betas[1:] > betas[:-1])
    assert float(alpha_bar[-1]) < 1e-4


@pytest.mark.parametrize(("options", "sequence"), [(3, 7), (5, 11)])
def test_token_conditioned_head_accepts_dynamic_option_and_sequence_axes(options, sequence):
    torch.manual_seed(7)
    model = _model().eval()
    output = model(
        torch.randn(2, options),
        torch.randn(2, options, sequence, 16),
        torch.ones(2, options, sequence, dtype=torch.bool),
        torch.tensor([3, 4]),
        torch.ones(2, options, dtype=torch.bool),
    )

    assert output["logits"].shape == (2, options)
    assert output["answerability"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()


def test_soft_targets_are_continuous_normalized_and_masked():
    labels = torch.tensor([0, 2])
    option_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])

    targets = soft_diffusion_targets(labels, option_mask, smoothing=0.1)

    torch.testing.assert_close(targets.sum(dim=-1), torch.ones(2))
    assert torch.all((targets > 0) == option_mask)
    torch.testing.assert_close(targets[torch.arange(2), labels], torch.tensor([0.9, 0.9]))


def test_token_conditioned_soft_target_loss_strictly_decreases():
    torch.manual_seed(7)
    batch, options, sequence, hidden = 8, 4, 6, 16
    model = _model(hidden)
    sequence_hidden = torch.randn(batch, options, sequence, hidden)
    attention_mask = torch.ones(batch, options, sequence, dtype=torch.bool)
    option_mask = torch.ones(batch, options, dtype=torch.bool)
    labels = torch.arange(batch) % options
    targets = soft_diffusion_targets(labels, option_mask, smoothing=0.1)
    noisy_scores = targets * 0.7 + torch.randn_like(targets) * 0.3
    timestep = torch.full((batch,), 5, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []

    for _ in range(12):
        optimizer.zero_grad()
        output = model(
            noisy_scores,
            sequence_hidden,
            attention_mask,
            timestep,
            option_mask,
        )
        loss = torch.nn.functional.cross_entropy(output["logits"], targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert all(current < previous for previous, current in pairwise(losses))
