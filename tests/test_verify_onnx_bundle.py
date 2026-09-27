from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("transformers")

from vons.models import ModelConfig, build_model

_SPEC = importlib.util.spec_from_file_location(
    "verify_onnx_bundle", Path(__file__).parents[1] / "tools/verify_onnx_bundle.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_model_config_from_checkpoint = _MODULE._model_config_from_checkpoint


def test_token_conditioned_checkpoint_config_reconstructs_model_state() -> None:
    expected = ModelConfig(
        hidden_size=256,
        latent_size=64,
        diffusion_steps=64,
        inference_steps=8,
        abstain_threshold=0.55,
        diffusion_conditioning="token_cross_attention",
        attention_heads=4,
    )
    trained_head = build_model("diffusion", option_count=32, config=expected)
    checkpoint = {
        "backend": "diffusion",
        "hidden_size": 256,
        "config": {
            "diffusion_conditioning": "token_cross_attention",
            "latent_size": 64,
            "diffusion_steps": 64,
            "inference_steps": 8,
            "abstain_threshold": 0.55,
            "attention_heads": 4,
        },
    }

    reconstructed = _model_config_from_checkpoint(checkpoint)
    verifier_head = build_model("diffusion", option_count=32, config=reconstructed)
    verifier_head.load_state_dict(trained_head.state_dict())

    assert reconstructed == expected


def test_token_conditioned_checkpoint_uses_training_defaults() -> None:
    checkpoint = {
        "backend": "diffusion",
        "hidden_size": 256,
        "config": {"diffusion_conditioning": "token_cross_attention"},
    }

    reconstructed = _model_config_from_checkpoint(checkpoint)

    assert reconstructed.latent_size == 64
    assert reconstructed.diffusion_steps == 64
    assert reconstructed.inference_steps == 8
    assert reconstructed.diffusion_conditioning == "token_cross_attention"


def test_pooled_checkpoint_without_schedule_fields_uses_export_defaults() -> None:
    expected = ModelConfig(
        hidden_size=256,
        latent_size=32,
        diffusion_steps=32,
        inference_steps=4,
        diffusion_conditioning="pooled",
    )
    trained_head = build_model("diffusion", option_count=32, config=expected)
    checkpoint = {
        "backend": "diffusion",
        "hidden_size": 256,
        "config": {},
    }

    reconstructed = _model_config_from_checkpoint(checkpoint)
    verifier_head = build_model("diffusion", option_count=32, config=reconstructed)
    verifier_head.load_state_dict(trained_head.state_dict())

    assert reconstructed == expected
