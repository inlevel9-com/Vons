"""Compare an exported end-to-end ONNX bundle with its PyTorch checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

from vons.models import ModelConfig, build_model, cosine_beta_schedule


def _model_config_from_checkpoint(checkpoint: dict[str, object]) -> ModelConfig:
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise TypeError("checkpoint config must be a mapping")
    backend = str(checkpoint["backend"])
    conditioning = str(config.get("diffusion_conditioning", "pooled"))
    token_conditioned = backend == "diffusion" and conditioning == "token_cross_attention"
    return ModelConfig(
        hidden_size=int(checkpoint["hidden_size"]),
        latent_size=int(config.get("latent_size", 64 if token_conditioned else 32)),
        diffusion_steps=int(config.get("diffusion_steps", 64 if token_conditioned else 32)),
        inference_steps=int(config.get("inference_steps", 8 if token_conditioned else 4)),
        abstain_threshold=float(config.get("abstain_threshold", 0.55)),
        diffusion_conditioning=conditioning,
        attention_heads=int(config.get("attention_heads", 4)),
    )


def verify(checkpoint_path: Path, bundle_path: Path, *, max_length: int = 32) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["backend"] not in {"direct", "diffusion"}:
        raise ValueError("unsupported checkpoint backend")
    encoder_config = checkpoint["config"]["encoder"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    encoder = AutoModel.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision")).eval()
    backend = checkpoint["backend"]
    conditioning = str(checkpoint["config"].get("diffusion_conditioning", "pooled"))
    head = build_model(
        backend,
        option_count=checkpoint["option_count"],
        config=_model_config_from_checkpoint(checkpoint),
    )
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    texts = [
        "The user asked for weather in Seoul.\nQuestion: Which action?\nCandidate: call_weather",
        "The user asked for weather in Seoul.\nQuestion: Which action?\nCandidate: clarify",
    ]
    token_length = max_length if backend == "direct" else int(encoder_config.get("max_tokens", 512))
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=token_length, return_tensors="pt")
    encoded.setdefault("token_type_ids", torch.zeros_like(encoded["input_ids"]))
    options = (
        checkpoint["option_count"]
        if backend == "diffusion" and conditioning == "pooled"
        else len(texts)
    )
    if options > len(texts):
        for name, tensor in encoded.items():
            encoded[name] = torch.nn.functional.pad(tensor, (0, 0, 0, options - len(texts)))
    batched = {name: tensor.unsqueeze(0) for name, tensor in encoded.items()}
    option_mask = torch.zeros((1, options), dtype=torch.bool)
    option_mask[:, : len(texts)] = True
    with torch.no_grad():
        if backend == "direct":
            candidate_embeddings = encoder(
                input_ids=batched["input_ids"].reshape(options, token_length),
                attention_mask=batched["attention_mask"].reshape(options, token_length),
                token_type_ids=batched["token_type_ids"].reshape(options, token_length),
            ).last_hidden_state[:, 0, :].reshape(1, options, -1)
            pooled = (candidate_embeddings * option_mask.unsqueeze(-1)).sum(dim=1) / option_mask.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
            torch_output = head(candidate_embeddings, pooled, option_mask)
        elif conditioning == "token_cross_attention":
            sequence_hidden = encoder(
                input_ids=batched["input_ids"].reshape(options, token_length),
                attention_mask=batched["attention_mask"].reshape(options, token_length),
                token_type_ids=batched["token_type_ids"].reshape(options, token_length),
            ).last_hidden_state.reshape(1, options, token_length, -1)
            attention_mask = batched["attention_mask"]
            diffusion_steps = int(checkpoint["config"].get("diffusion_steps", 64))
            inference_steps = int(checkpoint["config"].get("inference_steps", 8))
            alpha_bars = torch.cumprod(1.0 - cosine_beta_schedule(diffusion_steps), dim=0)
            timesteps = torch.linspace(diffusion_steps - 1, 0, inference_steps).round().long()
            sample = torch.zeros((1, options))
            torch_output = {}
            for index, timestep in enumerate(timesteps):
                timestep_batch = torch.full((1,), int(timestep), dtype=torch.long)
                result = head(sample, sequence_hidden, attention_mask, timestep_batch, option_mask)
                predicted_x0 = torch.softmax(result["logits"], dim=-1)
                alpha = alpha_bars[timestep]
                predicted_noise = (sample - torch.sqrt(alpha) * predicted_x0) / torch.sqrt(
                    1 - alpha
                ).clamp_min(1e-6)
                previous_alpha = (
                    alpha_bars[timesteps[index + 1]]
                    if index + 1 < len(timesteps)
                    else sample.new_tensor(1.0)
                )
                sample = torch.sqrt(previous_alpha) * predicted_x0 + torch.sqrt(
                    1 - previous_alpha
                ) * predicted_noise
                torch_output = result
            torch_output = {
                "scores": torch_output["logits"],
                "answerability": torch_output["answerability"],
            }
        else:
            candidate_embeddings = encoder(
                input_ids=batched["input_ids"].reshape(options, token_length),
                attention_mask=batched["attention_mask"].reshape(options, token_length),
                token_type_ids=batched["token_type_ids"].reshape(options, token_length),
            ).last_hidden_state[:, 0, :].reshape(1, options, -1)
            weights = option_mask.to(candidate_embeddings.dtype).unsqueeze(-1)
            pooled = (candidate_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            diffusion_steps = int(checkpoint["config"].get("diffusion_steps", 32))
            inference_steps = int(checkpoint["config"].get("inference_steps", 4))
            betas = torch.linspace(1e-4, 0.02, diffusion_steps)
            alpha_bars = torch.cumprod(1.0 - betas, dim=0)
            timesteps = torch.linspace(len(alpha_bars) - 1, 0, inference_steps).round().long()
            sample = torch.zeros((1, checkpoint["option_count"]))
            for index, timestep in enumerate(timesteps):
                predicted_noise = head(sample, pooled, torch.full((1,), int(timestep), dtype=torch.long))["noise"]
                alpha = alpha_bars[timestep]
                previous_alpha = alpha_bars[timesteps[index + 1]] if index + 1 < len(timesteps) else sample.new_tensor(1.0)
                predicted_x0 = (sample - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
                sample = torch.sqrt(previous_alpha) * predicted_x0 + torch.sqrt(1 - previous_alpha) * predicted_noise
            torch_output = {
                "scores": sample.masked_fill(~option_mask, float("-inf")),
                "answerability": head(
                    torch.zeros_like(sample), pooled, torch.zeros((1,), dtype=torch.long)
                )["answerability"],
            }
    session = ort.InferenceSession(str(bundle_path), providers=["CPUExecutionProvider"])
    ort_feed = {
        "input_ids": batched["input_ids"].numpy(),
        "attention_mask": batched["attention_mask"].numpy(),
        "token_type_ids": batched["token_type_ids"].numpy(),
        "option_mask": option_mask.numpy(),
    }
    if backend == "diffusion":
        ort_feed["initial_noise"] = np.zeros((1, options), dtype=np.float32)
    ort_output = session.run(None, ort_feed)
    score_name = "logits" if backend == "direct" else "scores"
    score_delta = torch_output[score_name].numpy() - ort_output[0]
    finite_scores = np.isfinite(torch_output[score_name].numpy()) & np.isfinite(ort_output[0])
    max_score_delta = float(np.max(np.abs(score_delta[finite_scores]))) if finite_scores.any() else 0.0
    result = {
        "checkpoint": str(checkpoint_path),
        "bundle": str(bundle_path),
        "provider": "CPUExecutionProvider",
        "backend": backend,
        "max_abs_logits": max_score_delta,
        "max_abs_answerability": float(np.max(np.abs(torch_output["answerability"].numpy() - ort_output[1]))),
        "pass": bool(
            np.allclose(torch_output[score_name].numpy(), ort_output[0], atol=1e-5, rtol=1e-5, equal_nan=True)
            and np.allclose(torch_output["answerability"].numpy(), ort_output[1], atol=1e-5, rtol=1e-5)
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.checkpoint, args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
