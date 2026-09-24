"""Compare an exported Direct bundle with its PyTorch checkpoint on fixed inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

from vons.models import ModelConfig, build_model


def verify(checkpoint_path: Path, bundle_path: Path, *, max_length: int = 32) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["backend"] not in {"direct", "diffusion"}:
        raise ValueError("unsupported checkpoint backend")
    encoder_config = checkpoint["config"]["encoder"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    encoder = AutoModel.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision")).eval()
    backend = checkpoint["backend"]
    head = build_model(
        backend,
        option_count=checkpoint["option_count"],
        config=ModelConfig(
            hidden_size=checkpoint["hidden_size"],
            diffusion_steps=int(checkpoint["config"].get("diffusion_steps", 32)),
            inference_steps=int(checkpoint["config"].get("inference_steps", 4)),
        ),
    )
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    texts = [
        "The user asked for weather in Seoul.\nQuestion: Which action?\nCandidate: call_weather",
        "The user asked for weather in Seoul.\nQuestion: Which action?\nCandidate: clarify",
    ]
    token_length = max_length if backend == "direct" else int(encoder_config.get("max_tokens", 512))
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=token_length, return_tensors="pt")
    with torch.no_grad():
        candidate_embeddings = encoder(**encoded).last_hidden_state[:, 0, :].unsqueeze(0)
        option_mask = torch.zeros((1, checkpoint["option_count"]), dtype=torch.bool)
        if backend == "direct":
            candidate_embeddings = candidate_embeddings[:, :2]
            option_mask = torch.ones((1, 2), dtype=torch.bool)
            pooled = candidate_embeddings.mean(dim=1)
            torch_output = head(candidate_embeddings, pooled, option_mask)
        else:
            padding = checkpoint["option_count"] - candidate_embeddings.shape[1]
            candidate_embeddings = torch.cat((candidate_embeddings, torch.zeros((1, padding, candidate_embeddings.shape[-1]))), dim=1)
            option_mask[:, :2] = True
            pooled = candidate_embeddings[:, :2].mean(dim=1)
            betas = torch.linspace(1e-4, 0.02, int(checkpoint["config"].get("diffusion_steps", 32)))
            alpha_bars = torch.cumprod(1.0 - betas, dim=0)
            timesteps = torch.linspace(len(alpha_bars) - 1, 0, int(checkpoint["config"].get("inference_steps", 4))).round().long()
            sample = torch.zeros((1, checkpoint["option_count"]))
            for index, timestep in enumerate(timesteps):
                predicted_noise = head(sample, pooled, torch.full((1,), int(timestep), dtype=torch.long))["noise"]
                alpha = alpha_bars[timestep]
                previous_alpha = alpha_bars[timesteps[index + 1]] if index + 1 < len(timesteps) else sample.new_tensor(1.0)
                predicted_x0 = (sample - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
                sample = torch.sqrt(previous_alpha) * predicted_x0 + torch.sqrt(1 - previous_alpha) * predicted_noise
            torch_output = {
                "scores": sample.masked_fill(~option_mask, float("-inf")),
                "answerability": head(torch.zeros_like(sample), pooled, torch.zeros((1,), dtype=torch.long))["answerability"],
            }
    session = ort.InferenceSession(str(bundle_path), providers=["CPUExecutionProvider"])
    options = checkpoint["option_count"] if backend == "diffusion" else 2
    ort_feed = {
        "input_ids": encoded["input_ids"].numpy()[None, :, :] if backend == "direct" else np.pad(encoded["input_ids"].numpy()[None, :, :], ((0, 0), (0, options - 2), (0, 0))),
        "attention_mask": encoded["attention_mask"].numpy()[None, :, :] if backend == "direct" else np.pad(encoded["attention_mask"].numpy()[None, :, :], ((0, 0), (0, options - 2), (0, 0))),
        "token_type_ids": encoded["token_type_ids"].numpy()[None, :, :] if backend == "direct" else np.pad(encoded["token_type_ids"].numpy()[None, :, :], ((0, 0), (0, options - 2), (0, 0))),
        "option_mask": np.asarray([[True, True] + [False] * (options - 2)], dtype=np.bool_),
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
