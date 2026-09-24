"""Small reproducible training/evaluation/export runners.

The encoder is deliberately loaded from an explicit Hugging Face revision. The
decision heads stay in this repository so Direct and Diffusion share the same
candidate representation during the first comparison.
"""

from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path
from typing import Any

from .data import Example, dataset_manifest, read_jsonl
from .evaluation import Prediction, summarize, write_report
from .models import ModelConfig, build_model, ddim_sample, require_torch


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("training requires optional dependencies: pip install -e '.[train]'") from exc
    return AutoModel, AutoTokenizer


def _load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("training config must be a JSON object")
    return value


def _device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _encode_examples(tokenizer: Any, encoder: Any, examples: list[Example], device: Any, max_length: int = 512) -> tuple[Any, Any, Any, Any, list[Example]]:
    torch, _ = require_torch()
    option_embeddings: list[Any] = []
    pooled_embeddings: list[Any] = []
    targets: list[int] = []
    kept: list[Example] = []
    with torch.no_grad():
        for row in examples:
            texts = [f"{row.state}\nQuestion: {row.question}\nCandidate: {option}" for option in row.options]
            batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            output = encoder(**batch)
            hidden = output.last_hidden_state[:, 0, :]
            option_embeddings.append(hidden)
            pooled_embeddings.append(hidden.mean(dim=0))
            targets.append(-1 if row.label is None else row.options.index(row.label))
            kept.append(row)
    max_options = max((len(item.options) for item in kept), default=2)
    hidden_size = option_embeddings[0].shape[-1] if option_embeddings else getattr(encoder.config, "hidden_size", 256)
    padded = torch.zeros((len(kept), max_options, hidden_size), device=device)
    mask = torch.zeros((len(kept), max_options), dtype=torch.bool, device=device)
    pooled = torch.stack(pooled_embeddings) if pooled_embeddings else torch.empty((0, hidden_size), device=device)
    for index, values in enumerate(option_embeddings):
        padded[index, : values.shape[0]] = values
        mask[index, : values.shape[0]] = True
    return padded, pooled, torch.tensor(targets, dtype=torch.long, device=device), mask, kept


def _pad_option_axis(option_embeddings: Any, option_mask: Any, option_count: int) -> tuple[Any, Any]:
    if option_embeddings.shape[1] > option_count:
        raise ValueError(f"data contains {option_embeddings.shape[1]} candidates but option_count is {option_count}")
    if option_embeddings.shape[1] == option_count:
        return option_embeddings, option_mask
    padded_embeddings = option_embeddings.new_zeros((option_embeddings.shape[0], option_count, option_embeddings.shape[-1]))
    padded_mask = option_mask.new_zeros((option_mask.shape[0], option_count))
    padded_embeddings[:, : option_embeddings.shape[1]] = option_embeddings
    padded_mask[:, : option_mask.shape[1]] = option_mask
    return padded_embeddings, padded_mask


def train(config_path: str | Path) -> Path:
    config = _load_config(config_path)
    torch, nn = require_torch()
    AutoModel, AutoTokenizer = _require_transformers()
    seed = int(config.get("seed", 7))
    random.seed(seed)
    torch.manual_seed(seed)
    data_path = Path(config["data"]["path"])
    examples = read_jsonl(data_path)
    train_rows = [row for row in examples if row.split == "train"]
    if not train_rows:
        raise ValueError("training data has no train split")
    encoder_config = config["encoder"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    encoder = AutoModel.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    device = _device(torch, str(config.get("device", "auto")))
    encoder.to(device).eval()
    option_embeddings, pooled, targets, option_mask, kept = _encode_examples(tokenizer, encoder, train_rows, device, int(encoder_config.get("max_tokens", 512)))
    option_count = int(config.get("option_count", option_embeddings.shape[1]))
    option_embeddings, option_mask = _pad_option_axis(option_embeddings, option_mask, option_count)
    backend = str(config.get("backend", "direct"))
    model = build_model(
        backend,
        option_count=option_embeddings.shape[1],
        config=ModelConfig(
            hidden_size=option_embeddings.shape[-1],
            diffusion_steps=int(config.get("diffusion_steps", 32)),
            inference_steps=int(config.get("inference_steps", 4)),
            abstain_threshold=float(config.get("abstain_threshold", 0.55)),
        ),
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 5e-5)))
    epochs = int(config.get("epochs", 3))
    labels = targets.clamp_min(0)
    answerable_targets = torch.tensor([row.answerable for row in kept], dtype=torch.float32, device=device)
    valid = targets >= 0
    if not bool(valid.any()):
        raise ValueError("training data has no answerable examples")
    schedule = None
    if backend == "diffusion":
        from .models import linear_beta_schedule

        diffusion_steps = int(config.get("diffusion_steps", ModelConfig().diffusion_steps))
        betas = linear_beta_schedule(diffusion_steps).to(device)
        schedule = torch.cumprod(1.0 - betas, dim=0)
    batch_size = max(1, int(config.get("batch_size", 16)))
    for _ in range(epochs):
        permutation = torch.randperm(len(kept), device=device)
        for start in range(0, len(kept), batch_size):
            indices = permutation[start : start + batch_size]
            batch_labels = labels[indices]
            batch_valid = valid[indices]
            optimizer.zero_grad()
            if backend == "direct":
                output = model(option_embeddings[indices], pooled[indices], option_mask[indices])
                if bool(batch_valid.any()):
                    decision_loss = nn.functional.cross_entropy(output["logits"][batch_valid], batch_labels[batch_valid])
                else:
                    decision_loss = output["answerability"].sum() * 0.0
            else:
                target = torch.nn.functional.one_hot(batch_labels, num_classes=option_embeddings.shape[1]).float()
                timestep = torch.randint(0, len(schedule), (len(indices),), device=device)
                noise = torch.randn_like(target)
                alpha_bar = schedule[timestep].unsqueeze(-1)
                noisy = alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise
                output = model(noisy, pooled[indices], timestep)
                if bool(batch_valid.any()):
                    decision_loss = nn.functional.mse_loss(output["noise"][batch_valid], noise[batch_valid])
                else:
                    decision_loss = output["answerability"].sum() * 0.0
            answerability_loss = nn.functional.binary_cross_entropy_with_logits(output["answerability"], answerable_targets[indices])
            loss = decision_loss + 0.1 * answerability_loss
            loss.backward()
            optimizer.step()
    artifact_dir = Path(config.get("artifact_dir", "artifacts/pilot"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "backend": backend, "option_count": int(option_embeddings.shape[1]), "hidden_size": int(option_embeddings.shape[-1]), "config": config}, checkpoint)
    manifest = {"model_id": config.get("model_id", "vons-pilot"), "backend": backend, "encoder": encoder_config, "device": str(device), "platform": platform.platform(), "seed": seed, "rows": len(kept), "data": dataset_manifest(data_path, examples), "checkpoint": str(checkpoint), "created_at": time.time()}
    (artifact_dir / "model-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return checkpoint


def evaluate(checkpoint_path: str | Path, input_path: str | Path, output_path: str | Path) -> Path:
    torch, _ = require_torch()
    AutoModel, AutoTokenizer = _require_transformers()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    encoder_config = config["encoder"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    encoder = AutoModel.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    encoder.eval()
    rows = read_jsonl(input_path)
    option_embeddings, pooled, _targets, option_mask, kept = _encode_examples(tokenizer, encoder, rows, torch.device("cpu"), int(encoder_config.get("max_tokens", 512)))
    expected_options = int(checkpoint["option_count"])
    if option_embeddings.shape[1] > expected_options:
        raise ValueError(f"evaluation input has {option_embeddings.shape[1]} candidates but checkpoint supports {expected_options}")
    if option_embeddings.shape[1] < expected_options:
        option_embeddings, option_mask = _pad_option_axis(option_embeddings, option_mask, expected_options)
    model = build_model(
        checkpoint["backend"],
        option_count=checkpoint["option_count"],
        config=ModelConfig(
            hidden_size=checkpoint["hidden_size"],
            diffusion_steps=int(config.get("diffusion_steps", 32)),
            inference_steps=int(config.get("inference_steps", 4)),
            abstain_threshold=float(config.get("abstain_threshold", 0.55)),
        ),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictions: list[Prediction] = []
    with torch.no_grad():
        for index, row in enumerate(kept):
            started = time.perf_counter()
            row_embeddings = option_embeddings[index : index + 1]
            row_pooled = pooled[index : index + 1]
            row_mask = option_mask[index : index + 1]
            if checkpoint["backend"] == "direct":
                output = model(row_embeddings, row_pooled, row_mask)
                logits = output["logits"]
                answerability = torch.sigmoid(output["answerability"])[0]
            else:
                logits = ddim_sample(
                    model,
                    row_pooled,
                    option_count=expected_options,
                    diffusion_steps=int(config.get("diffusion_steps", 32)),
                    inference_steps=int(config.get("inference_steps", 4)),
                    seed=None if config.get("seed") is None else int(config["seed"]) + index,
                )
                answerability = torch.sigmoid(model(
                    torch.zeros_like(logits), row_pooled,
                    torch.zeros((1,), dtype=torch.long),
                )["answerability"])[0]
                logits = logits.masked_fill(~row_mask, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            values = probabilities[0, : len(row.options)]
            choice_index = int(values.argmax())
            confidence = float(values.max() * answerability)
            abstained = confidence < float(config.get("abstain_threshold", 0.55)) or float(answerability) < float(config.get("answerability_threshold", 0.5))
            predictions.append(Prediction(row.id, None if abstained else row.options[choice_index], {option: float(values[offset]) for offset, option in enumerate(row.options)}, confidence, abstained, (time.perf_counter() - started) * 1000))
    report = Path(output_path)
    report_config = {**config, "latency_scope": "decision_head_only; encoder_embeddings_precomputed"}
    write_report(report, config=report_config, summary=summarize(kept, predictions), predictions=predictions)
    return report


def export_head(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    """Export a fixed candidate-count decision head; encoder export is separate."""
    torch, _ = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = build_model(
        checkpoint["backend"],
        option_count=checkpoint["option_count"],
        config=ModelConfig(
            hidden_size=checkpoint["hidden_size"],
            diffusion_steps=int(config.get("diffusion_steps", 32)),
            inference_steps=int(config.get("inference_steps", 4)),
        ),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    class DirectExportWrapper(torch.nn.Module):
        def __init__(self, head: Any) -> None:
            super().__init__()
            self.head = head

        def forward(self, option_embeddings: Any, pooled: Any, option_mask: Any) -> tuple[Any, Any]:
            result = self.head(option_embeddings, pooled, option_mask)
            return result["logits"], result["answerability"]

    class DiffusionExportWrapper(torch.nn.Module):
        def __init__(self, head: Any) -> None:
            super().__init__()
            self.head = head

        def forward(self, noisy_scores: Any, pooled: Any, timestep: Any) -> tuple[Any, Any]:
            result = self.head(noisy_scores, pooled, timestep)
            return result["noise"], result["answerability"]

    if checkpoint["backend"] == "direct":
        wrapper = DirectExportWrapper(model).eval()
        inputs = (
            torch.zeros((1, checkpoint["option_count"], checkpoint["hidden_size"])),
            torch.zeros((1, checkpoint["hidden_size"])),
            torch.ones((1, checkpoint["option_count"]), dtype=torch.bool),
        )
        input_names = ["option_embeddings", "pooled", "option_mask"]
        output_names = ["logits", "answerability"]
        dynamic_axes = {
            "option_embeddings": {0: "batch"}, "pooled": {0: "batch"}, "option_mask": {0: "batch"},
            "logits": {0: "batch"}, "answerability": {0: "batch"},
        }
    elif checkpoint["backend"] == "diffusion":
        wrapper = DiffusionExportWrapper(model).eval()
        inputs = (
            torch.zeros((1, checkpoint["option_count"])),
            torch.zeros((1, checkpoint["hidden_size"])),
            torch.zeros((1,), dtype=torch.long),
        )
        input_names = ["noisy_scores", "pooled", "timestep"]
        output_names = ["noise", "answerability"]
        dynamic_axes = {
            "noisy_scores": {0: "batch"}, "pooled": {0: "batch"}, "timestep": {0: "batch"},
            "noise": {0: "batch"}, "answerability": {0: "batch"},
        }
    else:
        raise ValueError(f"unsupported backend {checkpoint['backend']!r}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        output,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )
    return output


def export_direct_bundle(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    """Export the Direct encoder and decision head as one ONNX graph.

    The graph keeps candidate count dynamic but preserves the same per-candidate
    BERT CLS representation used during training. Tokenizer files are copied
    beside the graph by the caller; this function does not invent token IDs or
    a browser runtime.
    """
    torch, _ = require_torch()
    AutoModel, AutoTokenizer = _require_transformers()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["backend"] != "direct":
        raise ValueError("end-to-end bundle export currently supports the direct backend only")
    config = checkpoint["config"]
    encoder_config = config["encoder"]
    encoder = AutoModel.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    encoder.eval()
    model = build_model(
        "direct",
        option_count=checkpoint["option_count"],
        config=ModelConfig(hidden_size=checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    class FullDirectExportWrapper(torch.nn.Module):
        def __init__(self, encoder_model: Any, head: Any) -> None:
            super().__init__()
            self.encoder = encoder_model
            self.head = head

        def forward(self, input_ids: Any, attention_mask: Any, token_type_ids: Any, option_mask: Any) -> tuple[Any, Any]:
            batch, options, sequence = input_ids.shape
            flat_ids = input_ids.reshape(batch * options, sequence)
            flat_attention = attention_mask.reshape(batch * options, sequence)
            flat_types = token_type_ids.reshape(batch * options, sequence)
            encoded = self.encoder(
                input_ids=flat_ids,
                attention_mask=flat_attention,
                token_type_ids=flat_types,
            ).last_hidden_state[:, 0, :]
            candidate_embeddings = encoded.reshape(batch, options, -1)
            weights = option_mask.to(candidate_embeddings.dtype).unsqueeze(-1)
            pooled = (candidate_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            result = self.head(candidate_embeddings, pooled, option_mask)
            return result["logits"], result["answerability"]

    wrapper = FullDirectExportWrapper(encoder, model).eval()
    sequence = int(encoder_config.get("max_tokens", 512))
    option_count = int(checkpoint["option_count"])
    inputs = (
        torch.zeros((1, option_count, sequence), dtype=torch.long),
        torch.ones((1, option_count, sequence), dtype=torch.long),
        torch.zeros((1, option_count, sequence), dtype=torch.long),
        torch.ones((1, option_count), dtype=torch.bool),
    )
    input_names = ["input_ids", "attention_mask", "token_type_ids", "option_mask"]
    output_names = ["logits", "answerability"]
    dynamic_axes = {
        name: {0: "batch"} for name in input_names + output_names
    }
    dynamic_axes["input_ids"][1] = "options"
    dynamic_axes["input_ids"][2] = "sequence"
    dynamic_axes["attention_mask"][1] = "options"
    dynamic_axes["attention_mask"][2] = "sequence"
    dynamic_axes["token_type_ids"][1] = "options"
    dynamic_axes["token_type_ids"][2] = "sequence"
    dynamic_axes["option_mask"][1] = "options"
    dynamic_axes["logits"][1] = "options"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        output,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )
    tokenizer = AutoTokenizer.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    tokenizer.save_pretrained(output.parent / "tokenizer")
    manifest = {
        "format": "onnx",
        "backend": "direct",
        "encoder": encoder_config,
        "checkpoint": str(checkpoint_path),
        "onnx": str(output),
        "tokenizer": str(output.parent / "tokenizer"),
        "option_count": option_count,
        "sequence_length": sequence,
        "input_names": input_names,
        "output_names": output_names,
        "scope": "end_to_end_encoder_plus_decision_head",
    }
    (output.parent / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output


def export_diffusion_bundle(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    """Export a fixed-step Diffusion encoder/head graph as one ONNX bundle."""
    torch, _ = require_torch()
    AutoModel, AutoTokenizer = _require_transformers()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["backend"] != "diffusion":
        raise ValueError("diffusion bundle export requires a diffusion checkpoint")
    config = checkpoint["config"]
    encoder_config = config["encoder"]
    encoder = AutoModel.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    encoder.eval()
    diffusion_steps = int(config.get("diffusion_steps", 32))
    inference_steps = int(config.get("inference_steps", 4))
    if not 2 <= inference_steps <= diffusion_steps:
        raise ValueError("inference_steps must be between 2 and diffusion_steps")
    model = build_model(
        "diffusion",
        option_count=checkpoint["option_count"],
        config=ModelConfig(
            hidden_size=checkpoint["hidden_size"],
            diffusion_steps=diffusion_steps,
            inference_steps=inference_steps,
        ),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    class FullDiffusionExportWrapper(torch.nn.Module):
        def __init__(self, encoder_model: Any, head: Any) -> None:
            super().__init__()
            self.encoder = encoder_model
            self.head = head
            betas = torch.linspace(1e-4, 0.02, diffusion_steps)
            self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Any,
            option_mask: Any,
            initial_noise: Any,
        ) -> tuple[Any, Any]:
            batch, options, sequence = input_ids.shape
            flat_ids = input_ids.reshape(batch * options, sequence)
            flat_attention = attention_mask.reshape(batch * options, sequence)
            flat_types = token_type_ids.reshape(batch * options, sequence)
            encoded = self.encoder(
                input_ids=flat_ids,
                attention_mask=flat_attention,
                token_type_ids=flat_types,
            ).last_hidden_state[:, 0, :]
            candidate_embeddings = encoded.reshape(batch, options, -1)
            weights = option_mask.to(candidate_embeddings.dtype).unsqueeze(-1)
            pooled = (candidate_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            sample = initial_noise
            timesteps = torch.linspace(diffusion_steps - 1, 0, inference_steps).round().to(torch.long)
            for index in range(inference_steps):
                timestep = timesteps[index]
                timestep_batch = torch.full((batch,), timestep, dtype=torch.long, device=sample.device)
                predicted_noise = self.head(sample, pooled, timestep_batch)["noise"]
                alpha = self.alpha_bars[timestep]
                previous_timestep = timesteps[index + 1] if index + 1 < inference_steps else torch.tensor(0, device=sample.device)
                previous_alpha = self.alpha_bars[previous_timestep] if index + 1 < inference_steps else sample.new_tensor(1.0)
                predicted_x0 = (sample - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
                sample = torch.sqrt(previous_alpha) * predicted_x0 + torch.sqrt(1 - previous_alpha) * predicted_noise
            sample = sample.masked_fill(~option_mask.bool(), float("-inf"))
            answerability = self.head(initial_noise, pooled, torch.zeros((batch,), dtype=torch.long, device=sample.device))["answerability"]
            return sample, answerability

    wrapper = FullDiffusionExportWrapper(encoder, model).eval()
    sequence = int(encoder_config.get("max_tokens", 512))
    option_count = int(checkpoint["option_count"])
    inputs = (
        torch.zeros((1, option_count, sequence), dtype=torch.long),
        torch.ones((1, option_count, sequence), dtype=torch.long),
        torch.zeros((1, option_count, sequence), dtype=torch.long),
        torch.ones((1, option_count), dtype=torch.bool),
        torch.zeros((1, option_count)),
    )
    input_names = ["input_ids", "attention_mask", "token_type_ids", "option_mask", "initial_noise"]
    output_names = ["scores", "answerability"]
    # Diffusion's first layer is parameterized by the configured candidate
    # count, so its option and sequence dimensions are intentionally fixed.
    # Batch remains dynamic for serving.
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        output,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )
    tokenizer = AutoTokenizer.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    tokenizer.save_pretrained(output.parent / "tokenizer")
    manifest = {
        "format": "onnx",
        "backend": "diffusion",
        "encoder": encoder_config,
        "checkpoint": str(checkpoint_path),
        "onnx": str(output),
        "tokenizer": str(output.parent / "tokenizer"),
        "option_count": option_count,
        "sequence_length": sequence,
        "diffusion_steps": diffusion_steps,
        "inference_steps": inference_steps,
        "input_names": input_names,
        "output_names": output_names,
        "scope": "end_to_end_encoder_plus_fixed_ddim_decision_head",
        "seed_contract": "host supplies deterministic initial_noise",
    }
    (output.parent / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output
