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
from .models import (
    ModelConfig,
    build_model,
    cosine_beta_schedule,
    ddim_sample,
    ddim_sample_token_conditioned,
    require_torch,
)


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


def _encode_examples(
    tokenizer: Any,
    encoder: Any,
    examples: list[Example],
    device: Any,
    max_length: int = 512,
    *,
    include_token_states: bool = False,
) -> tuple[Any, Any, Any, Any, list[Example], Any | None, Any | None]:
    torch, _ = require_torch()
    option_embeddings: list[Any] = []
    pooled_embeddings: list[Any] = []
    targets: list[int] = []
    kept: list[Example] = []
    token_states: list[Any] = []
    token_masks: list[Any] = []
    with torch.no_grad():
        for row in examples:
            texts = [f"{row.state}\nQuestion: {row.question}\nCandidate: {option}" for option in row.options]
            batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            output = encoder(**batch)
            hidden_states = output.last_hidden_state
            hidden = hidden_states[:, 0, :]
            option_embeddings.append(hidden)
            pooled_embeddings.append(hidden.mean(dim=0))
            if include_token_states:
                token_states.append(hidden_states)
                token_masks.append(batch["attention_mask"].bool())
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
    padded_token_states = None
    padded_token_masks = None
    if include_token_states:
        max_sequence = max((values.shape[1] for values in token_states), default=1)
        padded_token_states = torch.zeros(
            (len(kept), max_options, max_sequence, hidden_size), device=device
        )
        padded_token_masks = torch.zeros(
            (len(kept), max_options, max_sequence), dtype=torch.bool, device=device
        )
        for index, (values, values_mask) in enumerate(zip(token_states, token_masks, strict=True)):
            options, sequence = values.shape[:2]
            padded_token_states[index, :options, :sequence] = values
            padded_token_masks[index, :options, :sequence] = values_mask
    return (
        padded,
        pooled,
        torch.tensor(targets, dtype=torch.long, device=device),
        mask,
        kept,
        padded_token_states,
        padded_token_masks,
    )


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


def soft_diffusion_targets(
    labels: Any,
    option_mask: Any,
    *,
    smoothing: float = 0.1,
) -> Any:
    """Create masked continuous class targets for diffusion cross-entropy."""
    torch, _ = require_torch()
    if not 0.0 < smoothing < 1.0:
        raise ValueError("smoothing must be strictly between zero and one")
    if labels.ndim != 1 or option_mask.ndim != 2 or labels.shape[0] != option_mask.shape[0]:
        raise ValueError("labels and option_mask batch dimensions must agree")
    valid_counts = option_mask.sum(dim=-1)
    if bool((valid_counts < 2).any()):
        raise ValueError("soft diffusion targets require at least two valid options")
    if bool(((labels < 0) | (labels >= option_mask.shape[1])).any()):
        raise ValueError("diffusion labels must index the option axis")
    if not bool(option_mask.gather(1, labels.unsqueeze(1)).all()):
        raise ValueError("diffusion labels must select a valid option")
    off_target = smoothing / (valid_counts - 1).to(torch.float32)
    targets = off_target.unsqueeze(1) * option_mask.to(torch.float32)
    targets.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    return targets


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
    backend = str(config.get("backend", "direct"))
    diffusion_conditioning = str(
        config.get("diffusion_conditioning", "token_cross_attention")
    )
    if backend == "diffusion":
        config["diffusion_conditioning"] = diffusion_conditioning
        config.setdefault("diffusion_steps", 64)
        config.setdefault("inference_steps", 8)
        config.setdefault("noise_schedule", "cosine")
        config.setdefault("soft_target_smoothing", 0.1)
    encoder_config = config["encoder"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    encoder = AutoModel.from_pretrained(encoder_config["name"], revision=encoder_config.get("revision"))
    device = _device(torch, str(config.get("device", "auto")))
    encoder.to(device).eval()
    (
        option_embeddings,
        pooled,
        targets,
        option_mask,
        kept,
        sequence_hidden,
        sequence_attention,
    ) = _encode_examples(
        tokenizer,
        encoder,
        train_rows,
        device,
        int(encoder_config.get("max_tokens", 512)),
        include_token_states=(
            backend == "diffusion" and diffusion_conditioning == "token_cross_attention"
        ),
    )
    option_count = int(config.get("option_count", option_embeddings.shape[1]))
    if not (backend == "diffusion" and diffusion_conditioning == "token_cross_attention"):
        option_embeddings, option_mask = _pad_option_axis(
            option_embeddings, option_mask, option_count
        )
    model = build_model(
        backend,
        option_count=option_count,
        config=ModelConfig(
            hidden_size=option_embeddings.shape[-1],
            latent_size=int(config.get("latent_size", 64 if diffusion_conditioning == "token_cross_attention" else 32)),
            diffusion_steps=int(config.get("diffusion_steps", 64)),
            inference_steps=int(config.get("inference_steps", 8)),
            abstain_threshold=float(config.get("abstain_threshold", 0.55)),
            diffusion_conditioning=diffusion_conditioning,
            attention_heads=int(config.get("attention_heads", 4)),
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
        diffusion_steps = int(config.get("diffusion_steps", ModelConfig().diffusion_steps))
        if diffusion_conditioning == "token_cross_attention":
            betas = cosine_beta_schedule(diffusion_steps).to(device)
        else:
            from .models import linear_beta_schedule

            betas = linear_beta_schedule(diffusion_steps).to(device)
        schedule = torch.cumprod(1.0 - betas, dim=0)
    batch_size = max(1, int(config.get("batch_size", 16)))
    training_history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        epoch_losses: list[float] = []
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
                timestep = torch.randint(0, len(schedule), (len(indices),), device=device)
                if diffusion_conditioning == "token_cross_attention":
                    target = soft_diffusion_targets(
                        batch_labels,
                        option_mask[indices],
                        smoothing=float(config.get("soft_target_smoothing", 0.1)),
                    ).to(device)
                else:
                    target = torch.nn.functional.one_hot(
                        batch_labels, num_classes=option_embeddings.shape[1]
                    ).float()
                noise = torch.randn_like(target)
                alpha_bar = schedule[timestep].unsqueeze(-1)
                noisy = alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise
                if diffusion_conditioning == "token_cross_attention":
                    if sequence_hidden is None or sequence_attention is None:
                        raise RuntimeError("token-conditioned diffusion requires token states")
                    output = model(
                        noisy,
                        sequence_hidden[indices],
                        sequence_attention[indices],
                        timestep,
                        option_mask[indices],
                    )
                    if bool(batch_valid.any()):
                        finite_logits = output["logits"].masked_fill(
                            ~option_mask[indices], -1e4
                        )
                        decision_loss = nn.functional.cross_entropy(
                            finite_logits[batch_valid], target[batch_valid]
                        )
                    else:
                        decision_loss = output["answerability"].sum() * 0.0
                else:
                    output = model(noisy, pooled[indices], timestep)
                    if bool(batch_valid.any()):
                        decision_loss = nn.functional.mse_loss(
                            output["noise"][batch_valid], noise[batch_valid]
                        )
                    else:
                        decision_loss = output["answerability"].sum() * 0.0
            answerability_loss = nn.functional.binary_cross_entropy_with_logits(output["answerability"], answerable_targets[indices])
            loss = decision_loss + 0.1 * answerability_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        training_history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": sum(epoch_losses) / len(epoch_losses),
            }
        )
    artifact_dir = Path(config.get("artifact_dir", "artifacts/pilot"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "backend": backend, "option_count": option_count, "hidden_size": int(option_embeddings.shape[-1]), "config": config, "training_history": training_history}, checkpoint)
    manifest = {"model_id": config.get("model_id", "vons-pilot"), "backend": backend, "encoder": encoder_config, "device": str(device), "platform": platform.platform(), "seed": seed, "rows": len(kept), "data": dataset_manifest(data_path, examples), "checkpoint": str(checkpoint), "training_history": training_history, "created_at": time.time()}
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
    diffusion_conditioning = str(config.get("diffusion_conditioning", "pooled"))
    (
        option_embeddings,
        pooled,
        _targets,
        option_mask,
        kept,
        sequence_hidden,
        sequence_attention,
    ) = _encode_examples(
        tokenizer,
        encoder,
        rows,
        torch.device("cpu"),
        int(encoder_config.get("max_tokens", 512)),
        include_token_states=(
            checkpoint["backend"] == "diffusion"
            and diffusion_conditioning == "token_cross_attention"
        ),
    )
    expected_options = int(checkpoint["option_count"])
    if option_embeddings.shape[1] > expected_options:
        raise ValueError(f"evaluation input has {option_embeddings.shape[1]} candidates but checkpoint supports {expected_options}")
    if (
        option_embeddings.shape[1] < expected_options
        and diffusion_conditioning != "token_cross_attention"
    ):
        option_embeddings, option_mask = _pad_option_axis(option_embeddings, option_mask, expected_options)
    model = build_model(
        checkpoint["backend"],
        option_count=checkpoint["option_count"],
        config=ModelConfig(
            hidden_size=checkpoint["hidden_size"],
            latent_size=int(config.get("latent_size", 32)),
            diffusion_steps=int(config.get("diffusion_steps", 32)),
            inference_steps=int(config.get("inference_steps", 4)),
            abstain_threshold=float(config.get("abstain_threshold", 0.55)),
            diffusion_conditioning=diffusion_conditioning,
            attention_heads=int(config.get("attention_heads", 4)),
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
                if diffusion_conditioning == "token_cross_attention":
                    if sequence_hidden is None or sequence_attention is None:
                        raise RuntimeError("token-conditioned diffusion requires token states")
                    row_sequence = sequence_hidden[index : index + 1]
                    row_attention = sequence_attention[index : index + 1]
                    logits = ddim_sample_token_conditioned(
                        model,
                        row_sequence,
                        row_attention,
                        row_mask,
                        diffusion_steps=int(config.get("diffusion_steps", 64)),
                        inference_steps=int(config.get("inference_steps", 8)),
                        seed=(
                            None
                            if config.get("seed") is None
                            else int(config["seed"]) + index
                        ),
                    )
                    answerability = torch.sigmoid(
                        model(
                            torch.zeros_like(logits),
                            row_sequence,
                            row_attention,
                            torch.zeros((1,), dtype=torch.long),
                            row_mask,
                        )["answerability"]
                    )[0]
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
    diffusion_conditioning = str(config.get("diffusion_conditioning", "pooled"))
    model = build_model(
        checkpoint["backend"],
        option_count=checkpoint["option_count"],
        config=ModelConfig(
            hidden_size=checkpoint["hidden_size"],
            latent_size=int(config.get("latent_size", 32)),
            diffusion_steps=int(config.get("diffusion_steps", 32)),
            inference_steps=int(config.get("inference_steps", 4)),
            diffusion_conditioning=diffusion_conditioning,
            attention_heads=int(config.get("attention_heads", 4)),
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

    class TokenDiffusionExportWrapper(torch.nn.Module):
        def __init__(self, head: Any) -> None:
            super().__init__()
            self.head = head

        def forward(
            self,
            noisy_scores: Any,
            sequence_hidden: Any,
            attention_mask: Any,
            timestep: Any,
            option_mask: Any,
        ) -> tuple[Any, Any]:
            result = self.head(
                noisy_scores,
                sequence_hidden,
                attention_mask,
                timestep,
                option_mask,
            )
            return result["logits"], result["answerability"]

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
        if diffusion_conditioning == "token_cross_attention":
            example_options = min(4, int(checkpoint["option_count"]))
            wrapper = TokenDiffusionExportWrapper(model).eval()
            inputs = (
                torch.zeros((1, example_options)),
                torch.zeros((1, example_options, 64, checkpoint["hidden_size"])),
                torch.ones((1, example_options, 64), dtype=torch.bool),
                torch.zeros((1,), dtype=torch.long),
                torch.ones((1, example_options), dtype=torch.bool),
            )
            input_names = [
                "noisy_scores",
                "sequence_hidden",
                "attention_mask",
                "timestep",
                "option_mask",
            ]
            output_names = ["logits", "answerability"]
            dynamic_axes = {
                "noisy_scores": {0: "batch", 1: "options"},
                "sequence_hidden": {0: "batch", 1: "options", 2: "sequence_length"},
                "attention_mask": {0: "batch", 1: "options", 2: "sequence_length"},
                "timestep": {0: "batch"},
                "option_mask": {0: "batch", 1: "options"},
                "logits": {0: "batch", 1: "options"},
                "answerability": {0: "batch"},
            }
        else:
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
    encoder = AutoModel.from_pretrained(
        encoder_config["name"], revision=encoder_config.get("revision")
    )
    encoder.eval()
    conditioning = str(config.get("diffusion_conditioning", "pooled"))
    diffusion_steps = int(config.get("diffusion_steps", 32))
    inference_steps = int(config.get("inference_steps", 4))
    if not 2 <= inference_steps <= diffusion_steps:
        raise ValueError("inference_steps must be between 2 and diffusion_steps")
    model = build_model(
        "diffusion",
        option_count=checkpoint["option_count"],
        config=ModelConfig(
            hidden_size=checkpoint["hidden_size"],
            latent_size=int(config.get("latent_size", 32)),
            diffusion_steps=diffusion_steps,
            inference_steps=inference_steps,
            diffusion_conditioning=conditioning,
            attention_heads=int(config.get("attention_heads", 4)),
        ),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    class FullPooledDiffusionExportWrapper(torch.nn.Module):
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
            encoded = self.encoder(
                input_ids=input_ids.reshape(batch * options, sequence),
                attention_mask=attention_mask.reshape(batch * options, sequence),
                token_type_ids=token_type_ids.reshape(batch * options, sequence),
            ).last_hidden_state[:, 0, :]
            candidate_embeddings = encoded.reshape(batch, options, -1)
            weights = option_mask.to(candidate_embeddings.dtype).unsqueeze(-1)
            pooled = (candidate_embeddings * weights).sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1)
            sample = initial_noise
            timesteps = torch.linspace(
                diffusion_steps - 1, 0, inference_steps
            ).round().to(torch.long)
            for index in range(inference_steps):
                timestep = timesteps[index]
                timestep_batch = torch.full(
                    (batch,), timestep, dtype=torch.long, device=sample.device
                )
                predicted_noise = self.head(sample, pooled, timestep_batch)["noise"]
                alpha = self.alpha_bars[timestep]
                previous_alpha = (
                    self.alpha_bars[timesteps[index + 1]]
                    if index + 1 < inference_steps
                    else sample.new_tensor(1.0)
                )
                predicted_x0 = (
                    sample - torch.sqrt(1 - alpha) * predicted_noise
                ) / torch.sqrt(alpha)
                sample = torch.sqrt(previous_alpha) * predicted_x0 + torch.sqrt(
                    1 - previous_alpha
                ) * predicted_noise
            sample = sample.masked_fill(~option_mask.bool(), float("-inf"))
            answerability = self.head(
                initial_noise,
                pooled,
                torch.zeros((batch,), dtype=torch.long, device=sample.device),
            )["answerability"]
            return sample, answerability

    class FullTokenDiffusionExportWrapper(torch.nn.Module):
        def __init__(self, encoder_model: Any, head: Any) -> None:
            super().__init__()
            self.encoder = encoder_model
            self.head = head
            self.register_buffer(
                "alpha_bars",
                torch.cumprod(1.0 - cosine_beta_schedule(diffusion_steps), dim=0),
            )

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Any,
            option_mask: Any,
            initial_noise: Any,
        ) -> tuple[Any, Any]:
            batch, options, sequence = input_ids.shape
            encoded = self.encoder(
                input_ids=input_ids.reshape(batch * options, sequence),
                attention_mask=attention_mask.reshape(batch * options, sequence),
                token_type_ids=token_type_ids.reshape(batch * options, sequence),
            ).last_hidden_state.reshape(batch, options, sequence, -1)
            sample = initial_noise
            timesteps = torch.linspace(
                diffusion_steps - 1, 0, inference_steps
            ).round().to(torch.long)
            final_logits = sample
            answerability = sample[:, 0]
            for index in range(inference_steps):
                timestep = timesteps[index]
                timestep_batch = torch.full(
                    (batch,), timestep, dtype=torch.long, device=sample.device
                )
                result = self.head(
                    sample,
                    encoded,
                    attention_mask,
                    timestep_batch,
                    option_mask,
                )
                final_logits = result["logits"]
                answerability = result["answerability"]
                predicted_x0 = torch.softmax(final_logits, dim=-1)
                alpha = self.alpha_bars[timestep]
                predicted_noise = (
                    sample - torch.sqrt(alpha) * predicted_x0
                ) / torch.sqrt(1 - alpha).clamp_min(1e-6)
                previous_alpha = (
                    self.alpha_bars[timesteps[index + 1]]
                    if index + 1 < inference_steps
                    else sample.new_tensor(1.0)
                )
                sample = torch.sqrt(previous_alpha) * predicted_x0 + torch.sqrt(
                    1 - previous_alpha
                ) * predicted_noise
            return final_logits.masked_fill(
                ~option_mask.bool(), float("-inf")
            ), answerability

    sequence = int(encoder_config.get("max_tokens", 512))
    option_count = int(checkpoint["option_count"])
    example_options = min(4, option_count) if conditioning == "token_cross_attention" else option_count
    wrapper = (
        FullTokenDiffusionExportWrapper(encoder, model).eval()
        if conditioning == "token_cross_attention"
        else FullPooledDiffusionExportWrapper(encoder, model).eval()
    )
    inputs = (
        torch.zeros((1, example_options, sequence), dtype=torch.long),
        torch.ones((1, example_options, sequence), dtype=torch.long),
        torch.zeros((1, example_options, sequence), dtype=torch.long),
        torch.ones((1, example_options), dtype=torch.bool),
        torch.zeros((1, example_options)),
    )
    input_names = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "option_mask",
        "initial_noise",
    ]
    output_names = ["scores", "answerability"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
    if conditioning == "token_cross_attention":
        for name in ("input_ids", "attention_mask", "token_type_ids"):
            dynamic_axes[name].update({1: "options", 2: "sequence_length"})
        dynamic_axes["option_mask"][1] = "options"
        dynamic_axes["initial_noise"][1] = "options"
        dynamic_axes["scores"][1] = "options"
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
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_config["name"], revision=encoder_config.get("revision")
    )
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
        "dynamic_options": conditioning == "token_cross_attention",
        "dynamic_sequence_length": conditioning == "token_cross_attention",
        "diffusion_conditioning": conditioning,
        "noise_schedule": str(config.get("noise_schedule", "linear")),
        "diffusion_steps": diffusion_steps,
        "inference_steps": inference_steps,
        "input_names": input_names,
        "output_names": output_names,
        "scope": "end_to_end_encoder_plus_fixed_ddim_decision_head",
        "seed_contract": "host supplies deterministic initial_noise",
    }
    (output.parent / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output
