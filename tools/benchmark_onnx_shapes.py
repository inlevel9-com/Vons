"""Measure the four-cell ONNX shape/padding decomposition.

This is a CPU reference runner for G3. It intentionally keeps the historical
20-row benchmark untouched and writes raw samples for the matched Direct and
Diffusion shape cells described in ``docs/BENCHMARK_PROTOCOL.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from vons.data import Example, read_jsonl


@dataclass(frozen=True)
class ShapeCell:
    cell_id: str
    backend: str
    slots: int
    sequence_length: int
    live_candidates: int


def shape_cells(*, live_candidates: int, short_sequence: int, full_sequence: int, slots: int = 32) -> tuple[ShapeCell, ...]:
    if live_candidates < 2 or slots < live_candidates:
        raise ValueError("live_candidates must be at least 2 and no larger than slots")
    if short_sequence < 1 or full_sequence < short_sequence:
        raise ValueError("sequence buckets must be positive and ordered")
    return (
        ShapeCell("direct_live_short", "direct", live_candidates, short_sequence, live_candidates),
        ShapeCell("direct_padded_short", "direct", slots, short_sequence, live_candidates),
        ShapeCell("direct_padded_full", "direct", slots, full_sequence, live_candidates),
        ShapeCell("diffusion_padded_full", "diffusion", slots, full_sequence, live_candidates),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _runtime_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "vons.bundle.manifest/v1":
        return payload
    root = path.parent
    graph = next(item for item in payload["files"] if item["role"] == "model_graph")
    tokenizer_json = next(item for item in payload["files"] if item["role"] == "tokenizer" and item["path"].endswith("tokenizer.json"))
    metadata = payload["metadata"]
    return {
        "backend": metadata["backend"],
        "onnx": str(root / graph["path"]),
        "tokenizer": str(root / Path(tokenizer_json["path"]).parent),
        "option_count": metadata.get("option_count", 32),
        "sequence_length": metadata.get("sequence_length", 512),
        "bundle_manifest": str(path),
    }


def _options_for(row: Example, live_candidates: int) -> tuple[str, ...]:
    if len(row.options) >= live_candidates:
        return row.options[:live_candidates]
    values = list(row.options)
    base = values[0]
    for index in range(len(values), live_candidates):
        values.append(f"{base}__shape_fixture_{index}")
    return tuple(values)


def _noise(seed: int, case_index: int, slots: int) -> np.ndarray:
    generator = np.random.default_rng(seed + case_index)
    return generator.standard_normal((1, slots), dtype=np.float32)


def _token_arrays(tokenizer: Any, row: Example, options: tuple[str, ...], sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.zeros((len(options), sequence_length), dtype=np.int64)
    attention = np.zeros_like(ids)
    token_types = np.zeros_like(ids)
    padding_id = tokenizer.token_to_id("[PAD]")
    if padding_id is None:
        raise ValueError("bundle tokenizer has no [PAD] token")
    ids.fill(int(padding_id))
    for index, option in enumerate(options):
        text = f"{row.state}\nQuestion: {row.question}\nCandidate: {option}"
        encoded = tokenizer.encode(text)
        if len(encoded.ids) > sequence_length:
            raise ValueError(
                f"case {row.id} needs {len(encoded.ids)} tokens but the cell allows {sequence_length}"
            )
        ids[index, : len(encoded.ids)] = encoded.ids
        attention[index, : len(encoded.attention_mask)] = encoded.attention_mask
        token_types[index, : len(encoded.type_ids)] = encoded.type_ids
    return ids, attention, token_types


def _feed(
    token_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    options: tuple[str, ...],
    cell: ShapeCell,
    noise: np.ndarray,
    padding_id: int,
) -> tuple[dict[str, np.ndarray], dict[str, int | str]]:
    token_ids, attention, token_types = token_arrays
    ids = np.zeros((1, cell.slots, cell.sequence_length), dtype=np.int64)
    masks = np.zeros_like(ids)
    types = np.zeros_like(ids)
    ids.fill(int(padding_id))
    ids[0, : len(options)] = token_ids
    masks[0, : len(options)] = attention
    types[0, : len(options)] = token_types
    option_mask = np.asarray([[True] * len(options) + [False] * (cell.slots - len(options))], dtype=np.bool_)
    feed: dict[str, np.ndarray] = {
        "input_ids": ids,
        "attention_mask": masks,
        "token_type_ids": types,
        "option_mask": option_mask,
    }
    if cell.backend == "diffusion":
        feed["initial_noise"] = noise
    shape_info: dict[str, int | str] = {
        "live_candidates": len(options),
        "allocated_candidates": cell.slots,
        "live_tokens": int(np.sum(attention)),
        "sequence_length": cell.sequence_length,
    }
    return feed, shape_info


def _postprocess(outputs: list[np.ndarray], backend: str, live_candidates: int) -> None:
    scores = outputs[0][0, :live_candidates]
    if np.any(~np.isfinite(scores)):
        raise ValueError(f"{backend} returned non-finite live scores")
    shifted = scores - np.max(scores)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities)
    answerability = float(1.0 / (1.0 + np.exp(-float(outputs[1][0]))))
    if not np.all(np.isfinite(probabilities)) or not np.isfinite(answerability):
        raise ValueError(f"{backend} returned non-finite postprocessed values")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _cell_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("tokenize_ms", "prepare_feed_ms", "inference_and_readback_ms", "postprocess_ms", "total_request_ms")
    return {
        "samples": len(samples),
        **{
            field: {"p50": _percentile([float(item[field]) for item in samples], 0.5), "p95": _percentile([float(item[field]) for item in samples], 0.95)}
            for field in fields
        },
        "failures": sum(1 for item in samples if item["status"] != "ok"),
    }


def benchmark(
    direct_manifest: Path,
    diffusion_manifest: Path,
    input_path: Path,
    *,
    limit: int = 4,
    session_starts: int = 1,
    warmup: int = 2,
    repeats: int = 5,
    seed: int = 7,
    live_candidates: int = 4,
    short_sequence: int = 64,
    full_sequence: int = 512,
) -> dict[str, Any]:
    if limit < 1 or session_starts < 1 or warmup < 0 or repeats < 1:
        raise ValueError("limit/session_starts/repeats must be positive and warmup cannot be negative")
    cells = shape_cells(live_candidates=live_candidates, short_sequence=short_sequence, full_sequence=full_sequence)
    manifests = {"direct": _runtime_manifest(direct_manifest), "diffusion": _runtime_manifest(diffusion_manifest)}
    rows = read_jsonl(input_path)[:limit]
    if not rows:
        raise ValueError("input contains no rows")
    tokenizers: dict[str, Any] = {}
    tokenizer_hashes = {
        backend: _sha256_bytes((Path(manifest["tokenizer"]) / "tokenizer.json").read_bytes())
        for backend, manifest in manifests.items()
    }
    bundle_hashes = {
        backend: _sha256_bytes(Path(manifest["onnx"]).read_bytes())
        for backend, manifest in manifests.items()
    }
    raw_samples: list[dict[str, Any]] = []
    for backend, manifest in manifests.items():
        from tokenizers import Tokenizer

        tokenizers[backend] = Tokenizer.from_file(str(Path(manifest["tokenizer"]) / "tokenizer.json"))
    for session_index in range(session_starts):
        sessions: dict[str, Any] = {}
        for backend in ("direct", "diffusion"):
            sessions[backend] = ort.InferenceSession(manifests[backend]["onnx"], providers=["CPUExecutionProvider"])
        for cell in cells:
            tokenizer = tokenizers[cell.backend]
            padding_id = tokenizer.token_to_id("[PAD]")
            if padding_id is None:
                raise ValueError("bundle tokenizer has no [PAD] token")
            noise_by_case = {
                index: _noise(seed, index, cell.slots) for index in range(len(rows))
            }
            for row_index, row in enumerate(rows):
                options = _options_for(row, live_candidates)
                noise = noise_by_case[row_index]
                input_identity = _canonical_bytes({"state": row.state, "question": row.question, "options": options})
                text_hash = _sha256_bytes(input_identity)
                noise_hash = _sha256_bytes(noise.tobytes()) if cell.backend == "diffusion" else None
                for warmup_index in range(warmup):
                    token_arrays = _token_arrays(tokenizer, row, options, cell.sequence_length)
                    feed, _ = _feed(token_arrays, options, cell, noise, int(padding_id))
                    sessions[cell.backend].run(None, feed)
                for repeat_index in range(repeats):
                    started = time.perf_counter()
                    tokenize_started = time.perf_counter()
                    # Tokenization is kept separate from feed packing to expose sequence padding cost.
                    tokenized = _token_arrays(tokenizer, row, options, cell.sequence_length)
                    tokenize_ms = (time.perf_counter() - tokenize_started) * 1000
                    prepare_started = time.perf_counter()
                    feed, shape_info = _feed(tokenized, options, cell, noise, int(padding_id))
                    prepare_feed_ms = (time.perf_counter() - prepare_started) * 1000
                    inference_started = time.perf_counter()
                    output_values = sessions[cell.backend].run(None, feed)
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                    postprocess_started = time.perf_counter()
                    _postprocess(output_values, cell.backend, len(options))
                    postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
                    total_ms = (time.perf_counter() - started) * 1000
                    raw_samples.append(
                        {
                            "run_id": f"cpu-shape-{session_index}",
                            "session_start": session_index,
                            "case_id": row.id,
                            "repeat": repeat_index,
                            "warmup_excluded": warmup,
                            "cell": cell.cell_id,
                            "backend": cell.backend,
                            "requested_provider": "CPUExecutionProvider",
                            "observed_provider_status": "cpu_ort",
                            "bundle_hash": bundle_hashes[cell.backend],
                            "tokenizer_hash": tokenizer_hashes[cell.backend],
                            "input_hash": text_hash,
                            "noise_hash": noise_hash,
                            "platform": platform.platform(),
                            **shape_info,
                            "tokenize_ms": tokenize_ms,
                            "prepare_feed_ms": prepare_feed_ms,
                            "inference_and_readback_ms": inference_ms,
                            "postprocess_ms": postprocess_ms,
                            "total_request_ms": total_ms,
                            "status": "ok",
                            "error_kind": None,
                        }
                    )
        del sessions
    summaries = {
        cell.cell_id: _cell_summary([item for item in raw_samples if item["cell"] == cell.cell_id])
        for cell in cells
    }
    comparisons = {
        "direct_padding_candidates_ms": _difference(summaries, "direct_padded_short", "direct_live_short", "inference_and_readback_ms"),
        "direct_padding_sequence_ms": _difference(summaries, "direct_padded_full", "direct_padded_short", "inference_and_readback_ms"),
        "matched_diffusion_minus_direct_ms": _difference(summaries, "diffusion_padded_full", "direct_padded_full", "inference_and_readback_ms"),
    }
    return {
        "schema_version": "vons.shape-benchmark/v1",
        "method": "CPU ONNX Runtime shape decomposition; same live texts and masks across paired cells",
        "input": str(input_path),
        "rows": len(rows),
        "session_starts": session_starts,
        "warmup": warmup,
        "repeats": repeats,
        "seed": seed,
        "cells": [cell.__dict__ for cell in cells],
        "onnxruntime_version": ort.__version__,
        "platform": platform.platform(),
        "summaries": summaries,
        "comparisons": comparisons,
        "raw_samples": raw_samples,
    }


def _difference(summaries: dict[str, Any], right: str, left: str, field: str) -> float | None:
    right_value = summaries[right][field]["p50"]
    left_value = summaries[left][field]["p50"]
    if right_value is None or left_value is None:
        return None
    return float(right_value - left_value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-manifest", required=True, type=Path)
    parser.add_argument("--diffusion-manifest", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--session-starts", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--live-candidates", type=int, default=4)
    parser.add_argument("--short-sequence", type=int, default=64)
    parser.add_argument("--full-sequence", type=int, default=512)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(
        args.direct_manifest,
        args.diffusion_manifest,
        args.input,
        limit=args.limit,
        session_starts=args.session_starts,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        live_candidates=args.live_candidates,
        short_sequence=args.short_sequence,
        full_sequence=args.full_sequence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema_version": result["schema_version"], "rows": result["rows"], "summaries": result["summaries"], "comparisons": result["comparisons"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
