"""Measure cold load and end-to-end warm latency for an exported bundle."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from vons.data import read_jsonl


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def benchmark(bundle_manifest: Path, input_path: Path, limit: int) -> dict[str, object]:
    manifest = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer"])
    rows = read_jsonl(input_path)[:limit]
    load_started = time.perf_counter()
    session = ort.InferenceSession(manifest["onnx"], providers=["CPUExecutionProvider"])
    cold_load_ms = (time.perf_counter() - load_started) * 1000
    backend = manifest["backend"]
    option_count = int(manifest["option_count"])
    sequence_length = int(manifest["sequence_length"])
    rng = np.random.default_rng(7)
    latencies: list[float] = []
    for row in rows:
        texts = [f"{row.state}\nQuestion: {row.question}\nCandidate: {option}" for option in row.options]
        started = time.perf_counter()
        encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=sequence_length, return_tensors="np")
        if backend == "diffusion":
            pad = option_count - len(row.options)
            if pad < 0:
                raise ValueError("row has more candidates than the fixed diffusion bundle")
            feed = {
                "input_ids": np.pad(encoded["input_ids"][None, :, :], ((0, 0), (0, pad), (0, 0))),
                "attention_mask": np.pad(encoded["attention_mask"][None, :, :], ((0, 0), (0, pad), (0, 0))),
                "token_type_ids": np.pad(encoded["token_type_ids"][None, :, :], ((0, 0), (0, pad), (0, 0))),
                "option_mask": np.asarray([[True] * len(row.options) + [False] * pad], dtype=np.bool_),
                "initial_noise": rng.standard_normal((1, option_count), dtype=np.float32),
            }
        else:
            feed = {
                "input_ids": encoded["input_ids"][None, :, :],
                "attention_mask": encoded["attention_mask"][None, :, :],
                "token_type_ids": encoded["token_type_ids"][None, :, :],
                "option_mask": np.ones((1, len(row.options)), dtype=np.bool_),
            }
        session.run(None, feed)
        latencies.append((time.perf_counter() - started) * 1000)
    return {
        "backend": backend,
        "bundle": manifest["onnx"],
        "input": str(input_path),
        "rows": len(latencies),
        "provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "platform": platform.platform(),
        "candidate_slots": option_count if backend == "diffusion" else "row-defined",
        "sequence_length": sequence_length,
        "padding": "max_length",
        "inference_steps": manifest.get("inference_steps") if backend == "diffusion" else None,
        "cold_load_ms": cold_load_ms,
        "warm_p50_ms": percentile(latencies, 0.5) if latencies else None,
        "warm_p95_ms": percentile(latencies, 0.95) if latencies else None,
        "latency_scope": "tokenization plus ONNX Runtime inference",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(args.manifest, args.input, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
