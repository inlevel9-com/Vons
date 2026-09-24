"""Generate real, separate k-specific Mind2Web selections without reading labels.

Strict 512-token question budget: overflow is an observed error, never silently
truncated. Outputs stay local because candidate text comes from restricted data.
The lexical baseline bypasses Vons' input contract and is labeled accordingly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

QUESTION = "Which element should the agent interact with next to achieve the goal?"
KS = (5, 10, 20, 32)


class ModelOutputError(ValueError):
    """Invalid numerical evidence stops the comparison, preserving its journal."""


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def make_request(row: dict[str, Any], k: int) -> dict[str, Any]:
    if k not in KS:
        raise ValueError("k must be 5, 10, 20 or 32")
    if any(key in row for key in ("positive_ids", "target_id", "no_positive", "label")):
        raise ValueError("selector input must be label-free retrieval output")
    fingerprint = row.get("input_sha256")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError("retrieval input requires a valid input_sha256 before inference")
    candidates = row["candidates"]
    ids = [item["id"] for item in candidates]
    if any(not isinstance(key, str) or not key for key in ids) or len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique nonempty strings")
    if ids != row["generated_candidates"]:
        raise ValueError("candidate order disagrees with retriever output")
    selected = candidates[:k]
    if any(not isinstance(item["text"], str) for item in selected):
        raise ValueError("candidate text must be a string")
    return {"state": canonical(row["state"]), "question": QUESTION,
            "candidates": [{"id": item["id"], "text": item["text"]} for item in selected]}


def decode(scores: list[float], answerability_logit: float,
           ids: list[str], threshold: float = 0.55) -> dict[str, Any]:
    if len(scores) != len(ids) or not ids or not all(math.isfinite(x) for x in scores):
        raise ModelOutputError("model returned invalid scores")
    if not math.isfinite(answerability_logit):
        raise ModelOutputError("model returned non-finite answerability")
    pivot = max(scores)
    weights = [math.exp(value - pivot) for value in scores]
    total = sum(weights)
    probabilities = [value / total for value in weights]
    z = math.exp(-abs(answerability_logit))
    answerability = 1 / (1 + z) if answerability_logit >= 0 else z / (1 + z)
    index = max(range(len(ids)), key=probabilities.__getitem__)
    confidence = probabilities[index] * answerability
    reason = ("answerability_below_threshold" if answerability < 0.5 else
              "confidence_below_threshold" if confidence < threshold else None)
    return {"selection": ids[index] if reason is None else None,
            "status": "ok" if reason is None else "abstain", "reason": reason,
            "probabilities": dict(zip(ids, probabilities)), "confidence": confidence,
            "answerability": answerability, "raw_scores": scores,
            "answerability_logit": answerability_logit}


class DirectSelector:
    """Offline, verified shared pilot graph; no fitted calibration is implied."""

    def __init__(self, manifest: Path, expected_digest: str) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        spec = importlib.util.spec_from_file_location("shared_export", Path(__file__).with_name("export_shared_bundle.py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.verify_manifest(manifest, expected_digest=expected_digest)
        self.np = np
        self.manifest = manifest
        self.tokenizer = Tokenizer.from_file(str(manifest.parent / "tokenizer/tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 1
        settings.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(str(manifest.parent / "encoder.onnx"), settings,
                                           providers=["CPUExecutionProvider"])
        self.head = ort.InferenceSession(str(manifest.parent / "direct.onnx"), settings,
                                        providers=["CPUExecutionProvider"])
        self.provenance = {"backend": "direct", "model_manifest_sha256": expected_digest,
                           "tokenizer_sha256": file_hash(manifest.parent / "tokenizer/tokenizer.json"),
                           "ort_version": ort.__version__, "provider": "CPUExecutionProvider",
                           "calibration": "none; historical default thresholds 0.55/0.5",
                           "threads": 1, "sequence_padding": "longest candidate input",
                           "candidate_padding": "live", "input_budget_tokens": 512}

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        np = self.np
        candidates = request["candidates"]
        if not 2 <= len(candidates) <= 32:
            raise ValueError("candidate_count_outside_2_32")
        prefix = f'{request["state"]}\nQuestion: {request["question"]}'
        all_text = prefix + "\nCandidates:\n" + "\n".join(item["text"] for item in candidates)
        token_count = len(self.tokenizer.encode(all_text).ids)
        if token_count > 512:
            return {"selection": None, "status": "error", "reason": "input_overflow",
                    "question_tokens": token_count}
        encoded = [self.tokenizer.encode(prefix + "\nCandidate: " + item["text"])
                   for item in candidates]
        length = max(len(item.ids) for item in encoded)
        if length > 512:
            return {"selection": None, "status": "error", "reason": "candidate_input_overflow",
                    "question_tokens": token_count}
        ids = np.full((1, len(candidates), length), self.tokenizer.token_to_id("[PAD]"), dtype=np.int64)
        masks = np.zeros_like(ids)
        types = np.zeros_like(ids)
        for index, item in enumerate(encoded):
            ids[0, index, :len(item.ids)] = item.ids
            masks[0, index, :len(item.ids)] = item.attention_mask
            types[0, index, :len(item.ids)] = item.type_ids
        option_mask = np.ones((1, len(candidates)), dtype=np.bool_)
        feed = {"input_ids": ids, "attention_mask": masks, "token_type_ids": types,
                "option_mask": option_mask}
        embeddings, pooled = self.encoder.run(None, feed)
        scores, answerability = self.head.run(None, {"candidate_embeddings": embeddings,
                                                    "pooled": pooled, "option_mask": option_mask})
        result = decode(scores[0].tolist(), float(answerability[0]),
                        [item["id"] for item in candidates])
        result.update(question_tokens=token_count,
                      input_array_sha256=object_hash({key: value.tolist() for key, value in feed.items()}))
        return result


def predict_file(source: Path, output: Path, *, split: str, k: int,
                 select: Callable[[dict[str, Any]], dict[str, Any]],
                 provenance: dict[str, Any]) -> dict[str, Any]:
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise ValueError("refusing to overwrite prediction evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    source_digest = file_hash(source)
    config = {"split": split, "k": k, "provenance": provenance, "adapted": False,
              "question": QUESTION, "input_overflow": "preserve error; no truncation"}
    config_digest = object_hash(config)
    with source.open() as handle, partial.open("x") as destination:
        for line in handle:
            row = json.loads(line)
            if row["split"] != split:
                continue
            if row["id"] in seen:
                raise ValueError("duplicate retrieval row ID")
            seen.add(row["id"])
            request = make_request(row, k)
            started = time.perf_counter()
            try:
                result = select(request)
            except ModelOutputError:
                raise
            except Exception as exc:  # noqa: BLE001 -- retain per-row provider failures in the denominator
                result = {"selection": None, "status": "error", "reason": type(exc).__name__,
                          "error": str(exc)}
            elapsed = (time.perf_counter() - started) * 1000
            record = {"id": row["id"], "task_id": row["task_id"], "split": split,
                      "k": k, "generated_candidates": [item["id"] for item in request["candidates"]],
                      "request_sha256": object_hash(request), "config_sha256": config_digest,
                      "retrieval_input_sha256": row["input_sha256"], "elapsed_ms": elapsed,
                      **result}
            if record["selection"] is not None and record["selection"] not in record["generated_candidates"]:
                raise ValueError("selector returned an unknown candidate")
            counts[record["status"]] += 1
            destination.write(canonical(record) + "\n")
            destination.flush()
    if not seen:
        raise ValueError("no rows matched requested split")
    partial.rename(output)
    report = {"schema": "vons.mind2web-prediction/v1", "config": config,
              "config_sha256": config_digest, "source_sha256": source_digest,
              "predictions_sha256": file_hash(output), "runner_sha256": file_hash(Path(__file__)),
              "rows": len(seen), "counts": dict(counts), "selection_reused_across_k": False,
              "timing_scope": "tokenization/inference/decoding, not a controlled benchmark",
              "scope": "action target selection; not browser task success"}
    output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["test_task", "test_website", "test_domain"], required=True)
    parser.add_argument("--k", type=int, choices=KS, required=True)
    parser.add_argument("--backend", choices=["direct", "lexical_top1"], default="direct")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.backend == "direct":
        if args.manifest is None or not args.manifest_sha256:
            parser.error("Direct requires a shared manifest and externally pinned --manifest-sha256")
        select = DirectSelector(args.manifest, args.manifest_sha256)
        provenance = select.provenance
    else:
        def select(request: dict[str, Any]) -> dict[str, Any]:
            candidates = request["candidates"]
            return {"selection": candidates[0]["id"] if candidates else None,
                    "status": "ok" if candidates else "abstain", "reason": None}
        provenance = {"backend": "lexical_top1", "input_budget": "unrestricted retrieval baseline"}
    report = predict_file(args.retriever, args.output, split=args.split, k=args.k,
                          select=select, provenance=provenance)
    print(json.dumps({"rows": report["rows"], "counts": report["counts"]}))


if __name__ == "__main__":
    main()
