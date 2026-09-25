"""Build a checked run index for a complete Mind2Web prediction matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SPLITS = ("test_task", "test_website", "test_domain")
KS = (5, 10, 20, 32)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def build_index(prediction_dir: Path, retriever: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    runner_hashes: set[str] = set()
    source_hashes: set[str] = set()
    for split in SPLITS:
        for k in KS:
            prediction = prediction_dir / f"{split}-k{k}.jsonl"
            manifest_path = prediction.with_suffix(".manifest.json")
            if not prediction.is_file() or not manifest_path.is_file():
                raise ValueError(f"missing prediction cell: {split}/k={k}")
            manifest = _object(manifest_path)
            config = manifest.get("config")
            if not isinstance(config, dict) or config.get("split") != split or config.get("k") != k:
                raise ValueError(f"prediction manifest identity mismatch: {split}/k={k}")
            digest = file_hash(prediction)
            if manifest.get("predictions_sha256") != digest:
                raise ValueError(f"prediction hash mismatch: {split}/k={k}")
            rows = manifest.get("rows")
            counts = manifest.get("counts")
            if (
                isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows <= 0
                or not isinstance(counts, dict)
                or sum(counts.values()) != rows
            ):
                raise ValueError(f"invalid row counts: {split}/k={k}")
            runner_hash = manifest.get("runner_sha256")
            source_hash = manifest.get("source_sha256")
            if not isinstance(runner_hash, str) or re.fullmatch(r"[0-9a-f]{64}", runner_hash) is None:
                raise ValueError("invalid prediction runner hash")
            if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
                raise ValueError("invalid retrieval source hash")
            runner_hashes.add(runner_hash)
            source_hashes.add(source_hash)
            runs.append(
                {
                    "split": split,
                    "k": k,
                    "rows": rows,
                    "counts": counts,
                    "predictions_sha256": digest,
                }
            )
    retriever_hash = file_hash(retriever)
    if source_hashes != {retriever_hash}:
        raise ValueError("prediction manifests do not match the retriever source")
    if len(runner_hashes) != 1:
        raise ValueError("prediction matrix uses multiple runner versions")
    return {
        "schema": "vons.mind2web-prediction-index/v1",
        "runs": runs,
        "source_sha256": retriever_hash,
        "runner_sha256": next(iter(runner_hashes)),
        "scope": (
            "complete exact-k Direct action-target matrix with label-free context compression; "
            "no test-set adaptation or browser task execution"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--retriever", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite prediction index")
    index = build_index(args.prediction_dir, args.retriever)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(index, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"runs": len(index["runs"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
