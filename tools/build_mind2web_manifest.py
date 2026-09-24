"""Build a non-public-path provenance manifest for an approved Mind2Web split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _file_digest(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": byte_count}


def build_manifest(
    *,
    variant: str,
    source_revision: str,
    permitted_use: str,
    approval_record: str,
    preprocessing_revision: str,
    retriever_revision: str,
    splits: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    if not variant or not source_revision or not approval_record:
        raise ValueError("variant, source_revision and approval_record are required")
    if permitted_use not in {"evaluation_only", "evaluation_and_training"}:
        raise ValueError("permitted_use must be evaluation_only or evaluation_and_training")
    if not splits:
        raise ValueError("at least one split is required")
    names = [name for name, _ in splits]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("split names must be non-empty and unique")

    split_records: list[dict[str, int | str]] = []
    for name, path in splits:
        if not path.is_file():
            raise FileNotFoundError(path)
        split_records.append({"name": name, **_file_digest(path)})
    return {
        "schema": "vons.mind2web-provenance/v1",
        "variant": variant,
        "source_revision": source_revision,
        "permitted_use": permitted_use,
        "approval_record": approval_record,
        "preprocessing_revision": preprocessing_revision,
        "retriever_revision": retriever_revision,
        "splits": split_records,
        "raw_data_paths": "omitted from public artifacts",
        "scope": "Mind2Web candidate recall and conditional selection only",
    }


def _split_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("split must be NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--permitted-use", required=True, choices=["evaluation_only", "evaluation_and_training"])
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--preprocessing-revision", required=True)
    parser.add_argument("--retriever-revision", required=True)
    parser.add_argument("--split", action="append", type=_split_argument, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(
        variant=args.variant,
        source_revision=args.source_revision,
        permitted_use=args.permitted_use,
        approval_record=args.approval_record,
        preprocessing_revision=args.preprocessing_revision,
        retriever_revision=args.retriever_revision,
        splits=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
