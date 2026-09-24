"""Create a field-level claim ledger from an explicit frozen JSON specification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def resolve(document: Any, pointer: str) -> Any:
    for part in pointer.split("/")[1:]:
        key = part.replace("~1", "/").replace("~0", "~")
        document = document[int(key)] if isinstance(document, list) else document[key]
    return document


def build(spec: Path, output: Path) -> int:
    root = Path(__file__).resolve().parents[1]
    records = []
    for item in json.loads(spec.read_text())["claims"]:
        source = root / item["source_path"]
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["source_sha256"]:
            raise ValueError(f"claim source changed: {item['source_path']}")
        value = resolve(json.loads(raw), item["json_pointer"])
        operation = item.get("operation", "identity")
        if operation == "length":
            value = len(value)
        elif operation != "identity":
            raise ValueError("unsupported ledger operation")
        records.append({**item, "operation": operation,
                        "value_json": json.dumps(value, allow_nan=False, ensure_ascii=False)})
    if len({item["claim_id"] for item in records}) != len(records):
        raise ValueError("claim ids must be unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["claim_id", "source_path", "source_sha256",
                                                   "json_pointer", "operation", "value_json", "scope"])
        writer.writeheader()
        writer.writerows(records)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"claims": build(args.spec, args.output), "output": str(args.output)}))
