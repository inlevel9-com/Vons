"""Recover an exact historical bundle manifest from a Codex JSONL session log.

This is a provenance utility for preserving a manifest whose digest is already
embedded in a raw browser benchmark. It does not change the current bundle
manifest and refuses to write unless the recovered bytes match the expected
digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def recover(
    session_log: Path,
    *,
    ordinal: int,
    source_hash_in_log: str,
    replacement_source_hash: str,
    expected_manifest_hash: str,
    output: Path,
    bundle_root: Path | None = None,
) -> dict[str, Any]:
    """Recover one command output and validate its exact historical digest."""

    stdout: str | None = None
    with session_log.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record.get("ordinal") != ordinal:
                continue
            payload = record.get("payload", {})
            item = payload.get("item", {}) if isinstance(payload, dict) else {}
            candidate = item.get("stdout") if isinstance(item, dict) else None
            if isinstance(candidate, str):
                stdout = candidate
                break
            raise ValueError(f"ordinal {ordinal} has no command stdout (line {line_number})")
    if stdout is None:
        raise ValueError(f"ordinal {ordinal} was not found in {session_log}")

    source_count = stdout.count(source_hash_in_log)
    if source_count != 1:
        raise ValueError(f"expected one source hash in the session output, found {source_count}")
    raw = stdout.replace(source_hash_in_log, replacement_source_hash, 1).encode("utf-8")
    payload = json.loads(raw.decode("utf-8"))
    source_snapshot = payload.get("source", {}).get("source_snapshot", {})
    if source_snapshot.get("sha256") != replacement_source_hash:
        raise ValueError("historical manifest does not contain the replacement source hash")

    # The session output contains the pre-change source hash. The caller passes
    # the hash recorded in the browser run separately through the same option.
    actual_hash = _sha256_bytes(raw)
    if actual_hash != expected_manifest_hash:
        raise ValueError(f"recovered manifest hash {actual_hash} != expected {expected_manifest_hash}")
    if payload.get("summary", {}).get("manifest_bytes") != len(raw):
        raise ValueError("historical manifest summary does not match recovered byte count")
    bundle_files_match: bool | None = None
    if bundle_root is not None:
        bundle_files_match = True
        for item in payload.get("files", []):
            path = bundle_root / str(item["path"])
            if not path.is_file() or path.stat().st_size != item.get("bytes"):
                bundle_files_match = False
                break
            if _sha256_bytes(path.read_bytes()) != item.get("sha256"):
                bundle_files_match = False
                break
        if not bundle_files_match:
            raise ValueError("historical manifest file inventory does not match the supplied bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    return {
        "output": str(output),
        "bytes": len(raw),
        "sha256": actual_hash,
        "source_snapshot_sha256": replacement_source_hash,
        "session_ordinal": ordinal,
        "bundle_files_match": bundle_files_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-log", type=Path, required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--source-hash-in-log", required=True)
    parser.add_argument("--replacement-source-hash", required=True)
    parser.add_argument("--expected-manifest-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(recover(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
