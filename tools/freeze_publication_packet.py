"""Freeze an explicit evidence/source allowlist for both manuscript authors.

The packet contains copies of reports and source, never model weights or raw
restricted test data. Its manifest authenticates copies relative to the digest
delivered separately to the authors; it does not certify experimental quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(spec_path: Path, output: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(spec_path.read_text())
    if output.exists() or output.with_suffix(".zip").exists():
        raise ValueError("refusing to overwrite a frozen packet")
    inputs = spec["inputs"]
    if not inputs or len({item["path"] for item in inputs}) != len(inputs):
        raise ValueError("packet requires a unique explicit input list")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "packet"
        stage.mkdir()
        entries = []
        for item in inputs:
            relative = Path(item["path"])
            source = root / relative
            if (relative.is_absolute() or ".." in relative.parts
                    or not source.resolve().is_relative_to(root)
                    or relative.parts[:2] == ("data", "restricted")
                    or relative.suffix in {".pt", ".onnx", ".data", ".safetensors", ".pyc"}):
                raise ValueError(f"not an allowed evidence path: {relative}")
            digest = sha256(source)
            if item.get("sha256") != digest:
                raise ValueError(f"input changed since the specification: {relative}")
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256(target) != digest:
                raise ValueError(f"copy mismatch: {relative}")
            entries.append({"path": relative.as_posix(), "role": item["role"],
                            "bytes": target.stat().st_size, "sha256": digest})
        packet = {"schema": "vons.publication-evidence/v2", "project": "Vons",
                  "git_revision": None, "git_revision_reason": "workspace has no Git repository",
                  "spec_sha256": sha256(spec_path), "context": spec["context"], "files": entries}
        (stage / "packet.json").write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
        shutil.copyfile(spec_path, stage / "input-spec.json")
        for entry in entries:
            if sha256(root / entry["path"]) != entry["sha256"]:
                raise ValueError("source changed during freeze")
        stage.rename(output)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                entry = zipfile.ZipInfo(path.relative_to(output).as_posix(), (2026, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(entry, path.read_bytes())
    return {"directory": str(output), "archive": str(archive), "archive_sha256": sha256(archive),
            "packet_sha256": sha256(output / "packet.json"), "files": len(entries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.spec, args.output), indent=2))
