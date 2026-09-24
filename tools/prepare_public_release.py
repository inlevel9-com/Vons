"""Audit and export an explicit public source set without uploading anything.

Git ignore rules protect ordinary staging; this separate allowlist also protects
folder-based uploads. Pattern scanning is a heuristic, not a rights review or a
complete secret detector. Findings report paths and categories, never matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

DENIED_PARTS = {
    ".git", ".git_backup", ".venv", "node_modules", "__pycache__", "data",
    "artifacts", "reports", "research", "private", "restricted", "models",
    "weights", "checkpoints", "releases", ".aws", ".ssh", ".cache",
}
TEXT_SUFFIXES = {".py", ".ts", ".mjs", ".md", ".json", ".toml", ".yml", ".cff", ".html"}
TEXT_NAMES = {".gitignore", ".gitattributes", "LICENSE", "NOTICE"}
PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "provider-token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
        r"|hf_[A-Za-z0-9]{25,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,})\b"
    ),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "personal-home-path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "private-writing-session": re.compile(
        r"localhost[:]8766|/frames/[0-9a-f-]{36}|proj_[0-9a-f]{12}"
    ),
    "credential-url": re.compile(r"https?://[^\s/:@]+:[^\s/@]+@"),
}


def git_files(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "ls-files", "-z", "--cached", "--others",
         "--exclude-standard"], cwd=root, check=True, capture_output=True,
    )
    return {p for p in result.stdout.decode().split("\0") if p}


def inspect_file(root: Path, name: str, reviewed_binary: str | None = None) -> bytes:
    relative = PurePosixPath(name)
    if (relative.is_absolute() or ".." in relative.parts or "\\" in name
            or relative.as_posix() != name or not relative.parts
            or DENIED_PARTS.intersection(relative.parts)
            or any(p.startswith(".env") for p in relative.parts)):
        raise ValueError(f"disallowed path: {name}")
    source = root / name
    if any((root / Path(*relative.parts[:i])).is_symlink()
           for i in range(1, len(relative.parts) + 1)):
        raise ValueError(f"symlink is not public source: {name}")
    if not source.is_file() or not source.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"missing or external source: {name}")
    if source.stat().st_size > 20 * 1024 * 1024:
        raise ValueError(f"oversized source: {name}")
    content = source.read_bytes()
    binary_location = (
        relative.suffix == ".pdf" and relative.parts[:2] == ("docs", "publications")
    ) or (relative.suffix == ".png" and relative.parts[:2] == ("docs", "assets"))
    if binary_location:
        signature = b"%PDF-" if relative.suffix == ".pdf" else b"\x89PNG\r\n\x1a\n"
        if (not content.startswith(signature) or not reviewed_binary
                or hashlib.sha256(content).hexdigest() != reviewed_binary):
            raise ValueError(f"binary needs an exact reviewed digest: {name}")
        return content
    if relative.suffix not in TEXT_SUFFIXES and relative.name not in TEXT_NAMES:
        raise ValueError(f"unreviewed file type: {name}")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"non-text source: {name}") from exc
    if "\0" in text:
        raise ValueError(f"binary content in text file: {name}")
    for category, pattern in PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"{category}: {name}")
    return content


def audit(root: Path) -> tuple[dict, dict[str, bytes]]:
    spec = json.loads((root / "configs/public-release.json").read_text(encoding="utf-8"))
    names = spec["files"]
    if (not isinstance(names, list) or not names
            or not all(isinstance(n, str) for n in names) or len(set(names)) != len(names)):
        raise ValueError("release files must be a nonempty unique path list")
    candidates = git_files(root)
    if candidates != set(names):
        raise ValueError(json.dumps({"unexpected_public_paths": sorted(candidates - set(names)),
                                     "missing_or_ignored_paths": sorted(set(names) - candidates)}))
    content = {name: inspect_file(root, name, spec.get("reviewed_binaries", {}).get(name))
               for name in sorted(names)}
    return spec, content


def prepare(root: Path, output: Path | None = None, target: str = "github") -> dict:
    spec, content = audit(root)
    if target not in {"github", "huggingface"}:
        raise ValueError("unknown release target")
    if target == "huggingface":
        content["README.md"] = content["docs/HUGGING_FACE.md"].replace(b"](../", b"](")
    manifest = {
        "schema": "vons.public-source/v1", "target": target,
        "status": "prepared_local_only", "license_status": spec["license_status"],
        "weights_included": False, "datasets_included": False,
        "files": [{"path": name, "bytes": len(value),
                   "sha256": hashlib.sha256(value).hexdigest()}
                  for name, value in sorted(content.items())],
    }
    if output is not None:
        output = output.resolve()
        if output.exists():
            raise ValueError("refusing to overwrite a release directory")
        if output == root.resolve() or output in root.resolve().parents:
            raise ValueError("output must not replace the source tree")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            stage = Path(temporary) / "release"
            stage.mkdir()
            for name, value in content.items():
                destination = stage / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(value)
            (stage / "RELEASE_MANIFEST.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            stage.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target", choices=["github", "huggingface"], default="github")
    parser.add_argument("--output", type=Path, help="Create a new local export directory")
    args = parser.parse_args()
    try:
        result = prepare(args.root.resolve(), args.output, args.target)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Public release audit failed: {exc}\n")
    print(json.dumps({"status": result["status"], "files": len(result["files"]),
                      "bytes": sum(f["bytes"] for f in result["files"]),
                      "license_status": result["license_status"], "target": args.target}, indent=2))


if __name__ == "__main__":
    main()
