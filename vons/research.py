"""Build evidence packets for Claude Science and public research artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_evidence(root: str | Path) -> dict[str, Any]:
    directory = Path(root)
    files = []
    if directory.exists():
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.stat().st_size < 25 * 1024 * 1024:
                files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(), "git_revision": revision, "evidence_root": str(directory), "files": files}


def build_research_packet(*, results_dir: str | Path, output: str | Path, project: str = "Vons") -> Path:
    packet = {
        "project": project,
        "purpose": "Prepare evidence for Claude Science paper and Tech Report drafting.",
        "paper_title": "Vons: Compact Decision Models for Frontier-Agent Workflows",
        "documents": ["paper", "tech_report"],
        "claim_policy": {"measured": "Only values in result JSON and linked raw evidence.", "unmeasured": "Keep as TBD; never infer from design targets.", "confidence": "Do not treat model self-reported confidence as probability."},
        "evidence": collect_evidence(results_dir),
        "claude_science_prompt": "Draft an evidence-bound English document for Vons. Separate measured, simulated, design-target, and unavailable values. Do not invent benchmark results and list unresolved limitations.",
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    return destination
