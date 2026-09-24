from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "prepare_public_release", Path(__file__).parents[1] / "tools/prepare_public_release.py"
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
inspect_file = _MODULE.inspect_file
prepare = _MODULE.prepare


def source_tree(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Source\n")
    (tmp_path / "docs/HUGGING_FACE.md").write_text(
        "# Hub source preview\n[Model card](../docs/MODEL_CARD.md)\n"
    )
    (tmp_path / ".gitignore").write_text("data/\nreleases/\n.env\n")
    spec = {"license_status": "pending", "files": [
        ".gitignore", "README.md", "configs/public-release.json", "docs/HUGGING_FACE.md",
    ]}
    (tmp_path / "configs/public-release.json").write_text(json.dumps(spec))
    return tmp_path


def test_export_omits_private_data_and_replaces_hub_card(tmp_path: Path) -> None:
    root = source_tree(tmp_path)
    (root / "data").mkdir()
    (root / "data/secret.jsonl").write_text("private row")
    (root / ".env").write_text("SECRET=private")
    output = root / "releases/hub"
    manifest = prepare(root, output, "huggingface")
    assert (output / "README.md").read_text() == (
        "# Hub source preview\n[Model card](docs/MODEL_CARD.md)\n"
    )
    assert (output / "docs/HUGGING_FACE.md").read_text() == (
        root / "docs/HUGGING_FACE.md"
    ).read_text()
    assert not (output / "data").exists()
    assert not (output / ".env").exists()
    for entry in manifest["files"]:
        assert hashlib.sha256((output / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    assert (root / "data/secret.jsonl").read_text() == "private row"
    with pytest.raises(ValueError, match="overwrite"):
        prepare(root, output, "huggingface")


def test_unreviewed_and_force_tracked_files_block_export(tmp_path: Path) -> None:
    root = source_tree(tmp_path)
    (root / ".env").write_text("PRIVATE=value")
    subprocess.run(["git", "add", "-f", ".env"], cwd=root, check=True)
    with pytest.raises(ValueError, match="unexpected_public_paths"):
        prepare(root)


def test_new_public_file_requires_review(tmp_path: Path) -> None:
    root = source_tree(tmp_path)
    (root / "notes.md").write_text("unreviewed")
    with pytest.raises(ValueError, match="unexpected_public_paths"):
        prepare(root)


@pytest.mark.parametrize("name", ["../outside.md", "/outside.md", "data/example.md", ".env"])
def test_unsafe_allowlist_path_is_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="disallowed path"):
        inspect_file(tmp_path, name)


def test_symlink_cannot_escape_export(tmp_path: Path) -> None:
    (tmp_path / "original.md").write_text("private")
    (tmp_path / "link.md").symlink_to(tmp_path / "original.md")
    with pytest.raises(ValueError, match="symlink"):
        inspect_file(tmp_path, "link.md")


def test_secret_finding_does_not_echo_the_value(tmp_path: Path) -> None:
    token = "hf_" + "a" * 30
    (tmp_path / "source.py").write_text(f'TOKEN = "{token}"\n')
    with pytest.raises(ValueError, match="provider-token") as failure:
        inspect_file(tmp_path, "source.py")
    assert token not in str(failure.value)


def test_publication_binary_is_bound_to_reviewed_digest(tmp_path: Path) -> None:
    path = tmp_path / "docs/publications/paper.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"%PDF-example")
    name = path.relative_to(tmp_path).as_posix()
    with pytest.raises(ValueError, match="reviewed digest"):
        inspect_file(tmp_path, name)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert inspect_file(tmp_path, name, digest) == path.read_bytes()
    path.write_bytes(b"%PDF-changed")
    with pytest.raises(ValueError, match="reviewed digest"):
        inspect_file(tmp_path, name, digest)
