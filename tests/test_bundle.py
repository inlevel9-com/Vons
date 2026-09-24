from __future__ import annotations

import json
from pathlib import Path

import pytest

from vons.bundle import (
    BundleManifestError,
    _snapshot_hash,
    build_bundle_manifest,
    verify_bundle_manifest,
)


def _write_external_model(root: Path) -> tuple[Path, Path]:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    root.mkdir(parents=True, exist_ok=True)
    initializer = numpy_helper.from_array(__import__("numpy").ones(2048, dtype="float32"), name="weight")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
        [initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    graph_path = root / "model.onnx"
    onnx.save_model(model, graph_path, save_as_external_data=True, all_tensors_to_one_file=True, location="weights.onnx.data")
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return graph_path, tokenizer


def test_manifest_inventories_external_data_and_verifies_bytes(tmp_path: Path) -> None:
    graph, tokenizer = _write_external_model(tmp_path)
    output = tmp_path / "bundle-manifest.v1.json"
    build_bundle_manifest(graph, tokenizer, output)
    result = verify_bundle_manifest(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result["pass"] is True
    assert result["external_data_files"] == 1
    assert {item["role"] for item in payload["files"]} >= {"model_graph", "model_external_data", "tokenizer"}
    assert payload["summary"]["model_asset_bytes"] == sum(
        item["bytes"] for item in payload["files"] if item["model_asset"]
    )

    external = tmp_path / "weights.onnx.data"
    external.write_bytes(external.read_bytes() + b"changed")
    with pytest.raises(BundleManifestError, match="changed"):
        verify_bundle_manifest(output)


def test_manifest_rejects_missing_external_data_and_empty_tokenizer(tmp_path: Path) -> None:
    graph, tokenizer = _write_external_model(tmp_path)
    (tmp_path / "weights.onnx.data").unlink()
    with pytest.raises(BundleManifestError, match="external-data file does not exist"):
        build_bundle_manifest(graph, tokenizer, tmp_path / "manifest.json")

    graph, tokenizer = _write_external_model(tmp_path / "second")
    for child in tokenizer.iterdir():
        child.unlink()
    with pytest.raises(BundleManifestError, match="contains no regular files"):
        build_bundle_manifest(graph, tokenizer, tokenizer.parent / "manifest.json")


def test_manifest_rejects_external_data_traversal(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    initializer = helper.make_tensor("weight", TensorProto.FLOAT, [1], [1.0])
    initializer.data_location = TensorProto.EXTERNAL
    initializer.external_data.add(key="location", value="../outside.onnx.data")
    graph = helper.make_graph([], "test", [], [], [initializer])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    graph_path = tmp_path / "model.onnx"
    onnx.save_model(model, graph_path)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(BundleManifestError, match="traversal"):
        build_bundle_manifest(graph_path, tokenizer, tmp_path / "manifest.json")


def test_manifest_rejects_symlinked_tokenizer(tmp_path: Path) -> None:
    graph, tokenizer = _write_external_model(tmp_path)
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    (tokenizer / "escape.txt").symlink_to(target)

    with pytest.raises(BundleManifestError, match="symlink"):
        build_bundle_manifest(graph, tokenizer, tmp_path / "manifest.json")


def test_source_snapshot_ignores_generated_python_caches(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n", encoding="utf-8")
    ignored = source / "__pycache__"
    ignored.mkdir()
    (ignored / "module.cpython-313.pyc").write_bytes(b"first")
    first = _snapshot_hash(source)
    (ignored / "module.cpython-313.pyc").write_bytes(b"second")
    (source / ".pytest_cache").mkdir()
    (source / ".pytest_cache" / "state").write_bytes(b"cache")
    assert _snapshot_hash(source) == first


def test_source_snapshot_can_pin_an_explicit_file_list(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n", encoding="utf-8")
    (source / "config.json").write_text("{}\n", encoding="utf-8")

    first = _snapshot_hash(source, ["module.py"])
    assert first["scope"] == "explicit"
    assert first["paths"] == ["module.py"]
    (source / "config.json").write_text('{"changed": true}\n', encoding="utf-8")
    assert _snapshot_hash(source, ["module.py"]) == first
    (source / "module.py").write_text("value = 2\n", encoding="utf-8")
    assert _snapshot_hash(source, ["module.py"]) != first


def test_source_snapshot_rejects_explicit_paths_outside_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(BundleManifestError, match="escapes"):
        _snapshot_hash(source, [outside])
