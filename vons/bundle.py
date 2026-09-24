"""Build and verify content-addressed manifests for deployable ONNX bundles.

The exporter writes a small legacy manifest with semantic metadata, but that
manifest does not prove which bytes a browser will load.  This module treats
the ONNX graph, its external-data references, tokenizer files, and explicitly
selected runtime assets as a single bundle contract.  Source checkpoints and
other files left in an experiment directory are inventoried but excluded from
the deployable byte total unless explicitly selected as a runtime asset.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA_VERSION = "vons.bundle.manifest/v1"
MODEL_ASSET_ROLES = frozenset({"model_graph", "model_external_data", "tokenizer", "calibration"})
DISTRIBUTION_ROLES = MODEL_ASSET_ROLES | frozenset({"runtime_asset", "license_notice", "config"})


class BundleManifestError(ValueError):
    """Raised when a bundle cannot be safely enumerated or verified."""


def _require_onnx() -> Any:
    try:
        import onnx  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError("bundle manifest generation requires the optional 'onnx' package") from exc
    return onnx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _root_path(root: Path) -> Path:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise BundleManifestError(f"bundle root is not a directory: {root}")
    return resolved


def _safe_relative(root: Path, value: str | Path, *, label: str) -> Path:
    """Resolve a bundle-relative path and reject traversal and symlinks.

    ONNX external-data locations are specification strings, not arbitrary local
    paths.  Rejecting both POSIX and Windows absolute forms avoids accepting a
    Windows drive path when a manifest is generated on another platform.
    """

    raw = str(value).replace("\\", "/")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise BundleManifestError(f"{label} must be relative to the bundle: {value!r}")
    if any(part == ".." for part in posix.parts):
        raise BundleManifestError(f"{label} contains traversal: {value!r}")
    raw_candidate = root / Path(*posix.parts)
    if raw_candidate.is_symlink():
        raise BundleManifestError(f"{label} may not be a symlink: {value!r}")
    candidate = raw_candidate.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BundleManifestError(f"{label} escapes the bundle root: {value!r}") from exc
    return candidate


def _bundle_input(root: Path, value: str | Path, *, label: str) -> Path:
    """Accept a caller's absolute path only when it is inside the bundle."""

    candidate = Path(value)
    if candidate.is_absolute() or PureWindowsPath(str(value)).drive:
        try:
            relative = candidate.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise BundleManifestError(f"{label} is outside the bundle root: {value!r}") from exc
        return _safe_relative(root, relative, label=label)
    caller_relative = (Path.cwd() / candidate).resolve(strict=False)
    if caller_relative.exists():
        try:
            relative = caller_relative.relative_to(root)
        except ValueError as exc:
            raise BundleManifestError(f"{label} is outside the bundle root: {value!r}") from exc
        return _safe_relative(root, relative, label=label)
    return _safe_relative(root, value, label=label)


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(root).as_posix()
    except ValueError as exc:
        raise BundleManifestError(f"path is outside bundle root: {path}") from exc


def _regular_files(root: Path, directory: Path) -> list[Path]:
    if directory.is_symlink():
        raise BundleManifestError(f"bundle directory may not be a symlink: {directory}")
    if not directory.is_dir():
        raise BundleManifestError(f"expected directory: {directory}")
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise BundleManifestError(f"bundle file may not be a symlink: {path}")
        if path.is_file():
            _safe_relative(root, path.relative_to(root), label="bundle file")
            files.append(path)
    return files


def _external_locations(model: Any, onnx: Any) -> set[str]:
    """Return external-data locations from initializers and nested subgraphs."""

    locations: set[str] = set()

    def visit_tensor(tensor: Any) -> None:
        for item in tensor.external_data:
            if item.key == "location":
                if not item.value:
                    raise BundleManifestError("ONNX external_data has an empty location")
                locations.add(item.value)

    def visit_graph(graph: Any) -> None:
        for tensor in graph.initializer:
            visit_tensor(tensor)
        for sparse in getattr(graph, "sparse_initializer", []):
            visit_tensor(sparse.values)
            visit_tensor(sparse.indices)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        visit_graph(nested)
                elif attribute.type == onnx.AttributeProto.TENSOR:
                    visit_tensor(attribute.t)
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    for tensor in attribute.tensors:
                        visit_tensor(tensor)
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSOR:
                    visit_tensor(attribute.sparse_tensor.values)
                    visit_tensor(attribute.sparse_tensor.indices)
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSORS:
                    for sparse in attribute.sparse_tensors:
                        visit_tensor(sparse.values)
                        visit_tensor(sparse.indices)

    visit_graph(model.graph)
    return locations


def _shape(value_info: Any) -> list[int | str | None] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    result: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append(None)
    return result


def _value_infos(values: Iterable[Any], onnx: Any) -> list[dict[str, Any]]:
    result = []
    for value in values:
        tensor_type = value.type.tensor_type
        result.append(
            {
                "name": value.name,
                "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
                "shape": _shape(value),
            }
        )
    return result


def _graph_metadata(model: Any, onnx: Any, external_locations: Sequence[str]) -> dict[str, Any]:
    return {
        "ir_version": int(model.ir_version),
        "opset_imports": [
            {"domain": item.domain, "version": int(item.version)} for item in model.opset_import
        ],
        "inputs": _value_infos(model.graph.input, onnx),
        "outputs": _value_infos(model.graph.output, onnx),
        "external_data_locations": sorted(external_locations),
        "node_count": len(model.graph.node),
    }


def _git_revision(root: Path) -> dict[str, str | None]:
    try:
        value = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        value = ""
    return {"value": value or None, "reason": None if value else "not a git checkout or revision unavailable"}


def _snapshot_hash(
    root: Path | None,
    paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    if root is None:
        return {
            "sha256": None,
            "reason": "source root was not supplied",
            "scope": None,
            "paths": [],
        }
    source_root = root.resolve(strict=True)
    if not source_root.is_dir():
        raise BundleManifestError(f"source root is not a directory: {root}")
    digest = hashlib.sha256()
    ignored_directories = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
    files: list[tuple[Path, Path]] = []
    if paths:
        for supplied in paths:
            candidate = Path(supplied)
            if not candidate.is_absolute():
                candidate = source_root / candidate
            if candidate.is_symlink():
                raise BundleManifestError(f"source snapshot path may not be a symlink: {supplied}")
            resolved = candidate.resolve(strict=True)
            try:
                relative_path = resolved.relative_to(source_root)
            except ValueError as exc:
                raise BundleManifestError(
                    f"source snapshot path escapes the source root: {supplied}"
                ) from exc
            if not resolved.is_file():
                raise BundleManifestError(f"source snapshot path is not a file: {supplied}")
            if any(part in ignored_directories for part in relative_path.parts) or resolved.suffix == ".pyc":
                raise BundleManifestError(f"source snapshot path is ignored: {supplied}")
            files.append((relative_path, resolved))
        if len({relative for relative, _ in files}) != len(files):
            raise BundleManifestError("source snapshot paths must be unique")
        scope = "explicit"
    else:
        for path in source_root.rglob("*"):
            if not path.is_file() or path.is_symlink() or path.suffix == ".pyc":
                continue
            relative_path = path.relative_to(source_root)
            if any(part in ignored_directories for part in relative_path.parts):
                continue
            files.append((relative_path, path))
        scope = "all_eligible_files"
    for relative_path, path in sorted(files):
        relative = relative_path.as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "sha256": digest.hexdigest(),
        "reason": None,
        "scope": scope,
        "paths": [relative.as_posix() for relative, _ in sorted(files)],
    }


def _legacy_metadata(root: Path, output: Path) -> dict[str, Any]:
    candidate = root / "bundle-manifest.json"
    if candidate == output or not candidate.is_file():
        return {}
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _file_record(root: Path, path: Path, role: str, *, runtime_loaded: bool, model_asset: bool) -> dict[str, Any]:
    relative = _relative(root, path)
    size = path.stat().st_size
    return {
        "path": relative,
        "role": role,
        "bytes": size,
        "sha256": _sha256(path),
        "compression": {"encoding": "identity", "measured": False, "bytes": size},
        "runtime_loaded": runtime_loaded,
        "model_asset": model_asset,
    }


def _stable_manifest(manifest: dict[str, Any], output: Path, entries: Sequence[dict[str, Any]]) -> None:
    """Write the manifest until its reported own size reaches a fixed point."""

    payload_bytes = sum(int(item["bytes"]) for item in entries if item["model_asset"])
    distribution_bytes = sum(int(item["bytes"]) for item in entries if item["role"] in DISTRIBUTION_ROLES)
    manifest["summary"]["model_asset_bytes"] = payload_bytes
    manifest["summary"]["manifest_bytes"] = 0
    manifest["summary"]["complete_download_bytes"] = distribution_bytes
    output.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(8):
        output.write_bytes(_json_bytes(manifest))
        manifest_bytes = output.stat().st_size
        complete = distribution_bytes + manifest_bytes
        if (
            manifest["summary"]["manifest_bytes"] == manifest_bytes
            and manifest["summary"]["complete_download_bytes"] == complete
        ):
            return
        manifest["summary"]["manifest_bytes"] = manifest_bytes
        manifest["summary"]["complete_download_bytes"] = complete
    raise BundleManifestError("manifest size did not converge")


def build_bundle_manifest(
    onnx_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    model_manifest_path: str | Path | None = None,
    calibration_paths: Sequence[str | Path] = (),
    runtime_asset_paths: Sequence[str | Path] = (),
    license_paths: Sequence[str | Path] = (),
    config_paths: Sequence[str | Path] = (),
    source_root: str | Path | None = None,
    source_paths: Sequence[str | Path] = (),
) -> Path:
    """Create a deployable manifest and inventory every regular file in its root."""

    output = Path(output_path).resolve(strict=False)
    root = _root_path(output.parent)
    tokenizer = _bundle_input(root, tokenizer_path, label="tokenizer path")
    graph_path = _bundle_input(root, onnx_path, label="ONNX path")
    if not graph_path.is_file():
        raise BundleManifestError(f"ONNX path is not a file: {onnx_path}")
    if output.parent != root:
        raise BundleManifestError("manifest output must be directly inside the bundle root")

    onnx = _require_onnx()
    try:
        model = onnx.load_model(str(graph_path), load_external_data=False)
    except Exception as exc:  # onnx raises several protobuf-specific exceptions
        raise BundleManifestError(f"could not parse ONNX graph {graph_path}: {exc}") from exc
    external_locations = _external_locations(model, onnx)
    external_files: dict[Path, str] = {}
    for location in sorted(external_locations):
        path = _safe_relative(root, location, label="ONNX external-data location")
        if not path.is_file():
            raise BundleManifestError(f"ONNX external-data file does not exist: {location}")
        external_files[path] = location

    paths_by_role: dict[Path, str] = {graph_path: "model_graph"}
    if tokenizer.is_file():
        paths_by_role[tokenizer] = "tokenizer"
    else:
        tokenizer_files = _regular_files(root, tokenizer)
        if not tokenizer_files:
            raise BundleManifestError("tokenizer directory contains no regular files")
        for path in tokenizer_files:
            paths_by_role[path] = "tokenizer"
    for path in external_files:
        paths_by_role[path] = "model_external_data"

    explicit_groups = [
        (checkpoint_path, "source_checkpoint"),
        (model_manifest_path, "source_manifest"),
    ]
    for values, role in (
        (calibration_paths, "calibration"),
        (runtime_asset_paths, "runtime_asset"),
        (license_paths, "license_notice"),
        (config_paths, "config"),
    ):
        explicit_groups.extend((value, role) for value in values)
    for value, role in explicit_groups:
        if value is None:
            continue
        path = _bundle_input(root, value, label=f"{role} path")
        if not path.is_file():
            raise BundleManifestError(f"{role} path is not a file: {value}")
        paths_by_role[path] = role

    if checkpoint_path is None:
        candidate = root / "model.pt"
        if candidate.is_file():
            paths_by_role.setdefault(candidate, "source_checkpoint")
    if model_manifest_path is None:
        candidate = root / "model-manifest.json"
        if candidate.is_file():
            paths_by_role.setdefault(candidate, "source_manifest")

    all_files = _regular_files(root, root)
    records: list[dict[str, Any]] = []
    for path in all_files:
        if path == output:
            continue
        role = paths_by_role.get(path)
        if role is None:
            if path.suffix == ".onnx":
                role = "unselected_model_artifact"
            elif path.suffix == ".pt":
                role = "source_checkpoint"
            elif path.name.endswith(".onnx.data"):
                role = "unreferenced_model_artifact"
            elif path.name.endswith("manifest.json"):
                role = "source_manifest"
            else:
                role = "extra"
        records.append(
            _file_record(
                root,
                path,
                role,
                runtime_loaded=role in {"model_graph", "model_external_data", "tokenizer", "runtime_asset"},
                model_asset=role in MODEL_ASSET_ROLES,
            )
        )
    records.sort(key=lambda item: item["path"])
    if not any(item["role"] == "model_graph" for item in records):
        raise BundleManifestError("manifest inventory does not contain the selected ONNX graph")

    legacy = _legacy_metadata(root, output)
    metadata = {key: value for key, value in legacy.items() if key not in {"onnx", "tokenizer", "checkpoint"}}
    metadata["graph"] = _graph_metadata(model, onnx, sorted(external_locations))
    metadata["checkpoint_sha256"] = next(
        (item["sha256"] for item in records if item["role"] == "source_checkpoint"), None
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest": {
            "path": output.name,
            "sha256": None,
            "self_hash_excluded": True,
        },
        "bundle_root": ".",
        "metadata": metadata,
        "source": {
            "git_revision": _git_revision(root),
            "source_snapshot": _snapshot_hash(
                Path(source_root) if source_root is not None else None,
                source_paths or None,
            ),
            "builder": {"python": sys.version.split()[0], "platform": platform.platform()},
        },
        "files": records,
        "summary": {
            "model_asset_bytes": 0,
            "manifest_bytes": 0,
            "complete_download_bytes": 0,
            "model_asset_limit_bytes": 67_108_864,
            "within_model_asset_limit": False,
            "runtime_loaded_file_count": sum(item["runtime_loaded"] for item in records),
        },
    }
    _stable_manifest(manifest, output, records)
    manifest["summary"]["within_model_asset_limit"] = (
        manifest["summary"]["model_asset_bytes"] <= manifest["summary"]["model_asset_limit_bytes"]
    )
    output.write_bytes(_json_bytes(manifest))
    # Updating the boolean can alter the byte count; converge once more without
    # including the manifest in its own digest or model-asset total.
    _stable_manifest(manifest, output, records)
    return output


def verify_bundle_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Verify every recorded file, graph reference, and byte summary."""

    manifest_file = Path(manifest_path).resolve(strict=True)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise BundleManifestError(f"unsupported bundle manifest schema: {payload.get('schema_version')!r}")
    root = manifest_file.parent
    manifest_info = payload.get("manifest", {})
    if manifest_info.get("path") != manifest_file.name:
        raise BundleManifestError("manifest path does not match the verification target")
    records = payload.get("files")
    if not isinstance(records, list) or not records:
        raise BundleManifestError("manifest has no file inventory")
    seen: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            raise BundleManifestError("manifest file entry is not an object")
        relative = item.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise BundleManifestError(f"invalid or duplicate manifest path: {relative!r}")
        seen.add(relative)
        path = _safe_relative(root, relative, label="manifest file path")
        if not path.is_file():
            raise BundleManifestError(f"manifest file is missing: {relative}")
        actual_bytes = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_bytes != item.get("bytes") or actual_hash != item.get("sha256"):
            raise BundleManifestError(f"manifest file changed: {relative}")

    graph_item = next((item for item in records if item.get("role") == "model_graph"), None)
    if graph_item is None:
        raise BundleManifestError("manifest has no model_graph entry")
    onnx = _require_onnx()
    model = onnx.load_model(str(root / graph_item["path"]), load_external_data=False)
    external_locations = _external_locations(model, onnx)
    inventory_paths = {item["path"] for item in records}
    for location in external_locations:
        path = _safe_relative(root, location, label="ONNX external-data location")
        if _relative(root, path) not in inventory_paths:
            raise BundleManifestError(f"external-data file is not in inventory: {location}")

    summary = payload.get("summary", {})
    model_asset_bytes = sum(item["bytes"] for item in records if item.get("model_asset"))
    distribution_bytes = sum(item["bytes"] for item in records if item.get("role") in DISTRIBUTION_ROLES)
    if summary.get("model_asset_bytes") != model_asset_bytes:
        raise BundleManifestError("model_asset_bytes does not match the inventory")
    if summary.get("manifest_bytes") != manifest_file.stat().st_size:
        raise BundleManifestError("manifest_bytes does not match the manifest file")
    if summary.get("complete_download_bytes") != distribution_bytes + manifest_file.stat().st_size:
        raise BundleManifestError("complete_download_bytes does not match the inventory")
    if summary.get("within_model_asset_limit") != (
        model_asset_bytes <= int(summary.get("model_asset_limit_bytes", 67_108_864))
    ):
        raise BundleManifestError("within_model_asset_limit is stale")
    return {
        "manifest": str(manifest_file),
        "files": len(records),
        "external_data_files": len(external_locations),
        "model_asset_bytes": model_asset_bytes,
        "complete_download_bytes": distribution_bytes + manifest_file.stat().st_size,
        "pass": True,
    }


def build_and_verify(*args: Any, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
    """Build a manifest and immediately run the independent verifier."""

    output = build_bundle_manifest(*args, **kwargs)
    return output, verify_bundle_manifest(output)
