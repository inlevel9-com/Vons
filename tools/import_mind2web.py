"""Import the pinned official Mind2Web held-out tasks for local evaluation only.

Outputs are restricted: rows.jsonl contains labels; retriever.jsonl contains
only pre-action inputs and deterministic lexical ranks. Nothing executes a
browser action. Current operation/value and current/future action_reprs are
never model inputs. The history is the official, teacher-forced past history.

Example (large extraction/normalization requires an available compute window)::

    python tools/import_mind2web.py --input-dir data/restricted/mind2web/extracted \
        --split all --output-dir data/restricted/mind2web/normalized \
        --source-manifest data/restricted/mind2web/acquisition-manifest.json \
        --extract-archive data/restricted/mind2web/acquisition/test.zip

Reusing an extracted corpus omits --extract-archive. The source manifest must
be the acquisition record for the pinned ZIP; file hashes are filled during
bounded extraction. Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TextIO
from xml.etree.ElementTree import ParseError, XMLParser

DATASET_REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"
ARCHIVE_SHA256 = "8f5fbe72afab942fe97cdf7fb397e179885d89b5c16862288e9a14bc6d41ca89"
ARCHIVE_BYTES = 567745122
OFFICIAL_TASK_COUNTS = {"test_task": 252, "test_website": 177, "test_domain": 912}
OFFICIAL_FILE_COUNTS = {"test_task": 3, "test_website": 2, "test_domain": 10}
SPLITS = tuple(OFFICIAL_TASK_COUNTS)
MAX_ARCHIVE_BYTES = 1024**3
MAX_MEMBER_BYTES = 1024**3
MAX_EXTRACTED_BYTES = 8 * 1024**3
# Some official tasks contain over 128 Mi characters, mostly unused raw_html.
# A task cannot exceed its verified member's byte bound; do not cap it below that.
MAX_TASK_CHARS = MAX_MEMBER_BYTES
CHUNK_SIZE = 1024**2
# Allowlisted observable DOM attributes exclude annotation flags by construction.
TEXT_ATTRIBUTES = frozenset(
    {
        "alt",
        "aria_description",
        "aria_label",
        "aria_role",
        "class",
        "href",
        "id",
        "input_checked",
        "input_value",
        "label",
        "name",
        "option_selected",
        "placeholder",
        "role",
        "text_value",
        "title",
        "type",
        "value",
    }
)


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return value


def _text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value.strip() and not allow_empty):
        raise ValueError(f"{label} must be a string")
    return value


def _identifier(value: object, label: str) -> str:
    value = _text(value, label)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{label} contains unsupported identifier characters")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _clean(value: str) -> str:
    return " ".join(value.split())


def _attributes(value: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, raw in value.items():
        name = name.lower().replace("-", "_")
        if name in TEXT_ATTRIBUTES and isinstance(raw, (str, bool, int, float)):
            cleaned = _clean(str(raw))
            if cleaned:
                result[name] = cleaned
    return result


class _CandidateDOM:
    """XML parser target for pre-action DOM nodes, including iframe descendants.

    The official cleaned DOM is XML serialization. Browser HTML parsing treats
    iframe contents as raw text and silently loses released candidate nodes.
    """

    def __init__(self, candidate_ids: set[str]) -> None:
        self.candidate_ids = candidate_ids
        self.stack: list[tuple[str, str | None]] = []
        self.nodes: dict[str, tuple[str, dict[str, str], list[str]]] = {}

    def start(self, tag: str, attrs: dict[str, str]) -> None:
        self._boundary()
        node_id = attrs.get("backend_node_id")
        candidate_id = node_id if node_id in self.candidate_ids else None
        if candidate_id is not None:
            if candidate_id in self.nodes:
                raise ValueError("duplicate candidate node id in cleaned DOM")
            self.nodes[candidate_id] = (tag, _attributes(attrs), [])
        self.stack.append((tag, candidate_id))

    def end(self, tag: str) -> None:
        if self.stack[-1][0] != tag:
            raise ValueError("unbalanced cleaned DOM")
        self._boundary()
        self.stack.pop()

    def _boundary(self) -> None:
        for _, candidate_id in self.stack:
            if candidate_id is not None:
                self.nodes[candidate_id][2].append(" ")

    def data(self, data: str) -> None:
        if any(tag in {"script", "style"} for tag, _ in self.stack):
            return
        if data:
            for _, candidate_id in self.stack:
                if candidate_id is not None:
                    self.nodes[candidate_id][2].append(data)

    def close(self) -> None:
        return None


def candidate_universe(action: Mapping[str, object]) -> list[dict[str, str]]:
    """Merge the released pool without exposing annotation membership or order."""
    candidates: dict[str, Mapping[str, object]] = {}
    for field in ("pos_candidates", "neg_candidates"):
        for raw in _array(action.get(field), field):
            candidate = _object(raw, "candidate")
            node_id = _identifier(candidate.get("backend_node_id"), "backend_node_id")
            if node_id in candidates:
                raise ValueError("duplicate candidate id in released pool")
            candidates[node_id] = candidate
    cleaned_html = _text(action.get("cleaned_html"), "cleaned_html", allow_empty=True)
    dom = _CandidateDOM(set(candidates))
    if cleaned_html:
        parser = XMLParser(target=dom)
        try:
            parser.feed(cleaned_html)
            parser.close()
        except ParseError:
            raise ValueError("cleaned DOM is not valid XML serialization") from None
    if set(dom.nodes) != set(candidates):
        raise ValueError("released candidate missing from cleaned DOM")
    result: list[dict[str, str]] = []
    for node_id in sorted(candidates):
        tag, dom_attributes, content = dom.nodes[node_id]
        raw_attributes = candidates[node_id].get("attributes", "{}")
        if isinstance(raw_attributes, str):
            try:
                raw_attributes = json.loads(raw_attributes)
            except json.JSONDecodeError:
                raise ValueError("invalid candidate attributes JSON") from None
        attributes = _attributes(_object(raw_attributes, "candidate attributes"))
        attributes.update(dom_attributes)
        parts = [tag, *(f"{key}={attributes[key]}" for key in sorted(attributes)), "".join(content)]
        result.append({"id": node_id, "text": _clean(" ".join(parts))})
    return result


def lexical_rank(query: str, candidates: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    """Deterministic BM25 (k1=1.2, b=0.75); the interface has no labels."""
    ids = [candidate["id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("lexical candidates must have unique ids")
    query_terms = set(re.findall(r"\w+", query.casefold()))
    terms = [Counter(re.findall(r"\w+", candidate["text"].casefold())) for candidate in candidates]
    lengths = [sum(row.values()) for row in terms]
    average_length = sum(lengths) / len(lengths) if lengths else 1.0
    document_frequency: Counter[str] = Counter()
    for row in terms:
        document_frequency.update(set(row))
    ranked: list[dict[str, object]] = []
    for candidate, counts, length in zip(candidates, terms, lengths):
        score = 0.0
        for term in sorted(query_terms):
            frequency = counts[term]
            if frequency:
                inverse_frequency = math.log1p(
                    (len(candidates) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = frequency + 1.2 * (0.25 + 0.75 * length / average_length)
                score += inverse_frequency * frequency * 2.2 / denominator
        ranked.append({"id": candidate["id"], "text": candidate["text"], "score": score})
    ranked.sort(key=lambda row: (-float(row["score"]), str(row["id"])))
    return [{**row, "rank": rank} for rank, row in enumerate(ranked, start=1)]


def normalize_task(
    task: Mapping[str, object],
    split: str,
) -> Iterator[tuple[dict[str, object], dict[str, object]]]:
    if split not in SPLITS:
        raise ValueError("only official held-out splits are supported")
    if "split" in task and task["split"] != split:
        raise ValueError("task split disagrees with official source file")
    task_id = _identifier(task.get("annotation_id"), "annotation_id")
    website = _text(task.get("website"), "website")
    domain = _text(task.get("domain"), "domain")
    goal = _text(task.get("confirmed_task"), "confirmed_task")
    actions = _array(task.get("actions"), "actions")
    history = _array(task.get("action_reprs"), "action_reprs")
    if not actions or len(history) != len(actions):
        raise ValueError("actions and action_reprs must be nonempty and aligned")
    seen_actions: set[str] = set()
    for index, raw_action in enumerate(actions):
        action = _object(raw_action, "action")
        action_id = _identifier(action.get("action_uid"), "action_uid")
        if action_id in seen_actions:
            raise ValueError("duplicate action id within task")
        seen_actions.add(action_id)
        candidates = candidate_universe(action)
        previous_actions = [_text(item, "past action_repr") for item in history[:index]]
        state = {"goal": goal, "previous_actions": previous_actions}
        query = goal + (
            "\nPrevious actions:\n" + "\n".join(previous_actions) if previous_actions else ""
        )
        ranked = lexical_rank(query, candidates)
        identity = {
            "id": f"{task_id}:{action_id}",
            "task_id": task_id,
            "action_id": action_id,
            "split": split,
            "website": website,
            "domain": domain,
        }
        # Membership is consulted only after label-blind inputs/ranks are complete.
        positive_ids = sorted(
            _identifier(
                _object(item, "positive candidate").get("backend_node_id"), "backend_node_id"
            )
            for item in _array(action.get("pos_candidates"), "pos_candidates")
        )
        labels = {
            **identity,
            "candidate_ids": [item["id"] for item in candidates],
            "positive_ids": positive_ids,
            "no_positive": not positive_ids,
        }
        retrieval = {
            **identity,
            "state": state,
            "query": query,
            "candidates": ranked,
            "generated_candidates": [item["id"] for item in ranked],
            "input_sha256": _json_hash({"state": state, "candidates": candidates}),
        }
        yield labels, retrieval


def iter_tasks(handle: TextIO) -> Iterator[Mapping[str, object]]:
    """Decode one task at a time from an official JSON array with a size bound."""
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False

    def more(read_size: int = CHUNK_SIZE) -> bool:
        nonlocal buffer, eof
        chunk = handle.read(read_size)
        eof = not chunk
        buffer += chunk
        if len(buffer) > MAX_TASK_CHARS:
            raise ValueError("task JSON exceeds configured buffer bound")
        return bool(chunk)

    def strip_space() -> None:
        nonlocal buffer
        buffer = buffer.lstrip()
        while not buffer and not eof:
            more()
            buffer = buffer.lstrip()

    strip_space()
    if not buffer.startswith("["):
        raise ValueError("official task file must be a JSON array")
    buffer = buffer[1:]
    strip_space()
    if buffer.startswith("]"):
        buffer = buffer[1:]
    else:
        while True:
            read_size = CHUNK_SIZE
            while True:
                try:
                    task, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof or not more(read_size):
                        raise ValueError("invalid or truncated task JSON") from None
                    # Re-decoding an incomplete large task every MiB is quadratic.
                    read_size = min(read_size * 2, MAX_TASK_CHARS)
            yield _object(task, "task")
            buffer = buffer[end:]
            strip_space()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                break
            if not buffer.startswith(","):
                raise ValueError("invalid task array separator")
            buffer = buffer[1:]
            strip_space()
    strip_space()
    if buffer:
        raise ValueError("trailing content after task array")


def _member_split(name: str) -> str:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or "\\" in name
        or ".." in path.parts
        or path.as_posix() != name
    ):
        raise ValueError("unsafe or unexpected archive member path")
    split = path.parts[0]
    if split not in SPLITS or not re.fullmatch(rf"{split}_\d+\.json", path.name):
        raise ValueError("archive member has unknown split or filename")
    return split


def _validate_manifest(manifest: Mapping[str, object]) -> Mapping[str, object]:
    archive = _object(manifest.get("archive"), "archive")
    if (
        manifest.get("schema") != "vons.mind2web-acquisition/v1"
        or manifest.get("dataset") != "osunlp/Mind2Web"
        or manifest.get("dataset_revision") != DATASET_REVISION
        or archive.get("sha256") != ARCHIVE_SHA256
        or archive.get("bytes") != ARCHIVE_BYTES
        or archive.get("status") != "downloaded-and-hash-verified"
    ):
        raise ValueError("source manifest does not match the pinned official test archive")
    restrictions = _object(manifest.get("restrictions"), "restrictions")
    if (
        manifest.get("usage") != "evaluation-only"
        or restrictions.get("no_training") is not True
        or restrictions.get("no_unzipped_redistribution") is not True
    ):
        raise ValueError("source manifest must preserve evaluation-only usage restrictions")
    return archive


def validate_archive_members(infos: Sequence[zipfile.ZipInfo]) -> None:
    expected_names = {
        f"{split}/{split}_{index}.json"
        for split, count in OFFICIAL_FILE_COUNTS.items()
        for index in range(count)
    }
    names: set[str] = set()
    total = 0
    for info in infos:
        _member_split(info.filename)
        mode = info.external_attr >> 16
        if info.is_dir() or stat.S_ISLNK(mode):
            raise ValueError("archive contains a directory or symlink")
        if info.filename in names:
            raise ValueError("archive contains a duplicate member")
        names.add(info.filename)
        if not 0 < info.file_size <= MAX_MEMBER_BYTES:
            raise ValueError("archive member exceeds extraction size bound")
        total += info.file_size
    if names != expected_names or total > MAX_EXTRACTED_BYTES:
        raise ValueError("archive member set or total size differs from official bounds")


def extract_official_archive(archive_path: Path, input_dir: Path, manifest_path: Path) -> None:
    manifest = dict(_object(json.loads(manifest_path.read_text()), "source manifest"))
    archive = _validate_manifest(manifest)
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("archive exceeds download bound")
    if (
        archive_path.stat().st_size != archive["bytes"]
        or _sha256(archive_path) != archive["sha256"]
    ):
        raise ValueError("archive failed pinned size/hash verification")
    if input_dir.exists():
        raise ValueError("refusing to overwrite existing extraction directory")
    input_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as source:
        infos = sorted(source.infolist(), key=lambda info: info.filename)
        validate_archive_members(infos)
        if shutil.disk_usage(input_dir.parent).free < sum(info.file_size for info in infos):
            raise ValueError("insufficient free disk for bounded extraction")
        with tempfile.TemporaryDirectory(prefix=".mind2web-extract-", dir=input_dir.parent) as temp:
            staging = Path(temp) / "data"
            staging.mkdir()
            records: list[dict[str, object]] = []
            for info in infos:
                path = staging / info.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with source.open(info, pwd=b"mind2web") as reader, path.open("xb") as writer:
                    while chunk := reader.read(CHUNK_SIZE):
                        size += len(chunk)
                        if size > info.file_size:
                            raise ValueError("decompressed member exceeded declared size")
                        writer.write(chunk)
                        digest.update(chunk)
                if size != info.file_size:
                    raise ValueError("decompressed member size mismatch")
                records.append(
                    {
                        "path": info.filename,
                        "split": _member_split(info.filename),
                        "bytes": size,
                        "sha256": digest.hexdigest(),
                    }
                )
            staging.rename(input_dir)
    manifest["files"] = records
    manifest["extraction"] = {
        "status": "complete",
        "file_count": len(records),
        "uncompressed_bytes": sum(int(row["bytes"]) for row in records),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def import_split(
    input_dir: Path,
    splits: Sequence[str],
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, object]:
    if (
        not splits
        or len(splits) != len(set(splits))
        or any(split not in SPLITS for split in splits)
    ):
        raise ValueError("split selection must contain distinct official held-out split names")
    manifest = _object(json.loads(manifest_path.read_text(encoding="utf-8")), "source manifest")
    _validate_manifest(manifest)
    if _object(manifest.get("extraction"), "extraction").get("status") != "complete":
        raise ValueError("official archive extraction is incomplete")
    records = [_object(row, "file record") for row in _array(manifest.get("files"), "files")]
    paths: set[str] = set()
    for record in records:
        name = _text(record.get("path"), "source path")
        split = _member_split(name)
        if name in paths or record.get("split") != split:
            raise ValueError("duplicate source path or incorrect official split identity")
        paths.add(name)
    expected = {
        f"{split}/{split}_{index}.json"
        for split, count in OFFICIAL_FILE_COUNTS.items()
        for index in range(count)
    }
    if paths != expected:
        raise ValueError("source file manifest does not cover the official held-out corpus")
    selected = sorted(
        (row for row in records if row["split"] in splits), key=lambda row: str(row["path"])
    )
    if output_dir.exists():
        raise ValueError("refusing to overwrite existing output directory")
    input_root = input_dir.resolve()
    for record in selected:
        path = input_dir / str(record["path"])
        if path.is_symlink() or not path.resolve().is_relative_to(input_root):
            raise ValueError("source file resolves outside the input directory")
        if path.stat().st_size != record.get("bytes") or _sha256(path) != record.get("sha256"):
            raise ValueError("source file failed size/hash verification")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    task_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    no_positive_counts: Counter[str] = Counter()
    seen_tasks: set[str] = set()
    seen_actions: set[str] = set()
    with tempfile.TemporaryDirectory(prefix=".mind2web-import-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "data"
        staging.mkdir()
        rows_path = staging / "rows.jsonl"
        retriever_path = staging / "retriever.jsonl"
        with (
            rows_path.open("w", encoding="utf-8") as labels_out,
            retriever_path.open("w", encoding="utf-8") as retrieval_out,
        ):
            for record in selected:
                split = str(record["split"])
                with (input_dir / str(record["path"])).open(encoding="utf-8") as handle:
                    for task in iter_tasks(handle):
                        task_id = _identifier(task.get("annotation_id"), "annotation_id")
                        if task_id in seen_tasks:
                            raise ValueError("duplicate task id across input files or splits")
                        seen_tasks.add(task_id)
                        task_counts[split] += 1
                        for labels, retrieval in normalize_task(task, split):
                            action_id = str(labels["action_id"])
                            if action_id in seen_actions:
                                raise ValueError("duplicate action id across tasks or splits")
                            seen_actions.add(action_id)
                            provenance = {
                                "dataset_revision": DATASET_REVISION,
                                "source_file": record["path"],
                                "source_sha256": record["sha256"],
                            }
                            labels["provenance"] = provenance
                            retrieval["provenance"] = provenance
                            labels_out.write(_canonical(labels) + "\n")
                            retrieval_out.write(_canonical(retrieval) + "\n")
                            row_counts[split] += 1
                            no_positive_counts[split] += int(bool(labels["no_positive"]))
        if any(task_counts[split] != OFFICIAL_TASK_COUNTS[split] for split in splits):
            raise ValueError("task counts do not match complete official held-out splits")
        report: dict[str, object] = {
            "schema": "vons.mind2web-import/v1",
            "dataset_revision": DATASET_REVISION,
            "archive_sha256": ARCHIVE_SHA256,
            "source_manifest_sha256": _sha256(manifest_path),
            "importer_sha256": _sha256(Path(__file__)),
            "splits": list(splits),
            "task_counts": dict(task_counts),
            "row_counts": dict(row_counts),
            "no_positive_counts": dict(no_positive_counts),
            "source_files": selected,
            "outputs": [
                {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in (rows_path, retriever_path)
            ],
            "retriever": {
                "name": "lexical-bm25",
                "k1": 1.2,
                "b": 0.75,
                "tokenization": "Unicode word tokens, casefold, unique query terms",
                "tie_break": "lexicographic candidate id",
                "top_k": None,
            },
            "candidate_order": "lexicographic candidate id after merging the released pool",
            "history": "teacher-forced official action_reprs strictly before the current action",
            "usage": "evaluation-only; no training or public unzipped redistribution",
            "scope": "official held-out candidate retrieval; no model or browser task evaluation",
            "limitations": [
                "Released candidate-pool recall bounds the task; this does not generate DOM candidates.",
                "No training split was acquired; train/test identity overlap is not independently audited.",
                "Duplicate identities are checked across the selected splits only.",
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (staging / ".gitignore").write_text("*\n", encoding="utf-8")
        staging.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split", nargs="+", choices=[*SPLITS, "all"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--extract-archive", type=Path)
    args = parser.parse_args()
    if "all" in args.split and args.split != ["all"]:
        parser.error("all cannot be combined with another split")
    try:
        if args.extract_archive:
            extract_official_archive(args.extract_archive, args.input_dir, args.source_manifest)
        report = import_split(
            args.input_dir,
            SPLITS if args.split == ["all"] else args.split,
            args.output_dir,
            args.source_manifest,
        )
    except (ValueError, TypeError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
        # Deliberately omit source text and JSON decoder excerpts from public logs.
        parser.exit(2, f"Mind2Web import failed ({type(exc).__name__}); no evaluation claimed.\n")
    print(
        json.dumps(
            {
                "status": "imported",
                "task_counts": report["task_counts"],
                "row_counts": report["row_counts"],
                "scope": report["scope"],
            }
        )
    )


if __name__ == "__main__":
    main()
