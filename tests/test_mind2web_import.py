"""Handcrafted fixtures only; official restricted tasks never live in tests."""

import copy
import hashlib
import importlib.util
import io
import json
import stat
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vons.mind2web import Mind2WebExample


def _load_importer():
    path = Path(__file__).parents[1] / "tools" / "import_mind2web.py"
    spec = importlib.util.spec_from_file_location("vons_mind2web_importer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Mind2Web importer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


importer = _load_importer()


def _candidate(node_id):
    return {
        "backend_node_id": node_id,
        "tag": "button",
        "is_original_target": True,
        "is_top_level_target": True,
        "attributes": json.dumps({"is_original_target": "SECRET_GOLD_FLAG", "role": "button"}),
    }


def _task(task_id="task-a", action_id="action-a"):
    return {
        "annotation_id": task_id,
        "website": "fixture.example",
        "domain": "fixture-domain",
        "confirmed_task": "Find the blue train",
        "action_reprs": ["CURRENT_ACTION_SECRET", "FUTURE_ACTION_SECRET"],
        "actions": [
            {
                "action_uid": action_id,
                "raw_html": "RAW_HTML_SECRET",
                "cleaned_html": (
                    '<html><button backend_node_id="30" is_original_target="DOM_GOLD_SECRET">'
                    '<text>Blue train</text></button><button backend_node_id="10">'
                    '<text>Red bus</text></button><input backend_node_id="20" '
                    'aria_label="Blue train timetable"/></html>'
                ),
                "operation": {"op": "TYPE", "value": "OPERATION_VALUE_SECRET"},
                "pos_candidates": [_candidate("30"), _candidate("20")],
                "neg_candidates": [_candidate("10")],
            },
            {
                "action_uid": action_id + "-next",
                "cleaned_html": '<button backend_node_id="40">Continue</button>',
                "operation": {"op": "CLICK", "value": "NEXT_OPERATION_SECRET"},
                "pos_candidates": [],
                "neg_candidates": [_candidate("40")],
            },
        ],
    }


def _manifest():
    return {
        "schema": "vons.mind2web-acquisition/v1",
        "dataset": "osunlp/Mind2Web",
        "dataset_revision": importer.DATASET_REVISION,
        "usage": "evaluation-only",
        "restrictions": {"no_training": True, "no_unzipped_redistribution": True},
        "archive": {
            "sha256": importer.ARCHIVE_SHA256,
            "bytes": importer.ARCHIVE_BYTES,
            "status": "downloaded-and-hash-verified",
        },
        "extraction": {"status": "complete"},
        "files": [],
    }


class Mind2WebImportTests(unittest.TestCase):
    def test_multiple_positives_and_no_positive_preserve_contract(self):
        rows = list(importer.normalize_task(_task(), "test_task"))
        first = Mind2WebExample.from_mapping(rows[0][0])
        second = Mind2WebExample.from_mapping(rows[1][0])
        self.assertEqual(first.candidate_ids, ("10", "20", "30"))
        self.assertEqual(first.positive_ids, ("20", "30"))
        self.assertEqual(first.task_id, "task-a")
        self.assertEqual(first.action_id, "action-a")
        self.assertEqual(first.website, "fixture.example")
        self.assertEqual(first.domain, "fixture-domain")
        self.assertEqual(first.split, "test_task")
        self.assertTrue(second.no_positive)
        self.assertEqual(second.positive_ids, ())

    def test_current_future_operation_raw_html_and_flags_never_enter_retriever(self):
        rows = list(importer.normalize_task(_task(), "test_task"))
        first = json.dumps(rows[0][1])
        for secret in (
            "CURRENT_ACTION_SECRET",
            "FUTURE_ACTION_SECRET",
            "OPERATION_VALUE_SECRET",
            "RAW_HTML_SECRET",
            "SECRET_GOLD_FLAG",
            "DOM_GOLD_SECRET",
            "positive_ids",
            "no_positive",
            "is_original_target",
        ):
            self.assertNotIn(secret, first)
        self.assertEqual(rows[1][1]["state"]["previous_actions"], ["CURRENT_ACTION_SECRET"])
        self.assertNotIn("FUTURE_ACTION_SECRET", json.dumps(rows[1][1]))

    def test_relabeling_and_reordering_do_not_change_rank_or_input_hash(self):
        task = _task()
        original = next(importer.normalize_task(task, "test_task"))[1]
        changed = copy.deepcopy(task)
        action = changed["actions"][0]
        action["neg_candidates"], action["pos_candidates"] = (
            list(reversed(action["pos_candidates"])),
            action["neg_candidates"],
        )
        action["operation"] = {"op": "SELECT", "value": "OTHER_SECRET"}
        changed["action_reprs"] = ["DIFFERENT_CURRENT", "DIFFERENT_FUTURE"]
        for field in ("pos_candidates", "neg_candidates"):
            for candidate in action[field]:
                candidate["is_original_target"] = False
                candidate["is_top_level_target"] = False
        result = next(importer.normalize_task(changed, "test_task"))[1]
        self.assertEqual(original, result)

    def test_lexical_baseline_uses_dom_text_and_deterministic_ties(self):
        rows = list(importer.normalize_task(_task(), "test_task"))
        ranked = rows[0][1]["candidates"]
        self.assertIn("Blue train", ranked[0]["text"])
        self.assertEqual([item["rank"] for item in ranked], [1, 2, 3])
        self.assertEqual(ranked[-1]["id"], "10")
        candidates = [{"id": "z", "text": "same"}, {"id": "a", "text": "same"}]
        self.assertEqual(
            [row["id"] for row in importer.lexical_rank("unmatched", candidates)], ["a", "z"]
        )

    def test_xml_dom_preserves_candidates_inside_nested_iframes(self):
        task = _task()
        action = task["actions"][0]
        action["cleaned_html"] = (
            '<html><iframe><button backend_node_id="10"><text>Alpha&amp;Beta</text></button>'
            '<iframe><img backend_node_id="20" alt="Blue train"/></iframe>'
            '<img backend_node_id="30" alt="Blue timetable"/></iframe></html>'
        )
        candidates = importer.candidate_universe(action)
        self.assertEqual([row["id"] for row in candidates], ["10", "20", "30"])
        self.assertIn("Alpha&Beta", candidates[0]["text"])
        self.assertIn("Blue train", candidates[1]["text"])

    def test_empty_candidate_pool_is_preserved(self):
        task = _task()
        task["actions"][0]["pos_candidates"] = []
        task["actions"][0]["neg_candidates"] = []
        labels, retrieval = next(importer.normalize_task(task, "test_task"))
        self.assertTrue(labels["no_positive"])
        self.assertEqual(retrieval["generated_candidates"], [])

    def test_duplicate_candidate_and_missing_cleaned_node_fail_closed(self):
        task = _task()
        task["actions"][0]["neg_candidates"].append(_candidate("30"))
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            list(importer.normalize_task(task, "test_task"))
        task = _task()
        task["actions"][0]["cleaned_html"] = "<html></html>"
        with self.assertRaisesRegex(ValueError, "missing from cleaned DOM"):
            list(importer.normalize_task(task, "test_task"))

    def test_duplicate_dom_node_and_action_ids_fail_closed(self):
        task = _task()
        task["actions"][0]["cleaned_html"] = task["actions"][0]["cleaned_html"].replace(
            "</html>", '<button backend_node_id="30">Again</button></html>'
        )
        with self.assertRaisesRegex(ValueError, "duplicate candidate node"):
            list(importer.normalize_task(task, "test_task"))
        task = _task()
        task["actions"][1]["action_uid"] = task["actions"][0]["action_uid"]
        with self.assertRaisesRegex(ValueError, "duplicate action"):
            list(importer.normalize_task(task, "test_task"))

    def test_unknown_or_conflicting_splits_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "official held-out"):
            list(importer.normalize_task(_task(), "train"))
        task = _task()
        task["split"] = "test_domain"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            list(importer.normalize_task(task, "test_task"))

    def test_history_misalignment_fails_closed(self):
        task = _task()
        task["action_reprs"] = ["ONLY_ONE"]
        with self.assertRaisesRegex(ValueError, "aligned"):
            list(importer.normalize_task(task, "test_task"))

    def test_streaming_json_survives_small_chunk_boundaries(self):
        tasks = [_task(), _task("task-b", "action-b")]
        with patch.object(importer, "CHUNK_SIZE", 17):
            self.assertEqual(list(importer.iter_tasks(io.StringIO(json.dumps(tasks)))), tasks)
            self.assertEqual(list(importer.iter_tasks(io.StringIO(" [ ] \n"))), [])

    def test_streaming_large_task_reads_grow_without_changing_task_boundaries(self):
        class RecordingReader(io.StringIO):
            def __init__(self, value):
                super().__init__(value)
                self.read_sizes = []

            def read(self, size=-1):
                self.read_sizes.append(size)
                return super().read(size)

        tasks = [_task(), _task("second-task", "second-action")]
        tasks[0]["actions"][0]["raw_html"] = "unneeded source text " * 500
        reader = RecordingReader(json.dumps(tasks))
        with patch.object(importer, "CHUNK_SIZE", 32):
            self.assertEqual(list(importer.iter_tasks(reader)), tasks)
        self.assertLess(len(reader.read_sizes), 32)

    def test_streaming_json_rejects_malformed_and_oversized_inputs(self):
        for source in ("{}", "[{},]", "[{}", "[{}]{}", "[{]", "[,{}]"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                list(importer.iter_tasks(io.StringIO(source)))
        with (
            patch.object(importer, "MAX_TASK_CHARS", 20),
            self.assertRaisesRegex(ValueError, "buffer bound"),
        ):
            list(importer.iter_tasks(io.StringIO(json.dumps([_task()]))))

    def test_archive_guards_reject_traversal_unknown_and_duplicate_members(self):
        for name in (
            "../test_task_0.json",
            "/test_task/test_task_0.json",
            "test_task/../../escape.json",
            "test_task\\test_task_0.json",
            "train/train_0.json",
            "test_task//test_task_0.json",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                importer.validate_archive_members([zipfile.ZipInfo(name)])
        info = zipfile.ZipInfo("test_task/test_task_0.json")
        info.file_size = 1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            importer.validate_archive_members([info, info])

    def test_archive_guards_require_exact_members_and_bounded_regular_files(self):
        infos = []
        for split, count in importer.OFFICIAL_FILE_COUNTS.items():
            for index in range(count):
                info = zipfile.ZipInfo(f"{split}/{split}_{index}.json")
                info.file_size = 1
                infos.append(info)
        importer.validate_archive_members(infos)
        with self.assertRaisesRegex(ValueError, "member set"):
            importer.validate_archive_members(infos[:-1])
        infos[0].external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaisesRegex(ValueError, "symlink"):
            importer.validate_archive_members(infos)
        infos[0].external_attr = 0
        infos[0].file_size = importer.MAX_MEMBER_BYTES + 1
        with self.assertRaisesRegex(ValueError, "size bound"):
            importer.validate_archive_members(infos)

    def _fixture_files(self, root):
        input_dir = root / "input"
        manifest = _manifest()
        for index, split in enumerate(importer.SPLITS):
            name = f"{split}/{split}_0.json"
            path = input_dir / name
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps([_task(f"task-{index}", f"action-{index}")]))
            manifest["files"].append(
                {
                    "path": name,
                    "split": split,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        manifest_path = root / "source.json"
        manifest_path.write_text(json.dumps(manifest))
        return input_dir, manifest_path

    def test_import_outputs_provenance_and_contract_compatible_jsonl(self):
        # Fixture counts are patched locally; the production CLI has no count override.
        with (
            TemporaryDirectory() as directory,
            patch.object(importer, "OFFICIAL_FILE_COUNTS", dict.fromkeys(importer.SPLITS, 1)),
            patch.object(importer, "OFFICIAL_TASK_COUNTS", dict.fromkeys(importer.SPLITS, 1)),
        ):
            root = Path(directory)
            input_dir, manifest_path = self._fixture_files(root)
            output = root / "output"
            report = importer.import_split(input_dir, importer.SPLITS, output, manifest_path)
            self.assertEqual(report["task_counts"], dict.fromkeys(importer.SPLITS, 1))
            self.assertEqual(report["row_counts"], dict.fromkeys(importer.SPLITS, 2))
            for line in (output / "rows.jsonl").read_text().splitlines():
                row = json.loads(line)
                Mind2WebExample.from_mapping(row)
                self.assertEqual(row["provenance"]["dataset_revision"], importer.DATASET_REVISION)
            for record in report["outputs"]:
                self.assertEqual(
                    record["sha256"],
                    hashlib.sha256((output / record["path"]).read_bytes()).hexdigest(),
                )
            with self.assertRaisesRegex(ValueError, "overwrite"):
                importer.import_split(input_dir, importer.SPLITS, output, manifest_path)

    def test_tampering_and_incomplete_task_counts_create_no_output(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(importer, "OFFICIAL_FILE_COUNTS", dict.fromkeys(importer.SPLITS, 1)),
        ):
            root = Path(directory)
            input_dir, manifest_path = self._fixture_files(root)
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "task counts"):
                importer.import_split(input_dir, importer.SPLITS, output, manifest_path)
            self.assertFalse(output.exists())
            path = input_dir / "test_task/test_task_0.json"
            path.write_text(path.read_text().replace("blue", "pink"))
            with self.assertRaisesRegex(ValueError, "hash verification"):
                importer.import_split(input_dir, importer.SPLITS, output, manifest_path)
            self.assertFalse(output.exists())

    def test_duplicate_tasks_across_splits_fail_closed(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(importer, "OFFICIAL_FILE_COUNTS", dict.fromkeys(importer.SPLITS, 1)),
        ):
            root = Path(directory)
            input_dir, manifest_path = self._fixture_files(root)
            manifest = json.loads(manifest_path.read_text())
            for record in manifest["files"]:
                path = input_dir / record["path"]
                path.write_text(json.dumps([_task()]))
                record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                record["bytes"] = path.stat().st_size
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "duplicate task"):
                importer.import_split(input_dir, importer.SPLITS, root / "output", manifest_path)

    def test_manifest_rejects_unknown_revision_and_training_usage(self):
        manifest = _manifest()
        importer._validate_manifest(manifest)
        manifest["dataset_revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "pinned"):
            importer._validate_manifest(manifest)
        manifest = _manifest()
        manifest["usage"] = "training"
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            importer._validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
