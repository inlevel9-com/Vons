import hashlib
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_builder():
    path = Path(__file__).parents[1] / "tools" / "build_mind2web_manifest.py"
    spec = importlib.util.spec_from_file_location("vons_mind2web_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Mind2Web manifest builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


class Mind2WebManifestTests(unittest.TestCase):
    def test_manifest_hashes_splits_without_public_paths(self) -> None:
        with TemporaryDirectory() as directory:
            split_path = Path(directory) / "cross-task.jsonl"
            payload = b'{"id":"a"}\n'
            split_path.write_bytes(payload)
            manifest = builder.build_manifest(
                variant="original-mind2web",
                source_revision="rev-1",
                permitted_use="evaluation_only",
                approval_record="owner-approved-2026-09-24",
                preprocessing_revision="prep-1",
                retriever_revision="retriever-1",
                splits=[("cross-task", split_path)],
            )

            self.assertEqual(manifest["schema"], "vons.mind2web-provenance/v1")
            self.assertEqual(manifest["splits"][0]["bytes"], len(payload))
            self.assertEqual(manifest["splits"][0]["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(manifest["raw_data_paths"], "omitted from public artifacts")

    def test_manifest_rejects_duplicate_split_names(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "split.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                builder.build_manifest(
                    variant="original-mind2web",
                    source_revision="rev-1",
                    permitted_use="evaluation_only",
                    approval_record="approved",
                    preprocessing_revision="prep-1",
                    retriever_revision="retriever-1",
                    splits=[("test", path), ("test", path)],
                )


if __name__ == "__main__":
    unittest.main()
