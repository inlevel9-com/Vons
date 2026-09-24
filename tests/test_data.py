import tempfile
import unittest
from pathlib import Path

from vons.data import dataset_manifest, read_jsonl, smoke_examples, synthetic_examples, write_jsonl


class DataTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.jsonl"
            write_jsonl(path, smoke_examples())
            rows = read_jsonl(path)
            self.assertEqual(len(rows), 4)
            self.assertEqual(dataset_manifest(path, rows)["rows"], 4)

    def test_reproducible_generation(self) -> None:
        self.assertEqual([row.to_mapping() for row in synthetic_examples(20, seed=9)], [row.to_mapping() for row in synthetic_examples(20, seed=9)])


if __name__ == "__main__":
    unittest.main()

