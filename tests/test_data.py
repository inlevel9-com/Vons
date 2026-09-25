import tempfile
import unittest
from collections import Counter
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

    def test_synthetic_generation_balances_all_actions_and_rules(self) -> None:
        rows = synthetic_examples(200, seed=9)
        answerable = [row for row in rows if row.answerable]
        counts = Counter(row.label for row in answerable)

        self.assertEqual(set(counts), {"call_tool", "clarify", "refuse", "respond_directly"})
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        for count in counts.values():
            self.assertAlmostEqual(count / len(answerable), 0.25, delta=0.01)
        self.assertTrue(all(row.provenance.get("rule") for row in rows))
        self.assertGreater(len({row.options.index(row.label) for row in answerable}), 1)

    def test_manifest_reports_label_and_abstention_counts(self) -> None:
        rows = synthetic_examples(52)
        manifest = dataset_manifest("synthetic.jsonl", rows)

        self.assertEqual(sum(manifest["labels"].values()) + manifest["unanswerable_rows"], 52)
        self.assertEqual(set(manifest["labels"]), {"call_tool", "clarify", "refuse", "respond_directly"})


if __name__ == "__main__":
    unittest.main()
