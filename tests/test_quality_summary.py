import importlib.util
import tempfile
import unittest
from pathlib import Path

from vons.data import smoke_examples, write_jsonl
from vons.evaluation import Prediction, write_report


def _load_summary_tool():
    path = Path(__file__).parents[1] / "tools" / "summarize_quality.py"
    spec = importlib.util.spec_from_file_location("vons_quality_summary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load quality summary tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary_tool = _load_summary_tool()


class QualitySummaryTests(unittest.TestCase):
    def test_recomputes_saved_direct_and_diffusion_quality(self) -> None:
        root = Path(__file__).parents[1]
        data_path = root / "data/generated/pilot-test.jsonl"
        reports = [
            root / "reports/pilot-direct-test.json",
            root / "reports/pilot-diffusion-test.json",
        ]
        if not all(path.is_file() for path in [data_path, *reports]):
            self.skipTest("Historical pilot inputs are intentionally excluded from public source")
        result = summary_tool.build_quality_summary(data_path, reports)

        self.assertEqual(result["measurement_label"], "measured_synthetic_pilot")
        self.assertEqual(result["dataset"]["rows"], 200)
        self.assertEqual(result["reports"][0]["metrics"]["accuracy_answerable"], 1.0)
        self.assertEqual(result["reports"][1]["metrics"]["coverage"], 0.155)
        self.assertIsNone(result["external_evaluation"]["mind2web"]["candidate_recall"])
        self.assertEqual(result["external_evaluation"]["mind2web"]["status"], "not_evaluated")

    def test_recomputes_metrics_without_private_pilot_files(self) -> None:
        rows = smoke_examples()[:2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "synthetic.jsonl"
            write_jsonl(data_path, rows)
            reports = []
            for backend in ("direct", "diffusion"):
                predictions = [
                    Prediction(
                        row.id, row.label,
                        {option: float(option == row.label) for option in row.options},
                        1.0, backend == "diffusion" and index == 1,
                    )
                    for index, row in enumerate(rows)
                ]
                report_path = root / f"{backend}.json"
                write_report(report_path, config={"backend": backend, "seed": 7},
                             summary={"coverage": -1}, predictions=predictions)
                reports.append(report_path)
            result = summary_tool.build_quality_summary(data_path, reports, version="v2")

        self.assertEqual(result["dataset"]["rows"], 2)
        self.assertEqual(result["metric_version"], "vons.metrics/v2")
        direct, diffusion = result["reports"]
        self.assertEqual(direct["metrics"]["accuracy_answerable"], 1.0)
        self.assertEqual(direct["metrics"]["coverage"], 1.0)
        self.assertEqual(diffusion["metrics"]["coverage"], 0.5)
        self.assertEqual(diffusion["metrics"]["selective_risk"], 0.0)
        self.assertEqual(diffusion["abstained_rows"], 1)
        self.assertIsNone(result["external_evaluation"]["mind2web"]["candidate_recall"])

    def test_descriptive_stats_preserve_undefined_seed_metrics_as_null(self) -> None:
        rows = [
            {"backend": "direct", "metrics": {"accuracy_answerable": 1.0}},
            {"backend": "direct", "metrics": {"accuracy_answerable": None}},
        ]
        result = summary_tool.descriptive_seed_stats(rows)
        metric = result["direct"]["accuracy_answerable"]
        self.assertEqual(metric["mean"], 1.0)
        self.assertEqual(metric["defined_seed_count"], 1)
        self.assertEqual(metric["null_seed_count"], 1)
        self.assertEqual(metric["seed_count"], 2)


if __name__ == "__main__":
    unittest.main()
