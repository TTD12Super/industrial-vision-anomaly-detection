import json
import tempfile
import unittest
from pathlib import Path

import scripts.render_thesis_report as report


class ThesisReportTest(unittest.TestCase):
    def test_current_sources_are_normalized_for_report_figures(self) -> None:
        rows = report._classifier_rows(report.ROOT)
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(rows[-1]["defect_recall"], 18 / 23, places=6)

        heatmaps = report._heatmap_rows(report.ROOT)
        self.assertIn("DINO spatial decoder 448", {row["name"] for row in heatmaps})

        detectors, _ = report._detector_rows(report.ROOT)
        self.assertEqual({row["name"] for row in detectors}, {"YOLO-OBB", "RT-DETR"})

        sweep, _ = report._threshold_sweep(report.ROOT)
        self.assertAlmostEqual(sweep["selected_validation_gate"]["test_diagnostic"]["recall"], 0.9130434782608695)

    def test_render_writes_manifest_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figures"
            manifest = report.render(output, report.ROOT)
            self.assertEqual(manifest["status"], "supervised_development_diagnostic")
            self.assertFalse(manifest["model_run"])
            self.assertTrue((output / "qualitative_evidence_montage.png").is_file())
            saved = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["generated_files"].keys(), manifest["generated_files"].keys())
            with self.assertRaises(FileExistsError):
                report.render(output, report.ROOT)


if __name__ == "__main__":
    unittest.main()
