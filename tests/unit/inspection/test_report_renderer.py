from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.render_static_report import (
    CHART_NAMES,
    _confusion_matrix,
    _history,
    _operating_points,
    render_static_report,
    sha256_file,
)


class StaticReportRendererTests(TestCase):
    @staticmethod
    def _write_inputs(root: Path) -> tuple[Path, Path]:
        operating_points = root / "operating_points.json"
        operating_points.write_text(
            json.dumps(
                {
                    "protocol": "locked_test_diagnostic",
                    "operating_points": [
                        {
                            "name": "op_a",
                            "test_defect_recall": 0.80,
                            "test_good_fp": 0.10,
                            "calibration_threshold": 0.5,
                        },
                        {
                            "calibration_threshold": 0.7,
                            "validation_defect_recall": 0.90,
                            "validation_good_fp": 0.20,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        classifier_summary = root / "summary.json"
        classifier_summary.write_text(
            json.dumps(
                {
                    "protocol": "DINOv2 historical development",
                    "history": [
                        {"epoch": 1, "train_loss": 0.8, "validation_auroc": 0.7},
                        {"epoch": 2, "train_loss": 0.4, "validation_auroc": 0.9},
                    ],
                    "results": {
                        "test": {
                            "confusion_matrix_good_defect": [[8, 2], [1, 9]],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return operating_points, classifier_summary

    def test_happy_render_writes_three_charts_and_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            operating_points, classifier_summary = self._write_inputs(root)
            output = root / "rendered"
            manifest = render_static_report(operating_points, classifier_summary, output)
            self.assertEqual(manifest["status"], "historical_development_diagnostic")
            self.assertFalse(manifest["model_run"])
            self.assertFalse(manifest["threshold_selection"])
            for name in (*CHART_NAMES, "report_manifest.json"):
                self.assertTrue((output / name).is_file(), name)
            self.assertIn("not a fresh independent holdout", manifest["warning"])
            self.assertEqual(manifest["domains"], ["default"])

    def test_actual_cad_mvtec_fixture_is_kept_as_two_domains(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        actual_path = repo_root / "reports/experiments/dinov2_patchcore_lite_top1pct_operating_points_rnd3.json"
        payload = json.loads(actual_path.read_text(encoding="utf-8"))
        points = _operating_points(payload)
        self.assertEqual({point["domain"] for point in points}, {"CAD-SD", "MVTec screw"})
        self.assertEqual(len(points), 2 * len(payload["operating_points"]))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, classifier_summary = self._write_inputs(root)
            output = root / "actual-domains"
            manifest = render_static_report(actual_path, classifier_summary, output)
            self.assertEqual(manifest["domains"], ["CAD-SD", "MVTec screw"])

    def test_manifest_records_source_and_generated_hashes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            operating_points, classifier_summary = self._write_inputs(root)
            output = root / "rendered"
            manifest = render_static_report(operating_points, classifier_summary, output)
            self.assertEqual(
                manifest["source_files"]["operating_points"]["sha256"],
                sha256_file(operating_points),
            )
            self.assertEqual(
                manifest["source_files"]["classifier_summary"]["sha256"],
                sha256_file(classifier_summary),
            )
            for name in CHART_NAMES:
                self.assertEqual(
                    manifest["generated_files"][name]["sha256"],
                    sha256_file(output / name),
                )

    def test_conflicting_aliases_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            operating_points, _ = self._write_inputs(root)
            payload = json.loads(operating_points.read_text(encoding="utf-8"))
            payload["operating_points"][0]["validation_defect_recall"] = 0.81
            with self.assertRaisesRegex(ValueError, "conflicting aliases"):
                _operating_points(payload)

            payload["operating_points"][0]["validation_defect_recall"] = 0.80
            points = _operating_points(payload)
            self.assertEqual(points[0]["defect_recall"], 0.80)

    def test_history_and_confusion_values_are_strictly_validated(self) -> None:
        valid_history = [{"epoch": 1, "train_loss": 0.1, "validation_auroc": 0.9}]
        self.assertEqual(_history({"history": valid_history})[0]["epoch"], 1)
        for malformed in (
            [{"epoch": 0, "train_loss": 0.1, "validation_auroc": 0.9}],
            [{"epoch": 1, "train_loss": -0.1, "validation_auroc": 0.9}],
            [{"epoch": 1, "train_loss": 0.1, "validation_auroc": 1.1}],
            [{"epoch": 1.5, "train_loss": 0.1, "validation_auroc": 0.9}],
        ):
            with self.assertRaises(ValueError):
                _history({"history": malformed})
        with self.assertRaisesRegex(ValueError, "integer"):
            _confusion_matrix(
                {"results": {"test": {"confusion_matrix_good_defect": [[1.5, 2], [3, 4]]}}}
            )

    def test_missing_inputs_and_invalid_confusion_matrix_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            operating_points, classifier_summary = self._write_inputs(root)
            missing_points = root / "missing_points.json"
            missing_points.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "operating_points"):
                render_static_report(missing_points, classifier_summary, root / "missing-points-run")

            with self.assertRaises(FileNotFoundError):
                render_static_report(root / "does-not-exist.json", classifier_summary, root / "missing-run")

            invalid_summary = root / "invalid_summary.json"
            invalid_summary.write_text(
                json.dumps(
                    {
                        "history": [{"epoch": 1, "train_loss": 0.5, "validation_auroc": 0.8}],
                        "results": {"test": {"confusion_matrix_good_defect": [[1, 2, 3], [4, 5, 6]]}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "2x2"):
                render_static_report(operating_points, invalid_summary, root / "invalid-run")

    def test_output_overwrite_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            operating_points, classifier_summary = self._write_inputs(root)
            output = root / "rendered"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                render_static_report(operating_points, classifier_summary, output)

    def test_partial_write_cleans_temp_and_final_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            operating_points, classifier_summary = self._write_inputs(root)
            output = root / "failed-render"
            def fake_chart(_data: object, path: Path, _pyplot: object) -> None:
                path.write_bytes(b"chart")

            with patch("scripts.render_static_report._load_pyplot", return_value=object()), patch(
                "scripts.render_static_report._render_operating_points", side_effect=fake_chart
            ), patch(
                "scripts.render_static_report._render_learning_curve", side_effect=fake_chart
            ), patch(
                "scripts.render_static_report._render_confusion_matrix", side_effect=fake_chart
            ), patch("scripts.render_static_report._write_text") as writer:
                writer.side_effect = RuntimeError("simulated manifest write failure")
                with self.assertRaisesRegex(RuntimeError, "manifest write failure"):
                    render_static_report(operating_points, classifier_summary, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".failed-render.tmp-*")), [])

    def test_cli_help_does_not_initialize_matplotlib(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repo_root / "src")
        completed = subprocess.run(
            [sys.executable, str(repo_root / "scripts/render_static_report.py"), "--help"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--operating-points", completed.stdout)


if __name__ == "__main__":
    import unittest

    unittest.main()
