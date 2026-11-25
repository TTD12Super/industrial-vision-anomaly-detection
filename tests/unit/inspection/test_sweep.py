from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from inspection.sweep import (
    DEFAULT_TIE_BREAK,
    MAX_RECALL_TIE_BREAK,
    load_sweep_spec,
    select_candidate,
    sha256_file,
    write_sweep_report,
)


class SweepLedgerTests(TestCase):
    @staticmethod
    def _candidate(
        name: str,
        recall: float,
        fp: float,
        latency: float | None = None,
        role: str = "validation",
        threshold: float | None = None,
    ) -> dict:
        metrics = {"defect_recall": recall, "good_false_positive_rate": fp}
        if latency is not None:
            metrics["latency_ms"] = latency
        if threshold is not None:
            metrics["threshold"] = threshold
        return {
            "name": name,
            "method": "static",
            "metric_unit": "object",
            "artifact_paths": [f"{name}.pt"],
            "source_report_path": f"source_{name}.json",
            "metrics": metrics,
            "metric_numerators": {
                "defect_true_positive": int(round(recall * 1000)),
                "good_false_alarm_count": int(round(fp * 1000)),
            },
            "metric_denominators": {"defect_count": 1000, "good_count": 1000},
            "split_role": role,
            "holdout_used_for_selection": False,
        }

    def _spec(
        self,
        root: Path,
        candidates: list[dict],
        selection: dict | None = None,
        *,
        independent_holdout_locked: bool = True,
    ) -> Path:
        split_roles = {}
        for role in ("train", "validation", "diagnostic", "independent_holdout"):
            manifest = root / f"{role}.json"
            manifest.write_text(json.dumps({"role": role}), encoding="utf-8")
            split_roles[role] = {"manifest": manifest.name, "sha256": sha256_file(manifest)}

        config = root / "config.yaml"
        config.write_text("product_key: screw\n", encoding="utf-8")
        for candidate in candidates:
            for artifact in candidate["artifact_paths"]:
                (root / artifact).write_bytes(candidate["name"].encode("utf-8"))
            source_report = root / candidate["source_report_path"]
            source_report.write_text(json.dumps({"candidate": candidate["name"]}), encoding="utf-8")
            candidate["source_report_sha256"] = sha256_file(source_report)
            candidate["artifact_sha256"] = {
                artifact: sha256_file(root / artifact) for artifact in candidate["artifact_paths"]
            }

        raw = {
            "schema_version": 1,
            "experiment": "test_sweep",
            "independent_holdout_locked": independent_holdout_locked,
            "split_roles": split_roles,
            "selection": selection
            or {
                "target_defect_recall": 0.90,
                "good_false_positive_cap": 0.20,
                "tie_break": list(DEFAULT_TIE_BREAK),
            },
            "lineage": {
                "config_path": config.name,
                "config_sha256": sha256_file(config),
                "seed": 7,
                "device": "cpu",
            },
            "candidates": candidates,
        }
        path = root / "spec.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    @staticmethod
    def _raw(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _save_raw(path: Path, raw: dict) -> None:
        path.write_text(json.dumps(raw), encoding="utf-8")

    def test_happy_selection_requires_both_gates(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._spec(
                Path(directory),
                [self._candidate("low", 0.90, 0.05), self._candidate("best", 0.95, 0.20)],
            )
            result = select_candidate(load_sweep_spec(path))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["selected_candidate"]["name"], "best")
            self.assertEqual(result["target_defect_recall"], 0.90)

    def test_recall_below_target_is_not_eligible(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._spec(Path(directory), [self._candidate("below", 0.899, 0.01)])
            result = select_candidate(load_sweep_spec(path))
            self.assertEqual(result["status"], "not_passed")
            self.assertIsNone(result["selected_candidate"])

    def test_recall_equal_target_and_fp_equal_cap_pass(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._spec(Path(directory), [self._candidate("boundary", 0.90, 0.20)])
            result = select_candidate(load_sweep_spec(path))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["selected_candidate"]["name"], "boundary")

    def test_no_candidate_passes_has_no_selected_candidate(self) -> None:
        with TemporaryDirectory() as directory:
            candidates = [self._candidate("high_fp", 0.99, 0.21), self._candidate("low_recall", 0.89, 0.01)]
            result = select_candidate(load_sweep_spec(self._spec(Path(directory), candidates)))
            self.assertEqual(result["status"], "not_passed")
            self.assertIsNone(result["selected_candidate"])

    def test_candidate_order_does_not_change_selection(self) -> None:
        with TemporaryDirectory() as directory:
            candidates = [
                self._candidate("slow", 0.95, 0.10, 30),
                self._candidate("fast", 0.95, 0.10, 10),
            ]
            result = select_candidate(load_sweep_spec(self._spec(Path(directory), list(reversed(candidates)))))
            self.assertEqual(result["selected_candidate"]["name"], "fast")

    def test_diagnostic_and_holdout_rows_are_ledger_only(self) -> None:
        with TemporaryDirectory() as directory:
            candidates = [
                self._candidate("validation", 0.90, 0.10),
                self._candidate("diagnostic", 1.00, 0.00, role="diagnostic"),
                self._candidate("holdout", 1.00, 0.00, role="independent_holdout"),
            ]
            result = select_candidate(load_sweep_spec(self._spec(Path(directory), candidates)))
            self.assertEqual(result["selected_candidate"]["name"], "validation")

    def test_holdout_lock_and_candidate_selection_provenance_are_required(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root, [self._candidate("candidate", 0.90, 0.10)], independent_holdout_locked=False)
            with self.assertRaisesRegex(ValueError, "independent_holdout_locked"):
                load_sweep_spec(path)

            path = self._spec(root, [self._candidate("candidate", 0.90, 0.10)])
            raw = self._raw(path)
            del raw["candidates"][0]["holdout_used_for_selection"]
            self._save_raw(path, raw)
            with self.assertRaisesRegex(ValueError, "holdout_used_for_selection"):
                load_sweep_spec(path)

    def test_split_hashes_and_changed_split_are_checked(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root, [self._candidate("candidate", 0.90, 0.10)])
            (root / "validation.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split_roles.validation.sha256"):
                load_sweep_spec(path)

    def test_diagnostic_alias_requires_matching_test_manifest_and_declaration(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root, [self._candidate("candidate", 0.90, 0.10)])
            raw = self._raw(path)
            raw["lineage"]["diagnostic_alias_of"] = "test"
            raw["lineage"]["diagnostic_alias_manifest"] = str((root / "diagnostic.json").resolve())
            raw["lineage"]["diagnostic_alias_sha256"] = sha256_file(root / "diagnostic.json")
            self._save_raw(path, raw)
            load_sweep_spec(path)

            raw["lineage"]["diagnostic_alias_of"] = "wrong"
            self._save_raw(path, raw)
            with self.assertRaisesRegex(ValueError, "diagnostic_alias_of"):
                load_sweep_spec(path)

    def test_source_report_hash_and_artifact_are_checked_and_recorded(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root, [self._candidate("candidate", 0.90, 0.10)])
            spec = load_sweep_spec(path)
            manifest = write_sweep_report(spec, root / "run")
            candidate = spec["candidates"][0]
            self.assertEqual(
                manifest["candidate_provenance"]["candidate"]["source_report_sha256"]["expected"],
                candidate["source_report_sha256"],
            )
            self.assertEqual(
                manifest["candidate_provenance"]["candidate"]["source_report_sha256"]["actual"],
                candidate["source_report_sha256"],
            )
            self.assertEqual(
                manifest["candidate_provenance"]["candidate"]["artifact_hashes"]["expected"],
                manifest["candidate_provenance"]["candidate"]["artifact_hashes"]["actual"],
            )
            self.assertEqual(len(manifest["split_roles"]), 4)

            drift = self._spec(root, [self._candidate("drift", 0.90, 0.10)])
            drift_spec = load_sweep_spec(drift)
            (root / "drift.pt").write_bytes(b"content drift")
            with self.assertRaisesRegex(ValueError, "artifact_sha256"):
                write_sweep_report(drift_spec, root / "drift-run")

            changed = self._spec(root, [self._candidate("changed", 0.90, 0.10)])
            changed_raw = self._raw(changed)
            source_report = root / changed_raw["candidates"][0]["source_report_path"]
            source_report.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_report_sha256"):
                load_sweep_spec(changed)

            missing_hash = self._spec(root, [self._candidate("missing_hash", 0.90, 0.10)])
            missing_raw = self._raw(missing_hash)
            del missing_raw["candidates"][0]["source_report_sha256"]
            self._save_raw(missing_hash, missing_raw)
            with self.assertRaisesRegex(ValueError, "source_report_sha256"):
                load_sweep_spec(missing_hash)

            missing = self._spec(root, [self._candidate("missing", 0.90, 0.10)])
            (root / "missing.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                load_sweep_spec(missing)

    def test_selection_candidate_names_must_be_validation(self) -> None:
        with TemporaryDirectory() as directory:
            selection = {
                "target_defect_recall": 0.90,
                "good_false_positive_cap": 0.20,
                "candidate_names": ["diagnostic"],
            }
            path = self._spec(
                Path(directory), [self._candidate("diagnostic", 1.00, 0.00, role="diagnostic")], selection
            )
            with self.assertRaisesRegex(ValueError, "Only validation candidates"):
                load_sweep_spec(path)

    def test_metric_numerator_denominator_contract_is_checked(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root, [self._candidate("mismatch", 0.95, 0.10)])
            raw = self._raw(path)
            raw["candidates"][0]["metrics"]["defect_recall"] = 0.94
            self._save_raw(path, raw)
            with self.assertRaisesRegex(ValueError, "does not match TP/defect_count"):
                load_sweep_spec(path)

            path = self._spec(root, [self._candidate("missing_fields", 0.90, 0.10)])
            raw = self._raw(path)
            del raw["candidates"][0]["metric_numerators"]
            with self.assertRaisesRegex(ValueError, "metric_numerators"):
                self._save_raw(path, raw)
                load_sweep_spec(path)

    def test_fractional_and_boolean_counts_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root, [self._candidate("fractional", 0.90, 0.10)])
            raw = self._raw(path)
            raw["candidates"][0]["metric_denominators"]["defect_count"] = 10.5
            self._save_raw(path, raw)
            with self.assertRaisesRegex(ValueError, "integer count"):
                load_sweep_spec(path)

            path = self._spec(root, [self._candidate("boolean", 0.90, 0.10)])
            raw = self._raw(path)
            raw["candidates"][0]["metric_numerators"]["good_false_alarm_count"] = True
            self._save_raw(path, raw)
            with self.assertRaisesRegex(ValueError, "integer count"):
                load_sweep_spec(path)

    def test_config_hash_is_required_and_pinned(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root, [self._candidate("config", 0.90, 0.10)])
            raw = self._raw(path)
            raw["lineage"]["config_sha256"] = "0" * 64
            self._save_raw(path, raw)
            with self.assertRaisesRegex(ValueError, "lineage.config_sha256"):
                load_sweep_spec(path)

            path = self._spec(root, [self._candidate("config_drift", 0.90, 0.10)])
            spec = load_sweep_spec(path)
            (root / "config.yaml").write_text("product_key: changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lineage.config_sha256"):
                write_sweep_report(spec, root / "config-drift-run")

    def test_invalid_metric_unit_denominator_nan_and_range_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root, [self._candidate("invalid", 0.90, 0.10)])
            raw = self._raw(path)
            raw["candidates"][0]["metric_unit"] = "unknown"
            self._save_raw(path, raw)
            with self.assertRaisesRegex(ValueError, "metric_unit"):
                load_sweep_spec(path)

            path = self._spec(root, [self._candidate("invalid_denom", 0.90, 0.10)])
            raw = self._raw(path)
            raw["candidates"][0]["metric_denominators"]["good_count"] = 0
            self._save_raw(path, raw)
            with self.assertRaisesRegex(ValueError, "must be positive"):
                load_sweep_spec(path)

            path = self._spec(root, [self._candidate("invalid_nan", 0.90, 0.10)])
            raw = self._raw(path)
            raw["candidates"][0]["metrics"]["defect_recall"] = "NaN"
            self._save_raw(path, raw)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_sweep_spec(path)

            path = self._spec(root, [self._candidate("invalid_range", 0.90, 0.10)])
            raw = self._raw(path)
            raw["candidates"][0]["metrics"]["good_false_positive_rate"] = 1.1
            self._save_raw(path, raw)
            with self.assertRaisesRegex(ValueError, "range"):
                load_sweep_spec(path)

    def test_non_default_tie_break_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            selection = {
                "target_defect_recall": 0.90,
                "good_false_positive_cap": 0.20,
                "tie_break": ["lower_latency_ms", "lower_good_false_positive_rate", "lower_name"],
            }
            path = self._spec(Path(directory), [self._candidate("candidate", 0.90, 0.10)], selection)
            with self.assertRaisesRegex(ValueError, "tie_break"):
                load_sweep_spec(path)

    def test_classifier_tie_break_maximizes_recall_then_threshold(self) -> None:
        with TemporaryDirectory() as directory:
            selection = {
                "target_defect_recall": 0.90,
                "good_false_positive_cap": 0.20,
                "tie_break": list(MAX_RECALL_TIE_BREAK),
            }
            path = self._spec(
                Path(directory),
                [
                    self._candidate("low", 1.00, 0.10, threshold=0.30),
                    self._candidate("high", 1.00, 0.10, threshold=0.40),
                    self._candidate("too_high", 0.89, 0.01, threshold=0.50),
                ],
                selection,
            )
            result = select_candidate(load_sweep_spec(path))
            self.assertEqual(result["selected_candidate"]["name"], "high")

    def test_classifier_tie_break_requires_threshold_metric(self) -> None:
        with TemporaryDirectory() as directory:
            selection = {
                "target_defect_recall": 0.90,
                "good_false_positive_cap": 0.20,
                "tie_break": list(MAX_RECALL_TIE_BREAK),
            }
            path = self._spec(Path(directory), [self._candidate("missing", 0.90, 0.10)], selection)
            with self.assertRaisesRegex(ValueError, "metrics.threshold"):
                load_sweep_spec(path)

    def test_report_refuses_overwrite_and_partial_write_leaves_no_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root, [self._candidate("candidate", 0.90, 0.10)])
            spec = load_sweep_spec(path)
            output = root / "run"
            write_sweep_report(spec, output)
            with self.assertRaises(FileExistsError):
                write_sweep_report(spec, output)

            failed_output = root / "failed-run"
            with patch("inspection.sweep._write_report_file") as writer:
                writer.side_effect = RuntimeError("simulated partial write")
                with self.assertRaisesRegex(RuntimeError, "partial write"):
                    write_sweep_report(spec, failed_output)
            self.assertFalse(failed_output.exists())
            self.assertEqual(list(root.glob(".failed-run.tmp-*")), [])


if __name__ == "__main__":
    import unittest

    unittest.main()
