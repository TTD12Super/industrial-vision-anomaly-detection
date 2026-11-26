"""Tests for the Phase 4.7E offline threshold evaluation (synthetic fixtures).

No real MVTec data, feature bank, calibration artifact, or evaluation output is
touched: every fixture is built in a temporary directory from fake bytes, tiny
placeholder files, and patched scoring.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

ROOT = Path(__file__).resolve().parents[1]

STANDARD_COLUMNS = [
    "category",
    "train_images",
    "alpha",
    "index_k",
    "threshold",
    "alpha_feasible",
    "formula",
    "sorting_convention",
    "decision_rule",
    "seconds",
]


def write_thresholds_csv(directory: Path, rows: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "thresholds.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()
        for row in rows:
            payload = {
                "formula": "k = ceil((m + 1) * (1 - alpha))",
                "sorting_convention": "ascending",
                "decision_rule": "score > threshold",
                "seconds": "0.1",
                **row,
            }
            writer.writerow({column: payload.get(column, "") for column in STANDARD_COLUMNS})
    return path


def bank_entry(category: str, bank_path: Path, project_relative: str | None = None) -> dict:
    data = bank_path.read_bytes()
    return {
        "category": category,
        "path": project_relative or f"artifacts/dinov2_knn/banks/{category}/feature_bank.pt",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def write_manifest(path: Path, entries: list[dict], schema_version: str = "1.0") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": schema_version,
        "manifest_type": "feature_bank_sha256_baseline",
        "entries": entries,
        "gate": {"passed": True},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def entry(category: str, **overrides) -> "object":
    from model.evaluation.threshold_evaluation import ThresholdEntry

    base = {"threshold": 0.5, "alpha": 0.05, "index_k": 3, "train_images": 3}
    base.update(overrides)
    return ThresholdEntry(category=category, **base)


def result_for(
    category: str,
    scores,
    labels,
    threshold_value: float = 0.5,
    **entry_overrides,
):
    from model.evaluation.threshold_evaluation import evaluate_category_thresholds

    return evaluate_category_thresholds(
        category,
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=bool),
        entry(category, threshold=threshold_value, **entry_overrides),
    )


def load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_thresholds_cli", ROOT / "scripts" / "evaluate_thresholds.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixtureBase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="teval_"))
        self.addCleanup(shutil.rmtree, self.root, True)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestStrictDecisionRule(FixtureBase):
    def test_equality_predicted_normal(self):
        from model.evaluation.threshold_evaluation import predict_defect

        predicted = predict_defect([0.5, 0.5000001, 0.4999999], 0.5)
        self.assertEqual(predicted.tolist(), [False, True, False])

    def test_rule_is_strict_greater_than_never_ge(self):
        from model.evaluation.threshold_evaluation import predict_defect

        self.assertEqual(predict_defect([0.5], 0.5).tolist(), [False])
        self.assertEqual(predict_defect([0.5000001], 0.5).tolist(), [True])

    def test_equality_counts_as_normal_in_confusion(self):
        result = result_for(
            "bottle",
            scores=[0.5, 0.5, 0.9, 0.1],
            labels=[False, True, True, False],
            threshold_value=0.5,
        )
        # score == threshold: normal image -> TN, defective image -> FN
        self.assertEqual((result.tn, result.fn, result.fp, result.tp), (2, 1, 0, 1))

    def test_prediction_function_has_no_label_input(self):
        import inspect

        from model.evaluation.threshold_evaluation import predict_defect

        self.assertEqual(
            list(inspect.signature(predict_defect).parameters),
            ["image_scores", "threshold"],
        )


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestConfusionAndMetrics(FixtureBase):
    def test_confusion_counts(self):
        from model.evaluation.threshold_evaluation import confusion_counts

        labels = [False, False, True, True]
        predicted = [False, True, True, True]
        tp, tn, fp, fn = confusion_counts(labels, predicted)
        self.assertEqual((tp, tn, fp, fn), (2, 1, 1, 0))

    def test_confusion_shape_mismatch(self):
        from model.evaluation.threshold_evaluation import ThresholdDataError, confusion_counts

        with self.assertRaises(ThresholdDataError):
            confusion_counts([True, False], [True])

    def test_metric_formulas(self):
        from model.evaluation.threshold_evaluation import threshold_metrics

        metrics = threshold_metrics(tp=2, tn=3, fp=1, fn=1)
        self.assertAlmostEqual(metrics["fpr"], 1 / 4)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["specificity"], 3 / 4)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_precision_undefined_without_predicted_positives(self):
        from model.evaluation.threshold_evaluation import threshold_metrics

        metrics = threshold_metrics(tp=0, tn=5, fp=0, fn=3)
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["f1"])
        self.assertEqual(metrics["fpr"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["specificity"], 1.0)

    def test_f1_undefined_when_precision_plus_recall_is_zero(self):
        from model.evaluation.threshold_evaluation import threshold_metrics

        metrics = threshold_metrics(tp=0, tn=5, fp=3, fn=2)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertIsNone(metrics["f1"])

    def test_population_metrics_undefined_without_that_class(self):
        from model.evaluation.threshold_evaluation import threshold_metrics

        without_normals = threshold_metrics(tp=2, tn=0, fp=0, fn=1)
        self.assertIsNone(without_normals["fpr"])
        self.assertIsNone(without_normals["specificity"])
        without_defectives = threshold_metrics(tp=0, tn=4, fp=2, fn=0)
        self.assertIsNone(without_defectives["recall"])
        self.assertIsNone(without_defectives["f1"])

    def test_score_summary_and_guards(self):
        from model.evaluation.threshold_evaluation import (
            ThresholdDataError,
            score_summary,
        )

        summary = score_summary([1.0, 2.0, 3.0])
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)
        self.assertEqual(summary["mean"], 2.0)
        empty = score_summary([])
        self.assertIsNone(empty["min"])
        with self.assertRaises(ThresholdDataError):
            score_summary([1.0, float("nan")])


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestAggregation(FixtureBase):
    def _results(self):
        # Category A: perfect (TP=1, TN=9); Category B: mixed (TP=1, TN=2, FP=8, FN=9).
        a = result_for(
            "bottle",
            scores=[0.1] * 9 + [0.6],
            labels=[False] * 9 + [True],
        )
        b = result_for(
            "cable",
            scores=[0.9] * 8 + [0.1] * 2 + [0.6] + [0.2] * 9,
            labels=[False] * 10 + [True] * 10,
        )
        return a, b

    def test_category_isolation(self):
        from model.evaluation.threshold_evaluation import macro_threshold_metrics

        a_alone = result_for(
            "bottle", scores=[0.1] * 9 + [0.6], labels=[False] * 9 + [True]
        )
        a, b = self._results()
        self.assertEqual(a.metrics, a_alone.metrics)
        self.assertEqual((a.tp, a.tn, a.fp, a.fn), (1, 9, 0, 0))
        self.assertEqual((b.tp, b.tn, b.fp, b.fn), (1, 2, 8, 9))
        macro = macro_threshold_metrics([a, b])
        self.assertEqual(macro["fpr"]["categories_included"], 2)

    def test_macro_and_micro_are_different_and_labeled(self):
        from model.evaluation.threshold_evaluation import (
            macro_threshold_metrics,
            micro_threshold_metrics,
        )

        a, b = self._results()
        macro = macro_threshold_metrics([a, b])
        micro = micro_threshold_metrics([a, b])
        # macro fpr = mean(0.0, 8/10) = 0.4; micro fpr = pooled 8 / (8 + 11)
        self.assertAlmostEqual(macro["fpr"]["value"], (0.0 + 8 / 10) / 2)
        self.assertAlmostEqual(micro["fpr"], 8 / 19)
        self.assertNotAlmostEqual(macro["fpr"]["value"], micro["fpr"])
        self.assertAlmostEqual(micro["recall"], 2 / 11)
        self.assertAlmostEqual(macro["recall"]["value"], (1.0 + 1 / 10) / 2)
        self.assertEqual(micro["pooled_counts"], {"tp": 2, "tn": 11, "fp": 8, "fn": 9})

    def test_macro_excludes_undefined_without_zero_fill(self):
        from model.evaluation.threshold_evaluation import macro_threshold_metrics

        with_normal_only = result_for(
            "bottle", scores=[0.1, 0.2], labels=[False, False]
        )  # no defectives -> recall/f1 undefined, precision undefined (no predicted positives)
        defined = result_for(
            "cable", scores=[0.1, 0.6], labels=[False, True]
        )
        macro = macro_threshold_metrics([with_normal_only, defined])
        self.assertEqual(macro["recall"]["categories_included"], 1)
        self.assertEqual(macro["recall"]["categories_undefined"], ["bottle"])
        self.assertAlmostEqual(macro["recall"]["value"], defined.metrics["recall"])
        self.assertEqual(macro["precision"]["categories_included"], 1)
        self.assertIn("bottle", macro["precision"]["categories_undefined"])


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestThresholdLoading(FixtureBase):
    EXPECTED = ["bottle", "cable"]

    def _csv(self, rows, columns=None):
        directory = self.root / "calibration"
        if columns is None:
            return directory, write_thresholds_csv(directory, rows)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "thresholds.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return directory, path

    def _valid_rows(self):
        return [
            {"category": "bottle", "train_images": 3, "alpha": 0.05, "index_k": 3, "threshold": 0.5, "alpha_feasible": "true"},
            {"category": "cable", "train_images": 4, "alpha": 0.05, "index_k": 4, "threshold": 0.7, "alpha_feasible": "true"},
        ]

    def test_valid_load(self):
        from model.evaluation.threshold_evaluation import load_frozen_thresholds

        directory, _ = self._csv(self._valid_rows())
        loaded = load_frozen_thresholds(directory, self.EXPECTED)
        self.assertEqual(list(loaded), ["bottle", "cable"])
        self.assertEqual(loaded["bottle"].threshold, 0.5)
        self.assertEqual(loaded["cable"].alpha, 0.05)
        self.assertEqual(loaded["cable"].index_k, 4)

    def _assert_rejects(self, rows, needle, columns=None):
        from model.evaluation.threshold_evaluation import (
            ThresholdDataError,
            load_frozen_thresholds,
        )

        directory, _ = self._csv(rows, columns=columns)
        with self.assertRaises(ThresholdDataError) as ctx:
            load_frozen_thresholds(directory, self.EXPECTED)
        self.assertIn(needle, str(ctx.exception))

    def test_missing_file(self):
        from model.evaluation.threshold_evaluation import (
            ThresholdDataError,
            load_frozen_thresholds,
        )

        with self.assertRaises(ThresholdDataError):
            load_frozen_thresholds(self.root / "absent", self.EXPECTED)

    def test_missing_column(self):
        self._assert_rejects(
            self._valid_rows(),
            "missing columns",
            columns=["category", "train_images", "alpha", "index_k", "alpha_feasible"],
        )

    def test_missing_category_row(self):
        self._assert_rejects(self._valid_rows()[:1], "missing thresholds")

    def test_extra_category_row(self):
        extra = self._valid_rows() + [
            {"category": "screw", "train_images": 3, "alpha": 0.05, "index_k": 3, "threshold": 0.4, "alpha_feasible": "true"}
        ]
        self._assert_rejects(extra, "unexpected threshold categories")

    def test_duplicate_category_row(self):
        self._assert_rejects(self._valid_rows() + [self._valid_rows()[0]], "duplicate")

    def test_nan_threshold(self):
        rows = self._valid_rows()
        rows[0]["threshold"] = "nan"
        self._assert_rejects(rows, "finite")

    def test_infinite_threshold(self):
        rows = self._valid_rows()
        rows[0]["threshold"] = "inf"
        self._assert_rejects(rows, "finite")

    def test_wrong_alpha(self):
        rows = self._valid_rows()
        rows[0]["alpha"] = 0.01
        self._assert_rejects(rows, "alpha")

    def test_infeasible_row(self):
        rows = self._valid_rows()
        rows[0]["alpha_feasible"] = "false"
        self._assert_rejects(rows, "alpha-feasible")

    def test_invalid_index(self):
        rows = self._valid_rows()
        rows[0]["index_k"] = 9
        self._assert_rejects(rows, "index_k")


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestBankBaseline(FixtureBase):
    def setUp(self):
        super().setUp()
        self.bank_dir = self.root / "banks"
        self.bank_path = self.bank_dir / "bottle" / "feature_bank.pt"
        self.bank_path.parent.mkdir(parents=True)
        self.bank_bytes = b"synthetic feature bank bytes" * 32
        self.bank_path.write_bytes(self.bank_bytes)
        self.manifest_path = self.root / "bank_sha256_baseline.json"
        write_manifest(self.manifest_path, [bank_entry("bottle", self.bank_path)])

    def test_valid_verification_returns_hashes(self):
        from model.evaluation.threshold_evaluation import verify_bank_baseline

        verified = verify_bank_baseline(self.bank_dir, self.manifest_path, ["bottle"])
        self.assertEqual(verified["bottle"]["sha256"], hashlib.sha256(self.bank_bytes).hexdigest())
        self.assertEqual(verified["bottle"]["size_bytes"], len(self.bank_bytes))

    def test_hash_mismatch_refused(self):
        from model.evaluation.threshold_evaluation import (
            BaselineIntegrityError,
            verify_bank_baseline,
        )

        self.bank_path.write_bytes(self.bank_bytes + b"tampered")
        with self.assertRaises(BaselineIntegrityError) as ctx:
            verify_bank_baseline(self.bank_dir, self.manifest_path, ["bottle"])
        self.assertIn("sha256", str(ctx.exception))

    def test_size_mismatch_refused(self):
        from model.evaluation.threshold_evaluation import (
            BaselineIntegrityError,
            verify_bank_baseline,
        )

        entries = [bank_entry("bottle", self.bank_path)]
        entries[0]["size_bytes"] += 1
        write_manifest(self.manifest_path, entries)
        with self.assertRaises(BaselineIntegrityError) as ctx:
            verify_bank_baseline(self.bank_dir, self.manifest_path, ["bottle"])
        self.assertIn("size", str(ctx.exception))

    def test_missing_bank_refused(self):
        from model.evaluation.threshold_evaluation import (
            BaselineIntegrityError,
            verify_bank_baseline,
        )

        self.bank_path.unlink()
        with self.assertRaises(BaselineIntegrityError) as ctx:
            verify_bank_baseline(self.bank_dir, self.manifest_path, ["bottle"])
        self.assertIn("missing", str(ctx.exception))

    def test_malformed_manifest_refused(self):
        from model.evaluation.threshold_evaluation import (
            BaselineIntegrityError,
            verify_bank_baseline,
        )

        self.manifest_path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(BaselineIntegrityError):
            verify_bank_baseline(self.bank_dir, self.manifest_path, ["bottle"])

    def test_unsupported_schema_refused(self):
        from model.evaluation.threshold_evaluation import (
            BaselineIntegrityError,
            verify_bank_baseline,
        )

        write_manifest(
            self.manifest_path, [bank_entry("bottle", self.bank_path)], schema_version="9.9"
        )
        with self.assertRaises(BaselineIntegrityError) as ctx:
            verify_bank_baseline(self.bank_dir, self.manifest_path, ["bottle"])
        self.assertIn("schema_version", str(ctx.exception))

    def test_duplicate_and_missing_categories_refused(self):
        from model.evaluation.threshold_evaluation import (
            BaselineIntegrityError,
            verify_bank_baseline,
        )

        entry = bank_entry("bottle", self.bank_path)
        write_manifest(self.manifest_path, [entry, dict(entry)])
        with self.assertRaises(BaselineIntegrityError) as ctx:
            verify_bank_baseline(self.bank_dir, self.manifest_path, ["bottle"])
        self.assertIn("duplicate", str(ctx.exception))
        write_manifest(self.manifest_path, [])
        with self.assertRaises(BaselineIntegrityError) as ctx:
            verify_bank_baseline(self.bank_dir, self.manifest_path, ["bottle"])
        self.assertIn("missing categories", str(ctx.exception))

    def test_unexpected_path_refused(self):
        from model.evaluation.threshold_evaluation import (
            BaselineIntegrityError,
            verify_bank_baseline,
        )

        write_manifest(
            self.manifest_path,
            [bank_entry("bottle", self.bank_path, project_relative="elsewhere/feature_bank.pt")],
        )
        with self.assertRaises(BaselineIntegrityError) as ctx:
            verify_bank_baseline(self.bank_dir, self.manifest_path, ["bottle"])
        self.assertIn("path", str(ctx.exception))

    def test_run_refuses_before_creating_any_directory(self):
        import model.evaluation.threshold_evaluation as te

        self.bank_path.write_bytes(self.bank_bytes + b"tampered")
        run_root = self.root / "runs"
        with self.assertRaises(te.BaselineIntegrityError):
            te.run_threshold_evaluation(
                categories=["bottle"],
                calibration_run=self._calibration_run(),
                bank_baseline_manifest=self.manifest_path,
                processed_root=self.root / "processed",
                bank_dir=self.bank_dir,
                run_root=run_root,
            )
        self.assertFalse(run_root.exists())

    def _calibration_run(self) -> Path:
        directory = self.root / "calibration"
        write_thresholds_csv(
            directory,
            [
                {
                    "category": "bottle",
                    "train_images": 3,
                    "alpha": 0.05,
                    "index_k": 3,
                    "threshold": 0.5,
                    "alpha_feasible": "true",
                }
            ],
        )
        (directory / "calibration_config.json").write_text(
            json.dumps({"alpha": 0.05}), encoding="utf-8"
        )
        return directory


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestRunThresholdEvaluation(FixtureBase):
    """End-to-end synthetic run with a patched (fake) scoring path."""

    SCORES = [0.9, 0.2, 0.5, 0.7, 0.3]
    # test/good/a -> FP, test/good/b -> TN, test/good/e -> TN (equality),
    # test/crack/c -> TP, test/crack/d -> FN

    def _fixture(self):
        processed = self.root / "processed"
        for folder in (
            "bottle/train/good",
            "bottle/test/good",
            "bottle/test/crack",
        ):
            (processed / folder).mkdir(parents=True)
        (processed / "bottle/train/good/000.png").write_bytes(b"placeholder")
        for name in ("a", "b", "e"):
            (processed / f"bottle/test/good/{name}.png").write_bytes(b"placeholder")
        for name in ("c", "d"):
            (processed / f"bottle/test/crack/{name}.png").write_bytes(b"placeholder")

        bank_dir = self.root / "banks"
        bank_path = bank_dir / "bottle" / "feature_bank.pt"
        bank_path.parent.mkdir(parents=True)
        bank_path.write_bytes(b"synthetic bank payload" * 16)
        manifest_path = self.root / "bank_sha256_baseline.json"
        write_manifest(manifest_path, [bank_entry("bottle", bank_path)])

        calibration = self.root / "calibration"
        write_thresholds_csv(
            calibration,
            [
                {
                    "category": "bottle",
                    "train_images": 3,
                    "alpha": 0.05,
                    "index_k": 3,
                    "threshold": 0.5,
                    "alpha_feasible": "true",
                }
            ],
        )
        (calibration / "calibration_config.json").write_text(
            json.dumps({"alpha": 0.05}), encoding="utf-8"
        )
        return {
            "processed": processed,
            "bank_dir": bank_dir,
            "manifest": manifest_path,
            "calibration": calibration,
            "run_root": self.root / "runs",
        }

    def _fake_evaluator(self, scores, labels):
        from model.evaluation.evaluate import CategoryResult

        def fake(category, *, processed_root, bank_dir, extractor, batch_size,
                 limit=None, save_maps=False, compute_pixel_metrics=True):
            rows = [
                {
                    "category": category,
                    "image_path": f"bottle/test/{'good' if not label else 'crack'}/{i}.png",
                    "defect_type": "good" if not label else "crack",
                    "is_anomaly": int(label),
                    "anomaly_score": float(score),
                    "processing_time_s": 0.01,
                }
                for i, (score, label) in enumerate(zip(scores, labels))
            ]
            result = CategoryResult(category=category)
            result.test_images = len(rows)
            result.normal_images = int(sum(1 for r in rows if not r["is_anomaly"]))
            result.defective_images = int(sum(1 for r in rows if r["is_anomaly"]))
            result.rows = rows
            result.pipeline_seconds = 0.05
            return result

        return fake

    class _DummyExtractor:
        def __init__(self, device=None, **kwargs):
            self.device = "cpu"
            self.patch_grid = (19, 19)
            self.pad_right = 10
            self.pad_bottom = 10

    def _run(self, fixture, scores, labels, **kwargs):
        import model.evaluation.threshold_evaluation as te

        with mock.patch(
            "model.evaluation.evaluate.evaluate_category",
            self._fake_evaluator(scores, labels),
        ), mock.patch(
            "model.inference.feature_extractor.DINOv2FeatureExtractor",
            self._DummyExtractor,
        ):
            return te.run_threshold_evaluation(
                categories=["bottle"],
                calibration_run=fixture["calibration"],
                bank_baseline_manifest=fixture["manifest"],
                processed_root=fixture["processed"],
                bank_dir=fixture["bank_dir"],
                run_root=fixture["run_root"],
                reference_eval_dir=None,
                **kwargs,
            )

    def test_full_synthetic_run_writes_isolated_artifacts(self):
        fixture = self._fixture()
        run_dir, results = self._run(fixture, self.SCORES, [False, False, False, True, True])
        produced = sorted(path.name for path in run_dir.iterdir())
        for expected_name in (
            "threshold_metrics.csv",
            "confusion_matrices.csv",
            "summary.json",
            "config.json",
            "bank_hashes_before.json",
            "bank_hashes_after.json",
            "report.md",
            "failures.json",
        ):
            self.assertIn(expected_name, produced)

        result = results[0]
        self.assertEqual((result.tn, result.fp, result.fn, result.tp), (2, 1, 1, 1))
        self.assertAlmostEqual(result.metrics["fpr"], 1 / 3)
        self.assertAlmostEqual(result.metrics["recall"], 1 / 2)

        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["alpha"], 0.05)
        self.assertEqual(summary["test_totals"], {
            "images": 5, "normal": 3, "defective": 2, "tp": 1, "tn": 2, "fp": 1, "fn": 1,
        })
        self.assertIn("marginal guarantee", summary["guarantee_note"])
        self.assertIn("not a macro average", summary["micro"]["definition"])

        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["alpha"], 0.05)
        self.assertIn("score > threshold", config["decision_rule"])
        self.assertEqual(config["thresholds"]["bottle"]["threshold"], 0.5)
        self.assertTrue(config["decision_rule"].count("strict"))

        before = json.loads((run_dir / "bank_hashes_before.json").read_text(encoding="utf-8"))
        after = json.loads((run_dir / "bank_hashes_after.json").read_text(encoding="utf-8"))
        self.assertEqual(before, after)
        self.assertEqual(
            before["bottle"],
            hashlib.sha256((fixture["bank_dir"] / "bottle" / "feature_bank.pt").read_bytes()).hexdigest(),
        )

        failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
        self.assertEqual(failures, [])
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("Macro averages", report)
        self.assertIn("Micro averages (pooled counts", report)
        self.assertIn("0.500000", report)
        self.assertNotIn("anomaly_maps", "".join(produced))

        with (run_dir / "threshold_metrics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["threshold"], "0.5")
        self.assertEqual(rows[0]["predicted_defect_when_normal"], "1")
        self.assertEqual(rows[0]["predicted_normal_when_normal"], "2")

    def test_refuses_to_overwrite_existing_run_dir(self):
        import model.evaluation.threshold_evaluation as te

        fixture = self._fixture()
        real_datetime = te.datetime

        class FixedDateTime:
            @classmethod
            def now(cls, tz=None):
                return real_datetime(2026, 1, 2, 3, 4, 5, 6, tzinfo=tz)

        scores, labels = self.SCORES, [False, False, False, True, True]
        with mock.patch.object(te, "datetime", FixedDateTime):
            run_dir, _ = self._run(fixture, scores, labels)
            self.assertTrue(run_dir.is_dir())
            with self.assertRaises(FileExistsError):
                self._run(fixture, scores, labels)
        self.assertTrue((run_dir / "summary.json").is_file())

    def test_labels_cannot_affect_predictions_or_threshold(self):
        from model.evaluation.threshold_evaluation import evaluate_category_thresholds

        scores = np.array([0.1, 0.5, 0.9, 0.4])
        labels_a = np.array([False, False, True, True])
        labels_b = np.array([True, True, False, False])  # fully permuted labels
        threshold = entry("bottle", threshold=0.5)
        first = evaluate_category_thresholds("bottle", scores, labels_a, threshold)
        second = evaluate_category_thresholds("bottle", scores, labels_b, threshold)
        np.testing.assert_array_equal(first.predicted, second.predicted)
        self.assertEqual(
            first.predicted.tolist(), [False, False, True, False]
        )  # only scores+threshold matter
        self.assertEqual(first.threshold_entry.threshold, 0.5)
        self.assertEqual(second.threshold_entry.threshold, 0.5)
        # confusion changes with labels, predictions never do
        self.assertNotEqual(
            (first.tp, first.tn, first.fp, first.fn),
            (second.tp, second.tn, second.fp, second.fn),
        )
        self.assertEqual((first.fp, second.fp), (0, 1))


class TestCliSurface(unittest.TestCase):
    def test_cli_has_no_threshold_or_alpha_override(self):
        cli = load_cli_module()
        args = cli.parse_args([])
        forbidden = ("threshold", "alpha", "label", "mask", "test_dir", "eval_run")
        for name in vars(args):
            self.assertFalse(
                any(token in name for token in forbidden),
                f"CLI option {name!r} could alter frozen thresholds or inputs",
            )
        self.assertTrue(args.preflight_only is False)
        self.assertTrue(hasattr(cli, "preflight"))

    def test_cli_requires_no_category_selection(self):
        cli = load_cli_module()
        args = cli.parse_args(["--preflight-only"])
        self.assertTrue(args.preflight_only)


if __name__ == "__main__":
    unittest.main()
