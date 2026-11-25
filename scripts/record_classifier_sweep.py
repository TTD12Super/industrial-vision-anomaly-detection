"""Record the saved DINOv2 threshold sweep in the guarded sweep ledger.

The adapter does not train or infer.  It converts an existing validation-only
threshold sweep into the repository's immutable ledger format and preserves
the diagnostic result separately from the validation candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

# Keep the script usable both as `python scripts/...` and with PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspection.sweep import sha256_file, write_sweep_report


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "reports/products/screw/screw-full-dev-004/classification_manifest.csv"
THRESHOLD_REPORT = ROOT / "reports/products/screw/screw-full-dev-004/classifier_lastblock_patch_head/threshold_sweep.json"
MODEL = ROOT / "artifacts/products/screw/screw-full-dev-004/classifier_lastblock_patch_head/model.pt"
INDEPENDENT_HOLDOUT = ROOT / "datasets/external/mvtec_screws/extracted/mvtec_screws_test.json"


def _write_split_csv(source: Path, split: str, destination: Path) -> tuple[int, int]:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No rows found for split={split}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    defect_count = sum(row["label"] == "DEFECT" for row in selected)
    good_count = sum(row["label"] == "GOOD" for row in selected)
    return defect_count, good_count


def _lineage_config(output_dir: Path, reconciled_report: Path) -> Path:
    path = output_dir / "inputs" / "lineage_config.json"
    payload = {
        "schema_version": 1,
        "experiment": "dinov2_lastblock_patch_head_threshold_sweep",
        "model": str(MODEL.relative_to(ROOT)),
        "model_sha256": sha256_file(MODEL),
        "classification_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "threshold_report": str(THRESHOLD_REPORT.relative_to(ROOT)),
        "reconciled_threshold_report": str(reconciled_report.relative_to(ROOT)),
        "selection": {
            "split": "validation",
            "target_defect_recall": 0.90,
            "good_false_positive_cap": 0.20,
            "tie_break": [
                "maximize_defect_recall",
                "lower_good_false_positive_rate",
                "higher_threshold",
            ],
        },
        "warning": "Development/diagnostic evidence; not a target-camera production benchmark.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _candidate_rows(
    threshold_data: dict[str, Any],
    defect_count: int,
    good_count: int,
    source_report_path: Path | None = None,
) -> list[dict[str, Any]]:
    candidates = []
    model_path = str(MODEL.resolve())
    model_hash = sha256_file(MODEL)
    source_report_path = source_report_path or THRESHOLD_REPORT
    source_report_path = source_report_path.resolve()
    source_report_hash = sha256_file(source_report_path)
    for index, point in enumerate(threshold_data["points"]):
        validation = point["validation"]
        recall = float(validation.get("recall", validation.get("defect_recall")))
        good_fp = float(validation["good_fp"])
        threshold = float(point["threshold"])
        defect_true_positive = validation.get("defect_true_positive")
        good_false_alarm_count = validation.get("good_false_alarm_count")
        if defect_true_positive is None and good_false_alarm_count is None:
            defect_true_positive = int(round(recall * defect_count))
            good_false_alarm_count = int(round(good_fp * good_count))
        elif defect_true_positive is None or good_false_alarm_count is None:
            raise ValueError("validation point must provide both exact count fields or neither")
        candidates.append(
            {
                "name": f"threshold_{index:03d}",
                "method": "dinov2_lastblock_patch_head",
                "metric_unit": "object",
                "artifact_paths": [model_path],
                "artifact_sha256": {model_path: model_hash},
                "source_report_path": str(source_report_path),
                "source_report_sha256": source_report_hash,
                "metrics": {
                    "defect_recall": recall,
                    "good_false_positive_rate": good_fp,
                    "threshold": threshold,
                },
                "metric_numerators": {
                    "defect_true_positive": int(defect_true_positive),
                    "good_false_alarm_count": int(good_false_alarm_count),
                },
                "metric_denominators": {
                    "defect_count": defect_count,
                    "good_count": good_count,
                },
                "split_role": "validation",
                "holdout_used_for_selection": False,
                "device": "mps",
            }
        )
    return candidates


def _reconcile_threshold_report(
    threshold_data: dict[str, Any], output_dir: Path, defect_count: int, good_count: int
) -> Path:
    """Materialize selection metadata and exact integer counts for the legacy report."""

    points = []
    for point in threshold_data["points"]:
        validation = dict(point["validation"])
        diagnostic = dict(point["test_diagnostic"])
        validation["defect_true_positive"] = int(round(float(validation["recall"]) * defect_count))
        validation["good_false_alarm_count"] = int(round(float(validation["good_fp"]) * good_count))
        points.append({**point, "validation": validation, "test_diagnostic": diagnostic})
    selected = threshold_data.get("selected_validation_gate")
    reconciled = {
        **threshold_data,
        "schema_version": "1.1",
        "source_report": str(THRESHOLD_REPORT.relative_to(ROOT)),
        "source_report_sha256": sha256_file(THRESHOLD_REPORT),
        "selection": {
            "target_defect_recall": 0.90,
            "good_false_positive_cap": 0.20,
            "selection_split_role": "validation",
            "tie_break": [
                "maximize_defect_recall",
                "lower_good_false_positive_rate",
                "higher_threshold",
            ],
        },
        "points": points,
        "metric_count_source": "legacy_rate_reconstruction_exact_fraction",
        "warning": "Derived from the legacy saved rates; rerun the trainer to obtain raw confusion counts.",
    }
    path = output_dir / "inputs" / "reconciled_threshold_sweep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reconciled, indent=2) + "\n", encoding="utf-8")
    return path


def record(output_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing sweep run: {output_dir}")
    if root != ROOT:
        raise ValueError("record_classifier_sweep currently requires the repository root")
    for path in (SOURCE_MANIFEST, THRESHOLD_REPORT, MODEL, INDEPENDENT_HOLDOUT):
        if not path.is_file():
            raise FileNotFoundError(path)

    inputs = output_dir / "inputs"
    counts = {}
    for split in ("train", "validation", "test"):
        counts[split] = _write_split_csv(SOURCE_MANIFEST, split, inputs / f"{split}.csv")
    threshold_data = json.loads(THRESHOLD_REPORT.read_text(encoding="utf-8"))
    reconciled_report = _reconcile_threshold_report(threshold_data, output_dir, *counts["validation"])
    # The historical diagnostic is the saved test result, not a second split.
    diagnostic_manifest = inputs / "test.csv"
    lineage_config = _lineage_config(output_dir, reconciled_report)
    candidates = _candidate_rows(threshold_data, *counts["validation"], reconciled_report)
    spec = {
        "schema_version": 1,
        "experiment": "dinov2_lastblock_patch_head_threshold_sweep",
        "independent_holdout_locked": True,
        "split_roles": {
            "train": {"manifest": str((inputs / "train.csv").resolve()), "sha256": sha256_file(inputs / "train.csv")},
            "validation": {"manifest": str((inputs / "validation.csv").resolve()), "sha256": sha256_file(inputs / "validation.csv")},
            "diagnostic": {"manifest": str(diagnostic_manifest.resolve()), "sha256": sha256_file(diagnostic_manifest)},
            "independent_holdout": {"manifest": str(INDEPENDENT_HOLDOUT.resolve()), "sha256": sha256_file(INDEPENDENT_HOLDOUT)},
        },
        "selection": {
            "target_defect_recall": 0.90,
            "good_false_positive_cap": 0.20,
            "selection_split_role": "validation",
            "tie_break": [
                "maximize_defect_recall",
                "lower_good_false_positive_rate",
                "higher_threshold",
            ],
        },
        "lineage": {
            "config_path": str(lineage_config.resolve()),
            "config_sha256": sha256_file(lineage_config),
            "seed": 42,
            "device": "mps",
            "command": "scripts/record_classifier_sweep.py",
            "diagnostic_alias_of": "test",
            "diagnostic_alias_manifest": str(diagnostic_manifest.resolve()),
            "diagnostic_alias_sha256": sha256_file(diagnostic_manifest),
            "source_metric_count_note": "Legacy source rates were converted to integer counts; raw confusion counts require a trainer rerun.",
        },
        "candidates": candidates,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    ledger = write_sweep_report(spec, output_dir / "ledger")
    selected_record = json.loads(
        (output_dir / "ledger" / "selected_candidate.json").read_text(encoding="utf-8")
    )
    selected_candidate = selected_record.get("selected_candidate")
    formal_point = None
    if selected_candidate is not None:
        selected_index = int(selected_candidate["name"].rsplit("_", 1)[1])
        formal_point = threshold_data["points"][selected_index]
    source_selected = threshold_data.get("selected_validation_gate")
    diagnostic_evidence = {
        "schema_version": 1,
        "status": "development_diagnostic" if formal_point is not None else "development_diagnostic_no_validation_gate",
        "ledger_selection": {
            "threshold": formal_point["threshold"],
            "validation": formal_point["validation"],
            "diagnostic": formal_point["test_diagnostic"],
            "selection_rule": spec["selection"],
        } if formal_point is not None else None,
        "source_report_selection": {
            "threshold": source_selected["threshold"],
            "validation": source_selected["validation"],
            "diagnostic": source_selected["test_diagnostic"],
        } if source_selected is not None else None,
        "source_report": str(reconciled_report.relative_to(ROOT)),
        "source_report_sha256": sha256_file(reconciled_report),
        "source_report_original": str(THRESHOLD_REPORT.relative_to(ROOT)),
        "source_report_original_sha256": sha256_file(THRESHOLD_REPORT),
        "diagnostic_alias_of": "test",
        "warning": "Diagnostic result is not an independent target-camera holdout; no-validation-gate runs must not select a runtime threshold.",
    }
    (output_dir / "diagnostic_evidence.json").write_text(json.dumps(diagnostic_evidence, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir.resolve()), "ledger": ledger, "diagnostic_evidence": diagnostic_evidence}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/experiments/screw_classifier_sweep_rnd1")
    args = parser.parse_args()
    print(json.dumps(record((ROOT / args.output_dir).resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
