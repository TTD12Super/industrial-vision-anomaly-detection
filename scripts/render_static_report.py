#!/usr/bin/env python3
"""Render report-ready charts from existing static-image JSON reports only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


CHART_NAMES = (
    "threshold_recall_vs_good_fp.png",
    "classifier_learning_curve.png",
    "classifier_confusion_matrix.png",
)
WARNING = (
    "Historical/development/diagnostic evidence rendered from existing JSON; "
    "not a fresh independent holdout and no model or threshold selection was performed."
)
ENVIRONMENT_LIMITATION = (
    "Chart rendering lazy-loads Matplotlib with the Agg backend; the first render may be delayed "
    "by environment font-cache initialization. Input validation and --help do not initialize it."
)
DOMAIN_SPECS = {
    "CAD-SD": (("cad_defect_recall",), ("cad_good_fp",)),
    "MVTec screw": (("mvtec_defect_recall",), ("mvtec_good_fp",)),
}


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _rate(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in the range [0, 1]")
    return number


def _integer_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer non-negative count")
    if value < 0:
        raise ValueError(f"{label} must be an integer non-negative count")
    return value


def _load_pyplot() -> Any:
    """Load Matplotlib only after JSON validation has completed."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _read_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {label} JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object: {path}")
    return path, payload


def _first_value(row: Mapping[str, Any], keys: tuple[str, ...], label: str) -> tuple[str, float]:
    values: list[tuple[str, float]] = []
    for key in keys:
        if key in row:
            values.append((key, _finite_number(row[key], f"{label}.{key}")))
    if values:
        first_key, first_value = values[0]
        if any(value != first_value for _, value in values[1:]):
            aliases = ", ".join(key for key, _ in values)
            raise ValueError(f"{label} has conflicting aliases: {aliases}")
        return first_key, first_value
    raise ValueError(f"{label} is missing one of: {', '.join(keys)}")


def _operating_points(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("operating_points")
    if not isinstance(rows, list) or not rows:
        raise ValueError("operating_points must be a non-empty list")
    points: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"operating_points[{index}] must be an object")
        label = f"operating_points[{index}]"
        point_label = None
        for key in ("label", "name", "operating_point"):
            if key in row:
                point_label = str(row[key])
                break
        if point_label is None:
            for key in ("threshold", "calibration_threshold", "defect_threshold", "allowed_good_fpr"):
                if key in row:
                    point_label = f"{key}={row[key]}"
                    break
        domain_points = []
        for domain, (recall_aliases, fp_aliases) in DOMAIN_SPECS.items():
            if any(key in row for key in (*recall_aliases, *fp_aliases)):
                recall_key, recall = _first_value(row, recall_aliases, f"{label}.{domain}")
                fp_key, good_fp = _first_value(row, fp_aliases, f"{label}.{domain}")
                domain_points.append(
                    {
                        "domain": domain,
                        "label": point_label or f"op_{index + 1}",
                        "defect_recall": _rate(recall, f"{label}.{recall_key}"),
                        "good_fp": _rate(good_fp, f"{label}.{fp_key}"),
                    }
                )
        if not domain_points:
            recall_key, recall = _first_value(
                row,
                ("test_defect_recall", "validation_defect_recall"),
                label,
            )
            fp_key, good_fp = _first_value(
                row,
                ("test_good_fp", "validation_good_fp"),
                label,
            )
            domain_points.append(
                {
                    "domain": str(payload.get("domain", "default")),
                    "label": point_label or f"op_{index + 1}",
                    "defect_recall": _rate(recall, f"{label}.{recall_key}"),
                    "good_fp": _rate(good_fp, f"{label}.{fp_key}"),
                }
            )
        for point in domain_points:
            points.append(point)
    return points


def _history(payload: Mapping[str, Any]) -> list[dict[str, float]]:
    rows = payload.get("history")
    if not isinstance(rows, list) or not rows:
        raise ValueError("classifier summary history must be a non-empty list")
    history: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"history[{index}] must be an object")
        label = f"history[{index}]"
        epoch = _integer_count(row.get("epoch"), f"{label}.epoch")
        if epoch <= 0:
            raise ValueError(f"{label}.epoch must be an integer greater than zero")
        train_loss = _finite_number(row.get("train_loss"), f"{label}.train_loss")
        if train_loss < 0.0:
            raise ValueError(f"{label}.train_loss must be non-negative")
        _, validation_auroc = _first_value(
            row,
            ("validation_auroc", "val_auroc"),
            label,
        )
        validation_auroc = _rate(validation_auroc, f"{label}.validation_auroc")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_auroc": validation_auroc,
            }
        )
    return history


def _confusion_matrix(payload: Mapping[str, Any]) -> list[list[int]]:
    try:
        matrix = payload["results"]["test"]["confusion_matrix_good_defect"]
    except (KeyError, TypeError) as error:
        raise ValueError("classifier summary is missing results.test.confusion_matrix_good_defect") from error
    if not isinstance(matrix, list) or len(matrix) != 2:
        raise ValueError("confusion matrix must be a 2x2 list")
    normalized: list[list[int]] = []
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("confusion matrix must be a 2x2 list")
        normalized_row: list[int] = []
        for column_index, value in enumerate(row):
            normalized_row.append(
                _integer_count(value, f"confusion_matrix[{row_index}][{column_index}]")
            )
        normalized.append(normalized_row)
    return normalized


def _render_operating_points(points: list[dict[str, Any]], path: Path, pyplot: Any) -> None:
    figure, axis = pyplot.subplots(figsize=(7.5, 5.0))
    try:
        for domain in sorted({point["domain"] for point in points}):
            domain_points = [point for point in points if point["domain"] == domain]
            x = [point["good_fp"] * 100.0 for point in domain_points]
            y = [point["defect_recall"] * 100.0 for point in domain_points]
            axis.plot(x, y, marker="o", linewidth=1.5, label=domain)
        axis.set_xlabel("GOOD false-positive rate (%)")
        axis.set_ylabel("DEFECT recall (%)")
        axis.set_title(
            "Operating points by domain: recall vs GOOD false positives\n"
            "Locked historical diagnostic points; not used for selection"
        )
        axis.legend(loc="best")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
    finally:
        pyplot.close(figure)


def _render_learning_curve(history: list[dict[str, float]], path: Path, pyplot: Any) -> None:
    figure, loss_axis = pyplot.subplots(figsize=(7.5, 5.0))
    try:
        epochs = [row["epoch"] for row in history]
        loss_line = loss_axis.plot(epochs, [row["train_loss"] for row in history], color="tab:blue", label="train loss")
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Train loss", color="tab:blue")
        loss_axis.tick_params(axis="y", labelcolor="tab:blue")
        auroc_axis = loss_axis.twinx()
        auroc_line = auroc_axis.plot(
            epochs,
            [row["validation_auroc"] for row in history],
            color="tab:orange",
            marker="o",
            label="validation AUROC",
        )
        auroc_axis.set_ylabel("Validation AUROC", color="tab:orange")
        auroc_axis.tick_params(axis="y", labelcolor="tab:orange")
        loss_axis.set_title("Classifier learning curve")
        loss_axis.grid(True, alpha=0.3)
        loss_axis.legend(loss_line + auroc_line, ["train loss", "validation AUROC"], loc="best")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
    finally:
        pyplot.close(figure)


def _render_confusion_matrix(matrix: list[list[int]], path: Path, pyplot: Any) -> None:
    figure, axis = pyplot.subplots(figsize=(5.5, 5.0))
    try:
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_xticks([0, 1], labels=["GOOD", "DEFECT"])
        axis.set_yticks([0, 1], labels=["GOOD", "DEFECT"])
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title("Test confusion matrix")
        maximum = max(max(row) for row in matrix) or 1.0
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                color = "white" if value > maximum * 0.5 else "black"
                axis.text(column_index, row_index, str(value), ha="center", va="center", color=color)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
    finally:
        pyplot.close(figure)


def _write_text(path: Path, content: str) -> None:
    """Small test seam for verifying cleanup after a partial write."""

    path.write_text(content, encoding="utf-8")


def render_static_report(
    operating_points_path: str | Path,
    classifier_summary_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render charts and publish one immutable report artifact directory."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing report directory: {output_dir}")
    operating_path, operating_payload = _read_json(operating_points_path, "operating-points report")
    classifier_path, classifier_payload = _read_json(classifier_summary_path, "classifier summary")
    points = _operating_points(operating_payload)
    history = _history(classifier_payload)
    matrix = _confusion_matrix(classifier_payload)
    pyplot = _load_pyplot()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    published = False
    try:
        _render_operating_points(points, temporary_dir / CHART_NAMES[0], pyplot)
        _render_learning_curve(history, temporary_dir / CHART_NAMES[1], pyplot)
        _render_confusion_matrix(matrix, temporary_dir / CHART_NAMES[2], pyplot)
        generated_files = {
            name: {
                "path": str(output_dir / name),
                "sha256": sha256_file(temporary_dir / name),
            }
            for name in CHART_NAMES
        }
        manifest = {
            "schema_version": "1.0",
            "source_files": {
                "operating_points": {"path": str(operating_path), "sha256": sha256_file(operating_path)},
                "classifier_summary": {"path": str(classifier_path), "sha256": sha256_file(classifier_path)},
            },
            "generated_files": generated_files,
            "domains": sorted({point["domain"] for point in points}),
            "source_protocol": {
                "operating_points": operating_payload.get("protocol", "unspecified"),
                "classifier_summary": classifier_payload.get("protocol", "unspecified"),
            },
            "status": "historical_development_diagnostic",
            "warning": WARNING,
            "environment_limitation": ENVIRONMENT_LIMITATION,
            "model_run": False,
            "threshold_selection": False,
        }
        _write_text(temporary_dir / "report_manifest.json", json.dumps(manifest, indent=2) + "\n")
        os.rename(temporary_dir, output_dir)
        published = True
        return manifest
    finally:
        if not published and temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operating-points", type=Path, required=True)
    parser.add_argument("--classifier-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = render_static_report(args.operating_points, args.classifier_summary, args.output_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
