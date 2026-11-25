"""Render report-ready figures from existing screw experiments.

This script is deliberately read-only with respect to models and datasets.  It
only consumes saved JSON reports and existing qualitative image artefacts.  The
generated manifest labels the figures as supervised-development diagnostics;
it does not turn historical test data into an independent benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _classifier_rows(root: Path) -> list[dict[str, Any]]:
    comparison_path = root / "reports/products/screw/screw-full-dev-002/classifier_comparison.md"
    image_ap_by_name: dict[str, float] = {}
    for line in _require(comparison_path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or "Method" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 8:
            try:
                image_ap_by_name[cells[0]] = float(cells[7])
            except ValueError:
                continue
    sources = [
        (
            "Frozen global + linear head",
            root / "reports/products/screw/screw-full-dev-001/classifier_metrics.json",
        ),
        (
            "Frozen global + regularized head",
            root / "reports/products/screw/screw-full-dev-002/classifier_linear_head/summary.json",
        ),
        (
            "Last block + patch-aware head",
            root / "reports/products/screw/screw-full-dev-002/classifier_lastblock_patch_head/summary.json",
        ),
        (
            "Last block + patch-aware (rnd4)",
            root / "reports/products/screw/screw-full-dev-004/classifier_lastblock_patch_head/summary.json",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, path in sources:
        data = _read_json(_require(path))
        result = data["splits"]["test"] if "splits" in data else data["results"]["test"]
        comparison_name = {
            "Frozen global + linear head": "Frozen global + Logistic Regression",
            "Frozen global + regularized head": "Frozen global + regularized head",
            "Last block + patch-aware head": "Last block + global/patch-aware head",
        }.get(name)
        rows.append(
            {
                "name": name,
                "accuracy": float(result["accuracy"]),
                "defect_recall": float(result.get("defect_recall", result.get("recall_defect"))),
                "auroc": float(result.get("auroc", result.get("image_auroc"))),
                "image_ap": float(result.get("image_average_precision", result.get("image_ap", image_ap_by_name.get(comparison_name, 0.0)))),
                "source": str(path.relative_to(root)),
            }
        )
    return rows


def _heatmap_rows(root: Path) -> list[dict[str, Any]]:
    comparison_path = root / "reports/products/screw/screw-full-dev-001/three_heatmap_comparison.json"
    comparison = _read_json(_require(comparison_path))
    labels = {
        "anomalyclip_baseline": "AnomalyCLIP",
        "relation_map_only": "Relation Map",
        "anomalyclip_plus_relation_map": "AnomalyCLIP + Relation",
        "supervised_patch_refiner": "Supervised refiner",
    }
    rows: list[dict[str, Any]] = []
    for key, label in labels.items():
        metrics = comparison["methods"][key]
        rows.append(
            {
                "name": label,
                "pixel_ap": float(metrics.get("test_pixel_average_precision", 0.0)),
                "pixel_recall": float(metrics["test_pixel_recall"]),
                "pixel_precision": float(metrics["test_pixel_precision"]),
                "good_fp": float(metrics["test_good_false_positive_rate"]),
                "source": str(comparison_path.relative_to(root)),
            }
        )

    decoder_path = root / "reports/products/screw/screw-full-dev-013/spatial_decoder_448_test.json"
    decoder = _read_json(_require(decoder_path))["metrics"]
    rows.append(
        {
            "name": "DINO spatial decoder 448",
            "pixel_ap": float(decoder["pixel_average_precision"]),
            "pixel_recall": float(decoder["pixel_recall"]),
            "pixel_precision": float(decoder["pixel_precision"]),
            "good_fp": float(decoder["good_false_positive_rate"]),
            "source": str(decoder_path.relative_to(root)),
        }
    )
    return rows


def _postprocess_rows(root: Path) -> tuple[list[dict[str, Any]], Path]:
    path = root / "reports/products/screw/screw-full-dev-002/active_refiner_localizer_methods_open7_open9_correct.json"
    methods = _read_json(_require(path))["methods"]
    rows = []
    for key in ("raw", "heatmap_open3", "heatmap_open5", "heatmap_open7", "heatmap_open9"):
        if key in methods:
            rows.append({"name": key.replace("heatmap_", ""), "pixel_ap": float(methods[key]["test"]["pixel_average_precision"])})
    return rows, path


def _detector_rows(root: Path) -> tuple[list[dict[str, Any]], Path]:
    path = root / "reports/experiments/weee_yolo_obb_rtdetr_fair_aabb_holdout_rnd4.json"
    if not path.is_file():
        path = root / "reports/experiments/weee_yolo_obb_rtdetr_fair_aabb_holdout_rnd3.json"
    data = _read_json(_require(path))
    rows = []
    labels = {"yolo_obb_projected_aabb": "YOLO-OBB", "rtdetr_aabb": "RT-DETR"}
    for key, label in labels.items():
        test = data["models"][key]["locked_test"]
        rows.append(
            {
                "name": label,
                "crop_recall": float(test["ground_truth_recall"]),
                "match_precision": float(test["prediction_match_precision"]),
                "latency_ms": float(test["latency_ms_per_image"]),
            }
        )
    return rows, path


def _anomaly_detector_rows(root: Path) -> tuple[list[dict[str, Any]], Path]:
    path = root / "reports/production/screw_comparison_rnd6/mvtec_ad_screw_benchmark/20261122T201103Z/comparison_summary.json"
    data = _read_json(_require(path))
    rows = []
    for name in ("exact_knn", "patchcore", "padim"):
        result = data["detectors"][name]
        rows.append(
            {
                "name": name,
                "image_ap": float(result["image_average_precision"]),
                "pixel_ap": float(result["pixel_average_precision"]),
                "p95_seconds": float(result["latency"]["p95_seconds"]),
            }
        )
    return rows, path


def _threshold_sweep(root: Path) -> tuple[dict[str, Any], Path]:
    reconciled = root / "reports/experiments/screw_classifier_sweep_rnd10/inputs/reconciled_threshold_sweep.json"
    path = reconciled if reconciled.is_file() else root / "reports/products/screw/screw-full-dev-004/classifier_lastblock_patch_head/threshold_sweep.json"
    return _read_json(_require(path)), path


def _plot_classifier(rows: list[dict[str, Any]], path: Path, plt: Any) -> None:
    labels = [row["name"] for row in rows]
    x = list(range(len(labels)))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for offset, key, title, color in [
        (-1.5, "accuracy", "Accuracy", "#4C78A8"),
        (-0.5, "defect_recall", "DEFECT recall", "#F58518"),
        (0.5, "auroc", "AUROC", "#54A24B"),
        (1.5, "image_ap", "Image AP", "#E45756"),
    ]:
        values = [row[key] for row in rows]
        bars = ax.bar([value + offset * width for value in x], values, width, label=title, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_title("Screw GOOD/DEFECT classifier comparison")
    fig.text(0.01, 0.955, "Supervised-development diagnostic; test split was used during development", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_heatmaps(rows: list[dict[str, Any]], path: Path, plt: Any) -> None:
    labels = [row["name"] for row in rows]
    x = list(range(len(labels)))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for offset, key, title, color in [
        (-1.5, "pixel_ap", "Pixel AP", "#4C78A8"),
        (-0.5, "pixel_recall", "Pixel recall", "#F58518"),
        (0.5, "pixel_precision", "Pixel precision", "#54A24B"),
        (1.5, "good_fp", "GOOD-only FP", "#E45756"),
    ]:
        values = [row[key] for row in rows]
        bars = ax.bar([value + offset * width for value in x], values, width, label=title, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score / rate")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_title("Heatmap/localization method comparison")
    fig.text(0.01, 0.955, "Pixel AP alone is insufficient; precision and GOOD false alarms are shown alongside recall", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_postprocess(rows: list[dict[str, Any]], path: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([row["name"] for row in rows], [row["pixel_ap"] for row in rows], marker="o", linewidth=2, color="#4C78A8")
    for index, row in enumerate(rows):
        ax.annotate(f"{row['pixel_ap']:.3f}", (index, row["pixel_ap"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_ylim(0, max(0.7, max(row["pixel_ap"] for row in rows) * 1.18))
    ax.set_ylabel("Pixel AP")
    ax.set_title("Heatmap post-processing sweep")
    fig.text(0.01, 0.955, "Threshold selection is validation-only; test values are diagnostic", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_detectors(rows: list[dict[str, Any]], path: Path, plt: Any) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    labels = [row["name"] for row in rows]
    colors = ["#4C78A8", "#F58518"]
    for ax, key, title, scale in [
        (axes[0], "crop_recall", "Crop recall", 1.0),
        (axes[1], "match_precision", "Matched crop precision", 1.0),
        (axes[2], "latency_ms", "Latency (ms/image)", 1.0),
    ]:
        values = [row[key] * scale for row in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=2)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle("Detector comparison on common AABB crop-coverage protocol", y=1.02)
    fig.text(0.5, -0.02, "YOLO-OBB is projected to AABB; this is not a direct OBB-quality comparison", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_anomaly_detectors(rows: list[dict[str, Any]], path: Path, plt: Any) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    labels = [row["name"] for row in rows]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for axis, key, title, fmt in (
        (axes[0], "image_ap", "Image AP", "%.3f"),
        (axes[1], "pixel_ap", "Pixel AP", "%.3f"),
        (axes[2], "p95_seconds", "p95 latency (s/image)", "%.3f"),
    ):
        values = [row[key] for row in rows]
        bars = axis.bar(labels, values, color=colors)
        axis.bar_label(bars, fmt=fmt, padding=2, fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=18)
    fig.suptitle("Screw anomaly-detector comparison on one locked test split", y=1.02)
    fig.text(0.5, -0.02, "MVTec AD screw supervised-development evidence; MPS; no detector fusion", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_sweep(data: dict[str, Any], path: Path, plt: Any) -> None:
    points = data["points"]
    validation = [(float(point["validation"]["good_fp"]), float(point["validation"]["recall"])) for point in points]
    diagnostic = [(float(point["test_diagnostic"]["good_fp"]), float(point["test_diagnostic"]["recall"])) for point in points]
    selected = data["selected_validation_gate"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([x for x, _ in validation], [y for _, y in validation], marker="o", markersize=3, linewidth=1.8, label="Validation", color="#4C78A8")
    ax.plot([x for x, _ in diagnostic], [y for _, y in diagnostic], marker="o", markersize=3, linewidth=1.8, label="Diagnostic", color="#F58518")
    ax.scatter([selected["validation"]["good_fp"]], [selected["validation"]["recall"]], s=100, marker="*", color="#1D4ED8", zorder=4, label="Selected on validation")
    ax.scatter([selected["test_diagnostic"]["good_fp"]], [selected["test_diagnostic"]["recall"]], s=100, marker="*", color="#C2410C", zorder=4, label="Same threshold on diagnostic")
    ax.axvline(0.20, linestyle="--", color="#98A2B3", linewidth=1.5, label="GOOD-FP cap 20%")
    ax.axhline(0.90, linestyle=":", color="#98A2B3", linewidth=1.5, label="Recall target 90%")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("GOOD false-positive rate")
    ax.set_ylabel("DEFECT recall")
    ax.set_title("DINOv2 classifier threshold sweep")
    fig.text(0.01, 0.955, "Threshold selected on validation; diagnostic curve is not an independent holdout", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_montage(root: Path, path: Path, plt: Any) -> list[str]:
    entries = [
        ("Detector validation prediction", root / "artifacts/products/screw/screw-supervised-dev-001/detector_dataset/training/run/val_batch0_pred.jpg"),
        ("Known-product overview", root / "runs/screw-open-set-candidate-rnd1/candidate_known_good_000/overview/objects_labeled.png"),
        ("Known-product heatmap overlay", root / "runs/screw-open-set-candidate-rnd1/candidate_known_good_000/objects/screw_01/heatmap_overlay.png"),
        ("Known-product status crop", root / "runs/screw-open-set-candidate-rnd1/candidate_known_good_000/objects/screw_01/crop_status.png"),
        ("UNKNOWN-product overview", root / "runs/screw-open-set-candidate-rnd1/candidate_mixed_unknown_003_clipped/overview/objects_labeled.png"),
        ("UNKNOWN-product crop", root / "runs/screw-open-set-candidate-rnd1/candidate_mixed_unknown_003_clipped/objects/screw_01/crop_status.png"),
    ]
    valid = [(label, _require(image)) for label, image in entries]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (label, image) in zip(axes.flat, valid):
        ax.imshow(plt.imread(image))
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle("Qualitative evidence from the image-first inspection pipeline", y=0.98)
    fig.text(0.5, 0.01, "These are smoke/development artefacts; captions do not imply independent accuracy", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [str(image.relative_to(root)) for _, image in valid]


def render(output_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    classifier = _classifier_rows(root)
    heatmaps = _heatmap_rows(root)
    postprocess, postprocess_source = _postprocess_rows(root)
    detectors, detector_source = _detector_rows(root)
    anomaly_detectors, anomaly_detector_source = _anomaly_detector_rows(root)
    threshold_sweep, threshold_source = _threshold_sweep(root)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    try:
        plt = _load_pyplot()
        _plot_classifier(classifier, temp_dir / "classifier_comparison.png", plt)
        _plot_heatmaps(heatmaps, temp_dir / "heatmap_method_comparison.png", plt)
        _plot_postprocess(postprocess, temp_dir / "heatmap_postprocess_sweep.png", plt)
        _plot_detectors(detectors, temp_dir / "detector_coverage_latency.png", plt)
        _plot_anomaly_detectors(anomaly_detectors, temp_dir / "anomaly_detector_comparison.png", plt)
        _plot_threshold_sweep(threshold_sweep, temp_dir / "classifier_threshold_sweep.png", plt)
        qualitative_sources = _plot_montage(root, temp_dir / "qualitative_evidence_montage.png", plt)
        source_paths = {
            row["source"]: root / row["source"] for row in classifier + heatmaps
        }
        source_paths[str(postprocess_source.relative_to(root))] = postprocess_source
        source_paths[str(detector_source.relative_to(root))] = detector_source
        detector_data = _read_json(detector_source)
        for model in detector_data.get("models", {}).values():
            manifest_name = model.get("locked_test", {}).get("crop_manifest", {}).get("manifest_path")
            if manifest_name:
                manifest_path = detector_source.parent / manifest_name
                if manifest_path.is_file():
                    source_paths[str(manifest_path.relative_to(root))] = manifest_path
        source_paths[str(anomaly_detector_source.relative_to(root))] = anomaly_detector_source
        source_paths[str(threshold_source.relative_to(root))] = threshold_source
        source_paths.update({source: root / source for source in qualitative_sources})
        comparison_path = root / "reports/products/screw/screw-full-dev-002/classifier_comparison.md"
        source_paths[str(comparison_path.relative_to(root))] = comparison_path
        manifest = {
            "schema_version": 1,
            "status": "supervised_development_diagnostic",
            "model_run": False,
            "threshold_selection": False,
            "warning": "Figures summarize saved development/diagnostic evidence; no independent target-camera holdout claim is made.",
            "source_files": {source: {"sha256": _sha256(path)} for source, path in sorted(source_paths.items())},
            "generated_files": {},
        }
        for image in sorted(temp_dir.glob("*.png")):
            manifest["generated_files"][image.name] = {"sha256": _sha256(image)}
        (temp_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temp_dir.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/experiments/thesis_figures_rnd1")
    args = parser.parse_args()
    manifest = render((ROOT / args.output_dir).resolve(), ROOT)
    print(json.dumps({"output_dir": str((ROOT / args.output_dir).resolve()), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
