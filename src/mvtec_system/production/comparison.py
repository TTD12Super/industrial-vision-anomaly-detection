"""Run and summarize independent anomaly-detector comparisons."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

from .pipeline import ProductionConfig, run_production


DETECTOR_ORDER = ("exact_knn", "patchcore", "padim")


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _comparison_provenance(summary: dict[str, Any]) -> dict[str, str | None]:
    provided = summary.get("comparison_provenance")
    if provided is not None:
        return dict(provided)
    manifest = summary.get("manifest", {})
    dataset_root = Path(summary.get("dataset_root", ""))
    report_dir = Path(summary.get("report_dir", ""))
    test_manifest = summary.get("test_manifest", {})
    return {
        "classification_manifest_sha256": _sha256_file(Path(manifest["manifest_path"]))
        if manifest.get("manifest_path")
        else None,
        "test_manifest_sha256": test_manifest.get("sha256"),
        "dataset_provenance_sha256": _sha256_file(dataset_root / "provenance.json"),
        "dataset_validation_sha256": _sha256_file(report_dir / "dataset_validation.json"),
    }


def summarize_comparison(
    records: Iterable[dict[str, Any]],
    roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a comparison report from completed independent production runs."""

    records = list(records)
    if not records:
        raise ValueError("At least one completed detector run is required")
    names = [record["detector"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Each detector may appear only once")
    missing = set(DETECTOR_ORDER) - set(names)
    if missing:
        raise ValueError(f"Missing detector runs: {sorted(missing)}")

    first = records[0]["summary"]
    dataset_name = first["dataset_name"]
    test_images = first["classification"]["total_test_images"]
    provenance = [_comparison_provenance(record["summary"]) for record in records]
    provenance_keys = tuple(provenance[0])
    for key in provenance_keys:
        values = [item.get(key) for item in provenance]
        if any(value is not None for value in values) and (
            any(value is None for value in values) or len(set(values)) != 1
        ):
            raise ValueError(f"Detector runs must share the same {key}")
    split_identity_verified = all(
        provenance[0].get(key) is not None
        for key in ("classification_manifest_sha256", "test_manifest_sha256")
    ) and all(
        item.get(key) == provenance[0].get(key)
        for item in provenance
        for key in ("classification_manifest_sha256", "test_manifest_sha256")
    )
    for record in records:
        summary = record["summary"]
        if summary["dataset_name"] != dataset_name:
            raise ValueError("Detector runs must use the same dataset")
        if summary["classification"]["total_test_images"] != test_images:
            raise ValueError("Detector runs must use the same test split")

    detector_results: dict[str, Any] = {}
    for record in records:
        name = record["detector"]
        summary = record["summary"]
        anomaly = summary["anomaly"]
        latency = summary["performance"]["inference"]
        detector_results[name] = {
            "report_dir": summary["report_dir"],
            "product_routing_accuracy": summary["classification"]["top1_accuracy_all"],
            "image_auroc": anomaly["image"]["ranking"]["auroc"],
            "image_average_precision": anomaly["image"]["ranking"]["average_precision"],
            "pixel_auroc": anomaly["pixel"]["ranking"]["auroc"],
            "pixel_average_precision": anomaly["pixel"]["ranking"]["average_precision"],
            "pixel_f1": anomaly["pixel"]["thresholded"].get("f1"),
            "image_thresholded_counts": anomaly.get("image", {}).get("thresholded", {}).get("counts"),
            "pixel_thresholded_counts": anomaly.get("pixel", {}).get("thresholded", {}).get("counts"),
            "latency": latency,
            "run_seconds": record["run_seconds"],
        }

    inference_seconds = sum(
        result["latency"]["total_seconds"] for result in detector_results.values()
    )
    combined = {
        "mode": "sequential_independent_detectors",
        "test_images": test_images,
        "total_inference_seconds": inference_seconds,
        "mean_seconds_per_image": inference_seconds / test_images if test_images else None,
        "official_fusion_enabled": False,
        "fusion_gate": "not_built_until_validation_or_reliability_improves",
    }
    return {
        "report_version": "detector-comparison-v1",
        "dataset_name": dataset_name,
        "test_images": test_images,
        "detectors": detector_results,
        "roles": roles
        or {
            "anomaly_image_baseline": "exact_knn",
            "heatmap_baseline": "patchcore",
        },
        "same_test_split": split_identity_verified,
        "split_identity_verified": split_identity_verified,
        "provenance": provenance[0],
        "combined": combined,
    }


def run_detector_comparison(
    config: ProductionConfig | None = None,
    detectors: Iterable[str] = DETECTOR_ORDER,
    output_root: str | Path = "reports/production/comparison",
) -> dict[str, Any]:
    """Run all independent detectors with one deterministic pipeline contract."""

    config = config or ProductionConfig()
    selected = tuple(detectors)
    if len(selected) != len(set(selected)):
        raise ValueError("Each detector may appear only once")
    if set(selected) != set(DETECTOR_ORDER):
        raise ValueError(f"Comparison requires exactly {DETECTOR_ORDER}")
    output_root = Path(output_root)
    run_root = output_root / "runs"
    records = []
    for detector in DETECTOR_ORDER:
        started = time.perf_counter()
        summary = run_production(
            ProductionConfig(
                datasets_root=config.datasets_root,
                dataset_root=config.dataset_root,
                processed_root=config.processed_root,
                report_root=run_root,
                artifact_root=config.artifact_root,
                image_size=config.image_size,
                validation_fraction=config.validation_fraction,
                seed=config.seed,
                batch_size=config.batch_size,
                k=config.k,
                detector=detector,
                orchestration="independent",
                detector_layers=config.detector_layers,
                neighborhood_size=config.neighborhood_size,
                coreset_ratio=config.coreset_ratio,
                coreset_max_points=config.coreset_max_points,
                padim_reduced_dim=config.padim_reduced_dim,
                covariance_regularization=config.covariance_regularization,
                target_acceptance_rate=config.target_acceptance_rate,
                classification_confidence_threshold=config.classification_confidence_threshold,
                device=config.device,
                artifact_namespace=detector,
                force=config.force,
            )
        )
        records.append(
            {
                "detector": detector,
                "summary": summary,
                "run_seconds": time.perf_counter() - started,
            }
        )

    comparison = summarize_comparison(records)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = output_root / comparison["dataset_name"] / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison["report_dir"] = str(report_dir.resolve())
    (report_dir / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "comparison_report.md").write_text(
        _markdown_report(comparison), encoding="utf-8"
    )
    return comparison


def _markdown_report(comparison: dict[str, Any]) -> str:
    lines = [
        f"# Detector comparison: `{comparison['dataset_name']}`",
        "",
        "All three detectors were run independently on the same deterministic test split.",
        "",
        "| Detector | Product routing | Image AUROC | Pixel AUROC | Pixel AP | Pixel F1 | Mean s/image | p95 s/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DETECTOR_ORDER:
        result = comparison["detectors"][name]
        latency = result["latency"]
        lines.append(
            f"| {name} | {result['product_routing_accuracy']:.4f} | "
            f"{result['image_auroc']:.4f} | {result['pixel_auroc']:.4f} | "
            f"{result['pixel_average_precision']:.4f} | {result['pixel_f1']:.4f} | "
            f"{latency['mean_seconds']:.4f} | {latency['p95_seconds']:.4f} |"
        )
    combined = comparison["combined"]
    lines.extend(
        [
            "",
            f"- Image anomaly baseline: **{comparison['roles']['anomaly_image_baseline']}**",
            f"- Heatmap baseline: **{comparison['roles']['heatmap_baseline']}**",
            f"- Combined mode: **{combined['mode']}**",
            f"- Combined inference time: **{combined['total_inference_seconds']:.3f} s**",
            "- Official fusion: **not enabled; requires a measured validation/reliability gain**",
            "",
        ]
    )
    return "\n".join(lines)
