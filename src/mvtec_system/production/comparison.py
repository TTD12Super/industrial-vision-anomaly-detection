"""Run and summarize independent anomaly-detector comparisons."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable

from .pipeline import ProductionConfig, run_production


DETECTOR_ORDER = ("exact_knn", "patchcore", "padim")


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
            "classification_accuracy": summary["classification"]["top1_accuracy_all"],
            "image_auroc": anomaly["image"]["ranking"]["auroc"],
            "image_average_precision": anomaly["image"]["ranking"]["average_precision"],
            "pixel_auroc": anomaly["pixel"]["ranking"]["auroc"],
            "pixel_average_precision": anomaly["pixel"]["ranking"]["average_precision"],
            "pixel_f1": anomaly["pixel"].get("thresholded", {}).get("f1"),
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
        "detectors": detector_results,
        "roles": roles
        or {
            "anomaly_image_baseline": "exact_knn",
            "heatmap_baseline": "patchcore",
        },
        "same_test_split": True,
        "combined": combined,
    }


def run_detector_comparison(
    config: ProductionConfig | None = None,
    detectors: Iterable[str] = DETECTOR_ORDER,
    output_root: str | Path = "reports/production/comparison",
) -> dict[str, Any]:
    """Run all independent detectors with one deterministic pipeline contract."""

    config = config or ProductionConfig()
    selected = tuple(dict.fromkeys(detectors))
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
                heatmap_refiner=config.heatmap_refiner,
                affinity_top_k=config.affinity_top_k,
                affinity_temperature=config.affinity_temperature,
                affinity_eta=config.affinity_eta,
                mask_postprocess=config.mask_postprocess,
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
        "| Detector | Classification | Image AUROC | Pixel AUROC | Pixel AP | Pixel F1 | Mean s/image | p95 s/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DETECTOR_ORDER:
        result = comparison["detectors"][name]
        latency = result["latency"]
        pixel_f1 = "n/a" if result["pixel_f1"] is None else f"{result['pixel_f1']:.4f}"
        lines.append(
            f"| {name} | {result['classification_accuracy']:.4f} | "
            f"{result['image_auroc']:.4f} | {result['pixel_auroc']:.4f} | "
            f"{result['pixel_average_precision']:.4f} | {pixel_f1} | "
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
