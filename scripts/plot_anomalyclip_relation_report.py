#!/usr/bin/env python3
"""Render reproducible comparison charts from an AnomalyCLIP relation summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCHCORE_SUMMARY = PROJECT_ROOT / "reports/experiments/goal_15_category_heatmap_methods_320/comparison_summary.json"
PATCHCORE_CSV = PROJECT_ROOT / "reports/experiments/goal_15_category_heatmap_methods_320/per_category_comparison.csv"
METRICS = (
    ("pixel_ap", "Macro Pixel AP"),
    ("pixel_auroc", "Macro Pixel AUROC"),
    ("f1", "Macro F1 at validation q=.99"),
    ("good_fp", "Macro GOOD false-positive pixel rate"),
)


def _validate(summary: dict[str, Any]) -> tuple[list[str], list[str]]:
    categories = summary.get("categories")
    metrics = summary.get("metrics", {})
    methods = list(metrics.get("per_category", {}))
    if not isinstance(categories, list) or len(categories) != 15 or len(set(categories)) != 15:
        raise ValueError("summary must contain 15 unique categories")
    if len(methods) < 2 or set(methods) != set(metrics.get("macro", {})):
        raise ValueError("summary must have matching per-category and macro method results")
    for method in methods:
        rows = metrics["per_category"][method]
        if len(rows) != len(categories) or {row.get("category") for row in rows} != set(categories):
            raise ValueError(f"{method} does not have one row for every category")
        for row in [metrics["macro"][method], *rows]:
            for key, _ in METRICS:
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{method}.{key} must be a finite metric in [0, 1]")
        for key, _ in METRICS:
            macro_value = float(metrics["macro"][method][key])
            category_mean = sum(float(row[key]) for row in rows) / len(rows)
            if not math.isclose(macro_value, category_mean, rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError(f"{method}.{key} macro value does not match the mean of its category values")
    return categories, methods


def _patchcore_reference(summary: dict[str, Any]) -> dict[str, Any]:
    reference = json.loads(PATCHCORE_SUMMARY.read_text(encoding="utf-8"))
    patchcore = reference["methods"]["raw_patchcore"]
    manifest_sha = summary.get("provenance", {}).get("manifest_sha256")
    if manifest_sha != patchcore.get("manifest_sha256"):
        raise ValueError("AnomalyCLIP and PatchCore references do not use the same split manifest")
    with PATCHCORE_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_categories = set(summary.get("categories", []))
    category_rows = [row for row in rows if row.get("category") in expected_categories]
    if len(category_rows) != 15 or {row["category"] for row in category_rows} != expected_categories:
        raise ValueError("PatchCore reference CSV must contain one Pixel AP row for each of 15 categories")
    per_category = {
        row["category"]: _checked_metric(row["raw_patchcore_pixel_ap"], f"PatchCore {row['category']} Pixel AP")
        for row in category_rows
    }
    per_category_auroc = {
        row["category"]: _checked_metric(row["raw_patchcore_pixel_auroc"], f"PatchCore {row['category']} Pixel AUROC")
        for row in category_rows
    }
    raw_summary = reference["methods"]["raw_patchcore"]
    macro_pixel_ap = _checked_metric(raw_summary["pixel_ap_macro_15"], "PatchCore macro Pixel AP")
    macro_pixel_auroc = _checked_metric(raw_summary["pixel_auroc_macro_15"], "PatchCore macro Pixel AUROC")
    _check_macro(macro_pixel_ap, per_category.values(), "PatchCore Pixel AP")
    _check_macro(macro_pixel_auroc, per_category_auroc.values(), "PatchCore Pixel AUROC")
    return {
        "macro_pixel_ap": macro_pixel_ap,
        "macro_pixel_auroc": macro_pixel_auroc,
        "per_category_pixel_ap": per_category,
        "manifest_sha256": manifest_sha,
    }


def _checked_metric(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return number


def _check_macro(macro: float, values: Any, label: str) -> None:
    category_mean = sum(values) / 15
    if not math.isclose(macro, category_mean, rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError(f"{label} summary value does not match the mean of its 15 category values")


def _render(summary: dict[str, Any], patchcore: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    categories, methods = _validate(summary)
    metrics = summary["metrics"]
    labels = [method.replace("_", " ").title() for method in methods]
    colors = ("#2563eb", "#f97316", "#16a34a", "#9333ea")
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (key, title) in zip(axes.flat, METRICS):
        values = [float(metrics["macro"][method][key]) * 100 for method in methods]
        bars = axis.bar(labels, values, color=colors[: len(methods)], width=0.62)
        axis.set_title(title)
        axis.set_ylabel("Percent")
        axis.set_ylim(0, max(5, max(values, default=0) * 1.18))
        axis.tick_params(axis="x", labelrotation=15)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    figure.suptitle("AnomalyCLIP relation-map ablation — exploratory MVTec test", fontsize=13)
    figure.tight_layout()
    overview = output_dir / "metrics_overview.png"
    figure.savefig(overview, dpi=160, bbox_inches="tight")
    plt.close(figure)

    y = np.arange(len(categories))
    all_methods = methods + ["raw_patchcore_reference"]
    all_labels = labels + ["Raw PatchCore (prior reference)"]
    width = 0.78 / len(all_methods)
    figure, axis = plt.subplots(figsize=(11, 8))
    for index, (method, label) in enumerate(zip(all_methods, all_labels)):
        if method == "raw_patchcore_reference":
            values = [float(patchcore["per_category_pixel_ap"][category]) * 100 for category in categories]
        else:
            rows = {row["category"]: row for row in metrics["per_category"][method]}
            values = [float(rows[category]["pixel_ap"]) * 100 for category in categories]
        offset = (index - (len(all_methods) - 1) / 2) * width
        axis.barh(y + offset, values, height=width, label=label, color=colors[index])
    axis.set_yticks(y, categories)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Pixel AP (%)")
    axis.set_title("Pixel AP by category")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    per_category = output_dir / "pixel_ap_by_category.png"
    figure.savefig(per_category, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return [overview.name, per_category.name]


def _write_report(summary_path: Path, output_dir: Path, patchcore: dict[str, Any] | None = None) -> list[str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    readme_path = output_dir / "README.md"
    if not readme_path.is_file():
        raise FileNotFoundError(f"benchmark report README is missing: {readme_path}")
    patchcore = patchcore or _patchcore_reference(summary)
    names = _render(summary, patchcore, output_dir)
    summary["visualizations"] = {
        "script": "scripts/plot_anomalyclip_relation_report.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": names,
        "patchcore_resize_comparison": "indicative_only: AnomalyCLIP 518-to-320 uses Pillow BILINEAR; the PatchCore reference uses PyTorch bilinear align_corners=False on its padded map",
    }
    runtime_path = output_dir / "runtime_environment.json"
    if runtime_path.is_file():
        summary["runtime_environment_file"] = runtime_path.name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = readme_path.read_text(encoding="utf-8")
    section = (
        "\n## Metric charts\n\n![Macro metrics](metrics_overview.png)\n\n"
        "![Pixel AP by category](pixel_ap_by_category.png)\n\n"
        "**PatchCore comparison caveat:** these bars are indicative only. AnomalyCLIP maps are resized 518→320 with Pillow `BILINEAR`; the PatchCore reference uses PyTorch bilinear interpolation with `align_corners=False` on its padded map.\n"
    )
    if runtime_path.is_file():
        section += "\n## Execution environment\n\nSee [`runtime_environment.json`](runtime_environment.json) for the captured Python, PyTorch, TorchVision, device, model-weight and manifest fingerprints.\n"
    if "## Metric charts" not in readme:
        readme_path.write_text(readme.rstrip() + "\n" + section, encoding="utf-8")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, nargs="?")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        categories = [f"category_{index}" for index in range(15)]
        methods = ("raw", "relation", "fusion")
        summary = {
            "categories": categories,
            "metrics": {
                "macro": {method: {key: 0.5 for key, _ in METRICS} for method in methods},
                "per_category": {
                    method: [{"category": category, **{key: 0.5 for key, _ in METRICS}} for category in categories]
                    for method in methods
                },
            },
        }
        inconsistent = json.loads(json.dumps(summary))
        inconsistent["metrics"]["macro"]["raw"]["pixel_ap"] = 0.6
        try:
            _validate(inconsistent)
        except ValueError:
            pass
        else:
            raise RuntimeError("macro/category consistency check failed to reject a mismatch")
        try:
            _check_macro(0.6, [0.5] * 15, "PatchCore Pixel AP")
        except ValueError:
            pass
        else:
            raise RuntimeError("PatchCore reference check failed to reject a macro mismatch")
        try:
            _checked_metric(float("nan"), "PatchCore Pixel AP")
        except ValueError:
            pass
        else:
            raise RuntimeError("PatchCore reference check failed to reject a non-finite metric")
        patchcore = {
            "macro_pixel_ap": 0.5,
            "macro_pixel_auroc": 0.5,
            "per_category_pixel_ap": {category: 0.5 for category in categories},
            "manifest_sha256": "self-check",
        }
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            (Path(directory) / "README.md").write_text("# synthetic\n", encoding="utf-8")
            names = _write_report(summary_path, Path(directory), patchcore)
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            readme = (Path(directory) / "README.md").read_text(encoding="utf-8")
            if len(names) != 2 or any((Path(directory) / name).stat().st_size < 1000 for name in names):
                raise RuntimeError("chart self-check failed")
            if saved["visualizations"]["files"] != names or "![Pixel AP by category]" not in readme:
                raise RuntimeError("report integration self-check failed")
        print("SELF-CHECK PASS: macro/category guard, 15 categories, 3 methods, report and charts")
        return 0
    if args.summary is None:
        parser.error("summary is required unless --self-check is used")
    summary_path = args.summary.resolve(strict=True)
    names = _write_report(summary_path, summary_path.parent)
    print(json.dumps({"summary": str(summary_path), "charts": names}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
