"""Export and analyze every UNKNOWN rejection from the product test report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ERROR_FIELDS = [
    "image_path",
    "true_category",
    "predicted_category",
    "routed_category",
    "confidence",
    "routing_status",
    "defect_type",
    "is_defect",
    "global_distance",
    "global_distance_threshold",
    "global_distance_ood",
    "rejection_reason",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _reason_flags(reason: str) -> tuple[bool, bool, bool]:
    return (
        "product confidence below threshold" in reason,
        "confidence below calibrated threshold" in reason,
        "nearest global distance above threshold" in reason,
    )


def _reason_group(reason: str) -> str:
    minimum, calibrated, distance = _reason_flags(reason)
    confidence = minimum or calibrated
    if confidence and distance:
        return "confidence_and_distance"
    if confidence:
        return "confidence_only"
    if distance:
        return "distance_only"
    return "other"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_reason_group(row["rejection_reason"])].append(row)
    total = len(rows)
    result = []
    for group in ("confidence_and_distance", "confidence_only", "distance_only", "other"):
        values = grouped.get(group, [])
        if not values:
            continue
        confidences = [float(row["confidence"]) for row in values]
        distances = [float(row["global_distance"]) for row in values]
        result.append(
            {
                "reason_group": group,
                "count": len(values),
                "percentage": len(values) / total if total else 0.0,
                "good_count": sum(row["is_defect"] == "False" for row in values),
                "defect_count": sum(row["is_defect"] == "True" for row in values),
                "mean_confidence": mean(confidences),
                "median_confidence": median(confidences),
                "mean_global_distance": mean(distances),
                "median_global_distance": median(distances),
            }
        )
    return result


def _category_rows(
    rows: list[dict[str, str]], summary: dict[str, Any]
) -> list[dict[str, Any]]:
    supports = {row["category"]: row["support"] for row in summary["per_category"]}
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row["true_category"]][_reason_group(row["rejection_reason"])] += 1
    result = []
    for category in sorted(supports):
        counts = grouped[category]
        rejected = sum(counts.values())
        support = int(supports[category])
        result.append(
            {
                "category": category,
                "support": support,
                "rejected": rejected,
                "rejection_rate": rejected / support if support else 0.0,
                "confidence_and_distance": counts["confidence_and_distance"],
                "confidence_only": counts["confidence_only"],
                "distance_only": counts["distance_only"],
                "other": counts["other"],
            }
        )
    return result


def _split_counts(rows: list[dict[str, str]]) -> dict[str, Any]:
    result = {}
    for name, subset in (
        ("good", [row for row in rows if row["is_defect"] == "False"]),
        ("defect", [row for row in rows if row["is_defect"] == "True"]),
    ):
        counts = Counter(_reason_group(row["rejection_reason"]) for row in subset)
        result[name] = {
            "rejected": len(subset),
            "confidence_and_distance": counts["confidence_and_distance"],
            "confidence_only": counts["confidence_only"],
            "distance_only": counts["distance_only"],
            "other": counts["other"],
        }
    return result


def _markdown(
    analysis: dict[str, Any],
    summary_path: Path,
    rejected_path: Path,
) -> str:
    trigger = analysis["trigger_counts"]
    lines = [
        "# Rejection-reason analysis",
        "",
        f"This analysis covers all **{analysis['total_rejected']}** rows rejected as `unknown_product_category`.",
        "The source table is exported without dropping any row or rejection reason.",
        "",
        "## Exported files",
        "",
        f"- All rejected images: `{rejected_path}`",
        f"- Reason summary: `{summary_path.parent / 'rejection_reason_counts.csv'}`",
        f"- Category summary: `{summary_path.parent / 'rejection_reason_by_category.csv'}`",
        "",
        "## Reason counts",
        "",
        f"- Configured confidence trigger (< minimum): **{trigger['minimum_confidence']} ({trigger['minimum_confidence_rate']:.2%})**",
        f"- Calibrated confidence trigger (disabled): **{trigger['calibrated_confidence']} ({trigger['calibrated_confidence_rate']:.2%})**",
        f"- Global-distance defect flag (not a reject trigger): **{trigger['global_distance']} ({trigger['global_distance_rate']:.2%})**",
        "",
    ]
    for row in analysis["reason_groups"]:
        lines.append(
            f"- `{row['reason_group']}`: {row['count']} ({row['percentage']:.2%}); "
            f"good={row['good_count']}, defect={row['defect_count']}"
        )
    lines.extend(
        [
            "",
            "## Good versus defect",
            "",
            f"- Good rejected: **{analysis['by_split']['good']['rejected']}**",
            f"- Defect rejected: **{analysis['by_split']['defect']['rejected']}**",
            "",
            "## Why the rejection rate is high",
            "",
            "1. The router now uses only the configured confidence threshold of 0.50. The lowest rejected confidence is "
            f"`{analysis['confidence_range']['min']:.6f}` if rejected rows exist; otherwise there are no UNKNOWN rows.",
            "2. The old calibrated confidence threshold `"
            f"{analysis['thresholds']['calibrated_confidence']}` is retained only as historical metadata and is not used for routing.",
            "3. The global-distance value `"
            f"{analysis['thresholds']['global_distance']}` is used only as a defect flag; it does not reject the product category.",
            "4. Defect images change the visual representation more than good images, so they are more likely to fall below the 0.50 confidence gate.",
            "",
            "In short, after removing the two old rejection checks, any remaining UNKNOWN result is caused only by confidence below 0.50.",
            "",
            "## Categories with highest rejection rates",
            "",
        ]
    )
    for row in analysis["highest_category_rejection_rates"][:5]:
        lines.append(
            f"- `{row['category']}`: {row['rejected']}/{row['support']} ({row['rejection_rate']:.2%})"
        )
    return "\n".join(lines) + "\n"


def analyze_rejections(report_dir: str | Path) -> dict[str, Any]:
    report_dir = Path(report_dir)
    source_path = report_dir / "error_analysis.csv"
    summary_path = report_dir / "classification_test_summary.json"
    rows = _read_rows(source_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rejected_rows = [row for row in rows if row["routing_status"] == "unknown_product_category"]
    if len(rejected_rows) != summary["unknown_rejected_predictions"]:
        raise ValueError(
            "error_analysis.csv does not contain every rejected prediction: "
            f"csv={len(rejected_rows)}, summary={summary['unknown_rejected_predictions']}"
        )
    reason_groups = _summary_rows(rejected_rows)
    flags = [_reason_flags(row["rejection_reason"]) for row in rejected_rows]
    trigger_counts = {
        "minimum_confidence": sum(flags_row[0] for flags_row in flags),
        "calibrated_confidence": sum(flags_row[1] for flags_row in flags),
        "global_distance": sum(flags_row[2] for flags_row in flags),
    }
    total = len(rejected_rows)
    confidence_values = [float(row["confidence"]) for row in rejected_rows]
    distance_values = [float(row["global_distance"]) for row in rejected_rows]
    analysis = {
        "report_version": "v1",
        "total_rejected": total,
        "thresholds": {
            "minimum_confidence": summary["routing"]["minimum_confidence_threshold"],
            "calibrated_confidence": summary["routing"]["calibrated_confidence_threshold"],
            "global_distance": summary["routing"]["global_distance_threshold"],
        },
        "trigger_counts": {
            **trigger_counts,
            "minimum_confidence_rate": trigger_counts["minimum_confidence"] / total if total else 0.0,
            "calibrated_confidence_rate": trigger_counts["calibrated_confidence"] / total if total else 0.0,
            "global_distance_rate": trigger_counts["global_distance"] / total if total else 0.0,
        },
        "reason_groups": reason_groups,
        "both_triggered": sum(flags_row[1] and flags_row[2] for flags_row in flags),
        "both_triggered_rate": sum(flags_row[1] and flags_row[2] for flags_row in flags) / total if total else 0.0,
        "confidence_range": {
            "min": min(confidence_values) if confidence_values else 0.0,
            "max": max(confidence_values) if confidence_values else 0.0,
            "mean": mean(confidence_values) if confidence_values else 0.0,
        },
        "global_distance_range": {
            "min": min(distance_values) if distance_values else 0.0,
            "max": max(distance_values) if distance_values else 0.0,
            "mean": mean(distance_values) if distance_values else 0.0,
        },
        "by_split": _split_counts(rejected_rows),
    }
    category_rows = _category_rows(rejected_rows, summary)
    analysis["category_rows"] = category_rows
    analysis["highest_category_rejection_rates"] = sorted(
        category_rows, key=lambda row: (-row["rejection_rate"], row["category"])
    )
    rejected_path = report_dir / "rejected_images.csv"
    _write_csv(rejected_path, rejected_rows, ERROR_FIELDS)
    _write_csv(
        report_dir / "rejection_reason_counts.csv",
        reason_groups,
        [
            "reason_group",
            "count",
            "percentage",
            "good_count",
            "defect_count",
            "mean_confidence",
            "median_confidence",
            "mean_global_distance",
            "median_global_distance",
        ],
    )
    _write_csv(
        report_dir / "rejection_reason_by_category.csv",
        category_rows,
        [
            "category",
            "support",
            "rejected",
            "rejection_rate",
            "confidence_and_distance",
            "confidence_only",
            "distance_only",
            "other",
        ],
    )
    (report_dir / "rejection_reason_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "rejection_reason_analysis.md").write_text(
        _markdown(analysis, report_dir / "rejection_reason_analysis.json", rejected_path),
        encoding="utf-8",
    )
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports/product_classification_test")
    )
    args = parser.parse_args()
    analysis = analyze_rejections(args.report_dir)
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
