"""Validation-only GOOD/DEFECT operating-point selection."""

from __future__ import annotations

from typing import Any


def select_operating_point(
    points: list[dict[str, Any]], target_recall: float, max_good_fp: float
) -> dict[str, Any]:
    """Select recall-first threshold candidates without reading diagnostic metrics."""

    eligible = [
        point
        for point in points
        if float(point["validation"]["recall"]) >= target_recall
        and float(point["validation"]["good_fp"]) <= max_good_fp
    ]
    if not eligible:
        return {
            "status": "no_validation_gate",
            "target_defect_recall": target_recall,
            "max_good_fp": max_good_fp,
            "selected": None,
            "candidate_count": 0,
        }
    selected = max(
        eligible,
        key=lambda point: (
            float(point["validation"]["recall"]),
            -float(point["validation"]["good_fp"]),
            float(point["threshold"]),
        ),
    )
    return {
        "status": "selected",
        "target_defect_recall": target_recall,
        "max_good_fp": max_good_fp,
        "selected": selected,
        "candidate_count": len(eligible),
        "selection_rule": "maximize validation defect recall, then minimize validation GOOD FP, then maximize threshold",
    }
