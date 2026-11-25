#!/usr/bin/env python3
"""Attach object-level ground-truth matches to a saved local-VLM benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from inspection.evaluation import polygon_iou


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    vlm_path = args.vlm_report.resolve()
    e2e_path = args.e2e_report.resolve()
    payload = json.loads(vlm_path.read_text(encoding="utf-8"))
    e2e = json.loads(e2e_path.read_text(encoding="utf-8"))
    per_image = {row["inspection_id"]: row for row in e2e["per_image"]}
    truths_by_image: dict[str, list[dict[str, Any]]] = {}
    with args.object_manifest.resolve().open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            truths_by_image.setdefault(row["image_id"], []).append(row)

    audited: list[dict[str, Any]] = []
    for item in payload["objects"]:
        inspection_path = Path(item["inspection_json"])
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        inspection_id = inspection["inspection_id"]
        image_id = per_image[inspection_id]["image_id"]
        prediction = next(row for row in inspection["objects"] if row["object_id"] == item["object_id"])
        candidates = [
            (
                polygon_iou(truth["obb_xyxyxyxy"], prediction["obb_xyxyxyxy"]),
                truth,
            )
            for truth in truths_by_image[image_id]
        ]
        best_iou, truth = max(candidates, key=lambda pair: pair[0])
        matched = best_iou >= args.iou_threshold
        audited.append(
            {
                "inspection_json": str(inspection_path),
                "inspection_id": inspection_id,
                "image_id": image_id,
                "object_id": item["object_id"],
                "ground_truth_object_id": truth["object_id"] if matched else None,
                "ground_truth_quality_label": truth["quality_label"] if matched else None,
                "obb_iou": float(best_iou),
                "matched": matched,
                "vlm_status": item["status"],
                "vlm_visible_defect_type": (item.get("report") or {}).get("visible_defect_type"),
                "vlm_heatmap_consistency": (item.get("report") or {}).get("heatmap_consistent_with_visible_evidence"),
                "vlm_needs_review": (item.get("report") or {}).get("needs_review"),
                "vlm_latency_ms": item["latency_ms"],
            }
        )
    matched = [row for row in audited if row["matched"]]
    output = {
        "schema_version": "1.0",
        "protocol": "local-VLM benchmark joined to object ground truth by maximum OBB IoU; exploratory report only",
        "vlm_report": str(vlm_path),
        "vlm_report_sha256": _sha256(vlm_path),
        "e2e_report": str(e2e_path),
        "e2e_report_sha256": _sha256(e2e_path),
        "object_manifest": str(args.object_manifest.resolve()),
        "iou_threshold": args.iou_threshold,
        "audited_objects": len(audited),
        "matched_objects": len(matched),
        "summary": {
            "matched_good": sum(row["ground_truth_quality_label"] == "GOOD" for row in matched),
            "matched_defect": sum(row["ground_truth_quality_label"] == "DEFECT" for row in matched),
            "defect_reports_with_visible_type": sum(
                row["ground_truth_quality_label"] == "DEFECT"
                and row["vlm_visible_defect_type"] not in {None, "unknown"}
                for row in matched
            ),
            "good_reports_needing_review": sum(
                row["ground_truth_quality_label"] == "GOOD" and row["vlm_needs_review"] is True
                for row in matched
            ),
        },
        "objects": audited,
        "warning": "This small sample audits report grounding, not VLM accuracy or production readiness.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-report", type=Path, required=True)
    parser.add_argument("--e2e-report", type=Path, required=True)
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be in (0, 1]")
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
