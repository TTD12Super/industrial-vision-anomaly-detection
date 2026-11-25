#!/usr/bin/env python3
"""Benchmark a local VLM against saved screw inspection evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from inspection.reporting import LocalVLMReporter, validate_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mime_type(path: Path) -> str:
    return "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"


def _evidence_paths(root: Path, object_id: str) -> list[Path]:
    candidates = sorted((root / "input").glob("original.*"))
    candidates.extend(
        root / "objects" / object_id / name
        for name in ("crop_original.png", "heatmap_overlay.png", "binary_mask.png")
    )
    paths = [path for path in candidates if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No VLM evidence exists for {object_id} under {root}")
    return paths


def _context(inspection: dict[str, Any], item: dict[str, Any], evidence: list[Path]) -> dict[str, Any]:
    return {
        "inspection_id": inspection["inspection_id"],
        "object_id": item["object_id"],
        "product_key": inspection["product_key"],
        "classifier_label": item.get("classifier_label"),
        "p_defect": item.get("p_defect"),
        "decision_reason": item.get("decision_reason"),
        "localization_status": item.get("localization_status"),
        "heatmap_path": item.get("heatmap_overlay"),
        "binary_mask_path": item.get("binary_mask"),
        "_evidence": [
            {
                "mime_type": _mime_type(path),
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
            for path in evidence
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    reporter = LocalVLMReporter(args.base_url, args.model, args.timeout_seconds, args.max_tokens)
    inspection_paths = [path.resolve() for path in args.inspection_json]
    results: list[dict[str, Any]] = []
    eligible_objects = 0
    for inspection_path in inspection_paths:
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        root = inspection_path.parent
        candidates = [
            item for item in inspection.get("objects", [])
            if item.get("operational_status") == "DEFECT"
        ][: args.max_objects]
        eligible_objects += len(candidates)
        for item in candidates:
            object_id = str(item["object_id"])
            evidence = _evidence_paths(root, object_id)
            context = _context(inspection, item, evidence)
            started = time.perf_counter()
            try:
                raw = reporter.generate(context)
                report = validate_report(raw, context)
                status = "completed"
                diagnostic = None
            except Exception as error:
                report = None
                status = "failed"
                diagnostic = f"{type(error).__name__}: {error}"
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            results.append(
                {
                    "inspection_json": str(inspection_path),
                    "inspection_sha256": _sha256(inspection_path),
                    "object_id": object_id,
                    "status": status,
                    "latency_ms": latency_ms,
                    "evidence": [
                        {"path": str(path), "sha256": _sha256(path), "mime_type": _mime_type(path)}
                        for path in evidence
                    ],
                    "report": report,
                    "diagnostic": diagnostic,
                }
            )
    latencies = [item["latency_ms"] for item in results if item["status"] == "completed"]
    output = {
        "schema_version": "1.0",
        "protocol": "saved deterministic inspection evidence -> local VLM -> validated English report",
        "inspection_jsons": [str(path) for path in inspection_paths],
        "model": args.model,
        "reporter_model_version": reporter.model_version,
        "base_url": args.base_url,
        "eligible_defect_objects": eligible_objects,
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "latency_ms": {
            "count": len(latencies),
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "objects": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection-json", type=Path, action="append", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen3-vl-4b")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.max_tokens < 32 or args.max_objects < 1:
        parser.error("timeout, max-tokens and max-objects must be positive")
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
