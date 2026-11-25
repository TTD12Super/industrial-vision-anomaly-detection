#!/usr/bin/env python3
"""Select a GOOD/DEFECT threshold from validation-only sweep metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspection.classifier_sweep import select_operating_point


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-defect-recall", type=float, default=0.90)
    parser.add_argument("--max-good-fp", type=float, default=0.20)
    args = parser.parse_args()
    payload = json.loads(args.sweep.read_text(encoding="utf-8"))
    result = {
        "schema_version": "1.0",
        "sweep": str(args.sweep.resolve()),
        "manifest": payload.get("manifest"),
        "model": payload.get("model"),
        "policy": select_operating_point(payload["points"], args.target_defect_recall, args.max_good_fp),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
