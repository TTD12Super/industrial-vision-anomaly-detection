#!/usr/bin/env python3
"""Run the reusable egg detection + anomaly demo CLI."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egg_demo.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
