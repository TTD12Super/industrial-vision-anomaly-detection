"""Run the complete dataset-driven industrial anomaly pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvtec_system.production import ProductionConfig, run_production


def _layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers:
        raise argparse.ArgumentTypeError("layers must contain at least one block number")
    return layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--report-root", type=Path, default=Path("reports/production"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/production"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--layers", type=_layers, default=(5, 8, 11))
    parser.add_argument("--neighborhood-size", type=int, default=3)
    parser.add_argument("--coreset-ratio", type=float, default=0.10)
    parser.add_argument("--coreset-max-points", type=int, default=50000)
    parser.add_argument("--padim-reduced-dim", type=int, default=100)
    parser.add_argument("--covariance-regularization", type=float, default=0.01)
    parser.add_argument("--mask-min-component-area", type=int, default=9)
    parser.add_argument("--mask-opening-kernel-size", type=int, default=3)
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda; auto-detect when omitted")
    parser.add_argument("--force", action="store_true", help="rebuild processed images and compatible artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_production(
        ProductionConfig(
            datasets_root=args.datasets_root,
            dataset_root=args.dataset_root,
            processed_root=args.processed_root,
            report_root=args.report_root,
            artifact_root=args.artifact_root,
            image_size=args.image_size,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            batch_size=args.batch_size,
            k=args.k,
            detector_layers=args.layers,
            neighborhood_size=args.neighborhood_size,
            coreset_ratio=args.coreset_ratio,
            coreset_max_points=args.coreset_max_points,
            padim_reduced_dim=args.padim_reduced_dim,
            covariance_regularization=args.covariance_regularization,
            mask_min_component_area=args.mask_min_component_area,
            mask_opening_kernel_size=args.mask_opening_kernel_size,
            device=args.device,
            force=args.force,
        )
    )
    print(json.dumps({"report_dir": summary["report_dir"], "final_report": str(Path(summary["report_dir"]) / "final_report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
