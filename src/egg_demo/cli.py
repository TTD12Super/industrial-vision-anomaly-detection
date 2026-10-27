"""Command-line entry points for the egg demo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .anomaly import PaDiMReference, build_padim_reference, evaluate_image_level
from .config import EggDemoConfig
from .data import inspect_quality_archive, prepare_freshness_dataset, prepare_quality_dataset, validate_yolo_dataset, write_local_yolo_yaml
from .detection import YoloEggDetector
from .pipeline import EggAnomalyPipeline
from .production import build_egg_production
from .visualization import draw_results


def _config(path: str | None) -> EggDemoConfig:
    return EggDemoConfig.from_json(path) if path else EggDemoConfig()


def _paths_from_manifest(root: Path, split: str, label: str = "good") -> list[Path]:
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return [root / row["image_path"] for row in csv.DictReader(handle) if row["split"] == split and row["label"] == label]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--config")
    inspect.add_argument("--output", type=Path)

    prepare = sub.add_parser("prepare-quality")
    prepare.add_argument("--config")
    prepare.add_argument("--archive", type=Path)
    prepare.add_argument("--output-root", type=Path)
    prepare.add_argument("--image-size", type=int, default=256)
    prepare.add_argument("--seed", type=int, default=42)

    freshness = sub.add_parser("prepare-freshness")
    freshness.add_argument("--archive", type=Path, required=True)
    freshness.add_argument("--output-root", type=Path, required=True)
    freshness.add_argument("--image-size", type=int, default=256)
    freshness.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train-yolo")
    train.add_argument("--config")
    train.add_argument("--dataset-root", type=Path)
    train.add_argument("--weights", default="yolo11n.pt")
    train.add_argument("--output-root", type=Path, default=Path("artifacts/egg_demo/yolo"))
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=4)
    train.add_argument("--device", default=None)

    yolo_eval = sub.add_parser("evaluate-yolo")
    yolo_eval.add_argument("--config")
    yolo_eval.add_argument("--dataset-root", type=Path)
    yolo_eval.add_argument("--weights", required=True)
    yolo_eval.add_argument("--device", default=None)

    reference = sub.add_parser("build-reference")
    reference.add_argument("--config")
    reference.add_argument("--processed-root", type=Path, default=Path("datasets/processed/egg_quality"))
    reference.add_argument("--artifact-dir", type=Path, default=Path("artifacts/egg_demo/padim"))
    reference.add_argument("--encoder-artifact-root", type=Path, default=Path("artifacts/production/mvtec_ad"))
    reference.add_argument("--image-size", type=int, default=256)
    reference.add_argument("--batch-size", type=int, default=8)
    reference.add_argument("--device", default="cpu", help="Reserved for future non-CPU training; current DINO reference uses CPU")

    production = sub.add_parser("build-production")
    production.add_argument("--config")
    production.add_argument("--quality-root", type=Path, default=Path("datasets/processed/egg_quality"))
    production.add_argument("--base-artifact-root", type=Path, default=Path("artifacts/production/mvtec_ad"))
    production.add_argument("--output-root", type=Path, default=Path("artifacts/production/mvtec_ad_egg"))
    production.add_argument("--encoder-artifact-root", type=Path, default=Path("artifacts/production/mvtec_ad"))
    production.add_argument("--batch-size", type=int, default=8)
    production.add_argument("--target-acceptance-rate", type=float, default=0.99)
    production.add_argument("--device", default="cpu")

    anomaly_eval = sub.add_parser("evaluate-anomaly")
    anomaly_eval.add_argument("--config")
    anomaly_eval.add_argument("--processed-root", type=Path, default=Path("datasets/processed/egg_quality"))
    anomaly_eval.add_argument("--artifact-dir", type=Path, default=Path("artifacts/egg_demo/padim"))
    anomaly_eval.add_argument("--encoder-artifact-root", type=Path, default=Path("artifacts/production/mvtec_ad"))
    anomaly_eval.add_argument("--device", default="cpu")

    infer = sub.add_parser("infer")
    infer.add_argument("--config")
    infer.add_argument("--image", type=Path)
    infer.add_argument("--directory", type=Path)
    infer.add_argument("--yolo-weights", required=True)
    infer.add_argument("--padim-artifact", type=Path, default=Path("artifacts/egg_demo/padim"))
    infer.add_argument("--encoder-artifact-root", type=Path, default=Path("artifacts/production/mvtec_ad"))
    infer.add_argument("--output-root", type=Path, default=Path("reports/egg_demo"))
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--confidence", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config(getattr(args, "config", None))
    if args.command == "inspect":
        report = {
            "yolo": validate_yolo_dataset(config.detection_dataset),
            "quality": inspect_quality_archive(config.quality_archive),
            "compatibility": {
                "full_egg_boxes_in_quality_dataset": False,
                "pixel_masks_in_quality_dataset": inspect_quality_archive(config.quality_archive)["has_pixel_masks"],
                "note": "Quality archive has image-level Good/Bad folders only; detection transfer needs measured boxes on quality images.",
            },
        }
        output = args.output
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "prepare-quality":
        report = prepare_quality_dataset(args.archive or config.quality_archive, args.output_root or config.quality_processed, args.image_size, args.seed)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "prepare-freshness":
        report = prepare_freshness_dataset(args.archive, args.output_root, args.image_size, args.seed)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "train-yolo":
        dataset_root = args.dataset_root or config.detection_dataset
        data_yaml = write_local_yolo_yaml(dataset_root, args.output_root / "data.yaml")
        detector = YoloEggDetector(args.weights, args.device)
        result = detector.train(data_yaml, project=str(args.output_root), name="train", epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device)
        print(json.dumps({"data_yaml": str(data_yaml), "result": str(result)}, indent=2))
        return 0
    if args.command == "evaluate-yolo":
        dataset_root = args.dataset_root or config.detection_dataset
        data_yaml = write_local_yolo_yaml(dataset_root, Path("artifacts/egg_demo/yolo/eval_data.yaml"))
        result = YoloEggDetector(args.weights, args.device).evaluate(data_yaml, device=args.device, split="val")
        print(json.dumps({"data_yaml": str(data_yaml), "metrics": str(result)}, indent=2))
        return 0
    if args.command == "build-reference":
        train_paths = _paths_from_manifest(args.processed_root, "train")
        validation_paths = _paths_from_manifest(args.processed_root, "validation")
        report = build_padim_reference(train_paths, validation_paths, args.artifact_dir, args.encoder_artifact_root, args.image_size, batch_size=args.batch_size)
        print(json.dumps({"artifact_dir": str(args.artifact_dir), "train_images": len(train_paths), "validation_images": len(validation_paths), "threshold": report["image_thresholds"]["categories"]["egg"]["threshold"]}, indent=2))
        return 0
    if args.command == "build-production":
        result = build_egg_production(
            args.quality_root,
            args.base_artifact_root,
            args.output_root,
            args.encoder_artifact_root,
            batch_size=args.batch_size,
            target_acceptance_rate=args.target_acceptance_rate,
            device=args.device,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "evaluate-anomaly":
        reference = PaDiMReference(args.artifact_dir, args.encoder_artifact_root, args.device)
        result = evaluate_image_level(
            reference,
            _paths_from_manifest(args.processed_root, "test", "good"),
            _paths_from_manifest(args.processed_root, "test", "bad"),
        )
        print(json.dumps(result, indent=2))
        return 0
    if not (args.image or args.directory) or (args.image and args.directory):
        raise SystemExit("infer requires exactly one of --image or --directory")
    pipeline = EggAnomalyPipeline(
        YoloEggDetector(args.yolo_weights, args.device),
        PaDiMReference(args.padim_artifact, args.encoder_artifact_root, args.device),
        args.confidence,
        config.crop_margin,
    )
    paths = [args.image] if args.image else sorted(path for path in args.directory.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    outputs = []
    for path in paths:
        results = pipeline.infer(path)
        image_output = args.output_root / path.name
        draw_results(path, results, image_output)
        outputs.append({"image": str(path), "output": str(image_output), "eggs": [{"box": result.detection.xyxy, "confidence": result.detection.confidence, "class": result.detection.class_name, "score": result.anomaly.score, "label": result.anomaly.label} for result in results]})
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "predictions.json").write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outputs, indent=2))
    return 0
