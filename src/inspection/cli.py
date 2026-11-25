"""Configuration-driven inspection commands shared by every product."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_screw_pilot
from .calibration import calibrate_saved_dinov2
from .config import load_product_config
from .dataset import validate_dataset
from .detector_calibration import calibrate_detector_confidence, measure_detector_confidence
from .evaluation import calibrate_localizer, calibrate_relation, evaluate_dino_refinement, evaluate_end_to_end, evaluate_localizer, evaluate_localizer_methods, evaluate_relation, train_supervised_refiner, train_supervised_rgb_refiner, train_synthetic_segmenter
from .product_recognition import build_product_gallery, calibrate_product_recognizer
from .runtime import build_pipeline
from .segad_lite import evaluate_segad_lite
from .dinov2_patchcore_lite import evaluate_dinov2_patchcore_lite
from .training import train_dinov2_logreg, train_yolo_obb


def _config(value: str) -> Any:
    return load_product_config(value)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_dataset_command(args: argparse.Namespace) -> int:
    report = validate_dataset(_config(args.config), verify_hashes=not args.skip_hashes)
    print(json.dumps(report.as_dict(), indent=2))
    return 0 if report.valid else 2


def bootstrap_screw_command(args: argparse.Namespace) -> int:
    print(json.dumps(bootstrap_screw_pilot(_config(args.config), args.source_root), indent=2))
    return 0


def calibrate_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    print(json.dumps({"classifier": calibrate_saved_dinov2(config), "detector": calibrate_detector_confidence(config)}, indent=2))
    return 0


def diagnose_detector_calibration_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    report = measure_detector_confidence(config)
    output = config.project_path(args.output or f"reports/products/{config.product_key}/detector_calibration_diagnostic.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output.resolve()), **report}, indent=2))
    return 0 if report["target_reached"] else 2


def inspect_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    pipeline = build_pipeline(config, args.runs_root)
    print(json.dumps(pipeline.inspect_path(args.image, args.inspection_id), indent=2))
    return 0


def export_bundle_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    destination = config.project_path(args.output or f"artifacts/products/{config.product_key}/{config.policy['model_bundle_version']}")
    output = destination / "model_bundle.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bundle manifest: {output}")
    calibration = config.project_path(config.policy["calibration_path"])
    if not calibration.is_file():
        raise FileNotFoundError("Run calibrate before export-bundle")
    tracked = {"config": config.source_path, "calibration": calibration}
    detector_weights = config.model("detector").get("weights")
    if detector_weights:
        path = config.project_path(detector_weights)
        if path.is_file():
            tracked["detector_weights"] = path
    classifier_dir = config.model("classifier").get("artifact_dir")
    if classifier_dir:
        path = config.project_path(classifier_dir) / "classifier" / "classifier.pt"
        if path.is_file():
            tracked["classifier_weights"] = path
    payload = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "model_bundle_version": config.policy["model_bundle_version"],
        "files": {name: {"path": str(path), "sha256": _file_hash(path)} for name, path in tracked.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def train_detector_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"artifacts/products/{config.product_key}/{config.policy['model_bundle_version']}/detector_dataset")
    print(json.dumps(train_yolo_obb(config, output, args.model, args.epochs, args.imgsz, args.device), indent=2))
    return 0


def train_classifier_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"artifacts/products/{config.product_key}/{config.policy['model_bundle_version']}/classifier_training")
    print(json.dumps(train_dinov2_logreg(config, output, args.batch_size, args.seed, args.device), indent=2))
    return 0


def build_product_gallery_command(args: argparse.Namespace) -> int:
    configs = [_config(value) for value in args.config]
    output = configs[0].project_path(args.output)
    print(json.dumps(build_product_gallery(configs, output, args.batch_size, args.device), indent=2))
    return 0


def calibrate_product_recognizer_command(args: argparse.Namespace) -> int:
    known_configs = [_config(value) for value in args.config]
    unknown_configs = [_config(value) for value in args.unknown_config]
    project_roots = {config.project_root for config in (*known_configs, *unknown_configs)}
    if len(project_roots) != 1:
        raise ValueError("Recognizer calibration configs must belong to the same project")
    project_config = known_configs[0]
    gallery_path = project_config.project_path(args.gallery)
    output = project_config.project_path(args.output)
    print(
        json.dumps(
            calibrate_product_recognizer(
                known_configs,
                unknown_configs,
                gallery_path,
                output,
                args.target_known_recall,
                args.batch_size,
                args.device,
            ),
            indent=2,
        )
    )
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    pipeline = build_pipeline(config, args.runs_root)
    output = config.project_path(args.output or f"reports/products/{config.product_key}/{config.policy['model_bundle_version']}/end_to_end.json")
    print(json.dumps(evaluate_end_to_end(config, pipeline, output, args.iou_threshold), indent=2))
    return 0


def evaluate_segad_lite_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"reports/products/{config.product_key}/{config.policy['model_bundle_version']}/segad_lite.json")
    print(
        json.dumps(
            evaluate_segad_lite(
                config,
                output,
                args.target_defect_recall,
                args.pixel_target_recall,
                args.interior_border_fraction,
                args.include_p_defect,
                device=args.device,
            ),
            indent=2,
        )
    )
    return 0


def evaluate_dinov2_patchcore_lite_command(args: argparse.Namespace) -> int:
    report_path = Path(args.output or "reports/experiments/dinov2_patchcore_lite.json")
    print(
        json.dumps(
            evaluate_dinov2_patchcore_lite(
                mvtec_root=args.mvtec_root,
                cad_root=args.cad_root,
                report_path=report_path,
                batch_size=args.batch_size,
                k=args.k,
                allowed_good_fpr=args.allowed_good_fpr,
                device=args.device,
                mixed_nominal=args.mixed_nominal,
                cad_memory_fraction=args.cad_memory_fraction,
                cad_split_salt=args.cad_split_salt,
                classifier_artifact_path=args.classifier_artifact_output,
            ),
            indent=2,
        )
    )
    return 0


def calibrate_localizer_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"artifacts/products/{config.product_key}/{config.policy['model_bundle_version']}/anomalyclip_calibration.json")
    print(json.dumps(calibrate_localizer(config, output, args.target_recall), indent=2))
    return 0


def evaluate_localizer_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"reports/products/{config.product_key}/{config.policy['model_bundle_version']}/anomalyclip_test.json")
    print(json.dumps(evaluate_localizer(config, output, args.threshold), indent=2))
    return 0


def calibrate_relation_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"artifacts/products/{config.product_key}/{config.policy['model_bundle_version']}/relation_calibration.json")
    print(json.dumps(calibrate_relation(config, output, args.target_recall), indent=2))
    return 0


def evaluate_relation_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"reports/products/{config.product_key}/{config.policy['model_bundle_version']}/relation_test.json")
    print(json.dumps(evaluate_relation(config, output), indent=2))
    return 0


def evaluate_localizer_methods_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"reports/products/{config.product_key}/{config.policy['model_bundle_version']}/localizer_methods.json")
    print(json.dumps(evaluate_localizer_methods(config, output, args.target_recall, args.active_refiner), indent=2))
    return 0


def evaluate_dino_refinement_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"reports/products/{config.product_key}/{config.policy['model_bundle_version']}/dino_refinement.json")
    print(json.dumps(evaluate_dino_refinement(config, output, args.target_fpr), indent=2))
    return 0


def train_supervised_refiner_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"artifacts/products/{config.product_key}/{config.policy['model_bundle_version']}/supervised_refiner_calibration.json")
    print(json.dumps(train_supervised_refiner(config, output, args.target_fpr), indent=2))
    return 0


def train_supervised_rgb_refiner_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"artifacts/products/{config.product_key}/{config.policy['model_bundle_version']}/supervised_rgb_refiner.json")
    print(json.dumps(train_supervised_rgb_refiner(config, output, args.target_fpr), indent=2))
    return 0


def train_synthetic_segmenter_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = config.project_path(args.output or f"artifacts/products/{config.product_key}/{config.policy['model_bundle_version']}/synthetic_segmenter.json")
    print(json.dumps(train_synthetic_segmenter(config, output, args.target_fpr), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-dataset")
    validate.add_argument("--config", required=True)
    validate.add_argument("--skip-hashes", action="store_true")
    validate.set_defaults(func=validate_dataset_command)
    bootstrap = commands.add_parser("bootstrap-screw-pilot")
    bootstrap.add_argument("--config", required=True)
    bootstrap.add_argument("--source-root", default="datasets/mvtec_ad_screw_bbox_obb")
    bootstrap.set_defaults(func=bootstrap_screw_command)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--config", required=True)
    calibrate.set_defaults(func=calibrate_command)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--image", required=True)
    inspect.add_argument("--inspection-id", required=True)
    inspect.add_argument("--runs-root", default="runs")
    inspect.set_defaults(func=inspect_command)
    bundle = commands.add_parser("export-bundle")
    bundle.add_argument("--config", required=True)
    bundle.add_argument("--output")
    bundle.set_defaults(func=export_bundle_command)
    detector = commands.add_parser("train-detector")
    detector.add_argument("--config", required=True)
    detector.add_argument("--model", default="yolo11n-obb.pt")
    detector.add_argument("--epochs", type=int, default=100)
    detector.add_argument("--imgsz", type=int, default=1024)
    detector.add_argument("--device")
    detector.add_argument("--output")
    detector.set_defaults(func=train_detector_command)
    detector_diagnostic = commands.add_parser("diagnose-detector-calibration")
    detector_diagnostic.add_argument("--config", required=True)
    detector_diagnostic.add_argument("--output")
    detector_diagnostic.set_defaults(func=diagnose_detector_calibration_command)
    classifier = commands.add_parser("train-classifier")
    classifier.add_argument("--config", required=True)
    classifier.add_argument("--output")
    classifier.add_argument("--batch-size", type=int, default=16)
    classifier.add_argument("--seed", type=int, default=42)
    classifier.add_argument("--device")
    classifier.set_defaults(func=train_classifier_command)
    gallery = commands.add_parser("build-product-gallery")
    gallery.add_argument("--config", action="append", required=True, help="Known product config; repeat for each gallery product")
    gallery.add_argument("--output", required=True)
    gallery.add_argument("--batch-size", type=int, default=16)
    gallery.add_argument("--device")
    gallery.set_defaults(func=build_product_gallery_command)
    recognizer_calibration = commands.add_parser("calibrate-product-recognizer")
    recognizer_calibration.add_argument("--config", action="append", required=True, help="Known product config; repeat as needed")
    recognizer_calibration.add_argument("--unknown-config", action="append", required=True, help="Held-out unknown product config; repeat as needed")
    recognizer_calibration.add_argument("--gallery", required=True)
    recognizer_calibration.add_argument("--output", required=True)
    recognizer_calibration.add_argument("--target-known-recall", type=float, required=True)
    recognizer_calibration.add_argument("--batch-size", type=int, default=16)
    recognizer_calibration.add_argument("--device")
    recognizer_calibration.set_defaults(func=calibrate_product_recognizer_command)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--output")
    evaluate.add_argument("--runs-root", default="runs")
    evaluate.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate.set_defaults(func=evaluate_command)
    segad = commands.add_parser("evaluate-segad-lite")
    segad.add_argument("--config", required=True)
    segad.add_argument("--output")
    segad.add_argument("--target-defect-recall", type=float, default=0.90)
    segad.add_argument("--pixel-target-recall", type=float, default=0.90)
    segad.add_argument("--interior-border-fraction", type=float, default=0.10)
    segad.add_argument("--include-p-defect", action="store_true")
    segad.add_argument("--device")
    segad.set_defaults(func=evaluate_segad_lite_command)
    patchcore = commands.add_parser("evaluate-dinov2-patchcore-lite")
    patchcore.add_argument("--mvtec-root", default="datasets/mvtec_ad/screw")
    patchcore.add_argument(
        "--cad-root",
        default="datasets/external/cad_sd/data/unpacked/Co-occurrence Anomaly Detection Screw Dataset",
    )
    patchcore.add_argument("--output")
    patchcore.add_argument("--batch-size", type=int, default=8)
    patchcore.add_argument("--k", type=int, default=1)
    patchcore.add_argument("--allowed-good-fpr", type=float, default=0.05)
    patchcore.add_argument("--mixed-nominal", action="store_true")
    patchcore.add_argument("--cad-memory-fraction", type=float, default=0.5)
    patchcore.add_argument("--cad-split-salt", default="cad_sd_train_0_good_v1")
    patchcore.add_argument("--classifier-artifact-output")
    patchcore.add_argument("--device")
    patchcore.set_defaults(func=evaluate_dinov2_patchcore_lite_command)
    localizer_calibration = commands.add_parser("calibrate-localizer")
    localizer_calibration.add_argument("--config", required=True)
    localizer_calibration.add_argument("--output")
    localizer_calibration.add_argument("--target-recall", type=float, default=0.90)
    localizer_calibration.set_defaults(func=calibrate_localizer_command)
    localizer_evaluation = commands.add_parser("evaluate-localizer")
    localizer_evaluation.add_argument("--config", required=True)
    localizer_evaluation.add_argument("--output")
    localizer_evaluation.add_argument("--threshold", type=float)
    localizer_evaluation.set_defaults(func=evaluate_localizer_command)
    relation_calibration = commands.add_parser("calibrate-relation")
    relation_calibration.add_argument("--config", required=True)
    relation_calibration.add_argument("--output")
    relation_calibration.add_argument("--target-recall", type=float, default=0.90)
    relation_calibration.set_defaults(func=calibrate_relation_command)
    relation_evaluation = commands.add_parser("evaluate-relation")
    relation_evaluation.add_argument("--config", required=True)
    relation_evaluation.add_argument("--output")
    relation_evaluation.set_defaults(func=evaluate_relation_command)
    methods = commands.add_parser("evaluate-localizer-methods")
    methods.add_argument("--config", required=True)
    methods.add_argument("--output")
    methods.add_argument("--target-recall", type=float, default=0.90)
    methods.add_argument("--active-refiner", action="store_true", help="evaluate the configured supervised patch refiner instead of the baseline map")
    methods.set_defaults(func=evaluate_localizer_methods_command)
    dino = commands.add_parser("evaluate-dino-refinement")
    dino.add_argument("--config", required=True)
    dino.add_argument("--output")
    dino.add_argument("--target-fpr", type=float, default=0.20)
    dino.set_defaults(func=evaluate_dino_refinement_command)
    refiner = commands.add_parser("train-supervised-refiner")
    refiner.add_argument("--config", required=True)
    refiner.add_argument("--output")
    refiner.add_argument("--target-fpr", type=float, default=0.20)
    refiner.set_defaults(func=train_supervised_refiner_command)
    rgb_refiner = commands.add_parser("train-supervised-rgb-refiner")
    rgb_refiner.add_argument("--config", required=True)
    rgb_refiner.add_argument("--output")
    rgb_refiner.add_argument("--target-fpr", type=float, default=0.20)
    rgb_refiner.set_defaults(func=train_supervised_rgb_refiner_command)
    synthetic = commands.add_parser("train-synthetic-segmenter")
    synthetic.add_argument("--config", required=True)
    synthetic.add_argument("--output")
    synthetic.add_argument("--target-fpr", type=float, default=0.20)
    synthetic.set_defaults(func=train_synthetic_segmenter_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
