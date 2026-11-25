# Screw YOLO-OBB pipeline

The first production component is a one-class detector: `0: screw`. It does not classify screw type or defect status. Its output is an oriented crop for the downstream GOOD/DEFECT model.

## Dataset layout

The WEEE Disassembly Screw dataset is stored under `datasets/external/weee_screw` with COCO JSON annotations and `train/val/test` images. Its segmentation polygons are converted to true minimum-area OBBs. Axis-aligned boxes are not silently treated as oriented labels.

MVTec Screws is stored under `datasets/external/mvtec_screws/extracted`. Its native `[row, col, width, height, phi]` boxes are converted with the MVTec coordinate convention. The default mapping keeps screw-like `type_001`–`type_006`, `type_008`, and `type_013`, while excluding the nut types `type_007`, `type_009`–`type_012`; use `--mvtec-types all` only when all fasteners are intentionally in scope.

## Convert and validate

```bash
python scripts/prepare_screw_yolo_obb.py \
  --weee-root datasets/external/weee_screw \
  --output-root datasets/screw_obb

python scripts/prepare_screw_yolo_obb.py \
  --weee-root datasets/external/weee_screw \
  --mvtec-root datasets/external/mvtec_screws/extracted \
  --output-root datasets/screw_obb_combined \
  --mvtec-types 1,2,3,4,5,6,8,13

python scripts/validate_screw_yolo_obb.py \
  --root datasets/screw_obb_combined \
  --render-dir reports/screw_yolo_obb/labels
```

When MVTec Screws is available, pass `--mvtec-root` in the same command. Use `--allow-axis-fallback` only after manually confirming that a source contains no native OBB/segmentation annotations.

## Train on the Windows GPU machine

```bash
pip install ultralytics opencv-python pillow numpy
python scripts/train_screw_yolo_obb.py \
  --data datasets/screw_obb/data.yaml \
  --model yolo11n-obb.pt \
  --epochs 100 \
  --imgsz 640 \
  --device 0
```

For small screws, repeat with `--imgsz 1024`. Select the checkpoint by validation recall/precision and then run:

```bash
python scripts/evaluate_screw_yolo_obb.py \
  --weights runs/screw_yolo_obb/combined/weights/best.pt \
  --data datasets/screw_obb/data.yaml \
  --device 0
```

The detector is evaluated separately on WEEE and MVTec when both sources are present. The important first metric is missed-screw rate/recall because a missed crop cannot reach DINOv2.
