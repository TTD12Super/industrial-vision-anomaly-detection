# Product onboarding

The common contract is configuration plus a versioned product dataset. Core pipeline source must not be edited for a conforming image product.

```bash
PYTHONPATH=src DAenv/bin/python -m inspection.cli validate-dataset \
  --config configs/products/<product>.yaml

PYTHONPATH=src DAenv/bin/python -m inspection.cli train-detector \
  --config configs/products/<product>.yaml --model yolo11n-obb.pt --epochs 100 --imgsz 1024

PYTHONPATH=src DAenv/bin/python -m inspection.cli train-classifier \
  --config configs/products/<product>.yaml

PYTHONPATH=src DAenv/bin/python -m inspection.cli calibrate \
  --config configs/products/<product>.yaml

PYTHONPATH=src DAenv/bin/python -m inspection.cli export-bundle \
  --config configs/products/<product>.yaml

PYTHONPATH=src DAenv/bin/python -m inspection.cli evaluate \
  --config configs/products/<product>.yaml
```

`train-detector` materializes a separate symlinked YOLO-OBB working directory. It never alters original images. `train-classifier` uses the manifest's OBB to create oriented object crops and trains frozen DINOv2 global embeddings plus a binary Logistic Regression head.

`calibrate` must read validation data only. Update the product config to point at the selected trained classifier and saved validation embeddings before calibration; do not overwrite a prior model bundle.

The first screw pilot also has a non-destructive adapter for the existing recovered labels:

```bash
PYTHONPATH=src DAenv/bin/python -m inspection.cli bootstrap-screw-pilot \
  --config configs/products/screw.yaml \
  --source-root datasets/mvtec_ad_screw_bbox_obb
```

The resulting product data uses symlinks to raw MVTec AD files and records that official test images are supervised-development data. It is not an untouched official benchmark.
