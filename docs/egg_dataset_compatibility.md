# Egg demo dataset compatibility report

## Sources and provenance

| Dataset | Local inspection | Intended use | License |
|---|---|---|---|
| [afshin-dini/Egg-Detection](https://huggingface.co/datasets/afshin-dini/Egg-Detection) | `datasets/raw/egg_detection` | Complete-egg YOLO detection | MIT |
| [Good and Bad Eggs Identification Image Dataset](https://data.mendeley.com/datasets/mdty358x8m/1) | `datasets/raw/egg_quality/Original_Images_Eggs.zip` | Image-level quality labels and normal reference candidates | CC BY 4.0 |

The Mendeley archive was downloaded from the original-images file. The separate
augmented archive is not used for fitting or threshold selection because its
augmentation lineage is not exposed per image; using it could leak transformed
copies across splits.

## Verified contents

### Dataset A: YOLO detection

- 49 train images and 2 validation images.
- 393 train boxes and 30 validation boxes.
- Classes are `white-egg` and `brown-egg`.
- Images are 3024×4032 JPEGs.
- Every image has a label file; all checked labels use normalized five-field
  YOLO rows and all coordinates are in `[0, 1]`.
- Images contain multiple eggs, so box count is larger than image count.

The expanded training script deliberately collapses both source classes into a
single `egg` class because the production detector only needs to locate eggs;
it does not classify shell color. It also preserves the supplementary
archive's official `train` and `valid` split so validation images are not
mixed into training.

### Dataset B: quality classification

- 1,000 original JPEGs: 500 under `Good Eggs` and 500 under `Bad Eggs`.
- Sampled images are 512×512.
- The archive contains image-level class folders only.
- No mask, segmentation, or ground-truth-region files were found.
- The split prepared by `egg_demo.py prepare-quality` uses only originals:
  good images 70% train, 15% validation, 15% test; bad images test only;
  deterministic seed 42.

## Compatibility decision

Dataset A can train a detector for complete eggs. Dataset B can evaluate
image-level Good/Bad anomaly decisions, but it does not provide verified
full-egg boxes or pixel defect masks. Therefore the current demo must not
report detection mAP on Dataset B or pixel anomaly metrics on Dataset B.

The two domains are visibly different: Dataset A contains multi-egg trays and
large 3024×4032 scenes, while Dataset B contains centered 512×512 single-egg
images on a dark background. A YOLO model trained only on Dataset A therefore
needs a measured transfer check on Dataset B before its boxes are trusted.
Without boxes in Dataset B, that transfer check is qualitative only; add a
small manually annotated hold-out set before claiming cross-dataset detection
accuracy.

## Leakage and limits

- Exact file-byte hashes were checked when preparing the archives; no source
  overlap is assumed without a perceptual-duplicate audit.
- Bad images are never used to fit PaDiM or calibrate its threshold.
- Pixel-level localization metrics are intentionally omitted for Dataset B.
- Dataset A's validation split has only two images, so its reported validation
  metrics are not a reliable final estimate. More labelled detection images or
  a new held-out set are required.
