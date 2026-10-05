# AI CONFIG — CONTINUE EXISTING MVTec AD PROJECT WITHOUT REBUILDING WORK

> **Repository:** `TTD12Super/industrial-vision-anomaly-detection`  
> **Target branch:** `develop`  
> **Current domain:** MVTec AD only  
> **Goal:** Continue and finish the existing project by reusing the current dataset, processed dataset, DINOv2 baseline, local artifacts, and notebook logic.  
> **Architecture target:** Shared frozen DINOv2 → Product Classification → Category Router → Category-specific DINOv2 + exact kNN anomaly detection.  
> **Critical principle:** **DO NOT restart the project from scratch. Reuse existing work first.**

---

# 1. READ THIS BEFORE MODIFYING ANYTHING

This repository already contains a working research baseline.

The AI agent MUST NOT assume this is a new project.

Before any implementation:

1. Inspect the current branch.
2. Inspect the entire repository tree.
3. Read:
   - `README.md`
   - `notebooks/01_data_exploration.ipynb`
   - `notebooks/02_first_model.ipynb`
   - `.gitignore`
   - this AI config.
4. Inspect the local filesystem for the existing MVTec dataset and generated artifacts.
5. Reuse existing valid files whenever possible.
6. Only regenerate data/artifacts when they are missing, incompatible, or fail validation.
7. Never delete or overwrite the existing raw dataset.
8. Never download MVTec AD again if the existing local dataset is valid.
9. Never create a second duplicate processed dataset if `datasets/mvtec_ad_processed` is valid.
10. Never rebuild all feature banks just because reusable banks already exist locally.

---

# 2. WHAT ALREADY EXISTS IN THE REPOSITORY

The `develop` branch already contains:

```text
industrial-vision-anomaly-detection/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_first_model.ipynb
├── model/
│   └── training/
│       └── train.py
├── artifacts/
│   └── dinov2_knn/
│       └── final_run/
│           ├── bottle/
│           ├── cable/
│           ├── capsule/
│           ├── carpet/
│           ├── grid/
│           ├── hazelnut/
│           ├── leather/
│           ├── metal_nut/
│           ├── pill/
│           ├── screw/
│           ├── tile/
│           ├── toothbrush/
│           ├── transistor/
│           ├── wood/
│           └── zipper/
└── .gitignore
```

Important:

`model/training/train.py` is currently empty.

Do not treat that empty file as evidence that the ML work does not exist.

The actual existing ML baseline is implemented in:

```text
notebooks/02_first_model.ipynb
```

---

# 3. IMPORTANT GITIGNORE BEHAVIOR

The repository `.gitignore` currently contains:

```text
# Datasets
**/datasets/

# Environment
**/DAENV/

# Notebook model outputs and downloaded backbone weights
artifacts/dinov2_knn/
.ipynb_checkpoints/
```

Therefore:

- the real MVTec images are expected to exist locally but are intentionally not committed,
- the processed MVTec dataset is also local,
- most generated feature banks/model outputs may exist locally but are ignored,
- GitHub cannot be used to determine whether those local ignored artifacts still exist.

The AI agent MUST inspect the local filesystem before regenerating anything.

---

# 4. EXISTING LOCAL DATASET LOCATIONS

The existing notebooks use exactly these paths:

```text
PROJECT_ROOT/
└── datasets/
    ├── mvtec_ad/
    └── mvtec_ad_processed/
```

Raw dataset:

```text
datasets/mvtec_ad
```

Processed dataset:

```text
datasets/mvtec_ad_processed
```

These paths are the canonical paths for the current project.

Do NOT introduce a new path such as:

```text
data/
dataset/
mvtec/
storage/
input_data/
```

unless explicitly approved.

---

# 5. EXISTING RAW DATASET EXPECTATIONS

`notebooks/01_data_exploration.ipynb` already validated the raw MVTec AD dataset.

Previously observed dataset statistics:

```text
Categories:          15
Training images:     3,629
Test images:         1,725
Ground-truth masks:  1,258
```

Categories:

```text
bottle
cable
capsule
carpet
grid
hazelnut
leather
metal_nut
pill
screw
tile
toothbrush
transistor
wood
zipper
```

Previous validation results:

```text
Train corrupted images:         0
Test corrupted images:          0
Ground-truth corrupted masks:   0
Missing masks:                  0
Unexpected masks:               0
Mask dimension mismatches:      0
```

The raw dataset was also checked for unintended modification.

This existing dataset is authoritative for the current project.

---

# 6. RAW DATASET IS READ-ONLY

Treat:

```text
datasets/mvtec_ad/
```

as READ-ONLY.

Never:

```text
❌ resize images in place
❌ rename files
❌ move files
❌ alter masks
❌ normalize and overwrite files
❌ add classifier split folders inside the raw dataset
❌ delete categories
```

Any additional project metadata must be stored elsewhere.

Recommended:

```text
configs/
manifests/
artifacts/
```

Do not mutate the original MVTec AD directory structure.

---

# 7. EXISTING PROCESSED DATASET

`notebooks/01_data_exploration.ipynb` already creates:

```text
datasets/mvtec_ad_processed/
```

The existing preprocessing baseline is:

```text
Raw image
    ↓
PIL decode
    ↓
RGB conversion
    ↓
Resize to 256 × 256
    ↓
LANCZOS for normal images
NEAREST for masks
    ↓
PNG output
```

Processed RGB images are expected to be:

```text
256 × 256
RGB
PNG
```

Masks are also resized to:

```text
256 × 256
```

using nearest-neighbour interpolation.

---

# 8. DO NOT PREPROCESS AGAIN BY DEFAULT

Before running any preprocessing:

Check:

```text
datasets/mvtec_ad_processed/
```

If it exists and passes the same validation already implemented in `01_data_exploration.ipynb`:

```text
REUSE IT.
```

Do not rewrite all processed files.

Only rerun preprocessing if:

```text
processed directory is missing
OR
files are incomplete
OR
validation fails
OR
preprocessing contract intentionally changes
```

If preprocessing intentionally changes, create a new versioned processed dataset rather than silently replacing the existing baseline.

Example:

```text
datasets/
├── mvtec_ad_processed/          # existing baseline, preserve
└── mvtec_ad_processed_v2/       # only if an approved new preprocessing version exists
```

Do not create `v2` merely for convenience.

---

# 9. DATASET PRE-FLIGHT CHECK

Every new task that depends on MVTec data should begin with a lightweight pre-flight check.

Pseudo-process:

```text
Does datasets/mvtec_ad exist?
        │
        ├── NO → STOP and report missing local dataset.
        │
        └── YES
              ↓
Validate 15 categories / expected structure
              ↓
Does datasets/mvtec_ad_processed exist?
        │
        ├── NO → run existing preprocessing workflow
        │
        └── YES
              ↓
Validate processed dataset
        │
        ├── PASS → reuse
        │
        └── FAIL → report exact mismatch before regenerating
```

Do not automatically download data from the internet.

---

# 10. EXISTING DINOv2 BASELINE

The current anomaly model already exists in:

```text
notebooks/02_first_model.ipynb
```

The current baseline is:

```text
MVTec train/good
        ↓
Frozen DINOv2 ViT-B/14
        ↓
Normalized patch embeddings
        ↓
Per-category normal feature bank
        ↓
Exact kNN
        ↓
Patch anomaly distances
        ↓
Image anomaly score
        ↓
Anomaly map
```

This is a valid working baseline.

Do NOT replace it.

---

# 11. EXISTING DINOv2 CONFIGURATION — PRESERVE IT

Current configuration:

```text
MODEL_NAME:
dinov2_vitb14

MODEL_REVISION:
7764ea0f912e53c92e82eb78a2a1631e92725fc8

FEATURE_KEY:
x_norm_patchtokens

Processed input:
256 × 256

ImageNet mean:
0.485, 0.456, 0.406

ImageNet std:
0.229, 0.224, 0.225

L2 normalize patch features:
true

K:
1

Distance:
Euclidean

Feature-bank sampling:
none

Image aggregation:
maximum patch anomaly score
```

Do not silently change:

```text
backbone
revision
input size
normalization
feature key
K
distance
aggregation
```

Any experiment that changes these must be recorded as a new experiment.

---

# 12. EXISTING DINOv2 WEIGHTS

The notebook sets the Torch Hub directory to:

```text
artifacts/dinov2_knn/torch_hub/
```

The first run downloads the pinned DINOv2 weights there.

Before downloading model weights:

```text
inspect artifacts/dinov2_knn/torch_hub/
```

If the correct pinned checkpoint exists:

```text
reuse it.
```

Existing notebook recorded checkpoint SHA-256:

```text
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
```

If a cached checkpoint exists, verify compatibility before downloading another copy.

---

# 13. EXISTING FEATURE-BANK LOGIC

The existing notebook already builds a separate normal bank for each category.

Example:

```text
bottle/train/good
       ↓
DINOv2 patch embeddings
       ↓
bottle feature bank

cable/train/good
       ↓
DINOv2 patch embeddings
       ↓
cable feature bank
```

Feature bank rules already implemented:

```text
✅ train/good only
✅ no test images
✅ one bank per category
✅ frozen DINOv2
✅ finite-value validation
✅ metadata saved with bank
```

Preserve these rules.

---

# 14. EXISTING LOCAL FEATURE BANKS MUST BE REUSED WHEN VALID

The notebook saves local feature banks as:

```text
<run_dir>/<category>/feature_bank.pt
```

The exact local run directory is created under:

```text
artifacts/dinov2_knn/run_<timestamp>/
```

The Git repository may not show these files because the artifact folder is ignored.

Before rebuilding a category bank:

1. Search local `artifacts/dinov2_knn/`.
2. Find existing `feature_bank.pt`.
3. Read its metadata.
4. Validate against the current baseline:
   - category,
   - model name,
   - model revision,
   - feature key,
   - image size,
   - normalization,
   - embedding dimension,
   - patch grid,
   - train image paths.
5. Reuse it if metadata matches.
6. Rebuild only if missing or incompatible.

Never blindly rebuild all 15 banks.

---

# 15. EXISTING ARTIFACTS

The current notebook can produce:

```text
feature_bank.pt
image_scores.csv
metrics.json
bank_metadata.json
anomaly_maps.npz
examples.png
examples_many.png
```

and run-level files such as:

```text
category_metrics.csv
macro_metrics.csv
dataset_summary.csv
run_config.json
```

Before recomputing an expensive experiment:

```text
inspect whether these outputs already exist locally.
```

If valid reusable outputs exist, consume them.

---

# 16. COMMITTED FINAL-RUN EVIDENCE

Git currently contains visual evidence under:

```text
artifacts/dinov2_knn/final_run/<category>/examples_many.png
```

for all 15 categories.

These are existing research evidence.

Do not delete them.

Do not replace them unless generating a clearly versioned new result.

---

# 17. EXISTING BASELINE RESULTS

The completed notebook evaluation previously produced approximately:

```text
Macro Image AUROC: 0.980325
Macro Image AP:    0.987077
Macro Pixel AUROC: 0.961282
Macro Pixel AP:    0.516239
```

These results are the baseline.

The current project continuation must preserve a way to reproduce or compare against this baseline.

Do not claim a new implementation is correct merely because it runs.

Verify that refactored output remains consistent with the notebook baseline within reasonable numerical tolerance.

---

# 18. WHAT MUST NOT BE REDONE

Unless validation proves otherwise, do NOT redo:

```text
❌ MVTec download
❌ raw dataset extraction
❌ raw dataset inspection from scratch
❌ 256×256 preprocessing
❌ all-category DINOv2 baseline exploration
❌ all-category heatmap proof of concept
❌ replacement of existing KNN algorithm
❌ new anomaly architecture
```

The project is now in a CONTINUATION phase.

---

# 19. NEW TARGET ARCHITECTURE

Continue from the existing baseline toward:

```text
                         INPUT IMAGE
                             │
                             ▼
                    Existing preprocessing
                             │
                             ▼
                       Shared DINOv2
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
       Global embedding               Patch embeddings
              │                             │
              ▼                             │
      Product classifier                    │
              │                             │
              ▼                             │
       Product category                     │
              │                             │
              ▼                             │
        Category router                     │
              │                             │
              ▼                             ▼
Existing category feature bank ◄──── Patch embeddings
              │
              ▼
       Existing exact kNN
              │
              ▼
       Patch anomaly scores
              │
       ┌──────┴──────┐
       ▼             ▼
Image score      Anomaly map
       │             │
       ▼             ▼
Threshold        Heatmap
       │             │
       └──────┬──────┘
              ▼
         Final result
```

---

# 20. SHARED DINOv2 PRINCIPLE

The new classifier must NOT introduce another independent DINOv2 implementation.

One shared DINOv2 forward pass should eventually produce:

```text
global representation
+
patch representation
```

Use:

```text
global representation
→ product classification

patch representation
→ anomaly detection
```

The current notebook patch-extraction behavior must remain supported.

---

# 21. PRODUCT CLASSIFICATION SOURCE DATA

Do not create a new image dataset.

Reuse:

```text
datasets/mvtec_ad_processed/<category>/train/good/
```

for product classification.

Each MVTec category becomes one class:

```text
bottle
cable
capsule
carpet
grid
hazelnut
leather
metal_nut
pill
screw
tile
toothbrush
transistor
wood
zipper
```

No extra downloaded classification dataset is required.

---

# 22. CLASSIFICATION SPLIT MUST BE METADATA ONLY

Do NOT physically move/copy images into new train/validation folders.

Create a manifest.

Recommended:

```text
manifests/
└── classification_split_v1.csv
```

Example:

```csv
image_path,category,split
bottle/train/good/000.png,bottle,train
bottle/train/good/001.png,bottle,validation
cable/train/good/000.png,cable,train
...
```

Requirements:

```text
deterministic
stratified by category
fixed random seed
no MVTec test image
```

This preserves the existing processed database exactly.

---

# 23. IMPORTANT: CLASSIFIER SPLIT DOES NOT CHANGE ANOMALY BANK SOURCE

The product-classification split is separate from the anomaly baseline.

For product classifier:

```text
train/good
→ classifier train/validation split
```

For anomaly feature bank baseline:

```text
ALL category/train/good
→ normal feature bank
```

Do not accidentally shrink the anomaly bank to the classifier training subset.

Preserve the existing anomaly baseline.

---

# 24. PRODUCT CLASSIFIER BASELINE

Use:

```text
Frozen DINOv2 global embeddings
        ↓
Linear classifier
        ↓
15 product categories
```

Start simple.

Do not:

```text
❌ fine-tune DINOv2 immediately
❌ train another CNN
❌ add ResNet/EfficientNet
❌ use test images during training
```

The goal is automatic category routing, not a new classification research project.

---

# 25. PRECOMPUTE AND REUSE GLOBAL EMBEDDINGS

Do not repeatedly run DINOv2 for classifier training.

Create versioned embedding artifacts from existing processed `train/good`.

Example:

```text
artifacts/
└── product_classifier/
    └── v1/
        ├── train_embeddings.pt
        ├── validation_embeddings.pt
        ├── class_mapping.json
        └── embedding_metadata.json
```

Before recomputing these:

```text
check whether compatible files already exist.
```

Reuse if valid.

Metadata should include:

```text
DINOv2 model name
DINOv2 revision
preprocessing version
source image paths
class mapping
embedding dimension
```

---

# 26. CATEGORY ROUTER

After classification:

```text
predicted category
      ↓
load/reuse correct feature bank
```

Example:

```text
bottle
→ existing bottle bank

tile
→ existing tile bank
```

Do not merge all 15 normal banks into a single baseline bank.

---

# 27. CLASSIFICATION CONFIDENCE

Do not always route blindly.

Classifier output:

```text
category
confidence
probabilities
```

Routing behavior:

```text
confidence >= calibrated threshold
→ use predicted category bank

confidence < threshold
→ return `unknown_product_category`
→ do not run misleading anomaly inference
```

Threshold selection must use validation data only.

## 27.1 OPEN-SET PRODUCT REJECTION

The classifier has 15 trained MVTec categories. Do **not** add an ordinary
16th `unknown` class without unknown training examples.

Instead, reject inputs that are outside the known product distribution.

Use the cached global embeddings from the classification `train` split as the
known-product reference set:

```text
nearest_global_distance
    = minimum distance from the input global embedding
      to the known classification global embeddings
```

Routing rule:

```text
max classifier confidence < calibrated confidence threshold
OR
nearest_global_distance > calibrated distance threshold
→ category = unknown_product_category
→ do not load any category feature bank
→ do not run anomaly inference
```

If neither reject condition is met:

```text
→ route to the predicted category feature bank
```

Both thresholds must be selected using classification validation data only.
The distance threshold should preserve a documented target acceptance rate for
known validation images. Unknown-product detection recall cannot be claimed
without a separate unknown-product evaluation set.

---

# 28. CURRENT ANOMALY THRESHOLD GAP

The current notebook deliberately keeps:

```text
IMAGE_THRESHOLD = None
PIXEL_THRESHOLD = None
```

Therefore existing baseline supports continuous ranking metrics but not a final:

```text
GOOD
or
DEFECT
```

decision.

This is unfinished work.

Do not fake a threshold such as `0.5`.

A proper calibration protocol must be designed first.

## 28.1 SELECTED ANOMALY CALIBRATION PROTOCOL

The implementation uses a normal-only quantile protocol:

1. keep a held-out validation split containing only known `train/good` images,
2. compute the continuous image score and continuous pixel score for every
   validation image,
3. set each threshold to the 99th percentile of its validation score
   distribution, using linear quantile interpolation,
4. classify `score >= threshold` as `DEFECT`; otherwise classify it as `GOOD`,
5. persist the threshold, target acceptance rate, observed acceptance rate,
   sample count, quantile method, and source split.

The 99% target is chosen because no legitimate defect calibration set is
available and the official MVTec test labels must remain final evaluation data.
This protocol is simple, reproducible, and controls the known-normal
false-reject rate. It cannot claim defect recall, AUROC, AP, or F1 during
threshold selection; those metrics are reported only after the thresholds are
fixed and evaluated on the final test split.

Other possible strategies were considered:

- supervised GOOD/DEFECT tuning would use defect labels and risk leaking final
  evaluation information;
- fitting a parametric tail distribution adds assumptions that are not needed
  for this baseline;
- a fixed value such as `0.5` is not meaningful because anomaly scores are
  distances whose scale depends on the category feature bank.

The reusable implementation is in `src/mvtec_system/thresholds.py`.

---

# 29. TEST DATA RULE

Official MVTec test labels may be used for:

```text
final evaluation
research reporting
heatmap comparison
```

They must NOT be used for:

```text
classifier training
classifier hyperparameter tuning
classification confidence threshold tuning
anomaly threshold tuning
K selection
distance selection
```

Keep the final test set final.

---

# 30. REFACTOR STRATEGY

Do not rewrite notebooks wholesale.

Extract tested reusable logic gradually.

Recommended final structure:

```text
src/
├── data/
│   ├── mvtec.py
│   ├── validation.py
│   └── manifests.py
│
├── features/
│   └── dinov2.py
│
├── classification/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── anomaly/
│   ├── feature_bank.py
│   ├── knn.py
│   ├── thresholds.py
│   └── heatmap.py
│
├── routing/
│   └── category_router.py
│
├── evaluation/
│   ├── classification_metrics.py
│   ├── anomaly_metrics.py
│   └── reports.py
│
└── inference/
    └── pipeline.py
```

The notebooks remain as research evidence.

---

# 31. MIGRATION RULE: ONE PIECE AT A TIME

When extracting notebook logic:

```text
1. Identify existing notebook function.
2. Add equivalent reusable module.
3. Add tests.
4. Compare module result with notebook result.
5. Only then update notebook to import/reuse module if appropriate.
```

Do not perform a massive rewrite.

---

# 32. DO NOT BREAK NOTEBOOK REPRODUCIBILITY

The following notebooks should remain runnable:

```text
notebooks/01_data_exploration.ipynb
notebooks/02_first_model.ipynb
```

Do not delete valuable outputs.

Do not replace them with empty shells.

---

# 33. LOCAL ARTIFACT DISCOVERY

Because ignored artifacts may exist only on the developer machine, any artifact-dependent task must first discover them locally.

Search:

```text
artifacts/dinov2_knn/
```

for:

```text
feature_bank.pt
metrics.json
image_scores.csv
anomaly_maps.npz
bank_metadata.json
run_config.json
category_metrics.csv
macro_metrics.csv
```

Report what is found before regenerating.

Prefer the latest compatible successful run, not simply the newest directory by timestamp.

Compatibility is more important than recency.

---

# 34. ARTIFACT COMPATIBILITY RULE

A reusable bank/artifact is compatible only if relevant metadata matches the active baseline.

At minimum verify:

```text
model_name
model_revision
feature_key
image size
normalization
embedding dimension
patch grid
category
source train paths
K/distance when relevant
```

If metadata mismatch occurs:

```text
do not reuse silently
```

Report the mismatch.

---

# 35. DATASET COMPATIBILITY RULE

Before reusing `mvtec_ad_processed`, validate:

```text
15 categories discovered
train/good exists
test exists
ground_truth exists
processed images decode
images are expected size
masks align with anomalous test images
```

Do not require SHA hashes to match an imagined external reference.

Use the existing notebook validation logic.

---

# 36. NO DUPLICATE DATA STORAGE

Do not create copies like:

```text
datasets/mvtec_classifier/
datasets/mvtec_knn/
datasets/mvtec_v2_copy/
datasets/classification_images/
```

for the same images.

Use manifests and metadata to create logical splits.

Image storage remains:

```text
datasets/mvtec_ad/
datasets/mvtec_ad_processed/
```

---

# 37. NO DATABASE SERVER REQUIRED

For this current MVTec project:

```text
PostgreSQL is NOT required.
SQLite is NOT required.
Qdrant is NOT required.
```

The "existing database/data source" is the local MVTec dataset directory plus artifact files.

Use filesystem + manifests + versioned artifacts.

Do not add database infrastructure without explicit approval.

---

# 38. CURRENT ROADMAP — CONTINUE FROM EXISTING STATE

The roadmap begins from what is already complete.

---

## Task 000 — Local Reuse Audit

Before new development, inspect:

```text
datasets/mvtec_ad/
datasets/mvtec_ad_processed/
artifacts/dinov2_knn/
```

Report:

```text
raw dataset present?
processed dataset present?
processed validation status?
cached DINOv2 present?
existing feature banks?
existing score files?
existing anomaly maps?
existing metrics?
```

Do not modify anything.

This task establishes the actual local state.

---

## Task 001 — Create Shared DINOv2 Module

Extract the existing DINOv2 behavior from `02_first_model.ipynb`.

Must preserve:

```text
dinov2_vitb14
pinned revision
existing preprocessing assumptions
patch feature behavior
L2 normalization
```

Add:

```text
global embedding output
```

for future product classification.

Do not change KNN yet.

---

## Task 002 — Verify Refactored Patch Features Against Notebook

Use existing processed data.

For a small deterministic sample:

```text
notebook patch features
vs
module patch features
```

Verify:

```text
shape
finite values
numerical closeness
```

Do not proceed if refactor changes baseline unexpectedly.

---

## Task 003 — Build Classification Manifest From Existing Data

Reuse:

```text
datasets/mvtec_ad_processed/*/train/good/
```

Create:

```text
manifests/classification_split_v1.csv
```

Split each category's `train/good` images independently with:

```text
train:      70%
validation: 30%
seed:       42
```

Use a deterministic shuffle and `round(0.30 * category_count)` validation
images, with at least one image in each split. Record paths relative to
`datasets/mvtec_ad_processed/` and include only `train/good` images.

Do not copy images.

Do not use official test images.

---

## Task 004 — Precompute Global DINOv2 Embeddings

Use the shared DINOv2 module.

Input:

```text
classification manifest
+
existing processed images
```

Output:

```text
versioned train/validation embedding artifacts
```

Reuse existing embeddings if compatible artifacts already exist.

---

## Task 005 — Train Linear Product Classifier

Train only the lightweight classification head.

Input:

```text
cached global embeddings
```

Output:

```text
classifier weights
class mapping
training config
```

DINOv2 remains frozen.

---

## Task 006 — Evaluate Product Classification

Measure:

```text
accuracy
macro precision
macro recall
macro F1
top-3 accuracy
confusion matrix
per-class metrics
```

Do model selection using classification validation data.

Only after decisions are fixed, report final classification performance on MVTec test images.

---

## Task 007 — Calibrate Classification Routing Confidence

Using validation data:

```text
choose confidence and nearest-global-distance behavior
```

Support:

```text
known category
unknown_product_category
```

Calibrate and persist:

```text
classification confidence threshold
nearest-global-distance threshold
known-validation acceptance rate
calibration protocol
```

Do not choose arbitrary thresholds. Do not use official MVTec test labels.

---

## Task 008 — Implement Category Router

Map:

```text
classifier prediction
→ existing category feature bank
```

The router must apply the open-set rejection rule before loading a bank:

```text
unknown_product_category
→ return unknown result
→ skip category-specific anomaly inference
```

The router should locate compatible local feature banks.

If a bank is missing:

```text
report missing bank
```

and rebuild only that category when requested/approved.

---

## Task 009 — Refactor Existing Feature-Bank Code

Move current notebook feature-bank logic to reusable module.

Do NOT change algorithm.

Support:

```text
build
save
load
metadata validation
reuse
```

Critical requirement:

```text
load existing compatible bank instead of rebuilding
```

---

## Task 010 — Refactor Existing Exact kNN

Reuse notebook implementation.

Preserve baseline:

```text
K = 1
Euclidean
chunked exact distance
max patch score for image
```

Add tests.

Do not optimize yet.

---

## Task 011 — Refactor Existing Anomaly Map / Heatmap

Reuse current interpolation and crop behavior.

Verify output shape remains:

```text
256 × 256
```

for current processed baseline.

---

## Task 012 — Define Anomaly Threshold Calibration Protocol

This is new research work.

Do not use official test labels to tune the threshold.

Before implementation:

1. propose calibration strategies,
2. document tradeoffs,
3. choose one justified protocol,
4. then implement.

Do not silently use test labels for convenience.

---

## Task 013 — Add Image-Level GOOD / DEFECT Decision

After threshold protocol is approved:

```text
continuous score
→ per-category threshold
→ GOOD / DEFECT
```

Persist threshold metadata.

The image score is the maximum patch score, matching the baseline definition
in `AI config/model.md`. Task 013 stores one normal-only calibrated threshold
per category and applies `score >= threshold` for `DEFECT`; lower scores are
`GOOD`. The decision output includes the category, continuous score, applied
threshold, label, and boolean decision.

---

## Task 014 — Add Pixel-Level Binary Mask Threshold

Only if justified by the calibration protocol.

Output:

```text
binary anomaly mask
```

without tuning against final test data.

Task 014 uses the same normal-only quantile protocol as image decisions, but
calibrates one pixel threshold per category from the flattened continuous
anomaly maps of held-out normal validation images. The persisted metadata marks
the artifact as `pixel_level`; applying `score >= threshold` returns a boolean
mask with the original `[images, height, width]` shape. Final test masks and
defect labels are used only for later evaluation.

---

## Task 015 — Expand Evaluation Metrics

Keep existing:

```text
Image AUROC
Image AP
Pixel AUROC
Pixel AP
```

Add when thresholds are valid:

```text
Image precision
Image recall
Image F1
Specificity
Confusion matrix

Pixel precision
Pixel recall
Pixel F1
Dice
IoU
```

AUPRO is optional after the core metrics are correct.

---

## Task 016 — Unified Automatic Inference Pipeline

Implement:

```text
image
  ↓
shared DINOv2
  ├── global embedding → classifier → category
  └── patch embeddings
                    ↓
          existing category bank
                    ↓
                   kNN
                    ↓
                anomaly result
```

Before routing the patch embeddings, apply the calibrated open-set check:

```text
global embedding
  ├── rejected → unknown_product_category, stop
  └── accepted → predicted category bank → kNN anomaly result
```

DINOv2 should run only once for the input image.

---

## Task 017 — CLI Inference

Target:

```bash
python predict.py --image path/to/image.png
```

No category argument required.

Output:

```text
product category or unknown_product_category
classification confidence
nearest-global distance
reject thresholds
anomaly score
threshold
GOOD / DEFECT
heatmap path
```

---

## Task 018 — Oracle Routing vs Automatic Routing

Evaluate two modes.

### Oracle

```text
known true category
→ correct bank
→ anomaly detection
```

### Automatic

```text
classifier
→ predicted category
→ bank
→ anomaly detection
```

Report the difference.

This measures the cost of classification mistakes.

---

## Task 019 — Reproduce Existing Baseline After Refactor

Before declaring success, verify the refactored anomaly implementation remains close to the existing notebook baseline.

Reference macro results:

```text
Image AUROC ≈ 0.980325
Image AP    ≈ 0.987077
Pixel AUROC ≈ 0.961282
Pixel AP    ≈ 0.516239
```

Investigate meaningful deviation.

Do not simply overwrite the old reference.

---

## Task 020 — Automated Tests

Add tests for:

```text
dataset discovery
processed dataset validation
classification manifest reproducibility
shared DINOv2 interface
global embedding shape
patch embedding shape
classifier output
category router
bank compatibility
bank loading/reuse
KNN numerical behavior
anomaly map dimensions
threshold behavior
metrics
unified inference
```

---

## Task 021 — Artifact Versioning

Do not destroy old runs.

Create a clean versioned structure for new work.

Example:

```text
artifacts/
├── dinov2_knn/
│   ├── existing historical runs
│   └── ...
│
└── mvtec_system/
    └── v1/
        ├── classifier/
        ├── embeddings/
        ├── feature_banks/
        ├── thresholds/
        ├── evaluation/
        └── config.json
```

Where possible, reference compatible old banks instead of copying huge artifacts.

---

## Task 022 — Performance Measurement

Measure:

```text
DINOv2 inference latency
classification latency
bank loading
KNN latency
total inference latency
RAM usage
feature-bank size
MPS / CPU / CUDA behavior
```

Do not optimize without profiling.

---

## Task 023 — Update Notebooks

After reusable modules are stable:

```text
notebooks can import reusable modules
```

Preserve the existing research history.

Do not delete existing visualization evidence.

---

## Task 024 — Final README and Report

Update README to accurately describe what the repository actually does.

Include:

```text
MVTec AD setup
existing dataset paths
preprocessing
shared DINOv2
product classification
category routing
category-specific KNN
thresholds
heatmaps
evaluation
CLI usage
results
reproduction steps
limitations
```

---

# 39. FIRST ACTION FOR AN AI CODING AGENT

Do NOT immediately implement Task 001.

First perform **Task 000 — Local Reuse Audit**.

Use this prompt:

```text
You are continuing the existing repository:
TTD12Super/industrial-vision-anomaly-detection

Branch target:
develop

This is NOT a new project.

Before changing any code, perform a read-only Local Reuse Audit.

Inspect:
- datasets/mvtec_ad/
- datasets/mvtec_ad_processed/
- artifacts/dinov2_knn/
- notebooks/01_data_exploration.ipynb
- notebooks/02_first_model.ipynb
- .gitignore
- requirements.txt

Important:
The dataset and most artifacts are intentionally ignored by Git, so inspect the LOCAL filesystem rather than assuming GitHub reflects the full project state.

Report:
1. Whether datasets/mvtec_ad exists.
2. Whether all 15 expected MVTec categories exist.
3. Whether datasets/mvtec_ad_processed exists.
4. Whether processed images match the existing 256x256 baseline.
5. Whether the processed dataset passes the existing notebook validation rules.
6. Whether cached DINOv2 weights exist under artifacts/dinov2_knn/torch_hub.
7. Every local feature_bank.pt that exists.
8. Every local metrics.json, image_scores.csv, anomaly_maps.npz, and run_config.json that exists.
9. Which artifact run(s) appear compatible with the current baseline.
10. What can be reused without recomputation.
11. What is actually missing.

Do NOT:
- download MVTec,
- preprocess data,
- rebuild feature banks,
- rerun DINOv2,
- change files,
- delete files,
- move files,
- push or merge.

Finish with a recommended smallest next step based on the actual local state.
```

---

# 40. AFTER TASK 000

Only after the audit should the AI agent continue.

If all current data/artifacts exist:

```text
reuse them
→ start Task 001
```

If processed dataset is missing but raw data exists:

```text
reuse raw
→ rerun ONLY existing preprocessing
→ validate
→ continue
```

If feature banks are missing:

```text
do not rebuild immediately
→ first refactor/load builder correctly
→ rebuild only required banks
```

If everything is present:

```text
do not run expensive baseline again
```

---

# 41. DEFINITION OF "REUSE"

Reuse means:

```text
Existing valid file
→ read it
→ validate metadata
→ use it
```

It does NOT mean:

```text
existing file
→ copy it into another folder
→ create duplicates
```

Prefer references/manifests over duplicate image storage.

---

# 42. DEFINITION OF DONE FOR THE CONTINUED PROJECT

The MVTec project is complete when:

```text
[ ] Existing raw dataset is preserved.
[ ] Existing processed dataset is reused.
[ ] Existing DINOv2 baseline remains reproducible.
[ ] Existing compatible feature banks can be loaded/reused.
[ ] Shared DINOv2 module exists.
[ ] Product classifier handles all 15 categories.
[ ] Product classifier does not use test data for training.
[ ] Classification confidence/routing is defined.
[ ] Inputs outside the known product distribution are rejected as unknown_product_category.
[ ] Unknown rejection thresholds are calibrated without official test labels.
[ ] Category router selects the correct bank.
[ ] KNN implementation is reusable outside notebook.
[ ] Anomaly threshold protocol is justified.
[ ] GOOD / DEFECT output exists.
[ ] Heatmap/localization remains available.
[ ] Existing ranking metrics remain available.
[ ] Thresholded metrics are added.
[ ] Oracle routing is evaluated.
[ ] Automatic routing is evaluated.
[ ] CLI inference works without manually passing category.
[ ] Tests cover core reusable code.
[ ] New artifacts are versioned without deleting old evidence.
[ ] README accurately documents the completed system.
```

---

# 43. NON-GOALS

Until the above project is complete, do NOT introduce:

```text
EfficientAD
FastFlow
PaDiM
Autoencoder
new anomaly backbone
Jade dataset
Jade service
FastAPI
PostgreSQL
Qdrant
RAG
LLM
recommendation system
e-commerce integration
```

Finish the current MVTec system first.

---

# 44. FINAL PRINCIPLE

The correct continuation strategy is:

```text
INSPECT
   ↓
REUSE EXISTING DATA
   ↓
REUSE EXISTING PROCESSED DATA
   ↓
REUSE EXISTING MODEL CACHE
   ↓
REUSE EXISTING FEATURE BANKS / SCORES WHEN COMPATIBLE
   ↓
REFACTOR WORKING NOTEBOOK LOGIC
   ↓
ADD PRODUCT CLASSIFICATION
   ↓
ADD CATEGORY ROUTING
   ↓
FINISH THRESHOLDING + METRICS
   ↓
BUILD UNIFIED CLI INFERENCE
   ↓
TEST + DOCUMENT
```

Not:

```text
DELETE
↓
REDOWNLOAD
↓
REPROCESS
↓
RETRAIN EVERYTHING
↓
START OVER
```

The existing research work is the foundation of the finished project.
