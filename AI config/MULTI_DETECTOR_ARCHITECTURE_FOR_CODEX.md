# ARCHITECTURE — MULTI-DETECTOR INDUSTRIAL ANOMALY SYSTEM

> Goal: Extend the current DINOv2 + Exact KNN system with two additional anomaly detectors:
>
> 1. **PatchCore-style**
> 2. **PaDiM-style**
>
> while preserving the existing baseline and making the whole project reusable for future factory datasets.

---

# 1. FINAL HIGH-LEVEL SYSTEM

```text
                           INPUT IMAGE
                               │
                               ▼
                     Image Preprocessing
                               │
                               ▼
                    Shared Frozen DINOv2
                         (ONE backbone)
                               │
             ┌─────────────────┴──────────────────┐
             │                                    │
             ▼                                    ▼
      Global Embedding                     Patch Features
             │                                    │
             ▼                                    │
     Product Classifier                           │
             │                                    │
             ▼                                    │
      Confidence Check                            │
             │                                    │
      ┌──────┴──────┐                             │
      │             │                             │
confidence < 0.50   confidence >= 0.50            │
      │             │                             │
      ▼             ▼                             │
    UNKNOWN      Category                         │
      │             │                             │
      │             ▼                             │
      │       Category Router                     │
      │             │                             │
      │             ▼                             │
      │      Selected Detector                    │
      │      exact_knn / patchcore / padim        │
      │             │                             │
      └──── STOP    └──────────────┬──────────────┘
                                   │
                                   ▼
                         Anomaly Score + Map
                                   │
                                   ▼
                         Category Threshold
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                       GOOD                DEFECT
                                               │
                                               ▼
                                            Heatmap
```

Important:

- DINOv2 must be loaded only once.
- Product recognition and anomaly detection reuse the same backbone.
- Product classification chooses the correct product/category.
- Anomaly detector decides whether that product is normal or defective.
- `UNKNOWN` must be separate from `DEFECT`.

---

# 2. DETECTOR REGISTRY

The anomaly system must expose three detectors through one common interface.

```text
                    Detector Registry
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
     Exact KNN        PatchCore          PaDiM
     baseline          style             style
```

Configuration decides which detector is active.

Example:

```yaml
anomaly:
  detector: patchcore
```

Supported values:

```text
exact_knn
patchcore
padim
```

---

# 3. COMMON DETECTOR INTERFACE

All detectors should follow the same contract.

Conceptual API:

```python
class AnomalyDetector:
    def fit(self, category, normal_images, feature_extractor, config):
        ...

    def predict(self, features):
        ...

    def save(self, artifact_dir):
        ...

    def load(self, artifact_dir):
        ...

    def validate_artifacts(self):
        ...
```

Common output:

```python
AnomalyResult(
    image_score=float,
    anomaly_map=array,
    metadata=dict,
)
```

The inference pipeline must not care which detector produced the result.

---

# 4. ARCHITECTURE A — EXISTING EXACT KNN BASELINE

## Training / Build

```text
Normal images
(train/good)
     │
     ▼
Shared Frozen DINOv2
     │
     ▼
Final Patch Embeddings
     │
     ▼
L2 Normalization
     │
     ▼
Store ALL normal patches
     │
     ▼
Category Feature Bank
```

Example:

```text
bottle/train/good
       ↓
DINOv2
       ↓
all bottle normal patch features
       ↓
bottle feature bank
```

## Inference

```text
Input image
    │
    ▼
DINOv2 Patch Embeddings
    │
    ▼
Exact Euclidean KNN
against category feature bank
    │
    ▼
Patch Distances
    │
    ├───────────────┐
    │               │
    ▼               ▼
Max score       Patch score grid
    │               │
    ▼               ▼
Image score      Upsample
                    │
                    ▼
                Anomaly map
```

Role:

```text
Question:
"Does every patch look similar to at least one known normal patch?"
```

Main strength:

```text
simple
strong baseline
easy to explain
high image-level performance
```

Main weakness:

```text
large feature bank
slow exact search
single-patch comparison can create local false positives
```

---

# 5. ARCHITECTURE B — PATCHCORE-STYLE

PatchCore remains a **memory-based nearest-neighbor detector**, but improves how normal patch representations are built.

## 5.1 PatchCore Training / Build

```text
                     NORMAL IMAGES
                          │
                          ▼
                 Shared Frozen DINOv2
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
       Layer 6         Layer 9        Layer 12
     fine details     texture/shape    semantic
           │              │              │
           └──────────────┬───────────────┘
                          ▼
                  Feature Alignment
                          │
                          ▼
                Multi-layer Fusion
                          │
                          ▼
              Local Neighborhood Aggregation
                    (example 3×3)
                          │
                          ▼
                  Patch Descriptors
                          │
                          ▼
                  Coreset Selection
                          │
                          ▼
          Representative Normal Memory Bank
```

### Layer 1 — Normal Images
Purpose: provide only normal training examples.

### Layer 2 — Frozen DINOv2
Purpose: convert pixels into visual representations while remaining frozen.

### Layer 3 — Multi-layer Features
Use configurable intermediate and final transformer layers.

Example interpretation:

```text
Layer 6  → fine local structures
Layer 9  → texture + shape
Layer 12 → semantic information
```

Purpose: combine local defect sensitivity with semantic robustness.

### Layer 4 — Feature Alignment
Purpose: align spatial feature shapes before fusion.

### Layer 5 — Multi-layer Fusion
Purpose: create one richer patch representation from multiple DINOv2 layers.

### Layer 6 — Local Neighborhood Aggregation

```text
      ┌───┬───┬───┐
      │ A │ B │ C │
      ├───┼───┼───┤
      │ D │ X │ E │
      ├───┼───┼───┤
      │ F │ G │ H │
      └───┴───┴───┘
```

Purpose:
- let patch `X` use neighboring context,
- reduce isolated false-positive noise,
- improve localization consistency.

### Layer 7 — Patch Descriptors
Purpose: output one context-aware feature vector per spatial location.

### Layer 8 — Coreset Selection

```text
500,000 normal patch descriptors
            ↓
      representative selection
            ↓
50,000 representative descriptors
```

Purpose:
- reduce RAM,
- reduce nearest-neighbor latency,
- remove redundant normal examples,
- preserve coverage of normal feature space.

Config example:

```yaml
patchcore:
  layers: [6, 9, 12]
  neighborhood_size: 3
  coreset_ratio: 0.10
  k: 1
```

## 5.2 PatchCore Inference

```text
                   INPUT IMAGE
                       │
                       ▼
                  DINOv2
                       │
            Selected Multi-Layers
                       │
                       ▼
               Feature Fusion
                       │
                       ▼
          Local Neighborhood Aggregation
                       │
                       ▼
               Query Descriptors
                       │
                       ▼
             KNN vs Coreset Bank
                       │
                       ▼
              Patch Anomaly Scores
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Image Score          Score Grid
                                  │
                                  ▼
                               Upsample
                                  │
                                  ▼
                              Anomaly Map
```

Main idea:

```text
"Does this local region look like any representative normal region?"
```

---

# 6. ARCHITECTURE C — PADIM-STYLE

PaDiM is a distribution-based detector.

It asks:

```text
"Does this feature belong to the normal statistical distribution
at this spatial position?"
```

## 6.1 PaDiM Training / Build

```text
                     NORMAL IMAGES
                          │
                          ▼
                 Shared Frozen DINOv2
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
       Layer 6         Layer 9        Layer 12
           │              │              │
           └──────────────┬───────────────┘
                          ▼
                  Feature Alignment
                          │
                          ▼
                  Multi-layer Fusion
                          │
                          ▼
                Dimension Reduction
                          │
                          ▼
            Features at Every Position
                          │
                          ▼
          ┌────────────────────────────┐
          │ For each spatial position │
          │                            │
          │ mean μ                     │
          │ covariance Σ               │
          └────────────────────────────┘
                          │
                          ▼
               Save Gaussian Statistics
```

### Layer 1 — Normal Images
Purpose: provide normal examples only.

### Layer 2 — DINOv2 Multi-layer Features
Purpose: extract visual features at multiple abstraction levels.

### Layer 3 — Feature Alignment + Fusion
Purpose: produce one spatial feature representation per position.

### Layer 4 — Dimension Reduction

```text
high-dimensional DINOv2 features
              ↓
        reduced representation
```

Purpose:
- make covariance estimation stable,
- reduce memory,
- reduce compute.

Config example:

```yaml
padim:
  layers: [6, 9, 12]
  reduced_dim: 100
  covariance_regularization: 0.01
```

### Layer 5 — Spatial Position Modeling

For a feature grid such as:

```text
19 × 19
```

every position gets its own normal statistical model.

Example:

```text
position (1,1)   → Gaussian A
position (1,2)   → Gaussian B
...
position (19,19) → Gaussian N
```

Purpose: learn what normally appears at each product location.

### Layer 6 — Mean and Covariance

For every position:

```text
normal features:
x1, x2, ..., xN

mean       = μ
covariance = Σ
```

Interpretation:

```text
μ → center of normal behavior
Σ → permitted normal variation
```

## 6.2 PaDiM Inference

```text
                    INPUT IMAGE
                        │
                        ▼
                      DINOv2
                        │
                        ▼
                Multi-layer Features
                        │
                        ▼
                Feature Alignment
                        │
                        ▼
                Dimension Reduction
                        │
                        ▼
              Feature at each position
                        │
                        ▼
              Mahalanobis Distance
            against μ and Σ at position
                        │
                        ▼
               Spatial Anomaly Scores
                        │
                ┌───────┴───────┐
                ▼               ▼
           Image Score       Score Grid
                                 │
                                 ▼
                              Upsample
                                 │
                                 ▼
                             Anomaly Map
```

Mahalanobis distance:

```text
d(x) = sqrt((x - μ)^T Σ^-1 (x - μ))
```

Small distance:

```text
fits normal distribution
→ likely normal
```

Large distance:

```text
statistically unusual
→ likely anomaly
```

---

# 7. CORE DIFFERENCE BETWEEN THE THREE DETECTORS

```text
Exact KNN:

query patch
   ↓
nearest raw normal patch
   ↓
distance
```

```text
PatchCore:

query context-aware descriptor
   ↓
nearest representative normal descriptor
   ↓
distance
```

```text
PaDiM:

query spatial feature
   ↓
normal Gaussian at same position
   ↓
Mahalanobis distance
```

---

# 8. COMPARISON

| Property | Exact KNN | PatchCore-style | PaDiM-style |
|---|---|---|---|
| Normal representation | All patch features | Representative coreset | Mean + covariance |
| Comparison | Euclidean nearest neighbor | KNN to coreset | Mahalanobis |
| Multi-layer | Current baseline: no | Yes | Yes |
| Local context | Limited | Yes | Position-based statistics |
| Memory bank | Large | Smaller | No KNN bank |
| Spatial alignment dependency | Medium | Medium | Higher |
| Search cost | High | Lower | No NN search |
| Main concept | nearest normal patch | representative normal memory | normal distribution |

---

# 9. SHARED POST-PROCESSING

All detectors should first produce a raw continuous anomaly map.

```text
Raw anomaly map
      │
      ▼
Gaussian smoothing
      │
      ▼
Pixel threshold
      │
      ▼
Morphological opening
      │
      ▼
Connected components
      │
      ▼
Remove tiny false-positive islands
      │
      ▼
Final binary mask
```

Important:

```text
Pixel AP must be calculated from continuous anomaly scores,
not from the final binary mask.
```

---

# 10. DATASET-AGNOSTIC PRODUCTION ARCHITECTURE

MVTec must not be hard-coded inside the ML detector logic.

```text
                     DATASET
                        │
                        ▼
                  DatasetAdapter
                        │
                        ▼
                Category Discovery
                        │
                        ▼
                 Shared DINOv2
                        │
               ┌────────┴────────┐
               ▼                 ▼
       Product Recognition   Patch Features
               │                 │
               ▼                 │
          Category Router        │
               │                 │
               └───────┬─────────┘
                       ▼
                 Detector Registry
              ┌────────┼────────┐
              ▼        ▼        ▼
         Exact KNN  PatchCore  PaDiM
              │        │        │
              └────────┼────────┘
                       ▼
               Anomaly Result
                       │
                       ▼
             Threshold + Heatmap
                       │
                       ▼
                 GOOD / DEFECT
```

---

# 11. DATASET ADAPTER

Core code should depend on a generic interface:

```python
class IndustrialDatasetAdapter:
    def categories(self):
        ...

    def normal_train_images(self, category):
        ...

    def validation_images(self, category):
        ...

    def test_images(self, category):
        ...

    def ground_truth_mask(self, sample):
        ...
```

Implement:

```text
MVTecDatasetAdapter
GenericFolderDatasetAdapter
```

---

# 12. NEW FACTORY WORKFLOW

A real factory should only need:

```text
new dataset
+
config file
```

Example:

```text
factory_dataset/
├── bearing/
│   ├── train/good/
│   ├── validation/good/
│   └── test/
│
├── gear/
│   ├── train/good/
│   ├── validation/good/
│   └── test/
│
└── connector/
    ├── train/good/
    ├── validation/good/
    └── test/
```

Then:

```bash
python build_system.py --config configs/factory.yaml
```

Build flow:

```text
Dataset
   ↓
Validate
   ↓
Discover categories
   ↓
DINOv2 features
   ↓
Product classifier
   ↓
Build selected detector artifacts
   ├─ Exact KNN banks
   ├─ PatchCore coresets
   └─ PaDiM statistics
   ↓
Threshold calibration
   ↓
Production artifacts
```

No detector source-code change should be required.

---

# 13. CONFIG-DRIVEN DETECTOR SELECTION

Example:

```yaml
dataset:
  type: generic_folder
  root: datasets/factory

backbone:
  name: dinov2_vitb14

classification:
  unknown_confidence_threshold: 0.50

anomaly:
  detector: patchcore

patchcore:
  layers: [6, 9, 12]
  neighborhood_size: 3
  coreset_ratio: 0.10
  k: 1

padim:
  layers: [6, 9, 12]
  reduced_dim: 100
  covariance_regularization: 0.01
```

Changing:

```yaml
anomaly:
  detector: padim
```

should switch detector behavior without changing application code.

---

# 14. RECOMMENDED SOURCE ARCHITECTURE

```text
src/
├── data/
│   ├── base.py
│   ├── mvtec.py
│   └── generic_folder.py
│
├── features/
│   └── dinov2.py
│
├── classification/
│   ├── model.py
│   └── router.py
│
├── anomaly/
│   ├── base.py
│   ├── registry.py
│   │
│   ├── exact_knn/
│   │   ├── detector.py
│   │   └── feature_bank.py
│   │
│   ├── patchcore/
│   │   ├── detector.py
│   │   ├── feature_fusion.py
│   │   ├── local_aggregation.py
│   │   └── coreset.py
│   │
│   └── padim/
│       ├── detector.py
│       ├── statistics.py
│       ├── reduction.py
│       └── mahalanobis.py
│
├── postprocessing/
│   ├── smoothing.py
│   ├── morphology.py
│   └── components.py
│
├── evaluation/
│   └── compare_detectors.py
│
└── inference/
    └── pipeline.py
```

---

# 15. ARTIFACT ARCHITECTURE

```text
artifacts/
└── systems/
    └── <dataset_name>/
        └── v1/
            ├── system_manifest.json
            ├── classifier/
            ├── routing/
            ├── thresholds/
            │
            └── detectors/
                ├── exact_knn/
                │   ├── category_A/
                │   └── category_B/
                │
                ├── patchcore/
                │   ├── category_A/
                │   │   └── coreset.pt
                │   └── category_B/
                │
                └── padim/
                    ├── category_A/
                    │   ├── mean.pt
                    │   └── covariance.pt
                    └── category_B/
```

---

# 16. FINAL INFERENCE FLOW

```text
                        IMAGE
                          │
                          ▼
                     Preprocess
                          │
                          ▼
                Shared DINOv2 Forward
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
      Global Embedding            Patch Features
              │                        │
              ▼                        │
      Product Classifier               │
              │                        │
      confidence < 0.50 ?              │
          │           │                │
         YES          NO               │
          │           │                │
          ▼           ▼                │
       UNKNOWN      Category            │
          │           │                │
         STOP         ▼                │
                 Category Router       │
                       │               │
                       └───────┬───────┘
                               ▼
                       Selected Detector
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
                Exact KNN   PatchCore   PaDiM
                    │          │          │
                    └──────────┼──────────┘
                               ▼
                       Raw Anomaly Map
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
               Image Score          Post-processing
                    │                     │
                    ▼                     ▼
            Category Threshold       Binary Mask
                    │                     │
              ┌─────┴─────┐               │
              ▼           ▼               │
            GOOD        DEFECT             │
                            └───────┬───────┘
                                    ▼
                                Final Result
```

---

# 17. WHAT CODEX MUST PRESERVE

Do not:

```text
- remove Exact KNN
- replace the historical baseline
- hard-code MVTec categories
- duplicate DINOv2 loaders
- use test labels for tuning
- delete notebooks
- overwrite old artifacts
```

The system must preserve three independently selectable detectors:

```text
Exact KNN
PatchCore-style
PaDiM-style
```

---

# 18. FINAL RESEARCH COMPARISON

The final project should support:

```text
                 SAME DATASET
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Exact KNN     PatchCore      PaDiM
        │            │            │
        └────────────┼────────────┘
                     ▼
                 Evaluation
```

Compare:

```text
Image AUROC
Image AP
Pixel AUROC
Pixel AP
Dice
IoU
F1
Latency
RAM
Artifact size
```

Do not assume a winner in advance.

Measure which detector works best for the active dataset.

---

# 19. FINAL PRODUCTION PRINCIPLE

```text
NEW FACTORY
    │
    ▼
NEW DATASET
    │
    ▼
CHANGE CONFIG
    │
    ▼
build_system.py
    │
    ▼
Automatically build:
- product classifier
- router
- selected anomaly detector
- detector artifacts
- thresholds
- production manifest
    │
    ▼
predict.py
```

The factory should not need to rewrite ML source code.

MVTec is only the current benchmark.

The reusable product is:

```text
INDUSTRIAL VISUAL ANOMALY DETECTION ENGINE
```
