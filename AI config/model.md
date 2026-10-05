# MODEL ARCHITECTURE SPECIFICATION

## Industrial Vision Anomaly Detection

This document defines the required model architecture for the Industrial Vision Anomaly Detection project.

Codex MUST read this document completely before implementing any model-related code.

This file is the source of truth for:

* model architecture
* feature extraction
* feature bank construction
* anomaly scoring
* defect localization
* evaluation
* model-related implementation constraints

Do NOT silently replace this architecture with another anomaly detection method.

---

# 1. Project Goal

The goal of the model is to detect visual anomalies in industrial images using a pretrained vision foundation model.

The system must support two levels of anomaly detection:

1. Image-level anomaly detection

   * Determine whether an image is likely GOOD or DEFECT.

2. Pixel/region-level defect localization

   * Determine where the anomalous region is located in the image.

The initial dataset is:

```text
MVTec AD
```

The initial model architecture is based on:

```text
DINOv2 ViT-B/14
```

DINOv2 is used as a frozen visual feature extractor.

The initial version MUST NOT fine-tune DINOv2.

---

# 2. Required Architecture

The required architecture is:

```text
                    MVTec AD
                       │
                       ▼
              DINOv2 ViT-B/14
                       │
                    Frozen
                       │
                       ▼
                 Patch Features
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
        Feature Bank         Test Image
      from TRAIN/GOOD             │
             │                    ▼
             │             DINOv2 ViT-B/14
             │                  Frozen
             │                    │
             │                    ▼
             │              Patch Features
             │                    │
             └───────────┬────────┘
                         ▼
                        kNN
                         │
                         ▼
                  Anomaly Scores
                    /         \
                   /           \
                  ▼             ▼
           Image-level      Patch-level
           Anomaly Score    Anomaly Scores
                  │             │
                  ▼             ▼
             GOOD/DEFECT    Anomaly Map
                                │
                                ▼
                       Defect Localization
```

Codex MUST preserve this architecture unless the user explicitly requests a change.

---

# 3. Core Architecture Principle

The model is NOT trained as a traditional supervised classifier.

There should initially be:

* no GOOD/DEFECT classification head
* no neural network classifier trained from scratch
* no end-to-end DINOv2 training
* no DINOv2 fine-tuning
* no synthetic anomaly generation
* no reconstruction network
* no segmentation network

Instead, anomaly detection is performed in feature space.

The main idea is:

```text
Normal training images
        ↓
DINOv2
        ↓
Normal patch embeddings
        ↓
Feature Bank
```

During inference:

```text
Test image
    ↓
DINOv2
    ↓
Patch embeddings
    ↓
Compare with normal Feature Bank
    ↓
Nearest-neighbor distance
    ↓
Anomaly score
```

A patch that is very different from normal training patches should receive a high anomaly score.

---

# 4. Dataset Usage

The dataset structure should already have been validated by the preprocessing stage.

Expected structure:

```text
datasets/
├── mvtec_ad/
│
└── mvtec_ad_processed/
```

Model-related code SHOULD use:

```text
datasets/mvtec_ad_processed/
```

rather than modifying or directly preprocessing the original raw dataset.

The raw dataset:

```text
datasets/mvtec_ad/
```

MUST remain unchanged.

---

# 5. Train Data

Only normal training images are used to construct the normal feature bank.

For MVTec AD:

```text
train/good/
```

is the reference normal dataset.

Example:

```text
datasets/mvtec_ad_processed/
└── bottle/
    └── train/
        └── good/
```

Codex MUST NOT include anomalous test images in the training feature bank.

The feature bank must represent normal visual patterns only.

---

# 6. Test Data

The test dataset contains:

```text
test/good/
```

and anomaly categories such as:

```text
test/broken_large/
test/broken_small/
test/contamination/
...
```

The exact anomaly classes differ between MVTec categories.

Do NOT hardcode anomaly class names globally.

Discover them from the dataset structure.

Test data MUST never be added to the training feature bank.

---

# 7. Ground Truth

MVTec anomaly masks are located under:

```text
ground_truth/
```

They are used only for localization evaluation.

Ground-truth masks MUST NOT be used to build the normal feature bank.

They MUST NOT be used during inference to calculate anomaly scores.

Their purpose is evaluation only.

---

# 8. Backbone

Use:

```text
DINOv2 ViT-B/14
```

as the initial feature extractor.

The backbone MUST initially remain frozen.

Conceptually:

```python
for parameter in model.parameters():
    parameter.requires_grad = False
```

The model should operate in:

```text
evaluation mode
```

during feature extraction and inference.

Do not implement fine-tuning unless explicitly requested later.

---

# 9. Feature Extraction

The model must extract PATCH-LEVEL features.

Do NOT rely only on the CLS/global token.

The anomaly localization pipeline requires spatial patch features.

For an input image:

```text
Image
  ↓
DINOv2
  ↓
Patch tokens
```

Conceptually:

```text
[N_patches, embedding_dimension]
```

For a batch:

```text
[B, N_patches, embedding_dimension]
```

The exact tensor dimensions MUST be inspected from the actual DINOv2 implementation instead of blindly hardcoding them.

---

# 10. Patch Features

Patch embeddings represent local visual regions of the image.

Example concept:

```text
Image
──────────────────────
│ p1 │ p2 │ p3 │ p4 │
──────────────────────
│ p5 │ p6 │ p7 │ p8 │
──────────────────────
│ p9 │...             │
──────────────────────
```

Each patch produces an embedding:

```text
patch → embedding vector
```

Anomaly detection compares these embeddings with embeddings extracted from normal training images.

This patch-level representation is required for localization.

---

# 11. Feature Bank

A normal feature bank must be created using:

```text
TRAIN / GOOD images
```

Workflow:

```text
Train GOOD Image 1
        ↓
DINOv2
        ↓
Patch Features
        │
Train GOOD Image 2
        ↓
DINOv2
        ↓
Patch Features
        │
       ...
        │
        ▼
Normal Feature Bank
```

Conceptually:

```text
feature_bank.shape

[number_of_normal_patches, embedding_dimension]
```

Every feature in the bank represents a visual patch observed in normal industrial samples.

---

# 12. Feature Bank Construction

The implementation should approximately follow:

```text
for each GOOD training image:

    load image

    apply required preprocessing

    run DINOv2

    extract patch features

    detach from computation graph

    move features to selected storage device

    append to feature bank
```

After all training images:

```text
concatenate patch embeddings
```

to construct the final feature bank.

---

# 13. Feature Bank Memory Management

The implementation MUST consider feature bank size.

Storing every patch from every image may consume significant memory.

The initial implementation MAY store all features if memory permits.

However, code should be structured so that later optimizations can be added, such as:

```text
random sampling
coreset sampling
feature subsampling
FAISS indexing
GPU nearest-neighbor search
```

Do NOT implement these optimizations prematurely unless necessary.

Correctness comes before optimization.

---

# 14. Feature Normalization

Before distance calculation, Codex should inspect whether L2 normalization is appropriate for the selected DINOv2 feature representation.

If used:

```text
feature
   ↓
L2 normalization
   ↓
normalized feature
```

The same normalization MUST be applied consistently to:

```text
training feature bank
test patch features
```

Do not normalize one side only.

The chosen behavior must be documented.

---

# 15. Test Feature Extraction

Inference pipeline:

```text
Test Image
    ↓
same preprocessing
    ↓
Frozen DINOv2
    ↓
Patch Features
```

Training and test images MUST use compatible preprocessing.

Do not accidentally change:

```text
image size
normalization
color space
feature layer
embedding format
```

between feature-bank generation and test inference.

---

# 16. k-Nearest Neighbor Anomaly Detection

The initial anomaly detector is based on nearest-neighbor distance.

For every test patch feature:

```text
test patch
     ↓
compare with
     ↓
normal feature bank
```

Calculate the distance to normal features.

Initial implementation should support:

```text
kNN
```

At minimum:

```text
k = 1
```

may be used as the baseline.

---

# 17. Patch Anomaly Score

For each test patch:

```text
anomaly_score(patch)
=
distance to nearest normal feature
```

Conceptually:

```text
s_i = min_j d(q_i, m_j)
```

where:

```text
q_i = test patch feature

m_j = normal memory-bank feature
```

Higher distance means:

```text
more anomalous
```

Lower distance means:

```text
more similar to normal training data
```

---

# 18. Distance Metric

The initial implementation should support an appropriate embedding-space distance such as:

```text
Euclidean distance
```

or another justified similarity/distance metric.

Do NOT randomly switch metrics.

The selected metric must be used consistently for:

```text
feature bank comparison
patch scoring
image scoring
evaluation
```

The implementation should document the selected metric.

---

# 19. Image-Level Anomaly Score

The patch anomaly scores must be converted into an image-level anomaly score.

Baseline:

```text
image anomaly score
=
maximum patch anomaly score
```

Conceptually:

```text
S_image = max(S_patch)
```

This allows a small localized defect to influence the final image prediction.

Alternative aggregation strategies may be researched later, but the initial implementation should remain simple and reproducible.

---

# 20. GOOD / DEFECT Prediction

Image-level prediction:

```text
image anomaly score
        ↓
threshold
        ↓
GOOD / DEFECT
```

Conceptually:

```text
score < threshold
→ GOOD

score >= threshold
→ DEFECT
```

Threshold selection MUST NOT use test labels in an invalid way that causes evaluation leakage.

Codex must clearly document how the threshold is selected.

If no justified threshold is available yet, anomaly scores should first be evaluated as continuous values before forcing binary classification.

---

# 21. Patch-Level Anomaly Map

Patch anomaly scores retain spatial information.

Example:

```text
Patch score grid

0.1  0.1  0.2  0.1
0.1  0.8  0.9  0.2
0.1  0.7  0.8  0.1
0.1  0.1  0.2  0.1
```

These scores should be reshaped according to the spatial DINOv2 patch grid.

Conceptually:

```text
[N_patches]
      ↓
[H_patch, W_patch]
```

Do NOT assume the patch-grid dimensions blindly.

Derive them from the actual input resolution and backbone behavior.

---

# 22. Anomaly Map Upsampling

The patch anomaly map must be resized back to the image resolution for visualization and localization.

Pipeline:

```text
Patch anomaly scores
        ↓
reshape
        ↓
Low-resolution anomaly map
        ↓
Interpolation
        ↓
Image-resolution anomaly map
```

The resulting anomaly map should align spatially with the input image.

---

# 23. Defect Localization

Localization output should allow:

```text
Original Image
      +
Anomaly Map
      ↓
Defect Location
```

The model should be able to highlight suspicious image regions.

Expected visualization later:

```text
Original Image

Ground Truth Mask

Predicted Anomaly Map

Overlay
```

These visualizations MUST remain inside notebooks or model evaluation outputs unless a separate dashboard task is explicitly requested.

DO NOT create a dashboard.

---

# 24. Localization Threshold

A localization threshold may later convert:

```text
continuous anomaly map
```

into:

```text
binary predicted anomaly mask
```

Example:

```text
anomaly map
     ↓
pixel threshold
     ↓
predicted mask
```

Thresholding is primarily useful for visualization and segmentation metrics.

Do not destroy the continuous anomaly map.

Always preserve continuous scores for evaluation.

---

# 25. Model Pipeline

The complete pipeline should be conceptually separated into:

```text
1. Load backbone

2. Extract normal features

3. Build feature bank

4. Extract test features

5. Run nearest-neighbor search

6. Calculate patch anomaly scores

7. Calculate image anomaly score

8. Generate anomaly map

9. Upsample anomaly map

10. Evaluate detection

11. Evaluate localization
```

Each component should have a clear responsibility.

Avoid writing the entire model pipeline inside one large function.

---

# 26. Suggested Python Modules

When model implementation begins, a structure similar to the following is recommended:

```text
src/
├── models/
│   ├── dinov2_backbone.py
│   ├── feature_extractor.py
│   └── anomaly_detector.py
│
├── features/
│   └── feature_bank.py
│
├── inference/
│   └── anomaly_inference.py
│
├── evaluation/
│   ├── image_metrics.py
│   └── localization_metrics.py
│
└── utils/
    └── visualization.py
```

Codex MUST inspect the existing repository structure before creating directories.

Do NOT create duplicate architecture if equivalent modules already exist.

---

# 27. Suggested Model Responsibilities

## `dinov2_backbone.py`

Responsible for:

```text
loading DINOv2
freezing model
setting evaluation mode
providing backbone inference
```

It should NOT contain:

```text
dataset loading
kNN
evaluation
visualization
```

---

## `feature_extractor.py`

Responsible for:

```text
extracting DINOv2 patch embeddings
handling patch token representation
returning spatial metadata when needed
```

---

## `feature_bank.py`

Responsible for:

```text
building the normal feature bank
storing embeddings
loading embeddings
saving embeddings if needed
```

---

## `anomaly_detector.py`

Responsible for:

```text
nearest-neighbor search
patch anomaly scores
image anomaly scores
```

---

## `anomaly_inference.py`

Responsible for orchestrating:

```text
image
→ feature extraction
→ kNN
→ score
→ anomaly map
```

---

## `visualization.py`

Responsible only for visualization such as:

```text
input image
ground truth
anomaly heatmap
overlay
```

It MUST NOT create a web dashboard.

---

# 28. Evaluation

The architecture should eventually support both:

```text
image-level evaluation
pixel-level/localization evaluation
```

---

# 29. Image-Level Metrics

Important image-level metrics may include:

```text
AUROC
Average Precision
Precision
Recall
F1 Score
```

AUROC should be prioritized when comparing continuous anomaly scores.

Do not rely only on accuracy because anomaly datasets may be imbalanced.

---

# 30. Pixel-Level Evaluation

Localization evaluation may include:

```text
Pixel AUROC
Pixel Average Precision
PRO / AUPRO
IoU
Dice
```

Only calculate metrics that are correctly implemented and justified.

Do not invent metric results.

---

# 31. Category-Level Evaluation

MVTec contains multiple industrial categories.

Evaluate categories separately.

Example:

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

Do NOT assume this list if the actual dataset differs.

Inspect the dataset.

The final evaluation should make it possible to report:

```text
Category | Image AUROC | Pixel AUROC | Other Metrics
```

---

# 32. No Data Leakage

This is a critical rule.

The implementation MUST prevent:

```text
test images entering feature bank
ground truth masks entering feature extraction
test anomaly labels affecting normal memory bank
```

Normal feature bank:

```text
TRAIN / GOOD ONLY
```

Evaluation:

```text
TEST ONLY
```

Ground truth:

```text
EVALUATION ONLY
```

---

# 33. Reproducibility

Set deterministic/random seeds where relevant.

Document:

```text
model version
input image size
DINOv2 variant
feature layer
embedding dimension
distance metric
k value
feature normalization
feature sampling
threshold strategy
```

A future user should be able to reproduce the same pipeline.

---

# 34. GPU Support

The model implementation should support:

```text
CUDA GPU
```

when available.

Example target environments may include:

```text
NVIDIA RTX 5060 Ti 16GB
NVIDIA RTX 5090 32GB
cloud GPU VM
Google Colab
```

However, code should not assume that CUDA always exists.

Use device detection.

Conceptually:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Do not hardcode a specific GPU.

---

# 35. Batch Processing

Feature extraction should support batching where practical.

However:

```text
correctness
```

takes priority over maximizing batch size.

Avoid GPU out-of-memory errors.

Batch size should be configurable.

---

# 36. Feature Bank Device

The implementation should clearly control where the feature bank is stored:

```text
CPU RAM
or
GPU VRAM
```

Do not accidentally copy the entire feature bank between CPU and GPU repeatedly for every patch.

Performance optimization should remain understandable and measurable.

---

# 37. Nearest Neighbor Implementation

The first implementation may use standard PyTorch operations.

Example conceptual choices:

```text
torch.cdist
```

or another transparent distance implementation.

Only introduce FAISS when:

```text
feature-bank size
or
inference performance
```

actually requires it.

Do not add FAISS solely because it is more advanced.

---

# 38. Optimization Roadmap

Optimization is NOT part of the baseline implementation unless explicitly requested.

Potential later improvements include:

```text
FAISS
coreset sampling
memory bank compression
multi-layer DINOv2 features
multi-scale inference
feature fusion
GPU indexing
approximate nearest neighbors
threshold calibration
anomaly-map smoothing
```

These are future experiments.

Do not implement them automatically.

---

# 39. Baseline First

Codex MUST first implement a working baseline:

```text
Frozen DINOv2
      ↓
Patch Features
      ↓
Normal Feature Bank
      ↓
1-NN
      ↓
Patch Scores
      ↓
Image Score
      ↓
Anomaly Map
```

Only after this pipeline works and is evaluated may more complex techniques be considered.

---

# 40. Notebook Requirement

Initial experimentation should be reproducible in a model notebook.

Recommended notebook:

```text
notebooks/02_model_baseline.ipynb
```

Possible logical sections:

```text
1. Environment Setup
2. Configuration
3. Dataset Loading
4. DINOv2 Loading
5. Patch Feature Inspection
6. Feature Bank Construction
7. Test Feature Extraction
8. kNN Anomaly Scoring
9. Image-Level Detection
10. Anomaly Map Generation
11. Ground Truth Comparison
12. Evaluation
13. Final Model Report
```

Codex should inspect existing notebooks before creating this file.

Do not overwrite existing useful work.

---

# 41. Notebook Cell Git Policy

If the existing project `CODEX.md` requires:

```text
ONE LOGICAL CELL
→ ONE COMMIT
→ ONE PUSH
```

the same policy MUST be respected for model experimentation.

A logical cell means a meaningful unit such as:

```text
load DINOv2
inspect patch embeddings
construct feature bank
implement kNN
generate anomaly map
evaluate image scores
```

Imports alone do not necessarily justify a dedicated commit.

Never push broken notebook state.

---

# 42. Required Verification After Each Model Step

After implementing a component, Codex MUST verify the actual output.

Examples:

### Backbone

Verify:

```text
model loaded successfully
correct DINOv2 variant
parameters frozen
device correct
```

### Patch Features

Verify:

```text
tensor dimensions
number of patches
embedding dimension
finite values
```

### Feature Bank

Verify:

```text
number of embeddings
embedding dimension
no NaN
no Inf
only train/good samples used
```

### kNN

Verify:

```text
distance tensor dimensions
finite scores
nearest-neighbor logic correct
```

### Anomaly Map

Verify:

```text
spatial dimensions
upsampling alignment
no invalid values
```

---

# 43. Sanity Checks

Before trusting metrics, Codex should perform simple sanity checks.

Expected general behavior:

```text
GOOD test image
→ relatively low anomaly score

obviously anomalous test image
→ relatively high anomaly score
```

This is not a substitute for proper evaluation.

It is only a pipeline validation step.

---

# 44. No Hardcoded Results

Codex MUST NEVER invent:

```text
AUROC
accuracy
threshold
anomaly score
feature dimension
GPU memory usage
training time
inference speed
```

All reported values must come from actual executed code.

---

# 45. Model Artifacts

Generated model-related artifacts may include:

```text
feature bank
evaluation CSV
model configuration
anomaly score files
visualization images
```

Large generated artifacts should NOT automatically be committed to Git.

Before committing, inspect:

```text
.gitignore
repository conventions
file sizes
```

The actual dataset and large feature-bank files should generally remain outside Git.

---

# 46. Suggested Artifact Structure

If compatible with the repository:

```text
artifacts/
└── dinov2_knn/
    ├── feature_bank/
    ├── evaluation/
    └── visualizations/
```

Do not create this structure blindly.

First inspect existing repository conventions.

---

# 47. Model Configuration

Avoid scattering constants throughout the notebook.

Use a configuration block.

Example:

```python
MODEL_NAME = "dinov2_vitb14"
IMAGE_SIZE = 256
K_NEIGHBORS = 1
BATCH_SIZE = 8
DEVICE = ...
```

Exact values must match preprocessing compatibility and actual implementation requirements.

---

# 48. Preprocessing Compatibility

Before implementing model feature extraction, Codex MUST read:

```text
AI_config/preprocessing_data.md
```

and inspect the preprocessing notebook.

The model pipeline must be compatible with the preprocessing decisions already made.

Do NOT silently preprocess the dataset differently.

If DINOv2 requires model-specific transformations beyond dataset preprocessing, clearly separate:

```text
dataset preprocessing
```

from:

```text
model input transformation
```

---

# 49. Required Files to Read Before Model Implementation

Before changing model code, Codex MUST inspect:

```text
CODEX.md
AI_config/preprocessing_data.md
AI_config/model.md
notebooks/
src/
datasets/
.gitignore
requirements files
pyproject.toml if present
git status
```

Only files that actually exist should be used.

Do not invent missing project files.

---

# 50. Conflict Resolution

Instruction priority:

```text
1. Explicit current user instruction
2. AI_config/model.md
3. AI_config/preprocessing_data.md
4. CODEX.md
5. Existing repository conventions
```

If there is a meaningful conflict between these sources:

DO NOT silently choose one.

Report the conflict before implementing the conflicting requirement.

---

# 51. Forbidden Architecture Changes

Without explicit user approval, Codex MUST NOT replace this baseline with:

```text
PatchCore library implementation
PaDiM
FastFlow
EfficientAD
Reverse Distillation
STFPM
GAN-based anomaly detection
autoencoder
U-Net
YOLO
supervised CNN classifier
DINOv2 fine-tuning
CLIP anomaly detection
Anomalib abstraction
```

These may be considered later as experimental comparison models.

The baseline architecture remains:

```text
DINOv2 + Feature Bank + kNN
```

---

# 52. Libraries

Preferred core stack:

```text
Python
PyTorch
torchvision
NumPy
Pandas
Pillow
Matplotlib
scikit-learn
```

DINOv2 loading may use the appropriate trusted implementation available in the repository/environment.

Do not add unnecessary dependencies.

Before installing a new package:

```text
check whether it is already installed
check existing environment files
justify why it is needed
```

---

# 53. No Dashboard Rule

The MODEL stage does NOT require a dashboard.

Terms such as:

```text
visualization
model report
anomaly map
evaluation report
statistics
heatmap
localization
```

MUST NOT be interpreted as permission to create a dashboard.

Allowed:

```text
Jupyter notebook outputs
Matplotlib figures
Pandas DataFrames
saved evaluation images
CSV reports
terminal output
```

Forbidden unless explicitly requested:

```text
Streamlit
Gradio
Dash
Flask UI
React UI
HTML dashboard
web application
interactive dashboard
```

Do NOT create:

```text
app.py
dashboard.py
streamlit_app.py
frontend/
dashboard/
```

for the model baseline.

---

# 54. No API Yet

The model stage does NOT require:

```text
FastAPI
Spring Boot integration
REST API
WebSocket
Kafka
Redis
database
```

The anomaly detector should first work as a standalone Python pipeline.

API integration will be a separate project phase.

---

# 55. No Camera / CCTV Yet

The baseline model stage does NOT require:

```text
camera capture
RTSP
OpenCV video loop
CCTV stream
real-time inference
continuous localization
frame buffering
```

The initial model must operate correctly on static MVTec images first.

Camera/CCTV inference will be implemented in a later phase.

---

# 56. No Training Requirement

Because DINOv2 remains frozen and kNN is non-parametric, the baseline does not perform traditional neural-network training.

The process commonly called "training" in this baseline consists mainly of:

```text
extracting normal features
+
constructing the normal feature bank
```

Do not create:

```text
optimizer
loss function
epochs
backpropagation
learning-rate scheduler
```

for the baseline architecture.

---

# 57. Expected Baseline Deliverables

The baseline model phase is complete when all applicable items below are satisfied:

* [ ] DINOv2 ViT-B/14 loads successfully.
* [ ] DINOv2 parameters are frozen.
* [ ] Model is in evaluation mode.
* [ ] Model device handling works.
* [ ] Training GOOD images load correctly.
* [ ] Test images load correctly.
* [ ] Patch features are extracted correctly.
* [ ] Patch feature dimensions are verified.
* [ ] Normal feature bank is constructed only from `train/good`.
* [ ] Feature bank contains no NaN or Inf values.
* [ ] Test patch features are extracted consistently.
* [ ] kNN search is implemented.
* [ ] Patch anomaly scores are generated.
* [ ] Image-level anomaly scores are generated.
* [ ] Patch scores are reconstructed into their spatial grid.
* [ ] Anomaly maps are generated.
* [ ] Anomaly maps are resized to image dimensions.
* [ ] Ground-truth masks are used only for evaluation.
* [ ] At least basic image-level evaluation works.
* [ ] At least basic localization evaluation works.
* [ ] GOOD and anomalous examples are visually inspected.
* [ ] No test images entered the normal feature bank.
* [ ] Raw dataset remains unchanged.
* [ ] Model configuration is documented.
* [ ] No invented metrics are reported.
* [ ] No dashboard was created.
* [ ] No API was created.
* [ ] No CCTV pipeline was created.

---

# 58. Baseline Completion Boundary

When the baseline model described in this file is complete:

STOP.

Do NOT automatically continue to:

```text
model optimization
fine-tuning
FAISS
coreset optimization
real-time CCTV
FastAPI
Spring Boot
React
dashboard
deployment
Docker
cloud deployment
```

These belong to later phases.

Wait for explicit user instruction.

---

# 59. Future Architecture

The planned future system may eventually become:

```text
Camera / CCTV
      │
      ▼
Frame Capture
      │
      ▼
Image Preprocessing
      │
      ▼
DINOv2
      │
      ▼
Patch Features
      │
      ▼
Normal Feature Bank + kNN
      │
      ▼
Anomaly Score
      │
      ├──────────────► GOOD / DEFECT
      │
      ▼
Anomaly Map
      │
      ▼
Defect Localization
      │
      ▼
Python Inference Service
      │
      ▼
FastAPI
      │
      ▼
Spring Boot Backend
      │
      ▼
Frontend
```

This diagram is a FUTURE target.

The baseline model task stops before:

```text
Camera
API
Backend
Frontend
Deployment
```

---

# 60. Absolute Rules

Codex MUST follow these rules:

```text
READ AI_config/model.md FIRST.

READ AI_config/preprocessing_data.md BEFORE MODEL IMPLEMENTATION.

USE DINOv2 ViT-B/14 AS A FROZEN FEATURE EXTRACTOR.

USE TRAIN/GOOD ONLY TO BUILD THE NORMAL FEATURE BANK.

EXTRACT PATCH FEATURES, NOT ONLY GLOBAL FEATURES.

USE kNN DISTANCE FOR THE BASELINE ANOMALY DETECTOR.

GENERATE BOTH IMAGE-LEVEL AND PATCH-LEVEL ANOMALY SCORES.

PRESERVE SPATIAL PATCH INFORMATION FOR DEFECT LOCALIZATION.

DO NOT FINE-TUNE DINOv2.

DO NOT CREATE A CLASSIFICATION HEAD.

DO NOT MODIFY RAW MVTec DATA.

DO NOT LEAK TEST DATA INTO THE FEATURE BANK.

DO NOT USE GROUND-TRUTH MASKS DURING INFERENCE.

DO NOT INVENT METRICS.

DO NOT CREATE A DASHBOARD.

DO NOT CREATE AN API.

DO NOT CREATE A CCTV PIPELINE.

IMPLEMENT AND VERIFY THE SIMPLE BASELINE FIRST.

STOP WHEN THE BASELINE MODEL PHASE IS COMPLETE.
```

---

# Final Baseline Architecture

```text
              TRAIN / GOOD
                   │
                   ▼
            DINOv2 ViT-B/14
                 Frozen
                   │
                   ▼
             Patch Features
                   │
                   ▼
             Feature Bank
                   │
                   │
                   │
Test Image         │
    │              │
    ▼              │
DINOv2 ViT-B/14    │
   Frozen          │
    │              │
    ▼              │
Patch Features     │
    │              │
    └───────┬──────┘
            ▼
           kNN
            │
            ▼
      Patch Distances
            │
       ┌────┴─────┐
       ▼          ▼
 Max / Aggregate  Spatial Grid
       │          │
       ▼          ▼
 Image Score   Anomaly Map
       │          │
       ▼          ▼
 GOOD/DEFECT   Localization
```

This is the required baseline architecture for the project.
