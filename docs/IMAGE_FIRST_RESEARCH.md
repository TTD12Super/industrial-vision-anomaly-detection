# Image-First Research Plan

## Scope and evidence policy

This document covers static images only. The target workflow is:

```text
image with multiple objects
        -> object detection
        -> per-object product recognition or UNKNOWN
        -> oriented/object crop
        -> GOOD/DEFECT decision
        -> heatmap/evidence
        -> VLM only for approved suspicious objects
```

The current repository is an image-first vertical slice, not a production or independent benchmark. The audit records known leakage, supervised-development status, weak localization quality, and an unavailable detector calibration artifact. See [AUDIT.md](AUDIT.md) and [screw.yaml](../configs/products/screw.yaml). No new benchmark result is claimed here.

The binary quality label remains `GOOD` or `DEFECT`. `REVIEW_REQUIRED` is an operational state, not a third classifier target. Product recognition is a separate decision:

- `KNOWN_PRODUCT`: product identity is accepted by a calibrated recognizer;
- `REVIEW_PRODUCT`: a known-product candidate exists but confidence/margin is insufficient;
- `UNKNOWN_PRODUCT`: no known product is accepted.

`UNKNOWN_PRODUCT` must not be routed to a product-specific anomaly model. It is saved as evidence and sent to the configured review path.

## Methods for GOOD/DEFECT and localization

### 1. DINOv2 with a supervised quality head

**Idea.** Extract frozen DINOv2 global or patch features from the object crop and train a supervised GOOD/DEFECT head. Start with logistic regression or a small linear head; calibrate the defect threshold on validation data.

**Use when.** There are labeled object-level GOOD/DEFECT examples and the main objective is very high defect recall with a controlled GOOD false-positive rate. It is the lowest-complexity supervised baseline and aligns with the existing repository classifier interface.

**Why it is scientifically defensible.** DINOv2 explicitly studies robust, reusable visual features that can be used with simple downstream classifiers.

**Risks.** A frozen general-purpose representation can preserve nuisance cues such as background, illumination, or product subtype. A small head can still overfit when the number of physical instances is small. Threshold calibration can overfit if validation images are near-duplicates of training images.

**Source.** [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

### 2. PatchCore

**Idea.** Build a memory bank of representative normal patch features and score test patches by nearest-neighbor distance. The maximum or aggregated patch score provides image-level anomaly evidence; patch scores provide a heatmap.

**Use when.** Defect labels are scarce, but verified GOOD crops are available. It is a strong nominal-only baseline for industrial anomaly detection and gives a useful comparison against supervised GOOD/DEFECT classification.

**Risks.** It can be sensitive to the coverage of the GOOD memory bank, crop alignment, illumination, and product variants. A broad heatmap can still have high anomaly recall but poor pixel precision. Memory-bank construction must use training GOOD crops only.

**Source.** [Towards Total Recall in Industrial Anomaly Detection / PatchCore, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)

### 3. PaDiM

**Idea.** Model the normal distribution of patch embeddings with a multivariate Gaussian and use a Mahalanobis-style distance for anomaly scoring and localization.

**Use when.** A compact one-class baseline is needed, normal data is available, and a probabilistic local model is useful. It is a natural comparison to PatchCore because both are nominal-only but make different assumptions about normal feature space.

**Risks.** Gaussian assumptions may be poor for multimodal products or strong viewpoint/lighting variation. Covariance estimation can be unstable with too few normal samples. Pixel maps inherit the spatial resolution and biases of the feature extractor.

**Source.** [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/abs/2011.08785)

### 4. DRAEM

**Idea.** Train a discriminative anomaly model using normal images and synthetically generated anomalies, learning both an anomaly decision and a localized anomaly representation.

**Use when.** Pixel masks are limited but realistic synthetic corruption can be designed for the product surface. It is especially relevant when the current approach produces diffuse heatmaps and a directly trained localization head is desired.

**Risks.** Synthetic-to-real mismatch is the central risk: scratches, dents, contamination, and geometric defects may not follow the chosen corruption process. A model can learn the synthetic artifact rather than the real defect. Synthetic data must be validated against held-out real defects, never against synthetic-only test data.

**Source.** [DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html)

### 5. AnomalyCLIP

**Idea.** Use object-agnostic CLIP prompts/features to detect and localize abnormal regions without product-specific defect training.

**Use when.** A zero-shot or auxiliary evidence baseline is needed, especially for comparing general anomaly cues against product-specific models. It should be evaluated as a localizer/evidence generator, not automatically as the GOOD/DEFECT authority.

**Risks.** Vision-language features may respond to object semantics, reflections, texture, or background rather than the defect. Prompt/checkpoint selection can overfit public benchmarks. The repository already records over-segmentation on the screw pilot; therefore its output should not override the classifier.

**Source.** [AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d7b50b8ac2c781a12f26155f48310d8d-Abstract-Conference.html)

### 6. SegAD

**Idea.** Use anomaly maps, segmentation maps and optional supervised classifier scores to form local statistics, then combine them with a boosted random-forest classifier for the final anomaly score.

**Use when.** Existing heatmaps contain useful signal but are too diffuse or noisy at image level. SegAD is attractive as a lightweight decision/refinement layer rather than a replacement for every localizer.

**Risks.** It requires a clean separation between training statistics and holdout images. If the underlying heatmap is biased, the boosted classifier can learn that bias. The method was introduced with the Valeo Anomaly Dataset, whose real production-line domain may differ substantially from screw images.

**Sources.** [Supervised Anomaly Detection for Complex Industrial Images, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Baitieva_Supervised_Anomaly_Detection_for_Complex_Industrial_Images_CVPR_2024_paper.html), [official SegAD/Valeo page](https://valeoai.github.io//publications/segad/)

## Open-set UNKNOWN design

Product recognition should not be implemented as “choose the closest known product.” A crop is accepted only if all required conditions hold:

1. top-1 confidence exceeds a validation threshold;
2. top-1 versus top-2 margin exceeds a validation threshold;
3. distance to the accepted product gallery/prototype is within a calibrated class-aware bound;
4. crop quality and geometry pass the recognition preconditions.

Otherwise return `REVIEW_PRODUCT` or `UNKNOWN_PRODUCT`.

The negative set must contain:

- products absent from the catalog;
- visually similar products from other classes;
- backgrounds and non-product objects;
- damaged/occluded views of known products;
- hard negatives that resemble a known product but should not use its anomaly bank.

Evaluation should report closed-set top-1/top-5 accuracy separately from open-set rejection: unknown AUROC, false-accept rate at a fixed known-product recall, OSCR/risk-coverage, and per-class rejection rates. OpenMax provides a classical reject-option formulation; selective classification formalizes the trade-off between accepted coverage and prediction risk. [OpenMax](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html), [Selective Classification](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)

An unknown object must not receive a product-specific GOOD/DEFECT label. It may receive a separate `unknown_object_review` disposition and evidence crop. VLM output, if requested for UNKNOWN, may describe visible appearance but cannot assign the product identity or alter the routing decision.

## YOLO-OBB versus DETR/RT-DETR

### YOLO-OBB baseline

The existing pipeline needs oriented corners because oriented cropping and inverse heatmap projection depend on them. YOLO-OBB is therefore the operational baseline. Its evaluation must include missed objects, unmatched false positives, OBB IoU, crop geometry error, confidence calibration, and latency—not only mAP.

### DETR family

DETR formulates detection as set prediction with bipartite matching and removes the traditional proposal/NMS pipeline, but the original model is not a fair low-data or latency baseline without controlled training and hardware measurement.

Source: [End-to-End Object Detection with Transformers, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf)

### RT-DETR

RT-DETR is a real-time DETR variant designed to reduce the speed cost of transformer detection and supports speed/accuracy adjustment through decoder depth. Its reported comparisons are on COCO; those results do not establish superiority on this small industrial OBB dataset.

Source: [DETRs Beat YOLOs on Real-time Object Detection, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html)

### Fair comparison rule

Use one of these protocols:

1. compare YOLO-OBB and RT-DETR using the same axis-aligned box target and the same downstream axis-aligned crop; or
2. compare YOLO-OBB with a rotated-box DETR/equivalent that outputs the same OBB representation.

Do not compare YOLO OBB mAP with RT-DETR horizontal-box mAP and then claim a better crop/localization system. Record:

- recall at the operating confidence;
- precision and unmatched false positives per image;
- box/OBB IoU at 0.50 and 0.75;
- missed-object count;
- crop corner/area/orientation error;
- p50/p95 single-image latency on the same hardware;
- model size and memory;
- downstream effect on classifier and heatmap metrics.

## Recommended experiment order

### Stage 0 — Validity gate

Before model comparisons:

- make the configured detector calibration artifact available or mark the pipeline not runnable;
- repair evaluation accounting for unmatched predictions;
- reserve a source-grouped independent holdout;
- verify no train/validation/holdout leakage;
- freeze preprocessing, crop geometry and metric definitions.

### Stage 1 — Quality classification

1. Frozen DINOv2 global feature + linear GOOD/DEFECT head.
2. Frozen DINOv2 patch-aware head.
3. Hard-negative/supervised-contrastive variant only if Stage 1 exposes systematic class confusion.
4. Calibrate the defect threshold and review band on validation only.

Prefer the simplest model meeting the defect-recall and GOOD false-positive gate. Do not select a model by AUROC alone.

### Stage 2 — Heatmap/localization

1. Existing AnomalyCLIP baseline.
2. PatchCore and PaDiM nominal-only baselines.
3. DRAEM/SimpleNet or another supervised/synthetic-anomaly localizer.
4. SegAD-style map-statistics decision/refinement layer.
5. Existing supervised patch refiner as a controlled ablation.

Keep a method only if it improves independent pixel AP and defect-region recall without exceeding the GOOD false-positive-area budget. A high-recall but very broad map remains evidence-only.

### Stage 3 — Product recognition and UNKNOWN

1. DINOv2 crop embedding + gallery/prototype retrieval.
2. Calibrated confidence, margin and distance rejection.
3. Hard-negative and held-out-unknown evaluation.
4. Product-specific anomaly routing only after recognition acceptance.

### Stage 4 — Detector comparison

1. Reproduce YOLO-OBB with fixed image size and calibration.
2. Train/evaluate RT-DETR on the same source-grouped split if an appropriate box representation is available.
3. Add rotated DETR/equivalent only if OBB quality is essential and horizontal boxes are not sufficient.
4. Judge the detector by downstream crop quality and latency as well as detection metrics.

## Selection rules

### Release candidate

Keep a method only when it satisfies the pre-agreed independent-holdout gates:

- DEFECT recall at least 90%;
- GOOD operational false-positive rate at most 20%;
- no silent UNKNOWN-to-known routing;
- detector false positives fully counted;
- heatmap pixel AP at least 60% only when mask quality and sample size justify that gate;
- evidence artifacts are complete and reproducible.

### Exploratory candidate

Preserve, but do not release, a method that improves ranking or defect recall while:

- GOOD false-positive rate is 20–25%;
- pixel AP is below the localization gate;
- synthetic-to-real transfer is unverified;
- results depend on a supervised-development split;
- latency or memory is not yet measured fairly.

### Reject

Reject or stop tuning a method when:

- it requires test-set threshold selection;
- it benefits from leaked source identities or derived crops across splits;
- it forces UNKNOWN into a known product;
- it improves only mAP but worsens missed-object count or crop quality;
- it improves heatmap recall by flagging most of the object/background;
- its gains disappear on a new camera, lot, defect type, or product subtype.

## Current repository boundary

The repository already contains interfaces and experiments for DINOv2 quality classification, AnomalyCLIP, PatchCore/PaDiM legacy modules, calibration, refinement, evidence storage and a mock VLM boundary. It does not currently provide a complete per-object product recognizer or a configured RT-DETR/rotated-DETR backend. Those are research candidates, not implemented features.

Existing repository metrics remain historical supervised-development evidence. This document intentionally adds no new benchmark result.
