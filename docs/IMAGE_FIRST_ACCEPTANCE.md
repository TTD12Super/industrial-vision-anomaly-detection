# Image-First Static Inspection: Acceptance Plan

**Status:** Plan and acceptance standard only. This document is not evidence that any gate has been achieved.

**Scope:** One static image may contain multiple objects/products. Video, live camera, tracking across frames and temporal aggregation are out of scope.

## 1. Intended workflow

```text
image
  -> object detector
  -> per-object crop
  -> product recognition
       -> KNOWN_PRODUCT -> product-specific quality model
       -> UNKNOWN       -> stop product-specific anomaly inference
       -> REVIEW_REQUIRED -> preserve evidence and flag review
  -> GOOD / DEFECT quality decision
  -> heatmap/localization for suspicious objects
  -> optional VLM evidence report
  -> JSON + overview + crops + evidence artifacts
```

The system MUST process each object independently. A failure or unknown result for one object MUST NOT hide successful results for other objects.

## 2. Status semantics

### 2.1 Product identity

Each detected object receives exactly one product-routing status:

- `KNOWN_PRODUCT`: product recognition passed its calibrated acceptance rule; the selected product bundle may run.
- `UNKNOWN`: the product is outside the known-product set, below the recognition threshold, or too ambiguous to route safely. No product-specific anomaly model may run.
- `REVIEW_REQUIRED`: product evidence is insufficient or conflicting and requires human review. It MUST NOT be silently converted to a known product.

`UNKNOWN` is not `GOOD`, `DEFECT`, or a defect prediction. Product recognition and quality classification are separate decisions.

The product recognizer MUST record the product candidate, score/distance, threshold version, model version, and rejection reason. The meaning of “similarity” MUST be documented as either a calibrated probability, embedding distance, or another explicit metric; an informal percentage is not an acceptance metric.

### 2.2 Quality status

Only a `KNOWN_PRODUCT` object may receive a product-specific quality decision:

- `GOOD`: classifier label `GOOD`, confidence below the review band.
- `REVIEW_REQUIRED`: classifier label remains `GOOD`, but the calibrated score is inside the uncertainty band.
- `DEFECT`: classifier label `DEFECT` at or above the product-specific defect threshold.
- `STAGE_FAILED`: a required stage failed; never treat it as `GOOD`.

`REVIEW_REQUIRED` is an operational disposition, not a third supervised quality class. The classifier label and operational status MUST be stored separately.

## 3. Functional acceptance requirements

### 3.1 Input and object accounting

For an image containing multiple objects, the result MUST include:

- the unchanged original image;
- one deterministic image-local object ID per detection;
- detector score and box/OBB coordinates;
- one crop per detected object;
- product-routing result per object;
- quality result per known object;
- stage status and failure reason where applicable.

The evaluator MUST account for all of the following:

- matched ground-truth objects;
- missed ground-truth objects (`FN`);
- unmatched predicted objects (`FP`);
- empty/negative images with false detections;
- known objects rejected as `UNKNOWN` or `REVIEW_REQUIRED`;
- detections whose crop geometry is invalid or materially misaligned.

No metric may silently discard unmatched predictions or missed objects.

### 3.2 Detector and crop

The baseline detector is YOLO-OBB. It MUST be evaluated for:

- object recall and precision;
- missed-object count;
- unmatched false detections per image;
- box/OBB IoU;
- orientation and crop geometry quality;
- crop-to-source inverse mapping;
- P50/P95 latency and memory on the declared hardware.

An RT-DETR/DETR candidate MAY be evaluated as an alternative detector. Standard RT-DETR outputs axis-aligned boxes; it is **not a direct replacement for YOLO-OBB** when oriented boxes and oriented crops are required. The candidate must either:

1. provide a validated OBB-compatible output; or
2. use an explicitly documented axis-aligned crop and pass a crop-quality/end-to-end comparison.

The comparison MUST use the same images, splits, preprocessing policy, evaluation matching rule, hardware, inference resolution and comparable training budget. mAP alone is insufficient.

### 3.3 Anomaly and heatmap evidence

For each known object, the quality model returns a calibrated `p_defect` and classifier label. Thresholds MUST be selected using validation data only.

For `DEFECT` objects, the system MUST attempt localization and store:

- continuous heatmap;
- heatmap overlay on the original-color crop;
- optional binary mask only when its threshold is validated;
- crop-to-source projection where applicable;
- localization status, method version and threshold provenance.

Heatmaps are evidence, not decision authority. A missing, empty, broad, or contradictory heatmap MUST NOT change `GOOD`/`DEFECT`.

`REVIEW_REQUIRED` localization MAY be run as supporting evidence. `UNKNOWN` must not be sent to a product-specific localizer.

### 3.4 VLM evidence

The VLM is an evidence/reporting component only. It MUST NOT:

- select or change the product identity;
- change `GOOD`, `DEFECT`, `UNKNOWN`, or `REVIEW_REQUIRED`;
- turn a heatmap into proof of a defect;
- block the inspection if unavailable or malformed.

Every classifier-`DEFECT` object receives a VLM attempt when the VLM feature is enabled. `REVIEW_REQUIRED` routing is policy-controlled. `UNKNOWN` is not sent to the VLM by default because product identity is unresolved.

The result MUST preserve `pending`, `completed`, and `failed` report states. A mock VLM validates orchestration only; it is not evidence of report quality, visual accuracy, cost, or latency.

## 4. Metrics and release gates

Metrics MUST be reported separately for:

1. oracle/ground-truth crops, to isolate classification/localization quality; and
2. detector-generated crops, to measure the integrated system.

### 4.1 Required metrics

#### Product recognition

- known-product top-1 accuracy;
- unknown rejection rate;
- false-known rate for unknown objects;
- ambiguous/review rate;
- score/distance calibration and threshold provenance.

#### Quality classification

- `DEFECT` recall;
- `GOOD` false-positive rate, with denominator explicitly defined as GOOD objects;
- precision, PR-AUC and AUROC where meaningful;
- review-band coverage;
- false negatives outside the review band;
- per-product and per-domain results.

#### Localization

- pixel AP;
- pixel precision and recall;
- Dice and IoU;
- GOOD-only pixel false-positive rate;
- predicted-area ratio;
- defect-region coverage;
- results by product, defect type and domain.

#### Detector

- object recall and precision;
- matched and unmatched counts;
- box/OBB IoU and geometry error;
- crop alignment quality;
- P50/P95 latency and memory.

### 4.2 Release gates

A release candidate MUST satisfy all applicable gates on an untouched independent holdout:

- `DEFECT` recall `>= 90%`;
- GOOD false-positive rate `<= 20%`;
- unknown objects are not routed into the wrong product anomaly model;
- unmatched false positives and missed objects are included in the reported metrics;
- detector crop geometry is valid for the accepted detector;
- JSON/evidence artifacts are complete and reproducible.

GOOD false-positive rates from `20%` to `25%` are exploratory only and MUST NOT be presented as release acceptance.

Pixel AP `>= 60%` is a conditional target, not an unconditional requirement. It may be used as a release gate only when:

- pixel masks are sufficiently complete, precise and correctly aligned;
- the holdout masks were never used for tuning;
- per-object and per-product sample counts are adequate;
- the report includes precision/recall, Dice/IoU and GOOD-only pixel false positives.

If those conditions are not met, localization is `not_evaluable` for release claims; a heatmap may still be retained as qualitative evidence.

## 5. Independent holdout and leakage rules

The independent holdout MUST be locked before model selection, threshold calibration, heatmap post-processing selection, or detector selection.

All derived records from the same source identity MUST remain in one split, including:

- crops from the same source image;
- augmentations of the same image;
- near-duplicate images;
- repeated captures of the same physical object;
- frames from the same capture session;
- masks and overlays derived from the same source.

The split report MUST record source IDs, capture groups, physical-instance IDs where available, hashes, domains and leakage checks. Official test data used for development or adaptation is supervised-development data, not an independent final holdout.

If the holdout is inspected to choose a method, threshold, post-processing rule, detector, or product bundle, it is no longer an independent holdout. A new locked holdout is required.

## 6. Repeat/stop loop

### Stop immediately

Stop the experiment and fix the protocol if any of the following occurs:

- missing or incompatible calibration/artifact;
- leakage or uncertain split identity;
- evaluator omits unmatched FP/FN;
- holdout data influences tuning;
- product identity is forced despite low confidence;
- a stage failure is silently converted to `GOOD`;
- reported artifacts cannot be reproduced from the recorded bundle.

### Repeat an experiment

Repeat only when:

- the holdout remains untouched;
- the change is isolated and documented;
- training/evaluation budget is comparable to the baseline;
- the reason for repetition is recorded;
- the same metric definitions and matching rules are retained.

### Promote a result

Promote only when the release gates pass, confidence intervals or uncertainty estimates are reported, failure cases are reviewed, and the selected model/bundle is versioned without overwriting prior evidence.

## 7. Required experiment record

Every experiment MUST record:

- product and bundle version;
- dataset/manifest hashes and split policy;
- train/validation/holdout membership;
- detector, classifier and localizer versions;
- preprocessing and crop geometry;
- thresholds and post-processing;
- seed, hardware and latency;
- all primary metrics and per-domain slices;
- unmatched FP/FN counts;
- known limitations and promotion decision.

This document defines the intended static-image acceptance standard. It does not claim that the current repository, screw dataset, YOLO baseline, localization method, or mock VLM has passed any of these gates.
