# Configurable Industrial Visual Inspection Platform — Requirements & Implementation Plan

**Version:** 2.0 (requirements clarification)
**Initial pilot:** Screw inspection
**Original working branch:** `codex/dinov2-good-defect`
**Document purpose:** Implementation contract for Codex and engineering review; not a claim of production readiness.
**Status key:** **CONFIRMED** = explicitly requested; **PROPOSED** = engineering design to validate; **TBD** = user decision or measured value still required.

> **Core requirement — CONFIRMED:** Build a reusable industrial visual inspection platform, *not* a screw-only application. A company must be able to provide a conforming dataset and product configuration, train/calibrate the required models, and inspect a new product **without editing pipeline source code**. This is a software extensibility requirement, **not** a promise of acceptable accuracy on arbitrary products. Each product/domain requires its own validated training data, model artifacts, thresholds, and acceptance tests.

## 0. Executive decisions and boundaries

| Topic | Requirement | Status |
|---|---|---|
| Product ambition | Production-oriented industrial inspection, with screw as the first pilot | CONFIRMED |
| Inputs | Single images, batches/images containing multiple objects, video, and eventually live camera streams | CONFIRMED |
| Quality semantics | GOOD = intact product; all observed undesirable anomalies/defects = DEFECT; exact defect policies configurable per product | CONFIRMED / policy details TBD |
| Optimization | Prioritize **DEFECT recall** (avoid missed defective items) over low false-alarm volume | CONFIRMED |
| Portability | Change dataset + configuration + retrain/recalibrate; no source edits | CONFIRMED |
| Detector | YOLO-OBB; evaluate multiple datasets/checkpoints, fine-tune and choose using validation | CONFIRMED |
| Decision model | Retain the existing DINOv2 supervised GOOD/DEFECT baseline for the first pilot | PROPOSED default; original model preserved |
| Known generalization limits | Accept initial screw-domain limitation; do not claim CAD-SD or unseen-domain robustness | CONFIRMED |
| Split management | Prefer checkpoint reuse; realign/retrain if leakage prevents a valid shared test | CONFIRMED |
| Data | Start with available datasets; expand later; there is **no proprietary camera dataset yet** | CONFIRMED |
| Evaluation | Per-stage and integrated end-to-end measurements | CONFIRMED |
| Localization | AnomalyCLIP supplies heatmap and optional binary mask; anomaly score is supplementary, not classification authority | CONFIRMED |
| Relation Map | AnomalyCLIP-only baseline first; include fusion only if useful under controlled evaluation | CONFIRMED |
| Uncertainty | Low-confidence GOOD must enter a third *operational state*, `REVIEW_REQUIRED`, without changing the classifier label | CONFIRMED |
| VLM | Describe visible anomalies, note disagreements, do **not** override DINOv2; initially cloud, local later | CONFIRMED |
| Cloud policy | Uploading imagery to an external VLM is allowed for this pilot; credentials and data rules still apply | CONFIRMED |
| Output | Overview image, individual status-colored crop images, defect heatmaps, detailed JSON and a compact PDF table | CONFIRMED |
| Integration | FastAPI is the backend; future website, desktop app and other clients consume the API | CONFIRMED |
| Review UX | Review flags/status in outputs only for v1; no human-review UI required now | CONFIRMED |
| Runtime/hardware | Deployment OS, GPU, VRAM, target FPS and latency | TBD (question 21) |
| Codex workflow | Full autonomous execution versus approval after each phase | TBD (question 22) |
| Source-control safety | No push, merge, release or destructive dataset changes without explicit permission | PROPOSED; consistent with prior plan |

### 0.1 Meaning of the three outcomes

**Very important:** The image classifier continues to output the binary label `GOOD` or `DEFECT`. `REVIEW_REQUIRED` is the **operational disposition**, not a third supervised classification target and not a new DINOv2 label.

| Classifier prediction | Confidence rule | `operational_status` | Work to perform |
|---|---|---|---|
| GOOD | Sufficiently far below defect decision threshold | `GOOD` | Save green-panel crop; stop heavy downstream steps |
| GOOD | Near the threshold / uncertain by validation-calibrated review band | `REVIEW_REQUIRED` | Save amber-panel crop; run AnomalyCLIP for supporting evidence; flag for human review; **do not relabel** |
| DEFECT | At or above decision threshold | `DEFECT` | Save red-panel crop; run AnomalyCLIP; enqueue VLM report; set `needs_review` as required by policy |
| No valid object detected | — | `NO_DETECTION` (inspection-level) | Record separately; never silently classify as GOOD |
| Critical component failure | — | `STAGE_FAILED` (affected object/inspection) | Preserve previous evidence and failure reason; never silently classify as GOOD |

**PROPOSED scoring convention:** `p_defect = P(DEFECT)` after product-specific calibration, `T_defect` is the binary classifier threshold, and `T_review_low` is the lower edge of the uncertain-GOOD band. Use:

- `p_defect >= T_defect` → classifier label `DEFECT`; status `DEFECT`.
- `T_review_low <= p_defect < T_defect` → classifier label `GOOD`; status `REVIEW_REQUIRED`.
- `p_defect < T_review_low` → classifier label `GOOD`; status `GOOD`.

Select both thresholds **only on validation data** with priority given to DEFECT recall. Save calibration version in the product model bundle; **never hard-code universal thresholds**. A high-confidence false negative can remain undetected by this band; quantify such failures on labeled evaluation datasets.

### 0.2 Delivery phases versus eventual product

The final platform should support video and live cameras. **Image inference is the first shippable vertical slice**, because camera integration, frame tracking, temporal aggregation, frame rates, edge hardware and transport protocols are not yet specified. Video/offline batch follows; live camera is a subsequent phase with independent acceptance tests. Do not advertise a camera production system before those gates pass.

## 1. System architecture

```text
       Image / image batch / video file / future live camera
                           |
                 FastAPI + ingestion layer
                           |
                Frame/sample acquisition
                           |
          Product configuration & model registry
                           |
                   YOLO-OBB detector
                           |
            OBB IDs + geometric crop + inverse
                           |
                DINOv2 binary classifier
                           |
               p_defect + calibrated thresholds
                           |
             +-------------+----------------+
             |             |                |
          GOOD       REVIEW_REQUIRED       DEFECT
             |             |                |
       Green crop     Amber crop        Red crop
             |             |                |
       Save result    AnomalyCLIP       AnomalyCLIP
             |         supporting map    + optional mask
             |             |                |
             |       Human-review flag   Async VLM adapter
             |             |                |
             +-------------+----------------+
                           |
           Artifacts + JSON + compact PDF summary
                           |
              FastAPI status/result endpoints
                           |
            Future Web UI / Desktop / integrations
```

**Decision ownership:** The binary classifier controls the GOOD/DEFECT label; the rules engine controls operational status; localization and the VLM **never silently override** either. Human-reviewed labels, when available in a later workflow, are stored separately.

**Module boundaries — PROPOSED:**

1. `ingestion`: validate uploads, frames, image metadata and stream provenance.
2. `product_registry`: resolve configuration, schema, model artifact versions and active product.
3. `detector`: a detector interface with a YOLO-OBB implementation.
4. `geometry`: OBB transform, oriented crop, inverse transform, coordinate mapping.
5. `classifier`: binary decision interface with frozen DINOv2 + Logistic Regression initial backend.
6. `decision_policy`: deterministic thresholds, state transitions and review flags.
7. `localizer`: AnomalyCLIP-only; optional relation-fusion implementation behind feature flag.
8. `reporter`: mock/cloud/local VLM adapters, JSON schema validator and PDF renderer.
9. `artifact_store`: immutable evidence files, manifest, metadata and run snapshots.
10. `api`: FastAPI routes, request validation, async jobs, health and progress.
11. `evaluation`: per-stage and integrated benchmark runners.
12. `training`: dataset validation, training, calibration and model export, driven by configuration.

Do not rewrite verified existing implementations just to match these suggested module names. Wrap/reuse them and refactor incrementally with tests.

## 2. Data contract: the critical no-code product portability requirement

### 2.1 Inputs a company must provide

A new object category must be onboarded using **a versioned dataset and a product YAML config**. For the **full pipeline** it needs:

| Asset | Required for what? | Notes |
|---|---|---|
| Original full images or frames | Object detection and integrated evaluation | Include scenes with multiple products when relevant |
| Product OBB annotations (or compatible boxes with a documented conversion) | Train/fine-tune YOLO-OBB | Each object instance must be labeled; negative/empty scenes recommended |
| Per-object `GOOD`/`DEFECT` ground truth | Train/evaluate the binary classifier | Labels must attach to object IDs, not blindly to whole images |
| Verified GOOD crops/instances | Normal prototype and optional relation localization | Exclude uncertain/polluted good examples |
| Object-level defect examples | Validate defect classification | Capture relevant defect appearances |
| Pixel defect masks | **Optional for inference**, needed for supervised localization metrics/threshold selection | Never fabricate segmentation ground truth |
| Independent validation and holdout identities | Model selection and unbiased evaluation | Require source grouping and leak checks |
| Domain metadata | Domain-specific analysis | Camera, lighting, production line, lot, capture session when available |

**Important constraints:** If a company provides only GOOD images, the existing supervised GOOD/DEFECT classifier cannot be trained as specified. If it provides no detection boxes, detector training cannot proceed. If it provides no defect masks, the system may still generate unvalidated heatmaps, but must not report measured pixel IoU/Dice or claim mask quality. Dataset acceptance should fail loudly with actionable reasons instead of editing code to work around missing labels.

### 2.2 Proposed canonical structure

```text
datasets/
  products/
    screw/
      dataset.yaml
      images/
        <source-relative-images>...
      masks/                  # optional
      manifests/
        images.jsonl          # full-image/frame records
        objects.jsonl         # one row per object, matched by image_id
        split_report.json
      provenance.json
    <new-product-key>/
      dataset.yaml
      images/...
      masks/...               # optional
      manifests/images.jsonl
      manifests/objects.jsonl
      provenance.json

configs/products/
  screw.yaml
  <new-product-key>.yaml

artifacts/products/
  <product-key>/<model-bundle-version>/
    model_bundle.json
    detector/...
    classifier/...
    localizer/...             # if used
    calibration.json
    preprocessing.json
    label_policy.json
```

`images.jsonl` schema — **PROPOSED required fields**:

```json
{
  "image_id": "screw_camera1_00001",
  "relative_path": "images/camera1/00001.png",
  "split": "train",
  "source_dataset": "mvtec_ad_screw",
  "capture_group_id": "camera1_session01",
  "domain_id": "camera1_daylight",
  "sha256": "<sha256-of-image-bytes>",
  "width": 1024,
  "height": 768
}
```

`objects.jsonl` schema — **PROPOSED required fields**:

```json
{
  "object_id": "screw_camera1_00001_obj01",
  "image_id": "screw_camera1_00001",
  "product_key": "screw",
  "obb_xyxyxyxy": [[100, 100], [220, 95], [222, 140], [102, 145]],
  "quality_label": "DEFECT",
  "defect_types": ["scratch"],
  "mask_relative_path": "masks/camera1/00001_obj01.png"
}
```

`defect_types` and `mask_relative_path` can be null/absent where unavailable. `GOOD`/`DEFECT` should follow the specific product's documented **quality policy**. Split placement belongs to the original capture/image identity: never split near-duplicate frames or crops of the same physical instance across train/validation/test.

### 2.3 Product configuration example

The interface below is a **target contract**, not a claim the current repository already implements these commands or fields.

```yaml
schema_version: 1
product_key: screw
dataset:
  root: datasets/products/screw
  image_manifest: manifests/images.jsonl
  object_manifest: manifests/objects.jsonl
  split_policy: grouped_by_source_and_capture
models:
  detector:
    backend: yolo_obb
    initial_checkpoint: null       # or validated prior checkpoint
    image_size: 1024
  classifier:
    backend: dinov2_logreg
    backbone: dinov2_vitb14
    frozen_backbone: true
  localizer:
    backend: anomalyclip
    relation_fusion_enabled: false
  reporter:
    backend: cloud_vlm            # mock for tests; cloud when configured
    async_enabled: true
policy:
  quality_labels: [GOOD, DEFECT]
  decision_threshold_from_calibration: true
  review_band_from_calibration: true
  optimize_for: defect_recall
outputs:
  overview: true
  per_object_status_crop: true
  heatmap_for_defect: true
  json: true
  compact_pdf: true
```

Dataset ingestion/validation, detector/classifier training, threshold calibration, evaluation, inference and output writing **must consume this configuration**. Hard-coded strings such as `screw`, `MVTec`, fixed data paths, defect classes or color conventions may appear only in sample/default configs, not in generic core logic.

### 2.4 Training orchestration — PROPOSED contract

```bash
python -m inspection.cli validate-dataset --config configs/products/screw.yaml
python -m inspection.cli train-detector --config configs/products/screw.yaml
python -m inspection.cli train-classifier --config configs/products/screw.yaml
python -m inspection.cli calibrate --config configs/products/screw.yaml
python -m inspection.cli evaluate --config configs/products/screw.yaml
python -m inspection.cli export-bundle --config configs/products/screw.yaml
```

Every step must record seed, code commit/hash when available, source manifest hash, splits, preprocessing, hyperparameters, metrics, thresholds, checkpoint hashes and dependencies. A new conforming product must use the **same commands**, changing only the dataset/config paths and generated model artifacts. If a new product needs an unsupported modality or fundamentally different annotation semantics, report an explicit capability limit instead of claiming no-code compatibility.

## 3. Existing screw pilot: datasets and model starting point

### 3.1 What the prior implementation plan records

- Current branch: `codex/dinov2-good-defect`.
- Existing screw detector code/documentation to **reuse**: `scripts/train_screw_yolo_obb.py`, `scripts/evaluate_screw_yolo_obb.py`, `scripts/validate_screw_yolo_obb.py`, `docs/Screw_YOLO.md`.
- Source detection dataset: `datasets/screw_obb_combined_screw_only_fixed_aug/`, combining WEEE Screw and MVTec Screws.
- Target adaptation dataset: `datasets/mvtec_ad_screw_bbox_obb/`, **120 raw images**, local split **90 train / 18 val / 12 test**, recovered OBB labels. Green overlays are annotation references; use clean images as inputs.
- DINOv2 artifact: `artifacts/production/dinov2_binary_screw/`; associated manifests: `reports/dinov2_binary_screw/`.
- Frozen initial classifier: DINOv2 ViT-B/14 global embeddings + Logistic Regression.
- These file paths, counts and benchmarks are **recorded from the existing plan** and must be verified against the current repository before execution.

| Prior recorded benchmark (not newly re-run) | Images | Accuracy | DEFECT recall | GOOD recall | AUROC |
|---|---:|---:|---:|---:|---:|
| MVTec AD/screw supervised holdout | 38 | 97.37% | 94.12% | 100% | 0.9972 |
| CAD-SD external | 376 | 44.15% | 100% | 0% | 0.5593 |

On CAD-SD, the baseline predicted **DEFECT for all 376 images**, including 210 GOOD cases. **Do not market the current model as cross-domain robust.** A customer wishing to run another product/domain must supply corresponding labeled data and retrain/validate a domain-specific model bundle.

### 3.2 Shared split, leakage and provenance

Prior plan states YOLO and DINOv2 split assignments differ: only **3 of 12** YOLO test images are in DINOv2 test; **8** are in DINOv2 train; **1** in validation; and **17** DINOv2 test images appear in the proposed YOLO target train split. Preserve historical per-model benchmark records, but do **not** treat them as an independent integrated-system test.

Before new end-to-end claims:

1. Canonicalize raw source IDs and hashes, group derived crops and augmented images with their parent raw image.
2. Put the same physical screw, session, near-duplicate frames and crop variants in a single split.
3. Reserve a **shared raw-image holdout** before any stage learns from it.
4. Reuse existing checkpoints when valid; otherwise retrain/realign the affected stage or procure a genuinely untouched holdout.
5. Treat MVTec AD official test images used for development/adaptation as **supervised development**, not unseen official-test evidence.
6. Record train/val/test composition, data licenses, available capture metadata and leakage-test results.

### 3.3 YOLO-OBB implementation

- Start from strongest **validated** detector checkpoint; run multiple controlled fine-tuning experiments across available source/target datasets.
- Start with `imgsz=1024` (experiment default, not a hard requirement); select using **target validation detection recall** and source-domain non-regression, while recording precision/mAP.
- Preserve original datasets read-only; put processed mixed training data in separate versioned locations.
- Return per-object OBB corners, confidence, deterministic object ID, and the oriented crop's transform/inverse transform.
- Benchmark on clean images, challenging multi-object scenes and empty/negative scenes where available.
- Save a visual overlay and a crop-mapping test to verify that a heatmap can be projected back to the original image.
- Metrics: per-domain detection precision/recall, mAP50, mAP50–95, missed-object count, false detections and crop geometry errors.

### 3.4 DINOv2 implementation and Q6 decision

**PROPOSED v1 choice:** Keep **supervised binary classification** (DINOv2 frozen features + Logistic Regression) rather than turning the classifier into a GOOD-only anomaly detector. This is the simplest way to preserve current benchmark and decision logic, and is **not** a claim that it is optimal for every product.

- Train using per-object verified GOOD/DEFECT samples in the product's training split.
- Keep the frozen current screw checkpoint as a reproducible baseline. Run raw/manual crops and actual YOLO crops on matched labels to expose preprocessing/domain shift.
- Calibrate `p_defect`, `T_defect`, and `T_review_low` **on validation only**. Optimize DEFECT recall and report trade-offs with false alarms and review volume.
- Store DINOv2 preprocessing, feature extraction strategy, classifier weights, class order, calibration and model versions.
- Keep any last-block global/patch classifier as **offline comparator**, not production authority, unless explicitly approved after evaluation.
- Do not silently change to a different classifier just to improve a headline metric; preserve reproducible baselines.

For the long-term platform, expose a `BinaryQualityClassifier` interface to support other training strategies later (e.g. alternative supervised or GOOD-only methods) without changing the FastAPI API, data contract or postprocessing.

## 4. Localization: AnomalyCLIP and optional Relation Map

### 4.1 AnomalyCLIP-only baseline — REQUIRED first

- Use a fixed/pinned upstream implementation and checkpoint; log its provenance and known training domains.
- Feed the same oriented crop (and preprocessing transform) used for inspection.
- Return `anomaly_score` **as secondary evidence only**, a continuous heatmap, and an optional binary mask **only when its threshold is validated**.
- Run on all DINOv2 `DEFECT` instances. Also run on `REVIEW_REQUIRED` / low-confidence GOOD instances, without changing their original classifier label.
- Map raw heatmaps to the crop and (when relevant) full-image coordinates using saved inverse geometry.
- If localization fails or the map is empty/contradictory, preserve the classifier result and mark missing/conflicting evidence. Never overwrite with GOOD.

### 4.2 Relation Map experiment — OPTIONAL adoption

Reference implementations: [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) and the [relation prototype method](https://www.mdpi.com/2079-9292/15/18/4293). Their exact compatibility with the chosen checkpoint/backend needs code-level verification.

- Construct normal prototypes from **verified GOOD training crops only**. Require consistent orientation (including screw head-to-tip) for positional prototypes.
- Check that matching token-order patch features `F ∈ R^(N×D)` and attention `A ∈ R^(N×N)` can actually be extracted for the required layers/heads. Under convention `A[i,j] = weight from patch j to i`, the contextual features are `M = A @ F`; log the convention and prove shapes through tests.
- If backend tensors are unavailable, record `INCOMPATIBLE_BACKEND` and retain AnomalyCLIP-only; do not invent attention maps.
- Compare base AnomalyCLIP map, relation-derived map and fused map. Fit any normalization, fusion weights and binary-mask threshold **only on validation**.
- Metrics on unchanged labeled holdout: pixel AP, Dice, IoU, defect-region coverage, false-positive pixels on GOOD, and results by defect type/domain where labels exist.
- Compare **ungated** localization on labeled examples and **gated** end-to-end localization after the classifier.
- Enable fusion for the deployed model bundle only if the improvement is validated and latency/complexity are acceptable. Otherwise keep AnomalyCLIP alone.

## 5. VLM: English evidence report, not decision authority

### 5.1 Responsibility and routing

**CONFIRMED:** All classifier-DEFECT screw crops must be eligible for VLM reporting. The VLM receives original visual evidence and, where useful, a heatmap overlay or detail crop. It describes **visible** suspected defects in English, may flag a contradiction, and must not relabel the object or claim the heatmap proves a defect.

- `DEFECT` → enqueue an asynchronous VLM report for **every detected defect object**. Failure of VLM must **not** block the inspection or erase the DEFECT classification.
- `REVIEW_REQUIRED` → AnomalyCLIP evidence and human-review flag; VLM reporting on this path is **not required in v1**, because the user explicitly asked that all DEFECT objects be reported. This can be made configurable later.
- `GOOD` → do not call VLM.
- Use `mock` backend in automated tests; use a cloud VLM after credentials/policy approval. The earlier plan proposes Gemini via Vertex AI and local Qwen2.5-VL as a future candidate. Model ID, current availability, endpoints and pricing must be verified at integration time, not presumed from this plan.

### 5.2 VLM JSON contract — PROPOSED

```json
{
  "report_version": "1.0",
  "inspection_id": "insp_001",
  "object_id": "screw_02",
  "language": "en",
  "classifier_label": "DEFECT",
  "visible_defect_type": "scratch",
  "defect_location": "screw_head",
  "description": "A narrow surface scratch is visible near the screw head.",
  "visible_evidence": ["thin irregular line", "localized surface discoloration"],
  "heatmap_consistent_with_visible_evidence": "unknown",
  "other_possible_explanation": "image reflection",
  "needs_review": true,
  "report_status": "completed"
}
```

`classifier_label` must be injected by the pipeline and cannot be authored/changed by the VLM. Unknown or unobservable defect type/location must use `unknown`, not fabricated specificity. Do not add severity until a **product-specific severity rubric** exists. Validate JSON shape, string limits, required fields and allowed values; on malformed output use `report_status=failed` and record diagnostic metadata.

### 5.3 Token optimization, caching, security

- Send only the clean crop and the minimum helpful visual evidence. Compare visual detail/resolution settings on subtle defects before fixing them.
- Cache validated output using SHA-256 over exact image/overlay bytes **plus** schema, prompt, model/version, inference options and any classifier evidence included in prompts. Similar-looking objects must not share a cache entry.
- Record model/provider, time, input/output tokens, billed cost, retries, cache hits and reporting latency. Avoid API calls on confirmed GOOD.
- Keep API keys on the server, never in client code or committed configuration. Use upload size limits, MIME verification, privacy-safe logs, and configurable image-retention/deletion policies.
- **Pilot permission to use cloud does not equal permission to upload later customer-confidential images.** Require customer/org approval and region/data-retention review before each deployment. Provide a future local/offline VLM adapter when imagery must remain on site.

## 6. Output specification and visual hierarchy

For one input image containing N detected screws/objects, provide the following:

1. **Overview image:** original frame with an oriented bounding box and deterministic identifier (`screw_01`, `screw_02`, …) for each detected object. Preserve the original source image unchanged. Status annotation may be added after classification.
2. **One status visualization per detected object:** use the crop at original color, displayed **inside a solid status-colored panel/background or border** (green = GOOD, red = DEFECT, amber = REVIEW_REQUIRED). Do not recolor the actual screw pixels, which could hide defects.
3. **One evidence map for every DEFECT:** `heatmap.png` and `heatmap_overlay.png`; optional validated `binary_mask.png`. Also save evidence for REVIEW_REQUIRED where localization runs.
4. **Detailed machine-readable JSON:** complete object results, scores, model versions, stage statuses, image references, localization outputs, VLM results or pending/failed state.
5. **Compact PDF:** one small overview table, with inspection ID/date/product, counts, and each object ID/status/confidence/suspected defect type/report state; include a brief note for pending reviews. Avoid substituting a PDF for detailed JSON.
6. **Per-defect English VLM report:** the VLM's validated detail is stored in JSON and can be summarized in the PDF. A separate PDF per object is **not required** in v1.

Example artifact tree — **PROPOSED**:

```text
runs/<inspection_id>/
  input/original.png
  overview/objects_labeled.png
  objects/
    screw_01/
      crop_original.png
      crop_status.png            # green panel
      result.json
    screw_02/
      crop_original.png
      crop_status.png            # red panel
      heatmap.png
      heatmap_overlay.png
      binary_mask.png            # optional
      vlm_report.json            # async; may be pending
      result.json
    screw_03/
      crop_original.png
      crop_status.png            # amber panel
      heatmap_overlay.png        # if localization succeeded
      result.json
  inspection.json
  summary.pdf
  manifest.json
```

### 6.1 Inspection JSON example

```json
{
  "schema_version": "1.0",
  "inspection_id": "insp_001",
  "product_key": "screw",
  "source_type": "image",
  "status": "completed_with_pending_reports",
  "model_bundle_version": "screw-001",
  "summary": {
    "detected_count": 3,
    "good_count": 1,
    "defect_count": 1,
    "review_required_count": 1,
    "no_detection": false
  },
  "overview_image": "overview/objects_labeled.png",
  "summary_pdf": "summary.pdf",
  "objects": [
    {
      "object_id": "screw_01",
      "detection_score": 0.99,
      "obb_xyxyxyxy": [[10, 10], [45, 10], [45, 25], [10, 25]],
      "classifier_label": "GOOD",
      "p_defect": 0.03,
      "operational_status": "GOOD",
      "needs_review": false,
      "status_crop": "objects/screw_01/crop_status.png",
      "localization_status": "not_requested",
      "vlm_status": "not_requested"
    },
    {
      "object_id": "screw_02",
      "detection_score": 0.98,
      "classifier_label": "DEFECT",
      "p_defect": 0.96,
      "operational_status": "DEFECT",
      "needs_review": true,
      "status_crop": "objects/screw_02/crop_status.png",
      "heatmap_overlay": "objects/screw_02/heatmap_overlay.png",
      "localization_status": "completed",
      "vlm_status": "pending"
    },
    {
      "object_id": "screw_03",
      "classifier_label": "GOOD",
      "p_defect": 0.48,
      "operational_status": "REVIEW_REQUIRED",
      "needs_review": true,
      "localization_status": "completed",
      "vlm_status": "not_requested"
    }
  ]
}
```

Values above are **illustrative**, not calibrated thresholds, actual predictions or real model metrics. All artifact paths in API results should be safe identifiers or authorized download links, **not unrestricted server filesystem paths**. Preserve API compatibility via `schema_version`.

### 6.2 Multi-object/sequence identity

- Within a still image, assign stable IDs in a reproducible spatial order, with documented tie-breaks.
- In a video/live stream, frame-level IDs are **not** automatically the same as physical-object IDs. Tracking/deduplication needs an explicit later design and validation; otherwise each frame is evaluated independently and repeated sightings are reported as such.
- For a failed single-object stage, isolate the failure: continue other objects and mark that object's stage explicitly.

## 7. FastAPI contract (design target, not existing implemented routes)

### 7.1 Initial still-image API

| Endpoint | Purpose | Expected behavior |
|---|---|---|
| `POST /api/v1/inspections` | Upload one image + `product_key` | Validate input; create inspection/job ID; perform or schedule image stages |
| `GET /api/v1/inspections/{inspection_id}` | Inspection status and JSON result | Show processing, completed, partial and failed states |
| `GET /api/v1/inspections/{inspection_id}/artifacts` | List allowed output artifacts | Return per-object images, overview, PDF and VLM reports |
| `GET /api/v1/inspections/{inspection_id}/artifacts/{artifact_id}` | Retrieve one authorized artifact | Prevent path traversal and disclosure across tenants |
| `GET /api/v1/products` | List deployed product bundles | Product key, bundle version and readiness |
| `GET /health` and `GET /health/ready` | Liveness and model/provider readiness | Distinguish server running from required model unavailable |

**Recommended behavior:** Return a job ID promptly; expose stage status while per-defect VLM calls are pending. Poll result endpoint first; WebSocket/SSE event streaming is optional. Maintain idempotency for client retries through request keys where practical. Require authentication and tenant isolation before processing proprietary enterprise data; pilot local auth policy **TBD**.

### 7.2 Video and future camera API

**Later-phase, not a v1 acceptance gate:** offline video upload/processing routes, per-frame outputs and aggregate report; live camera feed ingestion through approved transport adapter. The API contract must account for frame timestamps, dropped frames, backpressure, model throughput, queue limits and retention. Choice of `RTSP`, camera SDK, WebRTC or other transport is **TBD** and should remain behind an adapter.

## 8. Benchmark and acceptance metrics

### 8.1 Per-stage benchmarks — REQUIRED

| Stage | Required outputs and metrics |
|---|---|
| Detector | precision, recall, mAP50, mAP50–95, missed-object count, false detections on negative scenes, per-domain slice |
| Crop geometry | correct orientation, inverse transform, overlay alignment, crop bounds, image-to-crop object identity |
| DINOv2 classifier | confusion matrix, DEFECT recall/precision, GOOD recall, balanced accuracy, PR-AUC, AUROC when defined; actual YOLO-crop results |
| Review policy | review-band coverage, false negatives outside review band, queue size, false-alarm rate and per-domain results |
| AnomalyCLIP | pixel AP / Dice / IoU where masks exist, false-positive pixels on GOOD, defect-region coverage, runtime |
| Relation Map | same holdout metrics + delta against AnomalyCLIP-only; incremental latency and compatibility |
| VLM | valid JSON %, unsupported assertions %, evidence consistency, defect-description agreement with human-reviewed examples, time/tokens/cost |
| API/report | response and job correctness, output schema validity, PDF rendering, artifact permissions, failure and retry behavior |

### 8.2 Integrated metrics — REQUIRED

- **End-to-end DEFECT recall:** `number of GT defective objects both detected and classified DEFECT / total GT defective objects in full source images`. A missed detection counts as a miss.
- **Review-aware capture rate:** portion of GT defective objects detected and assigned either `DEFECT` **or** `REVIEW_REQUIRED` (report separately; never call REVIEW_REQUIRED a confirmed DEFECT).
- **End-to-end false-alarm volume:** GOOD objects flagged DEFECT or REVIEW_REQUIRED per domain and per inspected image.
- Localization quality **conditional on detected/classified defect objects** and **full-pipeline localization coverage**, reported separately.
- Per-image counts, per-object stage failures, total latency percentiles, throughput, VLM pending time and infrastructure utilization.
- For multi-domain testing, report each domain separately; do not average away a failing domain.

### 8.3 Initial acceptance gates

**Functional criteria (testable now):**

1. One full image with multiple screws produces the overview, reproducible IDs, original crop and correctly color-coded status panel for **every detected object**.
2. Every DEFECT object receives a heatmap attempt and asynchronous VLM report attempt; each attempt's final status is preserved.
3. Every uncertain-GOOD object is marked `REVIEW_REQUIRED`, **not silently relabeled DEFECT**.
4. GOOD outside the review band skips expensive localization/VLM steps.
5. Missing detections, no objects, localizer failure, provider failure and malformed VLM output have explicit non-GOOD failure statuses without hiding successful objects.
6. `inspection.json` validates against the published schema and `summary.pdf` accurately reflects stored results, including pending reports.
7. Given a conforming **second product dataset**, the same generic training/inference code paths execute **without source modifications**; all product-specific choices come from configuration and model bundle.
8. Independent test split and leakage validation pass for any new end-to-end performance claim.

**Numeric criteria (TBD; must be agreed and measured before production release):** minimum DEFECT recall, maximum false alarms/reviews, minimum detection recall, localization quality, image/video processing latency, camera FPS, VLM report accuracy, reliability, uptime, hardware, storage and retention. Prior pilot metrics are **baselines, not acceptance thresholds**.

## 9. Implementation roadmap (ordered by dependency)

### Phase 0 — Repository audit + requirements freeze

- Read current code, dataset directories, manifests and benchmarks; verify recorded assets and checkpoint files.
- Identify and inventory generic code vs screw-specific assumptions.
- Draft product data schema, model bundle contract, output JSON schema and configuration schema.
- Freeze a shared source-image split; produce a leakage report and provenance report.
- Deliver: `audit.md`, data-schema validator, known limitations and a signed-off requirement list.
- **Gate:** No unverified data/classifier leakage in the new integrated holdout.

### Phase 1 — YOLO-OBB refinement and reusable detector interface

- Adapt the existing screw training/evaluation code into configuration-driven entrypoints without breaking existing scripts.
- Fine-tune across existing available datasets, check annotation quality and save candidate checkpoints.
- Verify oriented crop transforms and inverse overlays; benchmark per-domain detection recall.
- Deliver: detector bundle, per-domain benchmark, example overview/crops, geometry tests.
- **Gate:** Deterministic crops; object-to-evidence mapping correct; comparison uses validation rather than holdout to choose checkpoints.

### Phase 2 — DINOv2 classifier and 3-state policy

- Reuse frozen baseline, measure real YOLO-crop classification separately, calibrate binary/review thresholds on validation.
- Implement `GOOD`, `DEFECT`, `REVIEW_REQUIRED`, `NO_DETECTION`, `STAGE_FAILED` with explicit state transitions.
- Deliver: model calibration, confusion matrices, false-negative review-band analysis, per-object JSON.
- **Gate:** Classification label and operational state never overwrite one another; cross-domain limitation documented.

### Phase 3 — AnomalyCLIP-only localization

- Pin and validate compatible AnomalyCLIP checkpoint; implement heatmap/mask projection.
- Evaluate on ground-truth masks where available and verify sample overlays visually.
- Deliver: heatmap for DEFECT and review-band cases, per-stage metrics and localizer failure test.
- **Gate:** Localization does not alter classification; empty or contradictory maps are preserved as evidence states.

### Phase 4 — Relation Map experiment

- Verify attention/feature availability, token order and shape conventions in unit tests.
- Build GOOD prototypes and test relation heatmap/fusion with validation-only tuning.
- Deliver: reproducible AnomalyCLIP-vs-fusion comparison and adoption decision.
- **Gate:** Fusion is optional; production default remains AnomalyCLIP-only until proven better.

### Phase 5 — VLM + JSON + compact PDF

- Implement a mock report adapter and cloud adapter with asynchronous queue, validation, retries, cache and privacy guards.
- Route all classifier DEFECT objects to VLM; save English reports without allowing label changes.
- Generate compact inspection summary PDF and verify JSON/PDF consistency, including pending states.
- Deliver: reporter adapter, example PDF, VLM test harness, cost/latency table.
- **Gate:** One VLM failure cannot lose, block or change the detector/classifier result.

### Phase 6 — FastAPI still-image end-to-end product

- Create and test documented image/status/artifact/health endpoints and product registry.
- Integrate stage jobs, all outputs and error handling, with an API schema for future web/desktop clients.
- Deliver: executable FastAPI service, OpenAPI documentation, smoke tests and representative examples.
- **Gate:** Client can inspect a full image and retrieve overview, per-object status panels, heatmaps, JSON and compact PDF.

### Phase 7 — Offline video, then live camera

- Add frame ingestion, timestamps, configurable sampling, workload controls and optional tracking/deduplication.
- Separate offline video validation from live camera claims. Test with representative videos and hardware.
- Deliver: video API, per-frame evidence, aggregate results, performance report; camera adapters only after environment and FPS targets are specified.
- **Gate:** Proven stable throughput/latency and controlled handling of dropped/duplicate frames before real-time use.

### Phase 8 — No-source-change product onboarding proof

- Prepare a **second** labeled product dataset with the common manifest format and config.
- Run dataset validation, detector/classifier training, calibration, inference and evaluation using the same generic commands.
- Record every source-file diff; acceptance requires **zero inspection/training core-code edits** (new config/data/artifacts allowed).
- Deliver: reproducible onboarding guide, tested product bundle, evaluation and limitations report.
- **Gate:** Second product is processed end to end and honestly benchmarked; no unsupported accuracy promise.

## 10. Testing and failure matrix

**Automated tests required:**

- Dataset: invalid OBB corners, missing GOOD/DEFECT labels, broken paths, mismatched object IDs, duplicates across splits, near-duplicate frame groups, missing masks and malformed config.
- Detector: multi-object, rotated objects, object cropped at boundary, 0 detections, false positives on empty image, per-object stable identity.
- Geometry: forward/inverse transform round-trip, color map alignment, masks resized with nearest-neighbor interpolation.
- Classifier: preprocessing parity, threshold boundary, uncertain GOOD routing, exact preservation of `classifier_label`, model bundle version mismatch.
- Localization: failure/timeout, inconsistent heatmap, invalid attention shape/token ordering, binary-mask threshold provenance.
- VLM: unavailable cloud, retries, rate limit, invalid JSON, missing report, hallucinated defect type, unknown location, stale cache invalidation, async result retrieval.
- API: invalid files, overlarge inputs, unauthorized artifacts, path traversal, concurrent jobs, idempotency, PDF rendering, schema validation.
- End-to-end: multiple objects with mixed GOOD/DEFECT/REVIEW_REQUIRED, all artifacts, one failed object stage, no detection, independent held-out metrics.
- Video (later): dropped frames, timestamp order, duplicate object tracking, backpressure and latency outliers.

Every experiment should write a machine-readable test/metrics report. Distinguish **not implemented**, **skipped due to missing data**, **failed**, **passed** and **unvalidated**; do not conflate them.

## 11. Operational, privacy and reproducibility requirements

- **PROPOSED** Python/FastAPI backend, adapter-based GPU workloads and optional worker queue. Specific deployment platform (Mac, Windows, Linux, on-prem/cloud), CUDA version, accelerator count and latency limits are **TBD**.
- Keep heavy training outside the API request process. Avoid running expensive models in event-loop request handlers; use controlled workers and concurrency limits.
- Track artifact lineage and make prior product bundles loadable/rollbackable. The API and UI should refer to product bundle IDs rather than local absolute model paths.
- Store only derived review states in v1 (`needs_review`, reason, timestamps). Do **not** build a full reviewer dashboard yet.
- Avoid user-visible unverified claims such as “works for all products”, “zero-shot” after training on test-domain examples, or “production ready” without a locked evaluation on target-site data.
- Respect source dataset licensing/usage. Require explicit permission and appropriate storage controls before sending new enterprise camera images to third-party VLM services.
- Default Git discipline: inspect repo status; work on a dedicated branch as appropriate; **do not commit, push, merge, delete source datasets or overwrite validated checkpoints without approval**. Do not change original test data to make metrics appear better.

## 12. Open questions and provisional defaults

Only these unresolved items should block final production commitments; they should not prevent building the still-image prototype.

| ID | Question | Provisional plan |
|---|---|---|
| Q6 | Training strategy for binary quality model? | **PROPOSED:** supervised GOOD/DEFECT DINOv2 + Logistic Regression first; backend remains replaceable |
| Q21a | Where will training and inference run (Mac/Windows/Linux/Colab/cloud)? | Environment-agnostic code; benchmark on actual selected hardware before deployment |
| Q21b | What GPU/VRAM and target image/video throughput? | **TBD:** no guaranteed FPS/latency, especially for VLM or video |
| Q22a | Should Codex implement phases automatically or stop for approval after each? | **PROPOSED:** phase gates; continue only after tests pass and report blockers clearly |
| Q22b | Exact Git commit/push permissions? | **PROPOSED:** no commit/push/merge without explicit approval |
| Q3-detail | Which marks are unacceptable for each new product (cosmetic vs functional)? | Conservative per-product written quality policy; all undesired visible anomalies as DEFECT for screw pilot |
| Metrics | DEFECT recall and false-alarm numerical thresholds? | Must be specified per company, camera and product; prioritize recall in validation |
| Camera | Transport/protocol and deduplication needs? | After still-image and offline-video proof; transport adapter TBD |
| Cloud ops | Exact VLM provider/model, quotas, region, image retention and budget? | Cloud first using adapter; verify official current service behavior at integration time |
| PDF | Exact paper size, branding and maximum number of rows? | **PROPOSED:** one compact inspection-level tabular PDF; paginate if needed |

## 13. Handoff checklist for Codex

1. **Audit before implementing.** Compare this plan to actual repo state, identify existing modules, dependencies, tests, model artifacts and stale paths. Report findings and do not assume source filenames still match.
2. Start with a **working screw still-image vertical slice** while building shared contracts; do not prematurely design around every possible factory scenario.
3. Keep the **product onboarding invariant** in code reviews: new product = new validated dataset + config + train/calibrate + model bundle; not `if product == "screw"` branches in core modules.
4. Run independent per-model metrics, then a leak-free integrated benchmark. Mark unavailable masks/GT as `not_evaluable`.
5. Preserve classifier decision and review state separately. VLM/relation map cannot silently change labels.
6. Build the exact output hierarchy (overview; colored individual crops; defect heatmaps; VLM reports; detailed JSON; compact PDF) and verify API clients can retrieve artifacts.
7. Prefer adding small tested adapters around existing implementation to broad rewrites. Keep legacy benchmark artifacts and document differences.
8. Keep unresolved deployment, live camera and accuracy targets **TBD** instead of inventing concrete specifications.
9. Provide at the end of every phase: changed files, commands, tests, benchmark results, artifacts, blockers, and next-phase recommendation.
10. **Never claim the platform is production-qualified on a new factory product without held-out, target-camera data and agreed numerical thresholds.**

---

### Source note

This document updates the provided **“Screw inspection implementation plan”** using the user's answers to clarification questions Q1–Q20. Recorded dataset paths, baseline metrics and previously noted split conflicts have been preserved as **historical information requiring repository verification**. This specification does not assert that the proposed API, data contract or pipeline has already been implemented.
