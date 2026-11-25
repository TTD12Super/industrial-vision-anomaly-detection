# Screw Image-First End-to-End Integration Plan

**Status:** Candidate integration implemented and exercised on real Torch/MPS inference; the end-to-end result is still a development prototype and fails the quality gates on the current detector-generated test split.

**Scope:** Static images containing multiple screws. Video, tracking and temporal aggregation remain out of scope.

## 1. Goal

Connect one screw inspection flow from a source image to deterministic quality evidence and an optional local VLM report:

```text
source image
  -> detector
  -> object crop + provenance
  -> screw routing / UNKNOWN / REVIEW
  -> GOOD / DEFECT decision
  -> heatmap and binary-mask evidence for suspicious objects
  -> optional grounded VLM report
  -> one JSON result plus human-readable artefacts
```

The VLM is an explanatory component. It must not change the detector, product-routing or GOOD/DEFECT decision.

## 2. One-authority module selection

The integrated flow must have one authoritative method per stage. Other methods remain benchmark challengers and are not run as competing decision authorities in the same production result.

| Stage | Selected integration candidate | Authority | Challenger or research-only options | Selection reason and limitation |
|---|---|---|---|---|
| Multi-object detection | YOLO-OBB | Detector boxes/corners | RT-DETR AABB | YOLO-OBB preserves oriented geometry and has much lower measured latency. Its WEEE recall is not yet a release gate; RT-DETR remains the recall challenger. |
| Crop/provenance | Deterministic OBB crop with explicit padding and manifest rows | Crop path and source-object join | AABB crop from RT-DETR | OBB is the current operational geometry contract. Every crop must retain source image ID, prediction index, GT index where available, confidence, box and hashes. |
| Product routing | DINOv2 gallery retrieval with calibrated rejection to `UNKNOWN`/`REVIEW_REQUIRED` | Routing status | Closed-world screw route | Known screw crops are routed to the screw bundle only after the gallery threshold/margin checks. A detector class is not silently treated as a quality label. |
| GOOD/DEFECT | Exact kNN with validation-selected recall-first threshold | Quality decision | DINOv2 last-block patch-aware head; frozen-global heads; DINOv2 + AnomalyCLIP cascade | Exact kNN is the current reproducible classifier authority. The recall-first threshold is selected on validation only. Its locked CAD-SD test evidence reaches defect recall `90.96%`, GOOD-FP `10.48%`, AUROC `0.9709` and AP `0.9658`; this is cross-dataset evidence, not a target-camera holdout. |
| Continuous heatmap | AnomalyCLIP input + supervised DINOv2 patch refiner | Heatmap evidence | PatchCore; Relation Map; high-resolution/spatial/RGB refiners | The supervised-refiner development sweep is the strongest saved localization candidate (test Pixel AP 0.4803 with `heatmap_open9`), and it runs through the current localizer contract. Its test split was used during development, so it is not production evidence. |
| Binary localization | The same supervised-refiner output, thresholded with validation calibration | Optional binary mask | PaDiM; raw PatchCore/DINO masks | This keeps one localization source for the heatmap and mask and avoids presenting PaDiM's weaker Pixel AP (0.0261) as a stronger boundary. The mask remains evidence only and never changes GOOD/DEFECT. |
| Report generation | Deterministic JSON/report builder first | Grounded facts and counts | Local VLM narrative | The deterministic record is the source of truth; narrative generation is optional and must be cacheable and reviewable. |
| VLM | Qwen3-VL-4B-Instruct Q4_K_M via llama.cpp/Ollama | Explanation only | Gemma 3 4B; Qwen2.5-VL-3B/7B; InternVL3-2B/8B | Small local multimodal model, Apache-2.0 model/repository, GGUF availability and support for Apple Silicon/CUDA. It is not a defect classifier. |

### Detector decision rule

YOLO-OBB is the operational candidate because the current contract needs oriented crop geometry and its measured latency is approximately 23 ms/image versus approximately 173 ms/image for the RT-DETR comparison. This is not a claim that YOLO-OBB has passed production: its locked WEEE object recall is only 58.37%. RT-DETR has higher recall (73.92%) but much lower matched precision and no native OBB output in the current comparison. A new independent holdout may change this selection.

### Heatmap decision rule

The integrated candidate now uses the supervised DINOv2 patch refiner over the AnomalyCLIP map. On the saved supervised-development comparison, `heatmap_open9` reached Pixel AP `0.4803`, while the same report's raw/opening alternatives were lower. PatchCore and PaDiM remain reproducible independent baselines (`0.0517` and `0.0261` Pixel AP respectively) and are not combined into a second decision authority. The `0.4803` result is not an independent target-camera result: it must be revalidated with a sealed holdout before any promotion.

## 3. Runtime contract

```text
image
  -> YOLO-OBB
  -> object records
       - source_image_id
       - object_id
       - prediction_index
       - confidence
       - OBB corners
       - crop path
       - manifest/hash lineage
  -> screw routing
       - KNOWN_PRODUCT
       - UNKNOWN
       - REVIEW_REQUIRED
  -> Exact kNN
       - GOOD
       - DEFECT
       - REVIEW_REQUIRED / STAGE_FAILED
  -> if DEFECT or REVIEW_REQUIRED:
       - supervised-refiner continuous heatmap
       - supervised-refiner binary mask when calibrated
       - source-coordinate overlays
  -> if policy allows and evidence exists:
       - local Qwen3-VL report
  -> aggregate JSON + overview + per-object evidence
```

Rules:

1. A detector miss remains a miss; it must not disappear from the report.
2. An unmatched prediction is preserved as a reviewable false-positive candidate.
3. `UNKNOWN` and `REVIEW_REQUIRED` are operational statuses, not GOOD or DEFECT labels.
4. A failed localizer cannot convert DEFECT to GOOD.
5. A VLM failure cannot change any deterministic decision.
6. The final report must distinguish `decision_source`, `heatmap_source`, `mask_source` and `vlm_source`.

## 4. Local VLM design

### 4.1 Candidate

Initial candidate: `Qwen/Qwen3-VL-4B-Instruct-GGUF`, language weights at `Q4_K_M` and the matching vision projector. The model card reports a 2.5 GB Q4_K_M language file and compatibility with llama.cpp, Ollama, CPU, CUDA and Apple Silicon. The official Qwen3-VL repository is Apache-2.0.

The runtime adapter must support either:

```text
llama-server / llama serve
  -> OpenAI-compatible local endpoint
```

or direct local invocation. The integration must not require a cloud API key.

The current product YAML remains on `mock` so existing tests and offline
inspection stay deterministic. To enable the local model after starting a
llama.cpp server or Ollama's OpenAI-compatible endpoint, change only the
reporter block:

```yaml
reporter:
  backend: local_vlm
  base_url: http://127.0.0.1:8080/v1
  model: qwen3-vl-4b
  timeout_seconds: 120
  max_tokens: 512
  workers: 1
  async_enabled: true
```

`async_enabled: true` is the API-friendly mode: the deterministic inspection
returns immediately and the client polls the inspection status while the VLM
report is written to the per-object artifact. The checked-in
`configs/experiments/screw_local_vlm.yaml` deliberately sets it to `false`
because that file is the static-image experiment runner; its command returns
only after `vlm_report.json` is durable. This does not change any classifier or
heatmap decision.

The local endpoint is optional. If it is unavailable, the deterministic JSON,
heatmap and binary-mask artifacts remain valid and the VLM report is marked
failed; the classifier decision is not changed.

The checked-in static local-VLM experiment points all DINOv2 consumers to
`artifacts/runtime/dinov2_clean`, whose materialized ViT-B/14 checkpoint is
provided by the source cache `artifacts/dinov2_knn` and has SHA-256
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`.
The otherwise equivalent `artifacts/production/mvtec_ad` copy is currently a
macOS `compressed,dataless` placeholder with zero allocated blocks, so it must
not be used for a local run until File Provider has materialized it.
The runtime view can be recreated with
`PYTHONPATH=src:scripts python scripts/prepare_dinov2_runtime.py`; it creates
an actual `trusted_list` and only links the verified repository/checkpoint, so
`torch.hub` does not read the iCloud placeholder. The script records the
checkpoint hash and is covered by `tests/unit/inspection/test_dinov2_runtime.py`.

For the selected anomaly pair there is a separate candidate configuration at
`configs/experiments/screw_patchcore_padim.yaml` (the filename is retained for
artifact continuity; its current localizer is the supervised-refiner path).
The existing
`configs/products/screw.yaml` intentionally remains on the historical
AnomalyCLIP path until the candidate is run on the same locked crop/holdout
protocol; this prevents a configuration switch from being mistaken for a
validated improvement.

### 4.2 VLM input policy

The VLM receives only suspicious objects:

- original image or a source-image view;
- object crop;
- heatmap overlay;
- binary mask overlay when available;
- deterministic JSON facts: product, classifier decision, score, threshold, localization status and artifact paths.

It must answer in structured JSON with:

```json
{
  "visible_defect_type": "crack|scratch|dent|unknown",
  "defect_location": "head|shaft|thread|unknown",
  "description": "",
  "visible_evidence": [],
  "heatmap_consistent_with_visible_evidence": "yes|no|unknown",
  "other_possible_explanation": "",
  "needs_review": true
}
```

The deterministic wrapper adds `inspection_id`, `object_id`, fixed English
language, fixed `classifier_label: DEFECT`, and `report_status`. Invalid or
missing provider fields are persisted as a failed report instead of being
treated as visual evidence.

The prompt must explicitly prohibit inventing a defect when the heatmap is broad, missing or inconsistent with the crop.

### 4.3 Cache and token control

The current local request cache key includes:

```text
reporter model_version (model identifier + prompt-template hash)
serialized deterministic context
exact bytes of the source/crop/heatmap/mask evidence
```

For the same inspection/object path, identical evidence must not trigger a
second local generation. The cache is an optimization and never replaces
provenance; cross-inspection deduplication is intentionally not claimed.
If a deployment changes the model file without changing its identifier, the
operator must update the configured model identifier/version before reuse;
the current adapter does not inspect a model file path itself.

## 5. Required implementation artifacts

The integration is not complete until it produces:

1. One callable static-image runner.
2. One per-image JSON result.
3. One object-level manifest with matched/missed/unmatched accounting.
4. Per-object crops and evidence overlays.
5. Deterministic latency by stage.
6. Optional local VLM report plus cache metadata.
7. A reproducible end-to-end benchmark report.
8. A figure pack containing architecture, object funnel, detector errors, classification confusion matrix, heatmap comparison, latency and qualitative evidence.

The local-VLM evidence benchmark is exposed by
`scripts/benchmark_screw_local_vlm.py`. It reads a saved `inspection.json`,
sends only `DEFECT` object evidence, measures request latency, validates the
English report schema and records the inspection/evidence SHA-256 values.

## 6. Verification gates

Before promoting the integrated candidate:

- run all modules on one known multi-screw image;
- verify every detected object has one result row;
- verify missed and unmatched detector records are retained;
- verify GOOD objects do not receive a production heatmap call;
- verify DEFECT/REVIEW objects receive localization attempt status;
- verify UNKNOWN objects do not enter screw-specific anomaly inference;
- verify VLM receives only eligible evidence;
- verify rerunning the same input yields the same deterministic JSON except runtime timestamps;
- run the complete unit/integration test suite;
- report the independent-holdout status separately from development evidence.

No production-readiness claim is allowed until the target-camera/independent holdout is locked and evaluated without tuning leakage.

## 8. Current verification snapshot

Verified in the current checkout:

- candidate readiness for `configs/experiments/screw_patchcore_padim.yaml` reports all configured files/calibrations present;
- the complete inspection unit suite passes 131/131 tests and the YOLO unit
  suite passes 3/3 tests under the temporary Python 3.15 environment (the
  Python 3.13 Torch import still stalls before these modules run);
- the local-VLM adapter test now covers OpenAI-compatible image payloads, validated JSON persistence and byte-identical report cache reuse (3 reporting tests pass);
- Python compilation passes; `git diff --check` is not currently conclusive because
  the pre-existing dirty worktree makes the command exceed a five-second check
  window;
- static image smoke flow produces original image, per-object crop/status/evidence artifacts, SHA-256 provenance and stage latency fields;
- the local runtime smoke was rerun successfully after isolating DINOv2 from the
  macOS iCloud `compressed,dataless` `trusted_list` placeholder. The clean
  runtime is `artifacts/runtime/dinov2_clean`; the full result is
  `runs/e2e_verify_local_vlm_sync_20261125/e2e_verify_local_vlm_sync_20261125/`.
  It completed detector → crop/provenance → DINOv2 routing → Exact kNN
  GOOD/DEFECT → AnomalyCLIP/supervised-refiner heatmap → binary mask → local
  Qwen3-VL report. This is a functional smoke, not a quality claim: the one
  fixture was classified DEFECT with `p_defect=0.99998`, while Qwen3-VL
  correctly returned `visible_defect_type=unknown` and `needs_review=true`
  because the visual evidence did not clearly support the classifier decision.
  Total image latency was `35.771 s`, including local VLM generation; the
  detector stage was `0.833 s`.
- a post-runtime-preparation smoke was rerun against the same fixture at
  `runs/e2e_verify_local_vlm_post_runtime_20251125/e2e_verify_local_vlm_post_runtime_20251125/`.
  It completed with the same stage contract, local Qwen3-VL status `completed`,
  total image latency `27.378 s`, and detector latency `0.719 s`. The VLM report
  again marked the deterministic DEFECT result for review because no visible
  defect type or localized evidence was supported. This confirms runtime
  reproducibility, not model quality.
- a second smoke used the real WEEE multi-object image
  `datasets/external/weee_screw/images/test/pc_13_663.png`, whose COCO test
  annotation contains two screw objects. The current detector emitted four
  candidates in `1.560 s`; all four were rejected by the calibrated DINOv2
  open-set router as `UNKNOWN_PRODUCT`, so none reached GOOD/DEFECT or VLM.
  This confirms the safety invariant that unknown/background candidates are
  not silently classified, but also exposes the unresolved WEEE detector/crop
  generalization problem: this is not evidence of multi-object recall.
- the same deterministic path was also exercised on
  `datasets/external/weee_screw/images/test/screws_22.png`, whose COCO test
  annotation contains 13 screw objects. It emitted four candidates in
  `0.992 s`; all four were rejected as `UNKNOWN_PRODUCT` and no classifier or
  localizer was invoked. This strengthens the failure diagnosis: the current
  integrated bundle is safe against silent false classification, but its
  current detector/gallery pair does not yet cover this WEEE scene.
- the `weee_adapt10` detector challenger improves the detector-only locked
  WEEE test to `66.51%` recall, `66.03%` matched-prediction precision and
  `26.7 ms/image` at its validation-selected confidence `0.30`. When replayed
  through the real crop/router flow on `screws_22.png`, its 16 candidates
  produced only `7–12 px` crops because the annotated screws are extremely
  small in the 1280×720 scene; all 16 were correctly rejected as
  `UNKNOWN_PRODUCT`. This is a data/scale limitation of using WEEE as a
  multi-object integration check, not evidence that the anomaly classifier can
  inspect those tiny crops. The locked benchmark is saved at
  `reports/experiments/weee_yolo_adapt10_rtdetr_fair_aabb_holdout_rnd1.json`.
- real YOLO-OBB + DINOv2 gallery + Exact kNN + supervised-refiner localizer completed on a known MVTec crop under the thread-limited MPS environment;
- a static smoke on `mvtec_ad_full_good_train_306_obj01` produced the full
  crop/heatmap/mask/JSON/PDF artefact set, but the GOOD object was classified
  DEFECT and its heatmap was spatially broad. This is a useful negative
  qualitative check and confirms that the current bundle is not production
  ready even though the flow is executable;
- the active-refiner aggregate report over all 78 test images is `reports/experiments/screw_e2e_active_refiner_rnd4_20251125.json`; it records 15/23 end-to-end defect captures, 37/55 GOOD false alarms and image p95 4.668 seconds;
- the selected Exact kNN patch-memory artifact is also evaluated in the locked
  mixed-nominal CAD-SD protocol at
  `reports/experiments/dinov2_patchcore_lite_mixed_top1_allowed03_rnd3.json`:
  CAD-SD test defect recall `90.96%` (151/166), GOOD false-positive rate
  `10.48%` (22/210), image AUROC `0.9709` and image AP `0.9658`. The artifact
  uses only CAD-SD train/0_good for its nominal memory/calibration split and
  does not use CAD-SD test scores for selection. This strengthens classifier
  generalization evidence, but it is not a target-camera end-to-end holdout.
- the previous PatchCore/PaDiM candidate remains available as a replayed baseline at `reports/experiments/screw_e2e_candidate_rnd6_20251125_replayed.json`.
- detector-generated crop calibration has been made reproducible by
  `scripts/benchmark_screw_detector_crops.py`; the train report contains 193
  KNOWN_PRODUCT crops (162 GOOD, 31 DEFECT), while 46 ground-truth objects are
  missed and 94 matched objects are rejected as UNKNOWN;
- `scripts/train_detector_crop_dinov2_logreg.py` and the optional
  `--classifier-artifact-dir` benchmark override provide a detector-crop
  classifier challenger, but it is not promoted until it is evaluated on the
  locked test split with a validation-frozen threshold. Its validation evidence
  does not meet the requested 90% DEFECT recall / 20% GOOD false-positive gate
  (validation: 100% / 38.46%), so it remains a challenger;
- with threshold `0.07482706755399704` selected on validation and then frozen,
  the detector-crop challenger reports 14/15 known matched DEFECT objects
  (93.33% recall) and 5/37 known matched GOOD objects flagged (13.51% FP) on
  the current development test. This is conditional classifier performance,
  not end-to-end performance: the same run has 9 missed GT objects and 17
  matched objects rejected as UNKNOWN. The test report is
  `reports/experiments/screw_detector_crop_logreg_test_rnd2.json`, and the
  threshold is not selected from that report.

Not yet verified in this checkout:

- the supervised-refiner bundle on an independent target-camera holdout;
- visual/reporting quality of Qwen3-VL across a representative labeled
  holdout; the current five-object audit is still exploratory and too small
  for an accuracy or production claim;
- detector recall and crop IoU after the current threshold on a target-camera multi-object holdout;
- whether the detector-crop challenger keeps its conditional result on an
  independent target-camera holdout; it must not replace Exact kNN before that
  check.

The first real local-VLM smoke is saved at
`reports/experiments/screw_local_vlm_benchmark_rnd1.json`: Qwen3-VL-4B
Q4_K_M completed one DEFECT evidence request in 13.93 seconds. It returned
`visible_defect_type=unknown`, `heatmap_consistent_with_visible_evidence=unknown`
and `needs_review=true` because the deterministic DEFECT decision was not
visually supported. This validates the reporting contract, not VLM defect
classification accuracy.

A five-object follow-up is saved at
`reports/experiments/screw_local_vlm_benchmark_rnd2.json`, with the object-level
ground-truth join in
`reports/experiments/screw_local_vlm_benchmark_rnd2_audit.json`. All 5/5 local
requests completed and validated, with mean latency 17.923 seconds and maximum
latency 18.297 seconds. All five predictions matched a ground-truth object at
OBB IoU >= 0.5: 3 were GOOD and 2 were DEFECT. The VLM named a visible defect
type for 0/2 DEFECT objects; all 3 matched GOOD objects were sent for review
because the upstream classifier had false positives. Therefore this run shows
that the VLM can act as a grounded review/reporting gate without inventing a
defect, but it does not yet improve classification or localization accuracy.
The audit is explicitly exploratory and must not be used as a production
quality claim.

The working MPS command must set the thread limits documented in the run notes
(`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `KMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1` and
`PYTORCH_ENABLE_MPS_FALLBACK=1`). Without those limits, this Python 3.13
environment can stall while importing Torch; this is an environment constraint,
not evidence of production readiness.

## 7. Research references used for the local VLM decision

- [Qwen3-VL official repository](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL-4B-Instruct-GGUF model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)
- [llama.cpp multimodal support](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- [Google Gemma local deployment guidance](https://gemma.ai/deployment/)

The optional local-VLM wiring is materialized in
`configs/experiments/screw_local_vlm.yaml` and points to an OpenAI-compatible
server at `http://127.0.0.1:8080/v1`. It is intentionally not the default
candidate. A local Qwen3-VL server was installed and benchmarked in rnd1/rnd2;
the quality result remains exploratory and is not a production claim.

These sources establish local runtime availability and model/tool compatibility. They do not establish defect-reporting accuracy for screw inspection; that must be measured on the project's own evidence set.
