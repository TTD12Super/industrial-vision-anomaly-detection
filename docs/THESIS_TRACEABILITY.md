# Thesis Requirements-to-Evidence Traceability

This matrix keeps the image-first screw inspection vertical slice defensible as
an IT graduation project. A “development” status means the implementation and
local evidence exist; it does not mean production readiness.

| ID | Requirement / use case | Implementation evidence | Acceptance evidence | Status and limitation |
|---|---|---|---|---|
| FR-01 | Validate a product dataset before training or inspection | `inspection.cli validate-dataset`, product YAMLs | validation command and dataset summary | Implemented; target-camera data still required |
| FR-02 | Detect screws in a multi-object image and crop each object | YOLO-OBB preparation/inference scripts | detector benchmark JSON, coverage, crop visual montage and latency chart | Development; WEEE/MVTec geometry shift remains open |
| FR-03 | Reject an unknown product instead of silently routing it to a screw bank | DINOv2 product recognition and `UNKNOWN_PRODUCT` disposition | known/unknown static smoke test | Prototype; multi-product target-camera gallery is incomplete |
| FR-04 | Classify each known screw crop as GOOD or DEFECT | DINOv2 patch-memory Exact kNN candidate and validation-only policy ledger | `screw-patch-memory-rnd3`, selected policy, threshold curve, and locked CAD-SD/MVTec test report | Cross-dataset evidence exists (CAD-SD defect recall 90.96%, GOOD-FP 10.48%); independent target-camera end-to-end score is still missing |
| FR-05 | Prefer high DEFECT recall while bounding GOOD false positives | `src/inspection/classifier_sweep.py`, `src/inspection/sweep.py` | gate `recall >= 0.90`, `GOOD-FP <= 0.20`, numerator/denominator contract | Implemented; legacy report counts are reconstructed until trainer rerun |
| FR-06 | Produce a defect-localization heatmap and binary evidence mask | AnomalyCLIP, supervised refiner, spatial decoder and morphology sweep | Pixel AP/precision/recall/Dice/IoU tables and qualitative montage | Development; high recall but localization precision is weak |
| FR-07 | Preserve evidence for human review | JSON result, crop, heatmap, mask and overview artifact paths | inspection API contract and report manifest | Implemented for static-image pilot; auth/tenant isolation absent |
| FR-08A | Queue/store a VLM request for eligible DEFECT evidence | VLM boundary, mock/cache reporter and local OpenAI-compatible adapter | mock/local adapter contract, local Qwen3-VL smoke and artifact references | Implemented for the static-image pilot; API remains asynchronous by configuration |
| FR-08B | Produce an evidence-grounded natural-language defect explanation | Real VLM provider plus crop/heatmap/mask evidence | five-object Qwen3-VL audit with latency and OBB-IoU joins | Exploratory only: 5/5 requests completed, but 0/2 DEFECT samples received a visible defect type |
| NFR-01 | Configuration-driven reproducibility | YAML configs, versioned artifacts, SHA-256 lineage | report manifest, sweep ledger, split hashes | Implemented for saved experiments |
| NFR-02 | Avoid leakage between train, validation and locked evidence | split-role guards, holdout lock, and no-tuning evaluator | unit tests and ledger manifest | Implemented at protocol level; independent labelled physical holdout still missing |
| NFR-03 | Keep runtime safe when a gate is not met | `not_passed`, no-pass handling, ungated fallback marker | sweep no-pass tests and trainer metadata | Implemented; fallback is exploratory and must not be promoted |
| NFR-04 | Provide report-ready engineering and research evidence | architecture/sequence diagrams, benchmark charts, qualitative figures | `docs/THESIS_REPORT_PACK.md` and figure manifest | Implemented for image-first scope |

## Use cases

| ID | Actor | Preconditions | Main flow | Alternate / failure flow | Linked requirements | Status |
|---|---|---|---|---|---|---|
| UC-01 | Operator | Static image and product configuration are available | Submit image → detect objects → crop → recognize product → classify quality → produce evidence artifacts | No detection, invalid crop or model failure becomes an explicit failed state | FR-01–FR-07 | Partial: component path exists; full acceptance run remains to be sealed |
| UC-02 | Operator | Product is outside the configured screw gallery | Detect/crop object → product recognizer compares against GOOD gallery → return `UNKNOWN_PRODUCT` | Unknown object is not routed to screw anomaly model | FR-02, FR-03, NFR-03 | Development smoke evidence exists |
| UC-03 | Operator / reviewer | Object is recognized as a known screw | DINOv2 applies the validation-selected threshold → return GOOD/DEFECT → retain score and provenance | Ungated or missing calibration must not become a production decision | FR-04, FR-05, NFR-03 | Development gate passes; independent labelled holdout missing |
| UC-04 | Reviewer | Object is classified or flagged for review | Open crop, heatmap, binary evidence and overview; optionally request VLM explanation | Heatmap/VLM cannot override the classifier; unavailable VLM remains an explicit pending/failed state | FR-06–FR-08, FR-07 | Evidence artifacts exist; real VLM quality not evaluated |
| UC-05 | ML engineer | Dataset/config/model artifacts are versioned | Validate dataset → train/calibrate → sweep validation candidates → lock ledger → render benchmark figures | Gate failure produces `not_passed`, not a silently promoted threshold | FR-01, FR-05, NFR-01–NFR-04 | Implemented for saved development experiments |

## Acceptance tests

| ID | Fixture / command | Expected result | Evidence | Status |
|---|---|---|---|---|
| AT-IMG-001 | Static image under `runs/screw-open-set-candidate-rnd1/` | Detector/crops and serialized inspection evidence are produced | overview, crop status, heatmap artifacts and report pack montage | Partial: smoke/development evidence, not a sealed end-to-end acceptance run |
| AT-IMG-MULTI-001 | WEEE COCO test image `pc_13_663.png` with two annotated objects | Multiple detector candidates are independently routed and unknown candidates are not classified | `runs/weee_two_object_vlm_smoke_20261125/weee_two_object_vlm_smoke_20261125/inspection.json` | Safety path passed: 4 candidates were rejected as `UNKNOWN_PRODUCT`; detector recall/precision failed as a generalization check and needs a target-camera detector holdout |
| AT-IMG-MULTI-002 | WEEE COCO test image `screws_22.png` with 13 annotated objects | The deterministic flow preserves per-object crop hashes and does not route untrusted candidates into quality classification | `runs/weee_multi_object_deterministic_smoke_20261125/weee_multi_object_deterministic_smoke_20261125/inspection.json` | Safety path passed: 4 candidates were rejected as `UNKNOWN_PRODUCT` in 0.992 s; coverage failed, so this remains a detector/gallery blocker |
| AT-DETECT-ADAPT-001 | WEEE locked 90-image test with `weee_adapt10` | A validation-selected detector challenger is compared under the same hashed split and latency protocol | `reports/experiments/weee_yolo_adapt10_rtdetr_fair_aabb_holdout_rnd1.json` | YOLO-OBB reached 66.51% recall / 66.03% precision / 26.7 ms; RT-DETR reached 73.92% / 42.27% / 215.7 ms. WEEE objects are often only 7–12 px in the full scene, so this does not validate anomaly quality |
| AT-E2E-CROP-001 | Current static multi-object fixture with detector-generated crops | Count detector misses and unmatched predictions, then measure routing, GOOD/DEFECT, heatmap and artifact continuity per object | Active candidate: `reports/experiments/screw_e2e_active_refiner_rnd4_20251125.json` plus per-image artifacts under `runs/e2e_active_refiner_benchmark_20251125/` | Passed as a development run over all 78 test images, but release gate fails: end-to-end DEFECT recall 65.22% and GOOD false-alarm rate 67.27%; current split is not an independent target-camera holdout |
| AT-DETECT-FAIR-001 | `scripts/evaluate_weee_geometry_fair.py` on the locked WEEE split | YOLO-OBB and RT-DETR are compared with the same source split, confidence grid and projected-AABB crop rule; per-object crop manifests include matched/missed/unmatched statuses | `reports/experiments/weee_yolo_obb_rtdetr_fair_aabb_holdout_rnd3.json` and detector figure | Passed for supervised-development crop coverage; not direct OBB quality or target-camera evidence |
| AT-ANOM-001 | `scripts/compare_detectors.py` on `mvtec_ad_screw_benchmark` | Exact kNN, PatchCore and PaDiM run independently on the same 160-image test split with per-stage latency and split-identity hashes | [comparison summary](../reports/production/screw_comparison_rnd6/mvtec_ad_screw_benchmark/20261122T201103Z/comparison_summary.json) and anomaly-detector figure | Passed for supervised-development comparison; target-camera holdout missing |
| AT-OPEN-001 | `benchmark_screw_open_set.py` with MVTec unknown object/type annotations | Unknown product types are rejected without silent screw routing | `artifacts/products/screw/screw-recognizer-rnd4/open_set_benchmark.json` | Development pass; not target-camera production evidence |
| AT-CLS-001 | Exact kNN patch-memory artifact with validation-only threshold | Validation gate is evaluated without tuning the locked test scores | [mixed CAD-SD/MVTec report](../reports/experiments/dinov2_patchcore_lite_mixed_top1_allowed03_rnd3.json) and artifact `screw-patch-memory-rnd3/classifier_top1_allowed03.pt` | Cross-dataset classifier evidence passes the recall-first target on CAD-SD; still not full target-camera evidence |
| AT-LEDGER-001 | Focused unit tests covering selector, provenance, no-pass, evaluator binding and matching | Unsupported selection, hash mismatch, threshold mismatch, one-class data and ambiguous matching are rejected | latest focused test output | Passed |
| AT-HEAT-001 | Saved localizer reports and figure renderer | Pixel AP/precision/recall/Dice/IoU and qualitative overlays are rendered | `reports/experiments/thesis_figures_rnd15/` | Passed as development evidence; independent mask holdout missing |
| AT-HOLDOUT-001 | Locked CSV with `image_path,label` and both GOOD/DEFECT | Evaluator verifies ledger threshold/model/manifest binding and writes exact confusion counts | `evaluate_dinov2_classifier_holdout.py` | Contract tested; blocked for independent claim until a valid labelled holdout exists |
| AT-VLM-001 | Mock/cache/local reporter path | VLM request/report boundary preserves classifier result and evidence references | mock/local adapter contract, sync/async reporter tests and per-object report artifacts | Passed as an integration contract; provider quality is separate |
| AT-VLM-E2E-001 | One static screw image with the local Qwen3-VL endpoint | The complete image-first flow finishes and persists classifier, heatmap, mask, provenance and grounded VLM JSON | `runs/e2e_verify_local_vlm_sync_20261125/e2e_verify_local_vlm_sync_20261125/inspection.json` and per-object artifacts | Functional smoke passed; one fixture only, 35.771 s total, and the VLM correctly abstained from naming a defect |
| AT-VLM-PROVIDER-001 | Configured real local Qwen3-VL provider with labelled evidence cases | Evaluate groundedness, unsupported assertions, latency, token cost and human agreement without changing the classifier label | `reports/experiments/screw_local_vlm_benchmark_rnd2.json` and `_audit.json` | Exploratory provider run: 5/5 completed, mean 17.923 s/object; too small for accuracy or production readiness |

## Additional non-functional requirements

| ID | Quality attribute | Evidence / measurement | Status |
|---|---|---|---|
| NFR-05 | Static-pipeline latency and memory are measurable per stage | Integrated replay report contains image/object latency summaries and per-object stage fields; local Qwen3-VL benchmark records request latency | Development; independent target-camera latency and memory envelope still missing |
| NFR-06 | One object failure must not erase other object results | Production result schema has per-object statuses; consolidated failure-injection acceptance record is missing | Partial |
| NFR-07 | JSON/image/PDF outputs remain consistent | Artifact paths and report manifest exist; one consolidated schema-consistency acceptance record is missing | Partial |
| NFR-08 | Artifact and model provenance are immutable and hash-pinned | Sweep ledger and report manifest store SHA-256 values; production promotion/rollback is not implemented | Development |
| NFR-09 | VLM data handling is explicit | Local endpoint uses image bytes only on loopback; cloud provider/key/retention remain unconfigured | Development; local deployment policy still needs review |
| NFR-10 | A second product can be onboarded through configuration | Product YAML boundary exists; second-product end-to-end acceptance record is missing | Not evaluated |
| NFR-11 | Runtime environment is reproducible | `scripts/prepare_dinov2_runtime.py`, pinned DINO revision, checkpoint SHA-256 and runtime-preparation unit test | Development; final sealed Python/environment lock and independent machine replay remain |

## Acceptance gate for the thesis demo

The demo is accepted as a static-image research prototype only when the
operator can submit one image, see detector boxes/crops, receive a GOOD/DEFECT
decision per crop, inspect heatmap/mask evidence, and retrieve the serialized
artifacts. A production claim additionally requires an independent target-camera
holdout, exact trainer confusion counts, and a configured VLM provider.

## Scope boundary

The current thesis slice is intentionally image-first. Video ingestion,
authentication, tenant isolation, cloud VLM deployment, and production model
promotion are future work rather than hidden acceptance assumptions.

## Independent classifier holdout contract

Use [`scripts/evaluate_dinov2_classifier_holdout.py`](../scripts/evaluate_dinov2_classifier_holdout.py)
with a locked CSV containing `image_path,label`, where `label` contains both
`GOOD` and `DEFECT`. Supply the threshold selected by the validation ledger;
also pass the ledger manifest with `--selection-ledger`. The evaluator verifies
the threshold, model hash and manifest hash against that ledger, rejects
one-class data and never tunes the threshold. The current
`datasets/external/mvtec_screws/extracted/mvtec_screws_test.json` contains
object/type annotations for unknown-product recognition, not GOOD/DEFECT labels,
so it is not a valid classifier holdout.
