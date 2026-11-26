# Clean holdout requirements — 15 MVTec categories

## Why the current scores are exploratory

The current 15-category experiments use the official MVTec AD test set. That set informed earlier method and resolution choices, so its scores cannot be treated as an untouched final estimate even though the current manifest and decoded-pixel audit find no exact cross-split duplicates. The exact-pixel audit does not establish absence of perceptual near-duplicates or acquisition-session leakage.

Current reference results:

- Heatmap baseline: raw PatchCore, **61.279% macro Pixel AP**. None of the tested screw-branch transfers improved Pixel AP.
- Product classification: frozen DINOv2 global embeddings with a multinomial logistic-regression head, **99.826% Top-1** on the exposed MVTec test; the three rejected predictions count as errors. This is a numeric pass for the current test only, not an unbiased generalization claim.
- The training embeddings cover 3,629 MVTec `train/good` images across 15 categories. A new split of these features could evaluate product-category recognition on held-out GOOD images, but not recognition of DEFECT images or anomaly localization.

## Proposed acceptance protocol — Project Owner approval pending

### Product classification

- Label every image with one of the 15 product categories, regardless of GOOD/DEFECT status.
- Report all-image Top-1, macro Top-1, per-category accuracy, GOOD/DEFECT breakdown, and reject rate. Count `UNKNOWN` or rejected predictions as incorrect in the primary accuracy.
- Proposed gate: **macro Top-1 above 90%**, with a group-aware 95% confidence interval lower bound above 90%. This confidence-bound requirement is a recommendation, not an approved project rule.

### Heatmap localization

- Use macro Pixel AP across the 15 categories as the primary threshold-free metric. Keep raw PatchCore as the comparator.
- Proposed improvement gate: at least **+1.0 percentage point** macro Pixel AP over raw PatchCore, with the paired group-level 95% confidence interval for the gain above zero and complete map coverage. This gate is a recommendation, not an approved project rule.
- Also report per-category Pixel AP, Pixel AUROC, and operating-point measures such as defect recall and GOOD-image false-positive pixel area.
- Calibrate thresholds on validation data only. Do not train, tune, or select a method using final holdout masks.

### Leakage and reproducibility controls

- Split by physical object and acquisition session. Keep frames from one object, video burst, or capture session in only one split.
- Freeze the final holdout before comparing methods. If its scores guide a code, threshold, or model change, it becomes development data and a new holdout is needed.
- Preserve original images and masks, split manifest, group IDs, source and capture metadata, source revision, model/config hashes, validation thresholds, per-image predictions, and heatmaps.
- Compute exact byte and decoded-pixel hashes. Use a reviewed perceptual-near-duplicate process and group/session checks; exact hashes alone are insufficient.

## Data needed to close the 15-category claim

Preferred source: a new production-camera or independently collected set covering all 15 labels: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, and `zipper`.

Proposed collection floor for planning, not a statistically proven minimum:

- 100 images per category: at least 50 GOOD and 50 DEFECT, from at least 10 independent object/session groups per category.
- Original-resolution image for every sample; a pixel-aligned defect mask for each DEFECT image. GOOD images need an explicit GOOD label and an empty mask or equivalent unambiguous annotation.
- Manifest columns: `sample_id`, image path, category, GOOD/DEFECT label, mask path, physical object/group ID, acquisition-session ID, source, camera, lighting, and capture date.
- Keep augmented copies out of the holdout. If unknown-product rejection is an acceptance target, provide separate non-target images and set an unknown false-accept requirement.
- If acceptance must include multi-product frames and detector crops, include the original frames and per-instance ground-truth boxes as an additional evaluation track.

More samples or groups may be required if the group-aware confidence intervals are too wide. The collection floor does not guarantee that the acceptance gates will be met.

## Public dataset options found so far

| Dataset | Verified relevant content | Appropriate use and limitation |
|---|---|---|
| [VisA](https://github.com/amazon-science/spot-diff) | Twelve object subsets, including `capsules`; image labels and pixel annotations for anomalies; CC BY 4.0. | Useful as an independent cross-dataset check for the capsule family, subject to a source/hash audit. It covers only a subset of the target labels, so it cannot validate 15-category accuracy or 15-category macro Pixel AP. |
| [Real-IAD](https://realiad4ad.github.io/Real-IAD/) | Thirty object categories, five views per sample, anomaly masks, and an approximately 53 GB 1024×1024 archive. Its object list includes toothbrush, transistor1, and zipper families. Defects are deliberately manufactured during dataset preparation. | Strong cross-dataset track for a few mapped product families, after reviewing the exact class mapping and keeping all views of each item in one split. It is not a full 15-category holdout and its manufactured anomalies do not establish performance on naturally occurring factory rejects. Research-only; the dataset card states CC BY-NC-SA 4.0. |
| [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) | Eight new scenarios with more than 8,000 high-resolution images; pixel masks are public for `TESTpub`, while larger private test labels are evaluated through the official server. | Independent anomaly-localization stress test under new scenarios and lighting, not a product-category test for the same 15 labels. |
| [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) | Five categories with structural and logical anomalies and pixel-precise annotations. | Useful as a separate logical-anomaly evaluation. Its scenes and task differ from the 15 MVTec AD categories, so report it separately. |
| [KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/) | 356 defective and 2,979 defect-free images of one industrial surface/product, with several surface-defect types. | Can probe binary GOOD/DEFECT transfer on a different product. It cannot support the 15-way product classifier; confirm annotation granularity before using it for pixel-level heatmap metrics. CC BY-NC-SA 4.0. |
| [ADNet](https://grainnet.github.io/ADNet.html) | A unified benchmark of 380 categories aggregated from 49 public datasets, with MVTec-style masks and text annotations. | Potential broad transfer track, not a turnkey clean holdout for this project: audit the released source lineage and exact image/content overlap before use. It does not itself provide a fresh, verified holdout for these exact 15 labels. |
| [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | All 15 target categories and pixel masks. | Retain as the reproducible in-domain benchmark, but label scores exploratory because its official test set has informed prior method and resolution choices. |

No reviewed public source has been verified to provide a fresh, independent holdout covering all 15 MVTec categories with defect masks. The practical public-data option is a **partial cross-dataset track** using VisA and selected Real-IAD product families; keep its per-dataset and per-category results separate from the MVTec 15-category score. For the full acceptance claim, collect new images from the intended camera/workflow, then freeze object/session-disjoint development and sealed test partitions before any tuning.

## Current blocker and user choice

The project contains cached DINOv2 embeddings for the current MVTec train/validation split. They are serialized as `.pt` and require Torch to repartition/load. The local `DAenv` import is stalled by iCloud-offloaded package files, and no environment hydration or replacement has been done. A clean GOOD-only classifier split is therefore not run yet; even if run, it would not validate DEFECT classification or heatmaps.

The Project Owner has been asked whether to provide a new 15-category holdout, use a partial public cross-dataset test, or keep the current numbers exploratory until new data are available.
