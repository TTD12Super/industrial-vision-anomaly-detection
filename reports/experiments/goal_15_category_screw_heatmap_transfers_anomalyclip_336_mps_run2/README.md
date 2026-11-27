# AnomalyCLIP relation-map benchmark

This is an isolated experiment; it does not change the production pipeline or classifier.

- Categories: all 15 MVTec categories
- Inference image size: 336
- Runtime versions: `{'python': '3.13.7', 'torch': '2.15.0', 'torchvision': '0.29.2', 'numpy': '2.5.5', 'scipy': '1.18.1', 'scikit-learn': '1.9.1', 'Pillow': '13.0.0', 'matplotlib': '3.11.2'}`
- Features: `[6, 12, 18, 24]`; DPAM layer: `20`; sigma: `4.0`
- Relation: `M=A@F`, patch L2 distance to a category prototype
- Fusion: `fuse_maps`, per-image normalization, beta=1.0
- Prototype fit: manifest `train` GOOD only
- Threshold calibration: manifest `validation` GOOD only, q=0.99
- Evaluation: official MVTec `test` only; metric size `320`
- CLIP cache: temporary local cache (not included in the repository)

- Resolved device: `mps`
- CLIP backbone: ViT-L/14 336px; SHA-256: `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`
- Pixel metric memory preflight: estimated `4336386048` bytes for the largest category ranking (17100800 pixels at 128 bytes/pixel plus 2 GiB runtime headroom); `7328546816` bytes available.

## Resolution fidelity

The completed local benchmark uses a 336px AnomalyCLIP input and resizes its maps to the shared 320×320 metric grid. The upstream MVTec test command uses `--image_size 518` and `--features_list 24` ([official test script](https://github.com/zqhang/AnomalyCLIP/blob/main/test.sh)); therefore this is a 336px transfer ablation, not a reproduction of the upstream-resolution evaluation. Attempts to score the 518px run on local MPS did not finish and produced no metrics; see the retained [518px run status](../goal_15_category_screw_heatmap_transfers_anomalyclip_518_mps_rerun5/RUN_STATUS.md). The effect of running AnomalyCLIP at 518px remains unmeasured.

Thresholds use validation maps resized to the metric grid. The sanitized [summary_public.json](summary_public.json) stores per-category/method validation counts, q=.99, linear-quantile order statistics (zero-based indices), and interpolation fractions without local machine paths.

## Localization scope

This is an **oracle-category localization ablation**: each test image uses its ground-truth category to select that category's GOOD prototype. It isolates heatmap quality and is not an end-to-end routed result; classification accuracy is evaluated separately.

## Frozen outputs

`anomalyclip_raw` is the upstream AnomalyCLIP map. `relation_map` is evaluated separately. `fusion` is only the fixed map fusion; it does not mix detector rankings with another detector's mask.

## Pixel AP interpretation

**Macro Pixel AP is the primary comparison metric**: it is the unweighted mean of the 15 category Pixel AP values, so each category contributes equally. Per-category Pixel AP/AUROC are computed independently. Pooled micro Pixel AP/AUROC are intentionally unavailable: ranking every test pixel together requires a large all-test allocation and raw score scales vary by category. The micro row still reports exact pooled thresholded confusion-count metrics after applying each category's own GOOD-validation threshold.

## Local-only score artifacts

The full `image_index.csv`, raw float32 score-map arrays, label map, and sample heatmap gallery remain local and are not committed. The repository version contains aggregate per-category metrics and the Pixel AP comparison chart; map coverage is `map_images / image_count`, and mask coverage is `mask_images / defect_image_count`.

## Output directory reservation

The run reserves `output_dir` with an exclusive directory creation before reading the manifest or loading the model. If a run fails, its directory is retained and may be empty or contain partial artifacts; inspect it before manually removing it. Reuse of any existing directory is rejected, so start a rerun with a new `--output-dir`.

## Provenance and caveats

- Manifest SHA-256: `fac2c931624430c83e1a96f2122691cd5ad319c5f984b43c0bccf5ac7029d432`
- Checkpoint SHA-256: `94ce202da3e6486a864b904fdfed5057de75846c5834e446fd1d2fe7f97acb44`
- Source SHA-256: `397c413633d0c683ae81cc767bb8fce85209afd1e2414526d0b9044ff68981ce`
- Experiment script SHA-256: `084fda9072e3a44ed168b3e3805e2df646500d726fe20dc5fc8f3214528da017`
- Project revision: `30cb69ef2f50fe77b8d7c681b92c73c9d46885a7`; tracked worktree dirty at capture: `True`
- No exact byte-hash overlap was found for train-vs-validation, train-vs-test, or validation-vs-test; perceptual duplicates are not ruled out.
- Processed split digests: {'train': '2fa60d9554d1e9540550baa06c64ca016a90507167781645436a6bb4240cb493', 'validation': 'bf326ff8f0128541500f149fc88abbe0a96976d5b9beb63d0eb43cfd6540da05', 'test': '0d8a5a58b20d36a7b1151549db2fb1c53cf8e60cb7a17997634d746c2d481c8c'}
- Test mask count/digest: 1258 / 03c7988ed3a0524e1ab762f05dae934c6f88677975e907a10c626e28bcc3fb78
- Upstream citation: the [official README](https://github.com/zqhang/AnomalyCLIP) and [test script](https://github.com/zqhang/AnomalyCLIP/blob/main/test.sh) identify the VisA-trained checkpoint used for MVTec evaluation.
- MVTec AD's official test split is public and has been exposed in prior research; these results are exploratory benchmark evidence, not a private production holdout.
- Test-mask files are read once for exact-byte split auditing before fit; pixel-mask values are opened for scoring only after prototypes and GOOD-only thresholds are fixed, and are never used for fitting or tuning.
- Pixel metrics resize the 336px output map to the processed 320px metric grid.
- The requested checkpoint's upstream training provenance is recorded as supplied; this script does not infer or rewrite its training dataset.
