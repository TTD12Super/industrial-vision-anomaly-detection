# Original Dinomaly ViTill-Small: 15-category feasibility pilot

This isolated experiment evaluates the **original CVPR 2025 Dinomaly ViTill-Small/14** implementation. It is not Dinomaly2 and does not alter production classification or routing. The pilot compares its localization maps with the project's **category-specific raw PatchCore** baseline.

## Leakage and dataset identity

- Split manifest: `artifacts/production_highres/mvtec_ad/classification_split.csv`, SHA-256 `fac2c931624430c83e1a96f2122691cd5ad319c5f984b43c0bccf5ac7029d432`.
- Saved decoded-pixel audit: `reports/experiments/goal_15_category_heatmap_methods_320/decoded_pixel_split_audit.json`; audit file identity and expected hashes are pinned in the runner.
- Expected full dataset inventory SHA-256: `76ecec7e1a11c95e244f345e7fe4c542cc9778fec5fa839e1f709229cc9df6eb`.
- Expected official-test inventory SHA-256: `a88785e62f357832c19188baddb3df95ca252e1257dc6050c26f728c23ac32be`.
- Expected train+validation decoded-pixel inventory SHA-256, checked before fitting: `ec86b00363eabe9ae981ef39015a92a00207c99c02ec4bcff11ce728c7007c63`.
- Expected counts: 5,354 images total: 2,540 train/good, 1,089 validation/good, and 1,725 official test images; the audit reports no exact cross-split pixel duplicates.
- The runner checks the saved audit identity and hashes actual train/validation decoded RGBA content against the pinned train+validation digest before fitting. It does not enumerate or decode official-test images until both models/checkpoints are saved and validation calibration is complete. Then it checks the live full-dataset and test inventory digests against the saved audit **before model inference**. Test labels and masks are attached only after both map sets have been generated.
- The official MVTec test split has been viewed in earlier project work. Every result is exploratory and is not an untouched-holdout estimate.

## Fixed 15-category router

This experiment does not retrain or change the product classifier. It uses the saved frozen-DINOv2 global-embedding plus multinomial-logistic-regression classifier to route PatchCore to one of 15 category banks. The existing MVTec test report records 99.826% top-1 accuracy over 1,725 images, counting all three rejected/UNKNOWN predictions as incorrect. That test was previously inspected, so this exceeds the numeric 90% goal on MVTec but is not evidence of 90% performance on unseen cameras, products, or an untouched holdout. See the [classification summary](../goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/classification_test_summary.json).

## Candidate and training schedule

The adapter uses `ViTill` from the original [Dinomaly repository](https://github.com/guojiajeremy/Dinomaly/tree/1f252be03a918789b19848f0ca37166e7c28dada), pinned to commit `1f252be03a918789b19848f0ca37166e7c28dada`. It selects the official runner's commented ViT-Small/14 option, target layers 2–9, the two four-layer fusion groups, one `bMlp(384, 1536, 384, drop=0.2)` bottleneck, and eight decoder blocks. The runner isolates and adapts this architecture because the upstream MVTec runner constructs test datasets before fit, evaluates test during training, and leaves checkpoint saving disabled.

Only the 2,540 train/good manifest rows enter fit. The run uses seed 42, batch size 8, and 5,000 optimizer steps as a **feasibility pilot only**. It cannot be used to reject Dinomaly or claim convergence. The original schedule is retained with its full 10,000-step horizon: StableAdamW at base LR `2e-3`, final LR `2e-4`, 100-step linear warmup, then cosine decay. The pilot stops halfway through that schedule; it does not reach the final LR. The schedule timing follows the pinned original `WarmCosineScheduler` formula.

The spatial protocol preserves the project's existing images and evaluator geometry. Normalized 320×320 inputs are replicate-padded only on bottom/right to 322×322 for the ViT/14 grid. Dinomaly residual maps are bilinear-resized to 322×322 with `align_corners=False` to match the project's interpolation convention, Gaussian-filtered with a 5×5 sigma-4 kernel, then cropped top-left to 320×320. This is an explicit project-matched adaptation: upstream uses `align_corners=True` and its MVTec evaluator resizes masks to 256, neither of which is used here. Both models are scored against the unchanged 320×320 masks.

## PatchCore comparison and calibration

PatchCore uses the project implementation and fits one normal memory bank for each category from that category's train/good images. The saved project classifier routes by raw top-1 prediction. Its confidence rejection does not suppress localization: a rejected item still uses `route.predicted_category`, so both methods must produce all 1,725 maps. Test ground-truth category is never used for routing or threshold selection.

For each method, the primary binary metrics use one global q=.99 threshold pooled from all validation GOOD maps. A secondary diagnostic preserves category-specific q=.99 thresholds, calibrated and selected by the classifier's **predicted** category on validation and test. Pixel AP and AUROC use continuous maps and are threshold-free. Per-category results use ground-truth category only to group predictions after inference.

## Windows RTX 4070 run

Use an isolated Python environment on the Windows GPU host. Choose a current CUDA-enabled PyTorch/TorchVision pair using the [official PyTorch selector](https://pytorch.org/get-started/locally/); do not copy the original repository's stale Python 3.8 / PyTorch 1.12 / CUDA 11.3 pins. The runner records Python, Torch, TorchVision, CUDA runtime, GPU and capability. It defaults to `cuda:0`, validates an explicit device index, and does not set `CUDA_LAUNCH_BLOCKING=1`. Pass the project DINOv2-Small pretrained weights in a local directory; network download is refused. The project DINOv2-B Torch Hub cache and matching classifier artifacts must also be present under the project artifact root.

From the repository root in PowerShell:

```powershell
$common = @(
  "--manifest", "artifacts/production_highres/mvtec_ad/classification_split.csv",
  "--data-root", "datasets/mvtec_ad_processed_320",
  "--decoded-pixel-audit", "reports/experiments/goal_15_category_heatmap_methods_320/decoded_pixel_split_audit.json",
  "--upstream-root", "D:/models/Dinomaly",
  "--dinomaly-weights", "D:/model-cache/dinomaly-weights",
  "--patchcore-artifact-root", "artifacts/production_highres/mvtec_ad",
  "--output-dir", "artifacts/experiments/goal_15_category_dinomaly_pilot/run_seed42_it5000",
  "--device", "cuda:0"
)

python scripts/benchmark_dinomaly_15cat.py @common --cuda-smoke-check
# Continue only when the smoke check passes and writes cuda_smoke_check.json.
python scripts/benchmark_dinomaly_15cat.py @common
```

The smoke command checks original ViTill forward/backward/optimizer, project DINOv2-B encoder, and classifier routing. It reads and hashes train/validation images but does not read test images. The full run requires the same source/data/weights/runtime fingerprint as the smoke marker.

## Outputs and statistical limits

- `dinomaly_vit_small_final.pt`, 15 category-specific `patchcore_category_banks/<category>/coreset.pt` banks, `training_loss.csv`.
- `provenance.json` and `validation_thresholds.json`, including candidate/source hashes, classifier identity, decoded-pixel audit identity and live verification, exact transforms, LR schedule and runtime.
- `score_maps/<category>.npz`, with maps, masks, test labels, predicted route and reject flag.
- `localization_metrics.csv` and `summary.json`: per-category, macro and micro Pixel AP/AUROC, global-threshold metrics, predicted-category-threshold diagnostics, coverage and GOOD/DEFECT predicted-mask area.
- `pixel_metrics_overview.png`: macro Pixel AP/AUROC and macro F1/GOOD false-positive pixel rate at the global validation q=.99 operating point.
- `pixel_ap_by_category.png`: Dinomaly and PatchCore Pixel AP for each of the 15 categories.
- `heatmap_gallery.png`: fixed representative GOOD and DEFECT examples for bottle, cable, screw and zipper; per-image normalization is for display only.
- Charts and the gallery are created only by the full benchmark after test inference; they are not produced by the CUDA smoke check. No paired group-bootstrap confidence intervals are reported because the manifest has no physical-object/acquisition group IDs, so image-level bootstrap intervals would overstate independence. The 5k result is a feasibility check only.
