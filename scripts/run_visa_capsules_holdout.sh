#!/usr/bin/env bash
# Freeze raw PatchCore as the transfer baseline; do not tune against the public VisA test masks.
# The production runner fits on train/good, calibrates on its held-out GOOD subset, then scores test once.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -eq 1 ]] || { printf 'Usage: PYTHON=/absolute/path/python bash %s <prepared-visa-capsules-root>\n' "$0" >&2; exit 2; }
DATASET_ROOT="$1"
DATASET_NAME="$(basename "$DATASET_ROOT")"
PYTHON="${PYTHON:-}"
CATEGORY="capsules"
PROCESSED_ROOT="${PROCESSED_ROOT:-${TMPDIR:-/tmp}/visa_capsules_processed_320}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${TMPDIR:-/tmp}/visa_capsules_artifacts}"
REPORT_ROOT="$ROOT/reports/experiments/goal_heatmap_external_visa/reproductions"
DINO_CACHE="$ROOT/artifacts/production_highres/mvtec_ad/torch_hub"
ENCODER_ARTIFACT_ROOT="$ARTIFACT_ROOT/$DATASET_NAME"
HUB_DIR="$ENCODER_ARTIFACT_ROOT/torch_hub"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
has_file() { [[ -n "$(find "$1" -type f -print -quit)" ]]; }
sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
count_images() {
    find "$1" -type f -print | awk 'tolower($0) ~ /\.(bmp|jpeg|jpg|png|tif|tiff|webp)$/ { count++ } END { print count + 0 }'
}

[[ -n "$PYTHON" ]] || fail "Set PYTHON to the isolated environment executable (for example /tmp/ivision_goal_env/bin/python); DAenv is not selected automatically."
[[ "$PYTHON" = /* ]] || fail "PYTHON must be an absolute path to the isolated environment executable: $PYTHON"
[[ -x "$PYTHON" ]] || fail "Python executable not found: $PYTHON"
[[ -d "$DATASET_ROOT/$CATEGORY/train/good" ]] || fail "Expected MVTec-style VisA capsules data under $DATASET_ROOT/$CATEGORY/{train/good,test,ground_truth}."
[[ -d "$DATASET_ROOT/$CATEGORY/test/good" ]] || fail "Missing test/good under $DATASET_ROOT/$CATEGORY."
[[ -d "$DATASET_ROOT/$CATEGORY/test/bad" ]] || fail "Missing test/bad under $DATASET_ROOT/$CATEGORY."
[[ -d "$DATASET_ROOT/$CATEGORY/ground_truth" ]] || fail "Missing ground_truth under $DATASET_ROOT/$CATEGORY."
[[ -d "$DATASET_ROOT/$CATEGORY/ground_truth/bad" ]] || fail "Missing ground_truth/bad under $DATASET_ROOT/$CATEGORY."
has_file "$DATASET_ROOT/$CATEGORY/train/good" || fail "No train/good images found."
has_file "$DATASET_ROOT/$CATEGORY/test/good" || fail "No test/good images found."
[[ "$(count_images "$DATASET_ROOT/$CATEGORY/train/good")" -eq 542 ]] || fail "Expected 542 train/good images in the verified VisA capsules archive."
[[ "$(count_images "$DATASET_ROOT/$CATEGORY/test/good")" -eq 60 ]] || fail "Expected 60 test/good images in the verified VisA capsules archive."

defect_dirs=0
bad_images=0
bad_masks=0
for test_dir in "$DATASET_ROOT/$CATEGORY/test"/*; do
    [[ -d "$test_dir" && "$(basename "$test_dir")" != good ]] || continue
    defect="$(basename "$test_dir")"
    [[ -d "$DATASET_ROOT/$CATEGORY/ground_truth/$defect" ]] || fail "Missing ground_truth/$defect."
    has_file "$test_dir" || fail "No test images in $defect."
    has_file "$DATASET_ROOT/$CATEGORY/ground_truth/$defect" || fail "No masks in ground_truth/$defect."
    images_in_class="$(count_images "$test_dir")"
    masks_in_class="$(count_images "$DATASET_ROOT/$CATEGORY/ground_truth/$defect")"
    bad_images=$((bad_images + images_in_class))
    bad_masks=$((bad_masks + masks_in_class))
    defect_dirs=$((defect_dirs + 1))
done
[[ "$defect_dirs" -eq 1 ]] || fail "Expected exactly one VisA capsules defect class, found $defect_dirs."
[[ "$bad_images" -eq 100 ]] || fail "Expected 100 test/bad images, found $bad_images."
[[ "$bad_masks" -eq 100 ]] || fail "Expected 100 ground-truth masks, found $bad_masks."

for entry in checkpoints facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8; do
    [[ -e "$DINO_CACHE/$entry" ]] || fail "Missing cached DINOv2 asset: $DINO_CACHE/$entry"
done
[[ -f "$DINO_CACHE/checkpoints/dinov2_vitb14_pretrain.pth" ]] || fail "Missing cached DINOv2 ViT-B/14 weights."

# run_production appends DATASET_NAME to --artifact-root; SharedDINOv2 reads ENCODER_ARTIFACT_ROOT/torch_hub.
# Reuse the pinned local DINOv2 cache while keeping this experiment's detector artifacts isolated.
mkdir -p "$HUB_DIR"
for entry in checkpoints facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8; do
    if [[ -L "$HUB_DIR/$entry" ]]; then
        [[ "$(readlink "$HUB_DIR/$entry")" == "$DINO_CACHE/$entry" ]] || fail "Unexpected cache link: $HUB_DIR/$entry"
    elif [[ -e "$HUB_DIR/$entry" ]]; then
        fail "Refusing to reuse unverified DINOv2 cache entry: $HUB_DIR/$entry"
    else
        ln -s "$DINO_CACHE/$entry" "$HUB_DIR/$entry"
    fi
done

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_root="$REPORT_ROOT/$run_id"
mkdir -p "$run_root"
project_commit="$(git -C "$ROOT" rev-parse HEAD)"
runner_sha="$(sha256 "$ROOT/scripts/run_visa_capsules_holdout.sh")"
entrypoint_sha="$(sha256 "$ROOT/scripts/run_production.py")"
pipeline_sha="$(sha256 "$ROOT/src/mvtec_system/production/pipeline.py")"
checkpoint_sha="$(sha256 "$DINO_CACHE/checkpoints/dinov2_vitb14_pretrain.pth")"
methods=(none dino_affinity heatmap_open5)
report_dirs=()
manifest="$ENCODER_ARTIFACT_ROOT/classification_split.csv"
expected_manifest_sha=""

for refiner in "${methods[@]}"; do
    run_report_root="$run_root/$refiner"
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$ROOT/scripts/run_production.py" \
        --dataset-root "$DATASET_ROOT" \
        --processed-root "$PROCESSED_ROOT" \
        --report-root "$run_report_root" \
        --artifact-root "$ARTIFACT_ROOT" \
        --image-size 320 \
        --validation-fraction 0.30 \
        --seed 42 \
        --batch-size 8 \
        --detector patchcore \
        --orchestration independent \
        --layers 5,8,11 \
        --coreset-ratio 0.10 \
        --coreset-max-points 50000 \
        --heatmap-refiner "$refiner" \
        --mask-postprocess raw

    report_category="$run_report_root/$DATASET_NAME"
    report_dir="$(find "$report_category" -mindepth 1 -maxdepth 1 -type d -print -quit)"
    [[ -n "$report_dir" ]] || fail "Production report directory was not created under $report_category."
    report_dirs+=("$report_dir")

    current_manifest_sha="$(sha256 "$manifest")"
    if [[ -n "$expected_manifest_sha" && "$current_manifest_sha" != "$expected_manifest_sha" ]]; then
        fail "Split manifest changed between methods; reports are not comparable."
    fi
    expected_manifest_sha="$current_manifest_sha"
done

audit="$run_root/exact_hash_split_audit.json"
"$PYTHON" "$ROOT/scripts/audit_decoded_pixel_split_duplicates.py" \
    --processed-root "$PROCESSED_ROOT" \
    --manifest "$manifest" \
    --output "$audit"
"$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); n=d["cross_split_identical_pixel_pair_count"]; print(f"Exact cross-split pixel duplicates: {n}"); sys.exit(n != 0)' "$audit" || fail "Exact decoded-pixel duplicates cross the train/validation/test split; inspect $audit."

comparison="$run_root/comparison.csv"
"$PYTHON" - "$comparison" "${methods[@]}" -- "${report_dirs[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

separator = sys.argv.index("--")
output = Path(sys.argv[1])
methods = sys.argv[2:separator]
report_dirs = sys.argv[separator + 1 :]
fields = [
    "method", "test_images", "pixel_auroc", "pixel_ap", "precision", "recall",
    "f1", "dice", "iou", "map_coverage", "mask_coverage", "report_dir",
]
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for method, report_dir in zip(methods, report_dirs):
        summary = json.loads((Path(report_dir) / "final_summary.json").read_text(encoding="utf-8"))
        anomaly = summary["anomaly"]
        detector = anomaly["detectors"]["patchcore"]
        ranking = detector["ranking"]
        thresholded = detector["thresholded"]
        writer.writerow({
            "method": method,
            "test_images": anomaly["test_images"],
            "pixel_auroc": ranking["auroc"],
            "pixel_ap": ranking["average_precision"],
            "precision": thresholded["precision"],
            "recall": thresholded["recall"],
            "f1": thresholded["f1"],
            "dice": thresholded["dice"],
            "iou": thresholded["iou"],
            "map_coverage": detector["map_coverage"],
            "mask_coverage": detector["mask_coverage"],
            "report_dir": report_dir,
        })
PY

for index in "${!methods[@]}"; do
    report_dir="${report_dirs[$index]}"
    {
        printf 'method=%s\n' "${methods[$index]}"
        printf 'run_id=%s\n' "$run_id"
        printf 'project_commit=%s\n' "$project_commit"
        printf 'runner_sha256=%s\n' "$runner_sha"
        printf 'production_entrypoint_sha256=%s\n' "$entrypoint_sha"
        printf 'pipeline_sha256=%s\n' "$pipeline_sha"
        printf 'dinov2_checkpoint_sha256=%s\n' "$checkpoint_sha"
        printf 'split_manifest_sha256=%s\n' "$expected_manifest_sha"
        printf 'validation_target_acceptance_rate=0.99\n'
        printf 'exact_hash_audit=%s\n' "$audit"
    } > "$report_dir/run_provenance.txt"
done

printf 'Reports:\n'
printf '  %s\n' "${report_dirs[@]}"
printf 'Comparison: %s\nExact-hash audit: %s\n' "$comparison" "$audit"
