# Static sweep ledger protocol

`src/inspection/sweep.py` is a selector and immutable ledger, not a model
runner. It records metrics and artifacts already produced by an experiment; it
does not train, infer, tune a model, or independently audit the source report.
Its checks are protocol guards only. A valid ledger therefore does not prove
that the reported metric provenance is correct.

The DINOv2 classifier trainer now performs inference once per split, evaluates
every validation score threshold, and writes `threshold_sweep.json`.
`scripts/record_classifier_sweep.py` converts that saved result into this
guarded ledger. The ledger remains model-free by design: the trainer is the
execution step and the ledger is the reproducibility gate.

## Split and lock rules

Every spec must set `independent_holdout_locked: true` and declare all four
roles. Each role is an object with an existing manifest and its expected
SHA-256. The loader verifies the declared hash; it does not replace a bad
expected hash with a newly computed value:

- `train`: data used to fit a model, gallery, or memory bank;
- `validation`: the only split allowed to select candidates and thresholds;
- `diagnostic`: development evidence that is never a selection source;
- `independent_holdout`: locked final evidence, never a selection source.

When a legacy report already calls its saved test evaluation “diagnostic”, the
ledger may use the `diagnostic` role only if lineage records
`diagnostic_alias_of: "test"`, `diagnostic_alias_manifest` and its
`diagnostic_alias_sha256`. The loader verifies that the alias path/hash matches
the locked diagnostic manifest. This is an explicit alias for provenance, not a
new sample and must not be counted as independent validation evidence.

The loader recomputes every split hash and rejects a missing or changed file.
`candidate_names` may name validation rows only. `lineage.config_path` and
`lineage.config_sha256` are required and pinned in the same way. Every
candidate must also set `holdout_used_for_selection: false`, provide a
non-empty list of existing `artifact_paths`, and provide an existing
`source_report_path` plus a mandatory `source_report_sha256`. Every artifact
path must have a matching expected hash in `artifact_sha256`; missing, extra,
or changed artifacts are rejected. The manifest records both expected and
actual split, config, artifact, and source-report hashes.

## Metric contract and selection gate

Each candidate declares `metric_unit` as one of `image`, `object`, `crop`,
`pixel`, or `session`. It must provide positive
`metric_denominators.defect_count` and `good_count`, plus
`metric_numerators.defect_true_positive` and
`good_false_alarm_count`. These four count fields are strict JSON integers:
booleans, fractions, negative values, and weighted aggregates are rejected.
The ledger checks, within a `1e-3` rounding tolerance:

```text
defect_recall = defect_true_positive / defect_count
good_false_positive_rate = good_false_alarm_count / good_count
```

Metrics must be finite and in `[0, 1]` except explicitly non-rate values such
as `latency_ms` and `memory_mb`, which must still be finite and non-negative.
The numerator and denominator are not invented or inferred by the ledger.

The selection section must contain both gates:

```json
"selection": {
  "target_defect_recall": 0.90,
  "good_false_positive_cap": 0.20,
  "tie_break": [
    "lower_good_false_positive_rate",
    "lower_latency_ms",
    "lower_name"
  ]
}
```

A validation candidate is eligible only when:

```text
defect_recall >= target_defect_recall
AND good_false_positive_rate <= good_false_positive_cap
```

If no candidate passes both gates, the result is `status: "not_passed"` and
`selected_candidate: null`; the selector never chooses the best candidate
under only one gate. The tie-break list is strict: an unsupported rule is
rejected rather than silently ignored. Threshold sweeps may explicitly use
the existing classifier operating-point policy:

```json
"tie_break": [
  "maximize_defect_recall",
  "lower_good_false_positive_rate",
  "higher_threshold"
]
```

That rule maximizes recall among candidates that pass both gates, then lowers
GOOD false positives, then chooses the higher threshold. Every candidate must
then provide `metrics.threshold`. Diagnostic and holdout rows remain in the
ledger but cannot be selected.

## Example spec shape

```json
{
  "schema_version": 1,
  "experiment": "classifier_thresholds",
  "independent_holdout_locked": true,
  "split_roles": {
    "train": {"manifest": "manifests/train.json", "sha256": "..."},
    "validation": {"manifest": "manifests/validation.json", "sha256": "..."},
    "diagnostic": {"manifest": "manifests/diagnostic.json", "sha256": "..."},
    "independent_holdout": {"manifest": "manifests/holdout.json", "sha256": "..."}
  },
  "selection": {
    "target_defect_recall": 0.90,
    "good_false_positive_cap": 0.20,
    "tie_break": [
      "lower_good_false_positive_rate",
      "lower_latency_ms",
      "lower_name"
    ]
  },
  "lineage": {
    "config_path": "configs/products/screw.yaml",
    "config_sha256": "...",
    "seed": 7,
    "device": "mps"
  },
  "candidates": [
    {
      "name": "dino_threshold_042",
      "method": "dinov2_logreg",
      "metric_unit": "object",
      "artifact_paths": ["artifacts/classifier.pt"],
      "artifact_sha256": {"artifacts/classifier.pt": "..."},
      "source_report_path": "reports/dino_threshold_042.json",
      "source_report_sha256": "...",
      "metrics": {"defect_recall": 0.91, "good_false_positive_rate": 0.18, "latency_ms": 12.4},
      "metric_numerators": {"defect_true_positive": 91, "good_false_alarm_count": 9},
      "metric_denominators": {"defect_count": 100, "good_count": 50},
      "split_role": "validation",
      "holdout_used_for_selection": false
    }
  ]
}
```

## Atomic output

Run:

```bash
PYTHONPATH=src DAenv/bin/python scripts/run_static_sweep.py \
  --spec path/to/sweep.json \
  --output reports/experiments/<new-run-id>
```

The command refuses an existing final output path. It writes
`manifest.json`, `candidates.jsonl`, `validation_summary.json`, and
`selected_candidate.json` into a temporary sibling directory, then renames
that directory into place only after all files are complete. A write failure
cleans the temporary directory and does not leave an incomplete final run.

The manifest records the spec/config lineage, all four split hashes,
candidate artifact/source-report hashes as expected/actual pairs, the
selection gates, and the explicit notice that this is a selector protocol
guard rather than an independent audit of metric provenance. Concurrent
invocation protection is intentionally deferred to P2; callers must allocate
unique output paths and avoid racing writers.

## Historical report renderer

`scripts/render_static_report.py` is a read-only renderer for existing JSON
evidence. It does not run a model, select a threshold, or create a new
benchmark. Given an operating-points JSON and a DINO classifier `summary.json`,
it writes three charts:

- `threshold_recall_vs_good_fp.png`: operating-point GOOD false-positive rate
  versus DEFECT recall, with separate labeled CAD-SD and MVTec screw series
  when the report contains `cad_*` and `mvtec_*` metrics. Flat
  `test_*`/`validation_*` schemas remain supported as one domain;
- `classifier_learning_curve.png`: train loss and validation AUROC by epoch;
- `classifier_confusion_matrix.png`: the supplied raw test GOOD/DEFECT 2x2
  integer-count confusion matrix.

The parser rejects conflicting aliases for one metric, malformed history
(`epoch` must be an integer greater than zero, loss non-negative, AUROC in
`[0, 1]`), and fractional/negative/normalized confusion matrices. Normalized
confusion matrices are outside this slice.

Each render directory also contains `report_manifest.json` with source and
generated-file SHA-256 values, source protocol strings, and an explicit
`historical_development_diagnostic` status. The manifest warns that the charts
are not fresh independent-holdout evidence. The renderer refuses an existing
output directory and publishes through a temporary sibling directory so a
plotting/write failure does not leave a partial report.

Matplotlib is lazy-imported only after JSON validation. CLI help and input
validation therefore do not initialize the font cache. Actual rendering uses
the Agg backend; on some Python/Matplotlib environments the first render may
still be delayed or blocked by font-cache initialization. That is an
environment limitation, not evidence that a benchmark ran.
