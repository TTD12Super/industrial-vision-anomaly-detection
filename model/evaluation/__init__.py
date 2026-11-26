"""Offline MVTec AD evaluation built on ``model.inference``.

Reference: ``notebooks/02_first_model.ipynb`` (metric cells 12, 62, 64, 66,
68, 74, 98). Uses lazy loading (PEP 562) so importing this package never
requires torch/Pillow/scikit-learn until an evaluation actually runs.
"""

from __future__ import annotations

_LAZY_EXPORTS = {
    "AUPRO_STATUS": "evaluate",
    "AGGREGATE_METRIC_KEYS": "threshold_evaluation",
    "BOUND_ASSUMPTIONS": "calibration",
    "BaselineIntegrityError": "threshold_evaluation",
    "CategoryCalibration": "calibration",
    "CategoryResult": "evaluate",
    "CategoryThresholdResult": "threshold_evaluation",
    "DECISION_RULE": "calibration",
    "DECISION_RULE_APPLIED": "threshold_evaluation",
    "DEFAULT_ALPHA": "calibration",
    "GUARANTEE_NOTE": "threshold_evaluation",
    "METRIC_DEFINITIONS": "metrics",
    "MAX_PIXEL_METRIC_GIB": "metrics",
    "PIXEL_METRICS_STATUS": "threshold_evaluation",
    "THRESHOLD_FORMULA": "calibration",
    "THRESHOLD_METRIC_DEFINITIONS": "threshold_evaluation",
    "THRESHOLD_STRATEGY": "evaluate",
    "ThresholdDataError": "threshold_evaluation",
    "ThresholdEntry": "threshold_evaluation",
    "ThresholdResult": "calibration",
    "build_run_config": "evaluate",
    "build_summary": "threshold_evaluation",
    "build_threshold_eval_config": "threshold_evaluation",
    "bank_hashes": "calibration",
    "calibrate_category": "calibration",
    "category_paths": "evaluate",
    "configuration_hash": "evaluate",
    "confusion_counts": "threshold_evaluation",
    "evaluate_category": "evaluate",
    "evaluate_category_thresholds": "threshold_evaluation",
    "evaluate_ranking": "metrics",
    "find_mask_path": "metrics",
    "image_files": "metrics",
    "image_labels": "metrics",
    "load_category_bank": "evaluate",
    "load_frozen_thresholds": "threshold_evaluation",
    "load_mask": "metrics",
    "load_masks": "metrics",
    "loo_image_scores": "calibration",
    "loo_patch_scores": "calibration",
    "macro_average": "metrics",
    "macro_threshold_metrics": "threshold_evaluation",
    "micro_threshold_metrics": "threshold_evaluation",
    "order_statistic_threshold": "calibration",
    "pixel_metric_memory_gib": "metrics",
    "pixel_metrics": "metrics",
    "predict_defect": "threshold_evaluation",
    "ranking_metrics": "metrics",
    "run_calibration": "calibration",
    "run_evaluation": "evaluate",
    "run_threshold_evaluation": "threshold_evaluation",
    "score_summary": "threshold_evaluation",
    "scoring_consistency": "threshold_evaluation",
    "sha256_file": "calibration",
    "threshold_metrics": "threshold_evaluation",
    "validate_bank_for_calibration": "calibration",
    "verify_bank_baseline": "threshold_evaluation",
    "write_aggregate_outputs": "evaluate",
    "write_category_outputs": "evaluate",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(f"{__name__}.{_LAZY_EXPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")


def __dir__():
    return sorted(__all__)
