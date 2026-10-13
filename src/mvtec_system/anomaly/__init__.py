"""Feature-bank anomaly scoring and inference maps."""

from .detectors import (
    AnomalyDetector,
    AnomalyResult,
    DETECTOR_REGISTRY,
    ExactKNNDetector,
    PaDiMDetector,
    PatchCoreDetector,
    create_detector,
    detector_names,
)
from .refinement import gate_heatmap, refine_binary_mask

__all__ = [
    "AnomalyDetector",
    "AnomalyResult",
    "DETECTOR_REGISTRY",
    "ExactKNNDetector",
    "PatchCoreDetector",
    "PaDiMDetector",
    "create_detector",
    "detector_names",
    "gate_heatmap",
    "refine_binary_mask",
]
