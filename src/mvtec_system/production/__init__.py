"""Dataset-driven production automation."""

from .comparison import run_detector_comparison, summarize_comparison
from .pipeline import ProductionConfig, infer_exact_first, run_production

__all__ = [
    "ProductionConfig",
    "infer_exact_first",
    "run_production",
    "run_detector_comparison",
    "summarize_comparison",
]
