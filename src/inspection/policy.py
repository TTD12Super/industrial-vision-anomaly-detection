"""Separate the binary quality label from the operational review state."""

from __future__ import annotations

from dataclasses import dataclass

from .interfaces import KNOWN_PRODUCT, REVIEW_PRODUCT, UNKNOWN_PRODUCT


GOOD = "GOOD"
DEFECT = "DEFECT"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
NOT_ASSESSED = "NOT_ASSESSED"
NO_DETECTION = "NO_DETECTION"
STAGE_FAILED = "STAGE_FAILED"


@dataclass(frozen=True)
class OperationalDecision:
    classifier_label: str
    operational_status: str
    p_defect: float
    needs_review: bool
    reason: str


@dataclass(frozen=True)
class DecisionPolicy:
    """Validation-calibrated decision and uncertain-GOOD review thresholds."""

    defect_threshold: float
    review_low_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.review_low_threshold < self.defect_threshold <= 1.0:
            raise ValueError("Expected 0 <= review_low_threshold < defect_threshold <= 1")

    def decide(self, p_defect: float) -> OperationalDecision:
        if not 0.0 <= p_defect <= 1.0:
            raise ValueError("p_defect must be in [0, 1]")
        if p_defect >= self.defect_threshold:
            return OperationalDecision(DEFECT, DEFECT, p_defect, True, "at_or_above_defect_threshold")
        if p_defect >= self.review_low_threshold:
            return OperationalDecision(GOOD, REVIEW_REQUIRED, p_defect, True, "within_uncertain_good_band")
        return OperationalDecision(GOOD, GOOD, p_defect, False, "below_review_band")
