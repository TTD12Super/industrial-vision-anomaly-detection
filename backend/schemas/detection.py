"""Detection response schemas.

The API returns a continuous ``anomaly_score`` only. There is deliberately
no ``is_anomaly`` / label field: the project has no calibrated threshold
(see ``model/inference``). Internal MongoDB ``_id`` and filesystem paths are
never exposed; only public URLs are returned.
"""

from pydantic import BaseModel, Field


class DetectionLinks(BaseModel):
    input_url: str = Field(description="URL of the stored input image (PNG).")
    heatmap_url: str = Field(description="URL of the anomaly heatmap (PNG).")
    overlay_url: str = Field(description="URL of the heatmap overlay (PNG).")


class DetectionResponse(BaseModel):
    prediction_id: str = Field(description="Unique ID of this detection.")
    category: str = Field(description="Requested MVTec category.")
    anomaly_score: float = Field(
        description="Continuous anomaly score (max patch distance). Higher = more anomalous."
    )
    processing_time_ms: int = Field(description="End-to-end request processing time.")
    result: DetectionLinks


class DetectionSummary(BaseModel):
    prediction_id: str
    category: str
    anomaly_score: float
    processing_time_ms: int
    created_at: str = Field(description="ISO-8601 creation timestamp (UTC).")
    result: DetectionLinks


class DetectionDetail(DetectionSummary):
    """Detail view (same public fields as the summary; paths stay internal)."""


class DetectionListResponse(BaseModel):
    items: list[DetectionSummary]
    total: int = Field(description="Total matching detections.")
    page: int
    page_size: int


class DeleteDetectionResponse(BaseModel):
    message: str
    prediction_id: str
