"""Common (non-detection) response schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    database: str


class CategoriesResponse(BaseModel):
    categories: list[str]
