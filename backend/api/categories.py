"""Category listing (single source of truth: Phase 1 config)."""

from fastapi import APIRouter

from ..schemas.common import CategoriesResponse

router = APIRouter(tags=["categories"])


@router.get("/api/categories", response_model=CategoriesResponse, summary="List categories")
def list_categories() -> CategoriesResponse:
    from model.inference.config import CATEGORIES

    return CategoriesResponse(categories=list(CATEGORIES))
