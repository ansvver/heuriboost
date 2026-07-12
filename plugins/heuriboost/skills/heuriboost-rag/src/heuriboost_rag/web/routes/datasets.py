"""Immutable dataset listing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request


router = APIRouter(prefix="/api", tags=["datasets"])


@router.get("/datasets")
def list_datasets(request: Request) -> list[dict[str, object]]:
    return request.app.state.import_service.list_datasets()
