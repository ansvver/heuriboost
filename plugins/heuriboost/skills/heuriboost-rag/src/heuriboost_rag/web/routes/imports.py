"""Upload, worksheet inspection, preview, and normalization endpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile

from ...importers.base import ImportOptions
from ..services.import_service import ImportService


router = APIRouter(prefix="/api", tags=["imports"])


def _service(request: Request) -> ImportService:
    return request.app.state.import_service


def _options(payload: Mapping[str, object]) -> ImportOptions:
    unknown = set(payload) - {"delimiter", "sheet_name", "header_row"}
    if unknown:
        raise ValueError("unknown import options: " + ", ".join(sorted(unknown)))
    return ImportOptions(
        delimiter=payload.get("delimiter"),
        sheet_name=payload.get("sheet_name"),
        header_row=payload.get("header_row", 1),
    )


@router.post("/imports", status_code=201)
async def create_import(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
    return await _service(request).create_upload(file)


@router.get("/imports/{upload_id}")
def get_import(upload_id: str, request: Request) -> dict[str, object]:
    return _service(request).get_upload(upload_id)


@router.get("/imports/{upload_id}/sheets")
def get_sheets(upload_id: str, request: Request) -> list[dict[str, object]]:
    return _service(request).sheets(upload_id)


@router.post("/imports/{upload_id}/preview")
def preview_import(upload_id: str, request: Request, payload: dict[str, object]) -> dict[str, object]:
    return _service(request).preview(upload_id, _options(payload))


@router.post("/imports/{upload_id}/normalize", status_code=201)
def normalize_import(upload_id: str, request: Request, payload: dict[str, Any]) -> dict[str, object]:
    mapping = payload.get("mapping")
    if not isinstance(mapping, Mapping):
        raise ValueError("mapping must be an object")
    role = payload.get("role")
    if not isinstance(role, str):
        raise ValueError("role must be a string")
    raw_options = payload.get("options", {})
    if not isinstance(raw_options, Mapping):
        raise ValueError("options must be an object")
    return _service(request).normalize(
        upload_id,
        role=role,
        mapping=mapping,
        options=_options(raw_options),
    )
