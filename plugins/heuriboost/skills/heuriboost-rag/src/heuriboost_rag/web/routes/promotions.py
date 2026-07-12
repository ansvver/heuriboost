"""Promotion write APIs."""

from __future__ import annotations

from fastapi import APIRouter, Header, Request


router = APIRouter(prefix="/api", tags=["promotions"])


@router.post("/promotions/{run_id}", status_code=201)
def promote_run(
    run_id: str,
    request: Request,
    payload: dict[str, object],
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, object]:
    if not idempotency_key:
        raise ValueError("Idempotency-Key is required")
    approved_by = payload.get("approved_by")
    if not isinstance(approved_by, str) or not approved_by:
        raise ValueError("approved_by is required")
    return request.app.state.promotion_service.promote(
        run_id,
        approved_by=approved_by,
        idempotency_key=idempotency_key,
    )


__all__ = ["router"]
