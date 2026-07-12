"""Release rollback APIs."""

from __future__ import annotations

from fastapi import APIRouter, Header, Request


router = APIRouter(prefix="/api", tags=["releases"])


@router.post("/releases/{run_id}/rollback", status_code=201)
def rollback_release(
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
    return request.app.state.promotion_service.rollback(run_id, approved_by=approved_by)


__all__ = ["router"]
