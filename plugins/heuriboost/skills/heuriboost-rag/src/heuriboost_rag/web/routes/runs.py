"""Durable run creation and lookup endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse


router = APIRouter(prefix="/api", tags=["runs"])


@router.post("/runs", status_code=201)
def create_run(request: Request, payload: dict[str, object], idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict[str, object]:
    if not idempotency_key:
        raise ValueError("Idempotency-Key is required")
    base = payload.get("base_dataset_id")
    cases = payload.get("production_cases_id")
    if not isinstance(base, str) or not isinstance(cases, str):
        raise ValueError("base_dataset_id and production_cases_id are required")
    return request.app.state.run_service.create_run(
        base,
        cases,
        requested_by="local-operator",
        idempotency_key=idempotency_key,
    )


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, object]:
    return request.app.state.run_service.get_run(run_id)


@router.post("/runs/{run_id}/cancel")
def cancel_run(
    run_id: str,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, object]:
    if not idempotency_key:
        raise ValueError("Idempotency-Key is required")
    return request.app.state.run_service.cancel(run_id)


@router.post("/runs/{run_id}/retry", status_code=201)
def retry_run(
    run_id: str,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, object]:
    if not idempotency_key:
        raise ValueError("Idempotency-Key is required")
    return request.app.state.run_service.retry(run_id)


@router.get("/runs/{run_id}/events")
def run_events(
    run_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    try:
        after_id = 0 if last_event_id is None else int(last_event_id)
    except ValueError as exc:
        raise ValueError("Last-Event-ID must be an integer") from exc
    if after_id < 0:
        raise ValueError("Last-Event-ID must be non-negative")
    events = [
        event
        for event in request.app.state.store.audit_events_after(after_id)
        if event["payload"].get("run_id") == run_id
    ]

    def stream():
        for event in events:
            payload = {
                "event_id": event["event_id"],
                "run_id": run_id,
                "type": event["event_type"],
                "occurred_at": event["occurred_at"],
                **event["payload"],
            }
            yield f"id: {event['event_id']}\nevent: run-event\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield ": heartbeat\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
