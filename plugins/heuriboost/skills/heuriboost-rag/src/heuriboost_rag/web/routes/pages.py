"""Server-rendered operational pages."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def workbench(request: Request) -> HTMLResponse:
    service = request.app.state.import_service
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="workbench.html",
        context={
            "datasets": service.list_datasets(),
            "runs": service.recent_runs(),
        },
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(run_id: str, request: Request) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="run_detail.html",
        context={"run": request.app.state.run_service.get_run(run_id)},
    )


@router.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="runs.html",
        context={"runs": request.app.state.import_service.recent_runs()},
    )


@router.get("/datasets", response_class=HTMLResponse)
def datasets_page(request: Request) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="datasets.html",
        context={"datasets": request.app.state.import_service.list_datasets()},
    )


@router.get("/models", response_class=HTMLResponse)
def models_page(request: Request) -> HTMLResponse:
    current = request.app.state.runtime.releases.read_current_model()
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="models.html",
        context={"current_model": current},
    )


@router.get("/gates", response_class=HTMLResponse)
def gates_page(request: Request) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="gates.html",
        context={"policy": request.app.state.runtime.policy},
    )


@router.get("/audit", response_class=HTMLResponse)
def audit_page(request: Request) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="audit.html",
        context={"events": request.app.state.store.audit_events_after(0, limit=200)},
    )


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={"config": request.app.state.config, "backend_name": request.app.state.runtime.backend.name},
    )
