"""FastAPI application factory for the local HeuriBoost Web Console."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import WebConfig
from .jobs.executor import LocalJobExecutor
from .routes import datasets, imports, pages, promotions, releases, reports, runs
from .runtime import build_runtime
from .security import CSRF_HEADER, SESSION_COOKIE, csrf_is_valid, has_session, reject, secure_headers
from .services.import_service import ImportService
from .services.promotion_service import PromotionService
from .services.report_service import ReportService
from .services.run_service import RunService
from .stores.sqlite import SQLiteStore
from ..reckless.errors import HeuriBoostError, PromotionConflictError


def create_app(config: WebConfig) -> FastAPI:
    """Create a local-only app with durable service hooks attached to state."""

    if not isinstance(config, WebConfig):
        raise TypeError("config must be a WebConfig")
    config.data_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = Path(__file__).with_name("templates")
    static_dir = Path(__file__).with_name("static")
    store = SQLiteStore(config.data_dir / "heuriboost.db")
    store.migrate()
    async def job_worker_loop(app: FastAPI) -> None:
        while True:
            try:
                job = await asyncio.to_thread(
                    app.state.job_executor.execute_next,
                    app.state.runtime.run_existing,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                app.state.store.append_audit_event(
                    "job.worker_error",
                    {"exception_type": type(exc).__name__},
                )
                await asyncio.sleep(config.job_worker_poll_seconds)
                continue
            if job is None:
                await asyncio.sleep(config.job_worker_poll_seconds)
            else:
                await asyncio.sleep(0)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        if config.job_worker_enabled:
            task = asyncio.create_task(job_worker_loop(app))
            app.state.job_worker_task = task
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(
        title="HeuriBoost Web Console",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.store = store
    app.state.runtime = build_runtime(config, store)
    app.state.import_service = ImportService(config, store)
    app.state.job_executor = LocalJobExecutor(store)
    app.state.run_service = RunService(store, app.state.job_executor, app.state.runtime)
    app.state.report_service = ReportService(app.state.runtime)
    app.state.promotion_service = PromotionService(app.state.runtime, app.state.report_service)
    app.state.templates = Jinja2Templates(directory=str(templates_dir))
    app.state.templates.env.globals["csrf_token"] = config.csrf_token
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.middleware("http")
    async def local_security(request: Request, call_next):
        if config.security_enabled and request.url.path != "/health":
            token = request.query_params.get("token")
            token_is_valid = token == config.session_token
            session_is_valid = has_session(request, config)
            if not token_is_valid and not session_is_valid:
                response = reject(401, "missing or invalid local session")
                secure_headers(response)
                return response
            if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not csrf_is_valid(request, config):
                response = reject(403, "missing or invalid CSRF token")
                secure_headers(response)
                return response
            response = await call_next(request)
            if token_is_valid:
                response.set_cookie(
                    SESSION_COOKIE,
                    str(config.session_token),
                    httponly=True,
                    samesite="strict",
                    secure=False,
                    max_age=config.session_token_ttl_seconds,
                )
            secure_headers(response)
            return response
        response = await call_next(request)
        secure_headers(response)
        return response

    @app.exception_handler(HeuriBoostError)
    async def heuriboost_error(_: Request, error: HeuriBoostError) -> JSONResponse:
        status = 409 if isinstance(error, PromotionConflictError) else 422
        return JSONResponse(status_code=status, content=error.to_dict())

    @app.exception_handler(FileNotFoundError)
    async def missing_resource(_: Request, error: FileNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"code": "NOT_FOUND", "message": str(error), "details": {}, "operator_action": "Check the selected record and try again."},
        )

    @app.exception_handler(ValueError)
    async def invalid_input(_: Request, error: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"code": "INVALID_INPUT", "message": str(error), "details": {}, "operator_action": "Correct the input and try again."},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(imports.router)
    app.include_router(datasets.router)
    app.include_router(runs.router)
    app.include_router(reports.router)
    app.include_router(promotions.router)
    app.include_router(releases.router)
    app.include_router(pages.router)
    return app


__all__ = ["create_app"]
