"""Report download and live approval pages."""

from __future__ import annotations

import base64
import hashlib
from html import escape
import re

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


router = APIRouter(tags=["reports"])


@router.get("/api/reports/{run_id}", response_class=HTMLResponse)
def offline_report(run_id: str, request: Request) -> HTMLResponse:
    return _report_response(request.app.state.report_service.html(run_id))


@router.get("/runs/{run_id}/report", response_class=HTMLResponse)
def live_report(run_id: str, request: Request) -> HTMLResponse:
    run = request.app.state.store.runs.get(run_id)
    html = request.app.state.report_service.html(run_id)
    report_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()
    return _report_response(_inject_live_controls(html, run_id=run_id, run_state=run.state, report_hash=report_hash))


def _inject_live_controls(html: str, *, run_id: str, run_state: str, report_hash: str) -> str:
    body = re.search(r"<body\b[^>]*>", html, flags=re.IGNORECASE)
    if body is None:
        raise RuntimeError("Pre Promote report HTML is missing a body element")
    control = (
        f'<section data-report-hash="{escape(report_hash, quote=True)}" '
        'style="max-width:1180px;margin:0 auto;padding:12px 16px;background:#fff;border:1px solid #111;border-top:0">'
        f"{_promotion_action(run_id, run_state)}"
        "</section>"
    )
    return html[: body.end()] + control + html[body.end() :]


def _promotion_action(run_id: str, run_state: str) -> str:
    if run_state != "READY_FOR_PROMOTION":
        return ""
    safe_run_id = escape(run_id, quote=True)
    return (
        f'<form method="post" action="/api/promotions/{safe_run_id}">'
        '<button type="submit">批准并 Promote</button>'
        "</form>"
    )


def _report_response(html: str) -> HTMLResponse:
    return HTMLResponse(html, headers={"Content-Security-Policy": _report_csp(html)})


def _report_csp(html: str) -> str:
    script_hashes = " ".join(_inline_script_hashes(html))
    script_src = "script-src 'self'" + (f" {script_hashes}" if script_hashes else "")
    return f"default-src 'self'; {script_src}; style-src 'self' 'unsafe-inline'"


_SCRIPT_RE = re.compile(r"<script\b(?P<attrs>[^>]*)>(?P<body>.*?)</script>", flags=re.IGNORECASE | re.DOTALL)
_SCRIPT_SRC_RE = re.compile(r"\bsrc\s*=", flags=re.IGNORECASE)
_SCRIPT_TYPE_RE = re.compile(r"""\btype\s*=\s*["']?(?P<type>[^"'\s>]+)""", flags=re.IGNORECASE)
_EXECUTABLE_SCRIPT_TYPES = {"", "text/javascript", "application/javascript", "module"}


def _inline_script_hashes(html: str) -> list[str]:
    hashes: list[str] = []
    for match in _SCRIPT_RE.finditer(html):
        attrs = match.group("attrs")
        if _SCRIPT_SRC_RE.search(attrs):
            continue
        type_match = _SCRIPT_TYPE_RE.search(attrs)
        script_type = "" if type_match is None else type_match.group("type").lower()
        if script_type not in _EXECUTABLE_SCRIPT_TYPES:
            continue
        digest = hashlib.sha256(match.group("body").encode("utf-8")).digest()
        hashes.append(f"'sha256-{base64.b64encode(digest).decode('ascii')}'")
    return hashes


__all__ = ["router"]
