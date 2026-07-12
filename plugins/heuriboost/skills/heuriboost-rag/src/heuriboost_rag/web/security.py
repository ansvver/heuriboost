"""Local Web Console security helpers."""

from __future__ import annotations

from typing import Protocol

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from .config import WebConfig


SESSION_COOKIE = "heuriboost_session"
CSRF_HEADER = "X-CSRF-Token"
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class IdentityProvider(Protocol):
    def current_identity(self, request: Request) -> str: ...


class LocalIdentityProvider:
    def __init__(self, username: str) -> None:
        self.username = username

    def current_identity(self, request: Request) -> str:
        return self.username


def secure_headers(response: Response) -> None:
    response.headers.setdefault("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")


def reject(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "code": "UNAUTHORIZED" if status_code == 401 else "FORBIDDEN",
            "message": message,
            "details": {},
            "operator_action": "Open the launch URL again and retry the action.",
        },
    )


def has_session(request: Request, config: WebConfig) -> bool:
    return request.cookies.get(SESSION_COOKIE) == config.session_token


def csrf_is_valid(request: Request, config: WebConfig) -> bool:
    return request.headers.get(CSRF_HEADER) == config.csrf_token


__all__ = [
    "CSRF_HEADER",
    "SESSION_COOKIE",
    "IdentityProvider",
    "LocalIdentityProvider",
    "csrf_is_valid",
    "has_session",
    "reject",
    "secure_headers",
]
