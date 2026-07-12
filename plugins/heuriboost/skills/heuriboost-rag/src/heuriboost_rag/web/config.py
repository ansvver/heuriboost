"""Immutable local Web Console configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import secrets
from types import MappingProxyType
from typing import Any, Mapping

import yaml


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _optional_string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_int(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _string(mapping: Mapping[str, object], key: str, default: str) -> str:
    value = mapping.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


@dataclass(frozen=True)
class WebConfig:
    data_dir: Path
    host: str = "127.0.0.1"
    port: int = 8787
    workspace_id: str = "default-rag"
    backend: str = "heuriboost_rag.web.runtime:DemoRepairBackend"
    promotion_target: str | None = None
    policy_path: Path | None = None
    backend_options: Mapping[str, object] | None = None
    default_locale: str = "zh-CN"
    security_enabled: bool = True
    shared_mode: bool = False
    max_upload_bytes: int = 100 * 1024 * 1024
    max_xlsx_uncompressed_bytes: int = 512 * 1024 * 1024
    max_xlsx_sheets: int = 32
    max_xlsx_rows: int = 250_000
    max_xlsx_columns: int = 256
    session_token_ttl_seconds: int = 3600
    job_worker_enabled: bool = True
    job_worker_poll_seconds: float = 1.0
    session_token: str | None = None
    csrf_token: str | None = None

    def __post_init__(self) -> None:
        data_dir = Path(self.data_dir).expanduser().resolve()
        object.__setattr__(self, "data_dir", data_dir)
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty string")
        if not isinstance(self.workspace_id, str) or not self.workspace_id:
            raise ValueError("workspace_id must be a non-empty string")
        if not isinstance(self.backend, str) or ":" not in self.backend:
            raise ValueError("backend must be an import path in module:object form")
        if self.promotion_target is not None and (
            not isinstance(self.promotion_target, str) or ":" not in self.promotion_target
        ):
            raise ValueError("promotion_target must be an import path in module:object form")
        if self.policy_path is not None:
            object.__setattr__(self, "policy_path", Path(self.policy_path).expanduser().resolve())
        if self.backend_options is None:
            object.__setattr__(self, "backend_options", MappingProxyType({}))
        elif not isinstance(self.backend_options, Mapping):
            raise ValueError("backend_options must be a mapping")
        else:
            object.__setattr__(self, "backend_options", MappingProxyType(dict(self.backend_options)))
        if not isinstance(self.default_locale, str) or not self.default_locale:
            raise ValueError("default_locale must be a non-empty string")
        if not isinstance(self.security_enabled, bool):
            raise ValueError("security_enabled must be a boolean")
        if not isinstance(self.job_worker_enabled, bool):
            raise ValueError("job_worker_enabled must be a boolean")
        if (
            isinstance(self.job_worker_poll_seconds, bool)
            or not isinstance(self.job_worker_poll_seconds, (int, float))
            or self.job_worker_poll_seconds <= 0
        ):
            raise ValueError("job_worker_poll_seconds must be a positive number")
        object.__setattr__(self, "job_worker_poll_seconds", float(self.job_worker_poll_seconds))
        if self.host not in _LOOPBACK_HOSTS and not self.shared_mode:
            raise ValueError("non-loopback host requires shared_mode")
        if isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        for name in (
            "max_upload_bytes",
            "max_xlsx_uncompressed_bytes",
            "max_xlsx_sheets",
            "max_xlsx_rows",
            "max_xlsx_columns",
            "session_token_ttl_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        token = self.session_token or secrets.token_urlsafe(32)
        if not isinstance(token, str) or not token:
            raise ValueError("session_token must be a non-empty string")
        object.__setattr__(self, "session_token", token)
        csrf_token = self.csrf_token or secrets.token_urlsafe(32)
        if not isinstance(csrf_token, str) or not csrf_token:
            raise ValueError("csrf_token must be a non-empty string")
        object.__setattr__(self, "csrf_token", csrf_token)

    @classmethod
    def for_test(cls, data_dir: Path) -> WebConfig:
        return cls(
            data_dir=Path(data_dir),
            session_token="heuriboost-web-test-token",
            csrf_token="heuriboost-web-test-csrf",
            security_enabled=False,
            job_worker_enabled=False,
        )

    def with_job_worker(self, *, enabled: bool, poll_seconds: float | None = None) -> WebConfig:
        values: dict[str, object] = {"job_worker_enabled": enabled}
        if poll_seconds is not None:
            values["job_worker_poll_seconds"] = poll_seconds
        return replace(self, **values)

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        data_dir: Path | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> WebConfig:
        try:
            loaded: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ValueError(f"cannot read Web Console config: {path}") from exc
        root = _mapping(loaded, "Web Console config")
        web = _mapping(root.get("web", {}), "web")
        backend_options = _mapping(root.get("backend_options", {}), "backend_options")
        configured_data_dir = _optional_string(root, "data_dir")
        configured_host = _optional_string(web, "host")
        configured_port = _optional_int(web, "port")
        shared_mode = web.get("shared_mode", False)
        if not isinstance(shared_mode, bool):
            raise ValueError("web.shared_mode must be a boolean")
        security_enabled = web.get("security_enabled", True)
        if not isinstance(security_enabled, bool):
            raise ValueError("web.security_enabled must be a boolean")
        job_worker_enabled = web.get("job_worker_enabled", True)
        if not isinstance(job_worker_enabled, bool):
            raise ValueError("web.job_worker_enabled must be a boolean")
        job_worker_poll_seconds = web.get("job_worker_poll_seconds", 1.0)
        return cls(
            data_dir=data_dir or Path(configured_data_dir or "~/.heuriboost"),
            host=host or configured_host or "127.0.0.1",
            port=port or configured_port or 8787,
            workspace_id=_string(root, "workspace_id", "default-rag"),
            backend=_string(root, "backend", "heuriboost_rag.web.runtime:DemoRepairBackend"),
            promotion_target=_optional_string(root, "promotion_target"),
            policy_path=None if root.get("policy") is None else Path(_string(root, "policy", "")),
            backend_options=backend_options,
            default_locale=_string(web, "default_locale", "zh-CN"),
            security_enabled=security_enabled,
            shared_mode=shared_mode,
            job_worker_enabled=job_worker_enabled,
            job_worker_poll_seconds=job_worker_poll_seconds,
        )

    @property
    def launch_url(self) -> str:
        if not self.security_enabled:
            return f"http://{self.host}:{self.port}/"
        return f"http://{self.host}:{self.port}/?token={self.session_token}"


__all__ = ["WebConfig"]
