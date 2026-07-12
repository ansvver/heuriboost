from __future__ import annotations

from collections.abc import Mapping
import errno
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import TYPE_CHECKING, Protocol, runtime_checkable
import uuid

from .contracts import DatasetRef, RepairRequest, _freeze_mapping, to_plain_data

if TYPE_CHECKING:
    from .policy import RecklessPolicy


_HASH_CHUNK_SIZE = 1024 * 1024
_FINGERPRINT_SCHEMA_VERSION = 1
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
)
_MACOS_SYSTEM_DIRECTORY_ALIASES = frozenset({"etc", "tmp", "var"})


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ExecutionIdentity:
    backend_version: str
    feature_names: tuple[str, ...]
    feature_version: str
    code_commit: str
    training_params: Mapping[str, object]
    random_seed: int

    def __post_init__(self) -> None:
        _required_string(self.backend_version, "backend_version")
        if not isinstance(self.feature_names, (list, tuple)):
            raise TypeError("feature_names must be an ordered sequence of strings")
        feature_names = tuple(self.feature_names)
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        for name in feature_names:
            _required_string(name, "feature_names item")
        object.__setattr__(self, "feature_names", feature_names)
        _required_string(self.feature_version, "feature_version")
        _required_string(self.code_commit, "code_commit")
        if not isinstance(self.training_params, Mapping):
            raise TypeError("training_params must be an explicitly supplied mapping")
        frozen_params = _freeze_mapping(self.training_params)
        json.dumps(
            to_plain_data(frozen_params),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        object.__setattr__(self, "training_params", frozen_params)
        if type(self.random_seed) is not int:
            raise TypeError("random_seed must be an integer")


@runtime_checkable
class ExecutionIdentityProvider(Protocol):
    def execution_identity(self) -> ExecutionIdentity: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: object) -> str:
    canonical = json.dumps(
        to_plain_data(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_run_fingerprint(
    request: RepairRequest,
    policy: RecklessPolicy,
    base_dataset: DatasetRef,
    production_cases: DatasetRef,
    execution_identity: ExecutionIdentity | None = None,
) -> str:
    if execution_identity is None:
        raise ValueError("execution_identity is required")
    if not isinstance(execution_identity, ExecutionIdentity):
        raise TypeError("execution_identity must be an ExecutionIdentity")

    try:
        policy_content_hash = policy.content_hash
        policy_schema_version = policy.version
    except AttributeError as exc:
        raise TypeError("policy must expose content_hash and version") from exc

    material = {
        "fingerprint_schema_version": _FINGERPRINT_SCHEMA_VERSION,
        "request": {
            "workspace_id": request.workspace_id,
            "base_dataset_id": request.base_dataset_id,
            "production_cases_id": request.production_cases_id,
            "policy_version": request.policy_version,
            "backend_name": request.backend_name,
        },
        "policy": {
            "content_hash": policy_content_hash,
            "effective_version": request.policy_version,
            "schema_version": policy_schema_version,
        },
        "datasets": {
            "base": {
                "content_hash": base_dataset.content_hash,
                "schema_hash": base_dataset.schema_hash,
            },
            "production_cases": {
                "content_hash": production_cases.content_hash,
                "schema_hash": production_cases.schema_hash,
            },
        },
        "backend": {
            "name": request.backend_name,
            "version": execution_identity.backend_version,
        },
        "features": {
            "names": list(execution_identity.feature_names),
            "version": execution_identity.feature_version,
        },
        "code_commit": execution_identity.code_commit,
        "training": {
            "params": execution_identity.training_params,
            "random_seed": execution_identity.random_seed,
        },
    }
    return canonical_json_hash(material)


def _require_secure_fs_primitives() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise RuntimeError("secure descriptor-relative filesystem operations unavailable")
    required_dir_fd = (os.open, os.mkdir, os.unlink)
    if any(operation not in os.supports_dir_fd for operation in required_dir_fd):
        raise RuntimeError("required dir_fd filesystem operations unavailable")


def _fsync_directory_fd(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            raise


def _absolute_directory_parts(directory: Path) -> tuple[str, ...]:
    parts = directory.parts[1:]
    if not parts or parts[0] not in _MACOS_SYSTEM_DIRECTORY_ALIASES:
        return parts
    try:
        target = os.readlink(os.path.join(os.sep, parts[0]))
    except OSError:
        return parts
    expected_relative = os.path.join("private", parts[0])
    expected_absolute = os.path.join(os.sep, expected_relative)
    if target in {expected_relative, expected_absolute}:
        return ("private", parts[0], *parts[1:])
    return parts


def _reject_symlinked_directory_component(
    parent_descriptor: int,
    name: str,
) -> None:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except (OSError, TypeError):
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"symlinked directory rejected: {name!r}")


def _open_directory_path(path: Path, *, create: bool) -> int:
    _require_secure_fs_primitives()
    directory = Path(path)
    if directory.is_absolute():
        descriptor = os.open("/", _DIRECTORY_FLAGS)
        parts = _absolute_directory_parts(directory)
    else:
        descriptor = os.open(".", _DIRECTORY_FLAGS)
        parts = directory.parts

    try:
        for part in parts:
            if part in {"", "."}:
                continue
            if part == "..":
                raise ValueError("parent traversal is not allowed")
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                created = False
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    created = True
                    _fsync_directory_fd(descriptor)
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                except OSError:
                    _reject_symlinked_directory_component(descriptor, part)
                    raise
                if created:
                    _fsync_directory_fd(child)
            except OSError:
                _reject_symlinked_directory_component(descriptor, part)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _serialized_json(payload: object) -> bytes:
    return (
        json.dumps(
            to_plain_data(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json_at(
    parent_descriptor: int,
    name: str,
    payload: object,
    *,
    replace: bool = True,
) -> None:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError(f"unsafe destination filename: {name!r}")
    serialized = _serialized_json(payload)
    temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
    temporary_exists = False
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_exists = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(
                temporary_name,
                name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            temporary_exists = False
        else:
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            os.unlink(temporary_name, dir_fd=parent_descriptor)
            temporary_exists = False
        _fsync_directory_fd(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass


def atomic_write_json(path: Path, payload: object) -> None:
    destination = Path(path)
    if not destination.name:
        raise ValueError("destination must name a file")
    parent_descriptor = _open_directory_path(destination.parent, create=True)
    try:
        _atomic_write_json_at(parent_descriptor, destination.name, payload)
    finally:
        os.close(parent_descriptor)
