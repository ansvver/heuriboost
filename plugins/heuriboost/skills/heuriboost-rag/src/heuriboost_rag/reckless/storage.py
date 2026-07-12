from __future__ import annotations

from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Iterator, Mapping, Protocol
import uuid

from .contracts import (
    ArtifactRef,
    DatasetRef,
    RepairRequest,
    RunRecord,
    StageManifest,
    to_plain_data,
)
from .errors import ArtifactIntegrityError, HeuriBoostError
from .hashing import (
    _atomic_write_json_at,
    _fsync_directory_fd,
    _open_directory_path,
    _require_secure_fs_primitives,
    sha256_file,
)
from .state import RunState, assert_transition

try:
    import fcntl
except ImportError:  # pragma: no cover - Unix is the supported deployment target.
    fcntl = None


_RECORD_SCHEMA_VERSION = 1
_MAX_PERSISTED_JSON_BYTES = 1024 * 1024
_MAX_PERSISTED_JSON_DEPTH = 64
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DATASET_FIELDS = frozenset(
    {
        "dataset_id",
        "role",
        "path",
        "content_hash",
        "schema_hash",
        "metadata",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "workspace_id",
        "base_dataset_id",
        "production_cases_id",
        "policy_version",
        "backend_name",
        "requested_by",
        "run_options",
    }
)
_RUN_FIELDS = frozenset(
    {
        "run_id",
        "state",
        "version",
        "request",
        "policy_hash",
        "input_hash",
        "metadata",
        "error",
    }
)
_ERROR_FIELDS = frozenset(
    {
        "code",
        "message",
        "stage",
        "run_id",
        "retryable",
        "details",
        "operator_action",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "stage",
        "input_hash",
        "artifacts",
        "started_at",
        "completed_at",
        "status",
        "duration_ms",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {"artifact_type", "path", "content_hash", "size_bytes"}
)
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_READ_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)


class _SecureRoot:
    def __init__(self, root: Path) -> None:
        _require_secure_fs_primitives()
        if fcntl is None:
            raise RuntimeError("fcntl.flock is required for secure local storage")
        absolute = Path(os.path.abspath(os.fspath(root)))
        self.path = absolute
        self._descriptor = _open_directory_path(absolute, create=False)

    def __del__(self) -> None:
        descriptor = getattr(self, "_descriptor", -1)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self._descriptor = -1

    def open_dir(
        self,
        parts: tuple[str, ...],
        *,
        create: bool,
        exclusive_last: bool = False,
    ) -> int:
        descriptor = os.dup(self._descriptor)
        try:
            for index, part in enumerate(parts):
                _validate_safe_id(part, "path component")
                is_last = index == len(parts) - 1
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise
                    created = False
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        if exclusive_last and is_last:
                            raise
                    else:
                        created = True
                        _fsync_directory_fd(descriptor)
                    try:
                        child = os.open(
                            part,
                            _DIRECTORY_FLAGS,
                            dir_fd=descriptor,
                        )
                    except OSError:
                        _reject_symlink_leaf(descriptor, part)
                        raise
                    if created:
                        _fsync_directory_fd(child)
                except OSError:
                    _reject_symlink_leaf(descriptor, part)
                    raise
                else:
                    if exclusive_last and is_last:
                        os.close(child)
                        raise FileExistsError(part)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def path_for(self, *parts: str) -> Path:
        return self.path.joinpath(*parts)


def _open_regular_file(parent_descriptor: int, name: str) -> int:
    _reject_symlink_leaf(parent_descriptor, name)
    try:
        descriptor = os.open(name, _FILE_READ_FLAGS, dir_fd=parent_descriptor)
    except OSError:
        _reject_symlink_leaf(parent_descriptor, name)
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError(f"record is not a regular file: {name!r}")
    return descriptor


def _open_child_directory(
    parent_descriptor: int,
    name: str,
    *,
    create: bool,
) -> int:
    _validate_safe_id(name, "directory name")
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            pass
        else:
            _fsync_directory_fd(parent_descriptor)
        try:
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
        except OSError:
            _reject_symlink_leaf(parent_descriptor, name)
            raise
        _fsync_directory_fd(descriptor)
        return descriptor
    except OSError:
        _reject_symlink_leaf(parent_descriptor, name)
        raise


def _read_bounded_json(descriptor: int, name: str) -> bytes:
    chunks = []
    size_bytes = 0
    while True:
        remaining = _MAX_PERSISTED_JSON_BYTES - size_bytes
        chunk = os.read(descriptor, min(1024 * 1024, remaining + 1))
        if not chunk:
            return b"".join(chunks)
        size_bytes += len(chunk)
        if size_bytes > _MAX_PERSISTED_JSON_BYTES:
            raise ValueError(
                f"JSON record exceeds {_MAX_PERSISTED_JSON_BYTES} bytes: {name}"
            )
        chunks.append(chunk)


def _validate_json_depth(raw: bytes, name: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue
        if byte == ord('"'):
            in_string = True
        elif byte in (ord("{"), ord("[")):
            depth += 1
            if depth > _MAX_PERSISTED_JSON_DEPTH:
                raise ValueError(
                    "JSON record exceeds maximum container nesting depth "
                    f"{_MAX_PERSISTED_JSON_DEPTH}: {name}"
                )
        elif byte in (ord("}"), ord("]")):
            depth = max(0, depth - 1)


def _hash_open_file(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return digest.hexdigest(), size_bytes
        digest.update(chunk)
        size_bytes += len(chunk)


def _strict_json_at(parent_descriptor: int, name: str) -> object:
    descriptor = _open_regular_file(parent_descriptor, name)
    try:
        raw = _read_bounded_json(descriptor, name)
    finally:
        os.close(descriptor)
    _validate_json_depth(raw, name)
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSON record: {name}") from exc


def _reject_symlink_leaf(parent_descriptor: int, name: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"symlinked storage record rejected: {name!r}")


@contextmanager
def _locked_file(parent_descriptor: int, name: str) -> Iterator[None]:
    _reject_symlink_leaf(parent_descriptor, name)
    descriptor = os.dup(parent_descriptor)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"lock parent is not a directory: {name!r}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _validate_safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"Unsafe {label}: {value!r}")
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON value is not allowed: {value}")


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite JSON value is not allowed: {value}")
    return parsed


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _mapping(
    value: object,
    *,
    label: str,
    fields: frozenset[str] | None = None,
) -> Mapping[str, object]:
    if not isinstance(value, MappingABC):
        raise ValueError(f"{label} must be a JSON object")
    if fields is not None and set(value) != fields:
        missing = sorted(fields - set(value))
        unknown = sorted(set(value) - fields)
        raise ValueError(
            f"Invalid {label} fields; missing={missing}, unknown={unknown}"
        )
    return value


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _json_mapping(value: object, label: str) -> Mapping[str, object]:
    mapping = _mapping(value, label=label)
    try:
        to_plain_data(mapping)
    except TypeError as exc:
        raise ValueError(f"{label} contains unsupported data") from exc
    return mapping


def _record_payload(value: object, label: str) -> Mapping[str, object]:
    wrapper = _mapping(
        value,
        label=f"{label} record",
        fields=frozenset({"schema_version", "record"}),
    )
    if type(wrapper["schema_version"]) is not int:
        raise ValueError(f"{label} schema_version must be an integer")
    if wrapper["schema_version"] != _RECORD_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {label} schema version: {wrapper['schema_version']!r}"
        )
    return _mapping(wrapper["record"], label=label)


def _dataset_from_data(value: object, expected_id: str) -> DatasetRef:
    data = _mapping(value, label="dataset", fields=_DATASET_FIELDS)
    dataset_id = _validate_safe_id(
        _string(data["dataset_id"], "dataset.dataset_id"),
        "dataset ID",
    )
    if dataset_id != expected_id:
        raise ValueError(
            f"Dataset record ID mismatch: expected {expected_id!r}, got {dataset_id!r}"
        )
    path = data["path"]
    if not isinstance(path, str):
        raise ValueError("dataset.path must be a string")
    return DatasetRef(
        dataset_id=dataset_id,
        role=_string(data["role"], "dataset.role"),
        path=Path(path),
        content_hash=_string(data["content_hash"], "dataset.content_hash"),
        schema_hash=_string(data["schema_hash"], "dataset.schema_hash"),
        metadata=_json_mapping(data["metadata"], "dataset.metadata"),
    )


def _request_from_data(value: object) -> RepairRequest:
    data = _mapping(value, label="repair request", fields=_REQUEST_FIELDS)
    return RepairRequest(
        workspace_id=_string(data["workspace_id"], "request.workspace_id"),
        base_dataset_id=_validate_safe_id(
            _string(data["base_dataset_id"], "request.base_dataset_id"),
            "base dataset ID",
        ),
        production_cases_id=_validate_safe_id(
            _string(
                data["production_cases_id"],
                "request.production_cases_id",
            ),
            "production cases ID",
        ),
        policy_version=_string(data["policy_version"], "request.policy_version"),
        backend_name=_string(data["backend_name"], "request.backend_name"),
        requested_by=_string(data["requested_by"], "request.requested_by"),
        run_options=_json_mapping(data["run_options"], "request.run_options"),
    )


def _error_from_data(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    data = _mapping(value, label="run.error", fields=_ERROR_FIELDS)
    _string(data["code"], "run.error.code")
    _string(data["message"], "run.error.message")
    _string(data["stage"], "run.error.stage")
    if data["run_id"] is not None:
        _validate_safe_id(
            _string(data["run_id"], "run.error.run_id"),
            "error run ID",
        )
    if type(data["retryable"]) is not bool:
        raise ValueError("run.error.retryable must be a boolean")
    _json_mapping(data["details"], "run.error.details")
    if not isinstance(data["operator_action"], str):
        raise ValueError("run.error.operator_action must be a string")
    return data


def _run_from_data(value: object, expected_id: str) -> RunRecord:
    data = _mapping(value, label="run", fields=_RUN_FIELDS)
    run_id = _validate_safe_id(
        _string(data["run_id"], "run.run_id"),
        "run ID",
    )
    if run_id != expected_id:
        raise ValueError(
            f"Run record ID mismatch: expected {expected_id!r}, got {run_id!r}"
        )
    state = _string(data["state"], "run.state")
    try:
        RunState(state)
    except ValueError:
        raise ValueError(f"Unknown run state: {state!r}") from None
    version = data["version"]
    if type(version) is not int or version < 1:
        raise ValueError("run.version must be a positive integer")
    return RunRecord(
        run_id=run_id,
        state=state,
        version=version,
        request=_request_from_data(data["request"]),
        policy_hash=_string(data["policy_hash"], "run.policy_hash"),
        input_hash=_string(data["input_hash"], "run.input_hash"),
        metadata=_json_mapping(data["metadata"], "run.metadata"),
        error=_error_from_data(data["error"]),
    )


def _write_record_at(
    parent_descriptor: int,
    name: str,
    record: DatasetRef | RunRecord,
) -> None:
    _reject_symlink_leaf(parent_descriptor, name)
    _atomic_write_json_at(
        parent_descriptor,
        name,
        {
            "schema_version": _RECORD_SCHEMA_VERSION,
            "record": to_plain_data(record),
        },
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _timestamp(value: object, label: str) -> str:
    timestamp = _string(value, label)
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return timestamp


class JsonDatasetRepository:
    def __init__(self, root: Path) -> None:
        self._secure_root = _SecureRoot(root)
        self.root = self._secure_root.path

    def _record_path(self, dataset_id: str) -> Path:
        safe_id = _validate_safe_id(dataset_id, "dataset ID")
        return self.root / "datasets" / safe_id / "dataset.json"

    def save(self, dataset: DatasetRef) -> DatasetRef:
        if not isinstance(dataset, DatasetRef):
            raise TypeError("dataset must be a DatasetRef")
        _dataset_from_data(to_plain_data(dataset), dataset.dataset_id)
        datasets_descriptor = self._secure_root.open_dir(
            ("datasets",),
            create=True,
        )
        try:
            with _locked_file(datasets_descriptor, ".repository.lock"):
                try:
                    dataset_descriptor = self._secure_root.open_dir(
                        ("datasets", dataset.dataset_id),
                        create=True,
                        exclusive_last=True,
                    )
                except FileExistsError:
                    raise FileExistsError(
                        f"Dataset already exists: {dataset.dataset_id!r}"
                    ) from None
                try:
                    _write_record_at(
                        dataset_descriptor,
                        "dataset.json",
                        dataset,
                    )
                except BaseException:
                    os.close(dataset_descriptor)
                    try:
                        os.rmdir(dataset.dataset_id, dir_fd=datasets_descriptor)
                        _fsync_directory_fd(datasets_descriptor)
                    except OSError:
                        pass
                    raise
                else:
                    os.close(dataset_descriptor)
        finally:
            os.close(datasets_descriptor)
        return dataset

    def get(self, dataset_id: str) -> DatasetRef:
        safe_id = _validate_safe_id(dataset_id, "dataset ID")
        dataset_descriptor = self._secure_root.open_dir(
            ("datasets", safe_id),
            create=False,
        )
        try:
            payload = _record_payload(
                _strict_json_at(dataset_descriptor, "dataset.json"),
                "dataset",
            )
        finally:
            os.close(dataset_descriptor)
        return _dataset_from_data(payload, dataset_id)


class DatasetRepository(Protocol):
    def get(self, dataset_id: str) -> DatasetRef: ...


class RunRepository(Protocol):
    def create(
        self,
        request: RepairRequest,
        policy_hash: str,
        input_hash: str,
    ) -> RunRecord: ...

    def get(self, run_id: str) -> RunRecord: ...

    def save(self, record: RunRecord) -> RunRecord: ...

    def transition(
        self,
        run_id: str,
        state: RunState,
        metadata: Mapping[str, object] | None = None,
    ) -> RunRecord: ...

    def fail(
        self,
        run_id: str,
        state: RunState,
        error: HeuriBoostError,
    ) -> RunRecord: ...


@dataclass(frozen=True)
class ResumeInspection:
    resumable: bool
    outcome: str
    reason: str

    def __post_init__(self) -> None:
        outcomes = frozenset({"resumable", "missing", "corrupt", "integrity", "io"})
        if self.outcome not in outcomes:
            raise ValueError(f"invalid resume inspection outcome: {self.outcome!r}")
        if self.resumable != (self.outcome == "resumable"):
            raise ValueError("resumable must match the resume inspection outcome")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("resume inspection reason must be a non-empty string")


class ArtifactStore(Protocol):
    root: Path

    def run_dir(self, run_id: str) -> Path: ...

    def complete_stage(
        self,
        run_id: str,
        stage: str,
        input_hash: str,
        artifacts: Mapping[str, Path],
    ) -> StageManifest: ...

    def inspect_resume(
        self,
        run_id: str,
        stage: str,
        input_hash: str,
    ) -> ResumeInspection: ...

    def load_completed_stage(
        self,
        run_id: str,
        stage: str,
        input_hash: str,
    ) -> StageManifest: ...

    def can_resume(self, run_id: str, stage: str, input_hash: str) -> bool: ...


@dataclass(frozen=True)
class OrchestratorStores:
    datasets: DatasetRepository
    runs: RunRepository
    artifacts: ArtifactStore


@dataclass(frozen=True)
class _PersistedStageManifest:
    manifest: StageManifest
    status: str
    duration_ms: int


@dataclass(frozen=True)
class _PreparedArtifact:
    ref: ArtifactRef
    source_descriptor: int
    source_metadata: os.stat_result
    source_path: Path


class JsonRunRepository:
    def __init__(self, root: Path) -> None:
        self._secure_root = _SecureRoot(root)
        self.root = self._secure_root.path

    def _record_path(self, run_id: str) -> Path:
        safe_id = _validate_safe_id(run_id, "run ID")
        return self.root / "runs" / safe_id / "run.json"

    @contextmanager
    def _write_lock(self) -> Iterator[int]:
        runs_descriptor = self._secure_root.open_dir(("runs",), create=True)
        try:
            with _locked_file(runs_descriptor, ".repository.lock"):
                yield runs_descriptor
        finally:
            os.close(runs_descriptor)

    def _read(self, run_id: str) -> RunRecord:
        safe_id = _validate_safe_id(run_id, "run ID")
        run_descriptor = self._secure_root.open_dir(
            ("runs", safe_id),
            create=False,
        )
        try:
            payload = _record_payload(
                _strict_json_at(run_descriptor, "run.json"),
                "run",
            )
        finally:
            os.close(run_descriptor)
        return _run_from_data(payload, run_id)

    @staticmethod
    def _assert_immutable_fields(stored: RunRecord, proposed: RunRecord) -> None:
        immutable_fields = (
            "run_id",
            "request",
            "policy_hash",
            "input_hash",
        )
        changed = [
            field_name
            for field_name in immutable_fields
            if getattr(stored, field_name) != getattr(proposed, field_name)
        ]
        if changed:
            raise ValueError(
                "Run immutable fields cannot change: " + ", ".join(changed)
            )

    def _save_locked(self, record: RunRecord) -> RunRecord:
        try:
            stored = self._read(record.run_id)
        except FileNotFoundError:
            raise ValueError(f"Unknown run record: {record.run_id!r}") from None
        self._assert_immutable_fields(stored, record)
        if record.version != stored.version:
            raise ValueError(
                f"Rejecting stale run version {record.version}; "
                f"stored version is {stored.version}"
            )
        if record == stored:
            raise ValueError("Run save contains no changes")
        if record.state != stored.state:
            assert_transition(stored.state, record.state)
        saved = replace(record, version=stored.version + 1)
        _run_from_data(to_plain_data(saved), saved.run_id)
        run_descriptor = self._secure_root.open_dir(
            ("runs", record.run_id),
            create=False,
        )
        try:
            _write_record_at(run_descriptor, "run.json", saved)
        finally:
            os.close(run_descriptor)
        return saved

    def create(
        self,
        request: RepairRequest,
        policy_hash: str,
        input_hash: str,
    ) -> RunRecord:
        if not isinstance(request, RepairRequest):
            raise TypeError("request must be a RepairRequest")
        _request_from_data(to_plain_data(request))
        _string(policy_hash, "policy_hash")
        _string(input_hash, "input_hash")
        with self._write_lock() as runs_descriptor:
            while True:
                run_id = f"run-{uuid.uuid4().hex}"
                try:
                    run_descriptor = self._secure_root.open_dir(
                        ("runs", run_id),
                        create=True,
                        exclusive_last=True,
                    )
                except FileExistsError:
                    continue
                break
            record = RunRecord(
                run_id=run_id,
                state=RunState.RECEIVED.value,
                version=1,
                request=request,
                policy_hash=policy_hash,
                input_hash=input_hash,
            )
            try:
                _write_record_at(run_descriptor, "run.json", record)
            except BaseException:
                os.close(run_descriptor)
                try:
                    os.rmdir(run_id, dir_fd=runs_descriptor)
                    _fsync_directory_fd(runs_descriptor)
                except OSError:
                    pass
                raise
            else:
                os.close(run_descriptor)
            return record

    def get(self, run_id: str) -> RunRecord:
        return self._read(run_id)

    def save(self, record: RunRecord) -> RunRecord:
        if not isinstance(record, RunRecord):
            raise TypeError("record must be a RunRecord")
        with self._write_lock():
            return self._save_locked(record)

    def transition(
        self,
        run_id: str,
        state: RunState,
        metadata: Mapping[str, object] | None = None,
    ) -> RunRecord:
        target = state if isinstance(state, RunState) else RunState(state)
        update = {} if metadata is None else _json_mapping(metadata, "metadata")
        with self._write_lock():
            stored = self._read(run_id)
            merged_metadata = {
                **to_plain_data(stored.metadata),
                **to_plain_data(update),
            }
            return self._save_locked(
                replace(
                    stored,
                    state=target.value,
                    metadata=merged_metadata,
                )
            )

    def fail(
        self,
        run_id: str,
        state: RunState,
        error: HeuriBoostError,
    ) -> RunRecord:
        if not isinstance(error, HeuriBoostError):
            raise TypeError("error must be a HeuriBoostError")
        target = state if isinstance(state, RunState) else RunState(state)
        if not (
            target.value.startswith("BLOCKED_")
            or target.value.endswith("FAILED")
            or target is RunState.FAILED_INTERNAL
        ):
            raise ValueError(f"fail target must be BLOCKED or FAILED: {target.value}")
        if error.code != target.value:
            raise ValueError(
                f"Error code {error.code!r} does not match failure state "
                f"{target.value!r}"
            )
        with self._write_lock():
            stored = self._read(run_id)
            return self._save_locked(
                replace(
                    stored,
                    state=target.value,
                    error=error.to_dict(),
                )
            )


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self._secure_root = _SecureRoot(root)
        self.root = self._secure_root.path

    def run_dir(self, run_id: str) -> Path:
        safe_id = _validate_safe_id(run_id, "run ID")
        descriptor = self._secure_root.open_dir(
            ("runs", safe_id),
            create=True,
        )
        os.close(descriptor)
        return self._secure_root.path_for("runs", safe_id)

    def _manifest_path(self, run_id: str, stage: str) -> Path:
        safe_stage = _validate_safe_id(stage, "stage")
        safe_run_id = _validate_safe_id(run_id, "run ID")
        return self._secure_root.path_for(
            "runs",
            safe_run_id,
            "stages",
            safe_stage,
            "stage_manifest.json",
        )

    def _artifact_parts(self, run_id: str, path: Path) -> tuple[str, ...]:
        safe_run_id = _validate_safe_id(run_id, "run ID")
        candidate = Path(path)
        if candidate.is_absolute():
            absolute = Path(os.path.abspath(os.fspath(candidate)))
            try:
                relative = absolute.relative_to(self.root)
            except ValueError:
                raise ValueError(f"Artifact path escapes store root: {path}") from None
        else:
            relative = candidate
        parts = relative.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"Artifact path is unsafe: {path}")
        if parts[:2] != ("runs", safe_run_id) or len(parts) < 3:
            raise ValueError(
                f"Artifact must reside under run {safe_run_id!r}: {path}"
            )
        return tuple(parts)

    def _open_artifact_source(
        self,
        run_id: str,
        path: Path,
    ) -> tuple[int, Path]:
        candidate = Path(path)
        source_path = candidate
        if candidate.is_absolute():
            absolute = Path(os.path.abspath(os.fspath(candidate)))
            try:
                relative = absolute.relative_to(self.root)
            except ValueError:
                if not absolute.name:
                    raise ValueError(f"Artifact path must name a file: {path}")
                parent_descriptor = _open_directory_path(
                    absolute.parent,
                    create=False,
                )
                try:
                    descriptor = _open_regular_file(
                        parent_descriptor,
                        absolute.name,
                    )
                finally:
                    os.close(parent_descriptor)
                return descriptor, absolute
            source_path = absolute
        else:
            relative = candidate
        parts = relative.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"Artifact path is unsafe: {path}")
        if parts[0] == "runs":
            parts = self._artifact_parts(run_id, relative)
        parent_descriptor = self._secure_root.open_dir(parts[:-1], create=False)
        try:
            descriptor = _open_regular_file(parent_descriptor, parts[-1])
        finally:
            os.close(parent_descriptor)
        return descriptor, source_path

    def _prepare_artifact(
        self,
        run_id: str,
        stage: str,
        artifact_type: str,
        path: Path,
    ) -> _PreparedArtifact:
        descriptor, source_path = self._open_artifact_source(run_id, path)
        try:
            before = os.fstat(descriptor)
            first_digest, first_size = _hash_open_file(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            second_digest, second_size = _hash_open_file(descriptor)
            self._assert_stable_artifact(
                before,
                os.fstat(descriptor),
                first_digest,
                first_size,
                second_digest,
                second_size,
                run_id=run_id,
                stage=stage,
                artifact_type=artifact_type,
                path=path,
            )
            snapshot_name = f"{artifact_type}-{second_digest}.snapshot"
            return _PreparedArtifact(
                ref=ArtifactRef(
                    artifact_type=artifact_type,
                    path=Path(
                        "runs",
                        run_id,
                        "stages",
                        stage,
                        "artifacts",
                        snapshot_name,
                    ),
                    content_hash=second_digest,
                    size_bytes=second_size,
                ),
                source_descriptor=descriptor,
                source_metadata=before,
                source_path=source_path,
            )
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _assert_stable_artifact(
        before: os.stat_result,
        after: os.stat_result,
        first_digest: str,
        first_size: int,
        second_digest: str,
        second_size: int,
        *,
        run_id: str,
        stage: str,
        artifact_type: str,
        path: Path,
    ) -> None:
        snapshot_fields = ("st_dev", "st_ino", "st_size")
        if (
            first_digest != second_digest
            or first_size != second_size
            or second_size != before.st_size
            or any(
                getattr(before, name) != getattr(after, name)
                for name in snapshot_fields
            )
        ):
            raise ArtifactIntegrityError(
                "artifact changed while it was being hashed",
                stage=stage,
                run_id=run_id,
                details={"artifact_type": artifact_type, "path": str(path)},
            )

    def _publish_artifact_snapshot(
        self,
        prepared: _PreparedArtifact,
        parent_descriptor: int,
        *,
        run_id: str,
        stage: str,
    ) -> None:
        artifact = prepared.ref
        source_descriptor = prepared.source_descriptor
        name = artifact.path.name
        temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
        temporary_exists = False
        destination_descriptor = -1
        try:
            destination_descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | os.O_NOFOLLOW,
                0o400,
                dir_fd=parent_descriptor,
            )
            temporary_exists = True
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            size_bytes = 0
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size_bytes += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_descriptor, view)
                    view = view[written:]
            copied_digest = digest.hexdigest()
            if (
                copied_digest != artifact.content_hash
                or size_bytes != artifact.size_bytes
            ):
                raise ArtifactIntegrityError(
                    "artifact changed while it was being snapshotted",
                    stage=stage,
                    run_id=run_id,
                    details={
                        "artifact_type": artifact.artifact_type,
                        "path": str(prepared.source_path),
                    },
                )
            self._assert_stable_artifact(
                prepared.source_metadata,
                os.fstat(source_descriptor),
                artifact.content_hash,
                artifact.size_bytes,
                copied_digest,
                size_bytes,
                run_id=run_id,
                stage=stage,
                artifact_type=artifact.artifact_type,
                path=prepared.source_path,
            )
            os.fsync(destination_descriptor)
            os.close(destination_descriptor)
            destination_descriptor = -1
            try:
                os.link(
                    temporary_name,
                    name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                try:
                    current = self._inspect_snapshot_at(
                        parent_descriptor,
                        artifact,
                        run_id=run_id,
                        stage=stage,
                    )
                except (OSError, ValueError) as exc:
                    raise ArtifactIntegrityError(
                        "content-addressed artifact snapshot is invalid",
                        stage=stage,
                        run_id=run_id,
                        details={"artifact_type": artifact.artifact_type},
                    ) from exc
                if current != artifact:
                    raise ArtifactIntegrityError(
                        "content-addressed artifact snapshot conflicts with existing file",
                        stage=stage,
                        run_id=run_id,
                        details={"artifact_type": artifact.artifact_type},
                    )
            os.unlink(temporary_name, dir_fd=parent_descriptor)
            temporary_exists = False
            _fsync_directory_fd(parent_descriptor)
        finally:
            if destination_descriptor >= 0:
                os.close(destination_descriptor)
            if temporary_exists:
                try:
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass

    def _inspect_snapshot_at(
        self,
        parent_descriptor: int,
        artifact: ArtifactRef,
        *,
        run_id: str,
        stage: str,
    ) -> ArtifactRef:
        descriptor = _open_regular_file(parent_descriptor, artifact.path.name)
        try:
            before = os.fstat(descriptor)
            first_digest, first_size = _hash_open_file(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            second_digest, second_size = _hash_open_file(descriptor)
            self._assert_stable_artifact(
                before,
                os.fstat(descriptor),
                first_digest,
                first_size,
                second_digest,
                second_size,
                run_id=run_id,
                stage=stage,
                artifact_type=artifact.artifact_type,
                path=artifact.path,
            )
        finally:
            os.close(descriptor)
        return ArtifactRef(
            artifact_type=artifact.artifact_type,
            path=artifact.path,
            content_hash=second_digest,
            size_bytes=second_size,
        )

    def _inspect_artifact(self, run_id: str, stage: str, artifact: ArtifactRef) -> ArtifactRef:
        parts = self._artifact_parts(run_id, artifact.path)
        parent_descriptor = self._secure_root.open_dir(parts[:-1], create=False)
        try:
            descriptor = _open_regular_file(parent_descriptor, parts[-1])
        finally:
            os.close(parent_descriptor)
        try:
            before = os.fstat(descriptor)
            first_digest, first_size = _hash_open_file(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            second_digest, second_size = _hash_open_file(descriptor)
            self._assert_stable_artifact(
                before,
                os.fstat(descriptor),
                first_digest,
                first_size,
                second_digest,
                second_size,
                run_id=run_id,
                stage=stage,
                artifact_type=artifact.artifact_type,
                path=artifact.path,
            )
        finally:
            os.close(descriptor)
        return ArtifactRef(
            artifact_type=artifact.artifact_type,
            path=artifact.path,
            content_hash=second_digest,
            size_bytes=second_size,
        )

    @staticmethod
    def _artifact_from_data(value: object) -> ArtifactRef:
        data = _mapping(value, label="artifact", fields=_ARTIFACT_FIELDS)
        artifact_type = _validate_safe_id(
            _string(data["artifact_type"], "artifact.artifact_type"),
            "artifact type",
        )
        path_value = data["path"]
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("artifact.path must be a non-empty string")
        path = Path(path_value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"artifact.path must be a safe relative path: {path}")
        content_hash = _string(data["content_hash"], "artifact.content_hash")
        if re.fullmatch(r"[0-9a-f]{64}", content_hash) is None:
            raise ValueError("artifact.content_hash must be a SHA-256 digest")
        size_bytes = data["size_bytes"]
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError("artifact.size_bytes must be a non-negative integer")
        return ArtifactRef(
            artifact_type=artifact_type,
            path=path,
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    @classmethod
    def _manifest_from_data(cls, value: object) -> _PersistedStageManifest:
        data = _mapping(value, label="stage manifest", fields=_MANIFEST_FIELDS)
        artifacts_data = data["artifacts"]
        if not isinstance(artifacts_data, list):
            raise ValueError("stage manifest artifacts must be a list")
        artifacts = tuple(cls._artifact_from_data(item) for item in artifacts_data)
        artifact_types = [artifact.artifact_type for artifact in artifacts]
        if artifact_types != sorted(artifact_types) or len(artifact_types) != len(
            set(artifact_types)
        ):
            raise ValueError("stage manifest artifacts must be uniquely sorted")
        started_at = _timestamp(data["started_at"], "stage manifest started_at")
        completed_at = _timestamp(
            data["completed_at"],
            "stage manifest completed_at",
        )
        if datetime.fromisoformat(completed_at) < datetime.fromisoformat(started_at):
            raise ValueError("stage manifest completed_at precedes started_at")
        stage = _validate_safe_id(
            _string(data["stage"], "stage manifest stage"),
            "stage",
        )
        status = _string(data["status"], "stage manifest status")
        duration_ms = data["duration_ms"]
        if type(duration_ms) is not int or duration_ms < 0:
            raise ValueError(
                "stage manifest duration_ms must be a non-negative integer"
            )
        return _PersistedStageManifest(
            manifest=StageManifest(
                stage=stage,
                input_hash=_string(
                    data["input_hash"],
                    "stage manifest input_hash",
                ),
                artifacts=artifacts,
                started_at=started_at,
                completed_at=completed_at,
            ),
            status=status,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _validate_stage_artifact_paths(
        run_id: str,
        stage: str,
        artifacts: tuple[ArtifactRef, ...],
    ) -> None:
        expected_parent = Path("runs", run_id, "stages", stage, "artifacts")
        for artifact in artifacts:
            if artifact.path.parent != expected_parent:
                raise ValueError(
                    "stage manifest artifact path is outside the declared stage: "
                    f"{artifact.path}"
                )

    def _read_existing_manifest(
        self,
        stage_descriptor: int,
        *,
        run_id: str,
        stage: str,
    ) -> _PersistedStageManifest | None:
        try:
            value = _strict_json_at(stage_descriptor, "stage_manifest.json")
        except FileNotFoundError:
            return None
        except (RecursionError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "existing stage manifest is invalid",
                stage=stage,
                run_id=run_id,
            ) from exc
        try:
            persisted = self._manifest_from_data(value)
            self._validate_stage_artifact_paths(
                run_id,
                stage,
                persisted.manifest.artifacts,
            )
            return persisted
        except (RecursionError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "existing stage manifest is invalid",
                stage=stage,
                run_id=run_id,
            ) from exc

    def _verify_existing_snapshots(
        self,
        stage_descriptor: int,
        artifacts: tuple[ArtifactRef, ...],
        *,
        run_id: str,
        stage: str,
    ) -> None:
        try:
            snapshot_descriptor = _open_child_directory(
                stage_descriptor,
                "artifacts",
                create=False,
            )
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "completed stage snapshot directory is unavailable",
                stage=stage,
                run_id=run_id,
            ) from exc
        try:
            for artifact in artifacts:
                try:
                    current = self._inspect_snapshot_at(
                        snapshot_descriptor,
                        artifact,
                        run_id=run_id,
                        stage=stage,
                    )
                except ArtifactIntegrityError:
                    raise
                except (OSError, ValueError) as exc:
                    raise ArtifactIntegrityError(
                        "completed stage artifact is unavailable or unsafe",
                        stage=stage,
                        run_id=run_id,
                        details={"artifact_type": artifact.artifact_type},
                    ) from exc
                if current != artifact:
                    raise ArtifactIntegrityError(
                        "completed stage artifact does not match its manifest",
                        stage=stage,
                        run_id=run_id,
                        details={"artifact_type": artifact.artifact_type},
                    )
        finally:
            os.close(snapshot_descriptor)

    def complete_stage(
        self,
        run_id: str,
        stage: str,
        input_hash: str,
        artifacts: Mapping[str, Path],
    ) -> StageManifest:
        _validate_safe_id(run_id, "run ID")
        _validate_safe_id(stage, "stage")
        _string(input_hash, "input_hash")
        if not isinstance(artifacts, MappingABC):
            raise TypeError("artifacts must be a mapping")
        stage_descriptor = self._secure_root.open_dir(
            ("runs", run_id, "stages", stage),
            create=True,
        )
        try:
            with _locked_file(stage_descriptor, ".stage.lock"):
                existing = self._read_existing_manifest(
                    stage_descriptor,
                    run_id=run_id,
                    stage=stage,
                )
                if existing is not None and existing.status != "COMPLETED":
                    raise ArtifactIntegrityError(
                        "existing stage manifest is not completed",
                        stage=stage,
                        run_id=run_id,
                        details={"status": existing.status},
                    )
                artifact_types = list(artifacts)
                for artifact_type in artifact_types:
                    _validate_safe_id(artifact_type, "artifact type")
                started_monotonic = time.monotonic_ns()
                started_at = datetime.now(timezone.utc).isoformat()

                if existing is not None:
                    candidate_refs = []
                    for artifact_type in sorted(artifact_types):
                        prepared = self._prepare_artifact(
                            run_id,
                            stage,
                            artifact_type,
                            artifacts[artifact_type],
                        )
                        try:
                            candidate_refs.append(prepared.ref)
                        finally:
                            os.close(prepared.source_descriptor)
                    candidate_refs = tuple(candidate_refs)
                    manifest = existing.manifest
                    if (
                        manifest.stage != stage
                        or manifest.input_hash != input_hash
                        or manifest.artifacts != candidate_refs
                    ):
                        raise ArtifactIntegrityError(
                            "stage completion conflicts with existing manifest",
                            stage=stage,
                            run_id=run_id,
                        )
                    self._verify_existing_snapshots(
                        stage_descriptor,
                        manifest.artifacts,
                        run_id=run_id,
                        stage=stage,
                    )
                    return manifest

                snapshot_descriptor = _open_child_directory(
                    stage_descriptor,
                    "artifacts",
                    create=True,
                )
                candidate_refs = []
                try:
                    for artifact_type in sorted(artifact_types):
                        prepared = self._prepare_artifact(
                            run_id,
                            stage,
                            artifact_type,
                            artifacts[artifact_type],
                        )
                        try:
                            self._publish_artifact_snapshot(
                                prepared,
                                snapshot_descriptor,
                                run_id=run_id,
                                stage=stage,
                            )
                            candidate_refs.append(prepared.ref)
                        finally:
                            os.close(prepared.source_descriptor)
                finally:
                    os.close(snapshot_descriptor)

                manifest = StageManifest(
                    stage=stage,
                    input_hash=input_hash,
                    artifacts=tuple(candidate_refs),
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                duration_ms = max(
                    0,
                    (time.monotonic_ns() - started_monotonic) // 1_000_000,
                )
                try:
                    _atomic_write_json_at(
                        stage_descriptor,
                        "stage_manifest.json",
                        {
                            **to_plain_data(manifest),
                            "status": "COMPLETED",
                            "duration_ms": duration_ms,
                        },
                        replace=False,
                    )
                except FileExistsError as exc:
                    raise ArtifactIntegrityError(
                        "stage manifest appeared during create-once publication",
                        stage=stage,
                        run_id=run_id,
                    ) from exc
                return manifest
        finally:
            os.close(stage_descriptor)

    def inspect_resume(
        self,
        run_id: str,
        stage: str,
        input_hash: str,
    ) -> ResumeInspection:
        _validate_safe_id(run_id, "run ID")
        _validate_safe_id(stage, "stage")
        _string(input_hash, "input_hash")
        try:
            stage_descriptor = self._secure_root.open_dir(
                ("runs", run_id, "stages", stage),
                create=False,
            )
        except FileNotFoundError:
            return ResumeInspection(
                resumable=False,
                outcome="missing",
                reason="stage manifest is missing",
            )
        except OSError as exc:
            return ResumeInspection(
                resumable=False,
                outcome="io",
                reason=f"unable to open stage manifest directory: {exc}",
            )
        try:
            try:
                persisted = self._manifest_from_data(
                    _strict_json_at(stage_descriptor, "stage_manifest.json")
                )
            except FileNotFoundError:
                return ResumeInspection(
                    resumable=False,
                    outcome="missing",
                    reason="stage manifest is missing",
                )
            except OSError as exc:
                return ResumeInspection(
                    resumable=False,
                    outcome="io",
                    reason=f"unable to read stage manifest: {exc}",
                )
            except (
                RecursionError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                return ResumeInspection(
                    resumable=False,
                    outcome="corrupt",
                    reason=f"stage manifest is invalid: {exc}",
                )
        finally:
            os.close(stage_descriptor)
        try:
            self._validate_stage_artifact_paths(
                run_id,
                stage,
                persisted.manifest.artifacts,
            )
        except ValueError as exc:
            return ResumeInspection(
                resumable=False,
                outcome="integrity",
                reason=f"stage manifest artifact path is invalid: {exc}",
            )
        if persisted.status != "COMPLETED":
            return ResumeInspection(
                resumable=False,
                outcome="integrity",
                reason="stage manifest is not completed",
            )
        manifest = persisted.manifest
        if manifest.stage != stage:
            return ResumeInspection(
                resumable=False,
                outcome="integrity",
                reason="stage manifest does not match the requested stage",
            )
        if manifest.input_hash != input_hash:
            return ResumeInspection(
                resumable=False,
                outcome="integrity",
                reason="stage manifest input hash does not match",
            )
        for artifact in manifest.artifacts:
            try:
                current = self._inspect_artifact(run_id, stage, artifact)
            except FileNotFoundError:
                return ResumeInspection(
                    resumable=False,
                    outcome="integrity",
                    reason=f"stage artifact is missing: {artifact.artifact_type}",
                )
            except (ArtifactIntegrityError, TypeError, ValueError) as exc:
                return ResumeInspection(
                    resumable=False,
                    outcome="integrity",
                    reason=f"stage artifact is invalid: {exc}",
                )
            except OSError as exc:
                return ResumeInspection(
                    resumable=False,
                    outcome="io",
                    reason=f"unable to inspect stage artifact: {exc}",
                )
            if current != artifact:
                return ResumeInspection(
                    resumable=False,
                    outcome="integrity",
                    reason=f"stage artifact does not match its manifest: {artifact.artifact_type}",
                )
        return ResumeInspection(
            resumable=True,
            outcome="resumable",
            reason="stage manifest and artifacts are valid",
        )

    def load_completed_stage(
        self,
        run_id: str,
        stage: str,
        input_hash: str,
    ) -> StageManifest:
        """Return a verified completed manifest with root-relative artifact refs.

        Callers must rebase each returned ``ArtifactRef.path`` against ``root``
        before handing it to a backend.  The relative paths are intentional: they
        keep the sealed manifest portable while preserving the storage boundary.
        """

        _validate_safe_id(run_id, "run ID")
        _validate_safe_id(stage, "stage")
        _string(input_hash, "input_hash")
        try:
            stage_descriptor = self._secure_root.open_dir(
                ("runs", run_id, "stages", stage),
                create=False,
            )
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "completed stage is unavailable",
                stage=stage,
                run_id=run_id,
            ) from exc
        try:
            with _locked_file(stage_descriptor, ".stage.lock"):
                persisted = self._read_existing_manifest(
                    stage_descriptor,
                    run_id=run_id,
                    stage=stage,
                )
                if persisted is None:
                    raise ArtifactIntegrityError(
                        "completed stage manifest is missing",
                        stage=stage,
                        run_id=run_id,
                    )
                if persisted.status != "COMPLETED":
                    raise ArtifactIntegrityError(
                        "completed stage manifest has invalid status",
                        stage=stage,
                        run_id=run_id,
                        details={"status": persisted.status},
                    )
                manifest = persisted.manifest
                if manifest.stage != stage:
                    raise ArtifactIntegrityError(
                        "completed stage manifest does not match the requested stage",
                        stage=stage,
                        run_id=run_id,
                    )
                if manifest.input_hash != input_hash:
                    raise ArtifactIntegrityError(
                        "completed stage manifest input hash does not match",
                        stage=stage,
                        run_id=run_id,
                    )
                self._verify_existing_snapshots(
                    stage_descriptor,
                    manifest.artifacts,
                    run_id=run_id,
                    stage=stage,
                )
                return manifest
        except ArtifactIntegrityError:
            raise
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "completed stage verification failed",
                stage=stage,
                run_id=run_id,
            ) from exc
        finally:
            os.close(stage_descriptor)

    def can_resume(self, run_id: str, stage: str, input_hash: str) -> bool:
        return self.inspect_resume(run_id, stage, input_hash).resumable
