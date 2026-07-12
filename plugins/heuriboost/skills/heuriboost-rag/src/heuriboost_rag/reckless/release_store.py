"""Immutable local release storage and atomic current-model activation."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any
import uuid

from .contracts import (
    ActivationResult,
    ArtifactRef,
    PromotionApproval,
    PromotionReceipt,
    ReleaseSnapshot,
    RollbackReceipt,
    RunRecord,
    to_plain_data,
)
from .errors import ArtifactIntegrityError, PromotionConflictError
from .hashing import canonical_json_hash


_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_JSON_SCHEMA_VERSION = 1
_COPY_CHUNK_SIZE = 1024 * 1024
_RELEASE_MANIFEST_NAME = "release_manifest.json"
_PROMOTION_RECEIPT_NAME = "promotion_receipt.json"
_PROMOTION_RECEIPT_HTML_NAME = "promotion_receipt.html"
_LEGACY_MIGRATION_MARKER_NAME = "legacy_migration.json"
_LEGACY_MIGRATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BootstrapReleaseSnapshot:
    """Verified legacy state ready to become the first immutable release."""

    run_id: str
    artifacts: tuple[ArtifactRef, ...]
    content_hash: str

    def __post_init__(self) -> None:
        _safe_component(self.run_id, "bootstrap run ID")
        if not self.artifacts:
            raise ValueError("bootstrap release requires at least one artifact")
        if len({artifact.artifact_type for artifact in self.artifacts}) != len(
            self.artifacts
        ):
            raise ValueError("bootstrap release artifact types must be unique")
        if re.fullmatch(r"[a-f0-9]{64}", self.content_hash) is None:
            raise ValueError("bootstrap release content_hash must be a SHA-256 digest")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str, label: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not (float("-inf") < parsed < float("inf")):
        raise ValueError("non-finite JSON number")
    return parsed


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        to_plain_data(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_create_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_regular_file(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"not a regular file: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode):
            raise ValueError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK_SIZE)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _json_from_file(path: Path) -> Mapping[str, object]:
    raw = _read_regular_file(path)
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
        parse_float=_strict_float,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _receipt_from_data(value: Mapping[str, object]) -> tuple[PromotionReceipt, str]:
    required = {
        "schema_version",
        "receipt_type",
        "idempotency_key",
        "run_id",
        "release_path",
        "promoted_at",
        "approved_by",
        "previous_model",
        "current_model",
        "release_manifest_hash",
        "receipt_json_path",
        "receipt_html_path",
        "report_hash",
        "decision_hash",
        "model_hash",
        "schema_hash",
    }
    if set(value) != required:
        raise ValueError("promotion receipt has an invalid schema")
    if value["schema_version"] != _JSON_SCHEMA_VERSION:
        raise ValueError("promotion receipt schema is unsupported")
    if value["receipt_type"] != "promotion":
        raise ValueError("release receipt is not a promotion receipt")
    previous = value["previous_model"]
    if previous is not None and not isinstance(previous, str):
        raise ValueError("promotion receipt previous_model is invalid")
    strings = (
        "idempotency_key",
        "run_id",
        "release_path",
        "promoted_at",
        "approved_by",
        "current_model",
        "release_manifest_hash",
        "receipt_json_path",
        "receipt_html_path",
    )
    if any(not isinstance(value[key], str) or not value[key] for key in strings):
        raise ValueError("promotion receipt has an invalid string field")
    return (
        PromotionReceipt(
            run_id=value["run_id"],
            release_path=Path(value["release_path"]),
            promoted_at=value["promoted_at"],
            approved_by=value["approved_by"],
            previous_model=previous,
            current_model=value["current_model"],
            release_manifest_hash=value["release_manifest_hash"],
            receipt_json_path=Path(value["receipt_json_path"]),
            receipt_html_path=Path(value["receipt_html_path"]),
        ),
        value["idempotency_key"],
    )


def _receipt_html(title: str, payload: Mapping[str, object]) -> bytes:
    json_text = _json_bytes(payload).decode("utf-8").replace("<", "\\u003c")
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>body{{margin:0;background:#f7f7f8;color:#111;font-family:Helvetica Neue,Helvetica,Arial,sans-serif}}main{{max-width:960px;margin:0 auto;background:#fff;border-left:1px solid #111;border-right:1px solid #111;min-height:100vh;padding:32px}}h1{{margin:0 0 24px;font-size:30px}}pre{{margin:0;padding:16px;border:1px solid #111;background:#f7f7f8;white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.5}}</style></head>
<body><main><h1>{title}</h1><pre id="receipt"></pre></main><script id="heuriboost-receipt-data" type="application/json">{json_text}</script><script>document.getElementById("receipt").textContent=JSON.stringify(JSON.parse(document.getElementById("heuriboost-receipt-data").textContent),null,2);</script></body></html>"""
    return html.encode("utf-8")


def _rollback_from_data(value: Mapping[str, object]) -> RollbackReceipt:
    required = {
        "schema_version",
        "receipt_type",
        "source_run_id",
        "rolled_back_at",
        "approved_by",
        "previous_model",
        "restored_model",
        "receipt_json_path",
        "receipt_html_path",
        "source_release_manifest_hash",
        "target_metadata",
    }
    if set(value) != required:
        raise ValueError("rollback receipt has an invalid schema")
    if value["schema_version"] != _JSON_SCHEMA_VERSION or value["receipt_type"] != "rollback":
        raise ValueError("rollback receipt is unsupported")
    strings = required - {"schema_version", "receipt_type", "target_metadata"}
    if any(not isinstance(value[key], str) or not value[key] for key in strings):
        raise ValueError("rollback receipt has an invalid string field")
    if not isinstance(value["target_metadata"], Mapping):
        raise ValueError("rollback receipt target_metadata is invalid")
    return RollbackReceipt(
        source_run_id=value["source_run_id"],
        rolled_back_at=value["rolled_back_at"],
        approved_by=value["approved_by"],
        previous_model=value["previous_model"],
        restored_model=value["restored_model"],
        receipt_json_path=Path(value["receipt_json_path"]),
        receipt_html_path=Path(value["receipt_html_path"]),
    )


class FileReleaseStore:
    """Owns immutable release publication and the mutable current-model pointer."""

    def __init__(
        self,
        root: Path,
        *,
        before_pointer_swap: Callable[[], None] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._releases_dir = self.root / "releases"
        self._releases_dir.mkdir(exist_ok=True)
        self._before_pointer_swap = before_pointer_swap

    @property
    def current_pointer_path(self) -> Path:
        return self.root / "current_model.json"

    @contextmanager
    def workspace_lock(self) -> Iterator[None]:
        lock_path = self.root / ".release.lock"
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _pointer_data(
        self,
        current_model: str,
        *,
        source_run_id: str,
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "schema_version": _JSON_SCHEMA_VERSION,
            "current_model": current_model,
            "source_run_id": source_run_id,
            "metadata": to_plain_data(metadata),
        }

    def read_current_model(self) -> str | None:
        try:
            data = _json_from_file(self.current_pointer_path)
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                "current model pointer is invalid",
                stage="PROMOTING",
            ) from exc
        required = {"schema_version", "current_model", "source_run_id", "metadata"}
        if set(data) != required or data["schema_version"] != _JSON_SCHEMA_VERSION:
            raise ArtifactIntegrityError(
                "current model pointer has an invalid schema",
                stage="PROMOTING",
            )
        current = data["current_model"]
        if not isinstance(current, str) or not current:
            raise ArtifactIntegrityError(
                "current model pointer has an invalid model reference",
                stage="PROMOTING",
            )
        return current

    def bootstrap_current_model(self, current_model: str | None) -> None:
        """Create the initial pointer once for tests, bootstrap, or migration."""

        with self.workspace_lock():
            existing = self.read_current_model()
            if existing is not None:
                if existing != current_model:
                    raise PromotionConflictError(
                        "current model pointer is already initialized",
                        stage="PROMOTING",
                        details={"existing": existing, "requested": current_model},
                    )
                return
            if current_model is None:
                return
            _safe_component(current_model, "current model")
            _atomic_replace_bytes(
                self.current_pointer_path,
                _json_bytes(
                    self._pointer_data(
                        current_model,
                        source_run_id="bootstrap",
                        metadata={"source": "bootstrap"},
                    )
                ),
            )

    def _write_current_model(
        self,
        current_model: str,
        *,
        source_run_id: str,
        metadata: Mapping[str, object],
    ) -> None:
        _safe_component(current_model, "current model")
        _atomic_replace_bytes(
            self.current_pointer_path,
            _json_bytes(
                self._pointer_data(
                    current_model,
                    source_run_id=source_run_id,
                    metadata=metadata,
                )
            ),
        )

    def _release_dir(self, run_id: str) -> Path:
        return self._releases_dir / _safe_component(run_id, "run ID")

    @property
    def legacy_migration_marker_path(self) -> Path:
        return self.root / _LEGACY_MIGRATION_MARKER_NAME

    @staticmethod
    def _legacy_marker_data(
        snapshot: BootstrapReleaseSnapshot,
        source: str,
    ) -> dict[str, object]:
        return {
            "schema_version": _LEGACY_MIGRATION_SCHEMA_VERSION,
            "source": source,
            "snapshot_hash": snapshot.content_hash,
            "run_id": snapshot.run_id,
        }

    def _read_legacy_marker(self) -> Mapping[str, object] | None:
        try:
            data = _json_from_file(self.legacy_migration_marker_path)
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                "legacy migration marker is invalid",
                stage="MIGRATION",
            ) from exc
        required = {"schema_version", "source", "snapshot_hash", "run_id"}
        if set(data) != required or data["schema_version"] != _LEGACY_MIGRATION_SCHEMA_VERSION:
            raise ArtifactIntegrityError(
                "legacy migration marker has an invalid schema",
                stage="MIGRATION",
            )
        try:
            _safe_component(data["source"], "migration source")
            _safe_component(data["run_id"], "bootstrap run ID")
        except ValueError as exc:
            raise ArtifactIntegrityError(
                "legacy migration marker has unsafe identifiers",
                stage="MIGRATION",
            ) from exc
        if not isinstance(data["snapshot_hash"], str) or re.fullmatch(
            r"[a-f0-9]{64}", data["snapshot_hash"]
        ) is None:
            raise ArtifactIntegrityError(
                "legacy migration marker has an invalid snapshot hash",
                stage="MIGRATION",
            )
        return data

    def legacy_migration_complete(self) -> bool:
        """Return whether migration is durable and a valid package release is active."""

        marker = self._read_legacy_marker()
        if marker is None:
            return False
        receipt_data = self.read_promotion_receipt(str(marker["run_id"]))
        if receipt_data is None:
            raise ArtifactIntegrityError(
                "legacy migration marker has no durable bootstrap receipt",
                stage="MIGRATION",
            )
        receipt, _ = receipt_data
        current = self.read_current_model()
        if current is None:
            return False
        if current == receipt.current_model:
            return True
        current_receipt = self.read_promotion_receipt(current)
        return (
            current_receipt is not None
            and current_receipt[0].current_model == current
        )

    def _verify_published_release(self, receipt: PromotionReceipt) -> None:
        expected_release = self._release_dir(receipt.run_id)
        if receipt.release_path != expected_release:
            raise ArtifactIntegrityError(
                "promotion receipt points outside the expected release directory",
                stage="PROMOTING",
                run_id=receipt.run_id,
            )
        manifest_path = expected_release / _RELEASE_MANIFEST_NAME
        try:
            raw = _read_regular_file(manifest_path)
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "release manifest is unavailable",
                stage="PROMOTING",
                run_id=receipt.run_id,
            ) from exc
        if _hash_bytes(raw) != receipt.release_manifest_hash:
            raise ArtifactIntegrityError(
                "release manifest no longer matches the promotion receipt",
                stage="PROMOTING",
                run_id=receipt.run_id,
            )
        try:
            manifest = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
                parse_float=_strict_float,
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                "release manifest is not strict JSON",
                stage="PROMOTING",
                run_id=receipt.run_id,
            ) from exc
        if not isinstance(manifest, Mapping) or manifest.get("run_id") != receipt.run_id:
            raise ArtifactIntegrityError(
                "release manifest belongs to a different run",
                stage="PROMOTING",
                run_id=receipt.run_id,
            )
        raw_artifacts = manifest.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ArtifactIntegrityError(
                "release manifest has no artifact index",
                stage="PROMOTING",
                run_id=receipt.run_id,
            )
        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, Mapping):
                raise ArtifactIntegrityError(
                    "release manifest contains an invalid artifact record",
                    stage="PROMOTING",
                    run_id=receipt.run_id,
                )
            path_value = raw_artifact.get("path")
            content_hash = raw_artifact.get("content_hash")
            size_bytes = raw_artifact.get("size_bytes")
            if (
                not isinstance(path_value, str)
                or not isinstance(content_hash, str)
                or not content_hash
                or type(size_bytes) is not int
                or size_bytes < 0
            ):
                raise ArtifactIntegrityError(
                    "release manifest contains invalid artifact fields",
                    stage="PROMOTING",
                    run_id=receipt.run_id,
                )
            path = Path(path_value)
            try:
                path.relative_to(expected_release)
            except ValueError as exc:
                raise ArtifactIntegrityError(
                    "release manifest artifact escapes its release directory",
                    stage="PROMOTING",
                    run_id=receipt.run_id,
                ) from exc
            try:
                artifact_bytes = _read_regular_file(path)
            except (OSError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    "published release artifact cannot be read",
                    stage="PROMOTING",
                    run_id=receipt.run_id,
                ) from exc
            if _hash_bytes(artifact_bytes) != content_hash or len(artifact_bytes) != size_bytes:
                raise ArtifactIntegrityError(
                    "published release artifact no longer matches the manifest",
                    stage="PROMOTING",
                    run_id=receipt.run_id,
                )

    def read_promotion_receipt(
        self,
        run_id: str,
    ) -> tuple[PromotionReceipt, str] | None:
        path = self._release_dir(run_id) / _PROMOTION_RECEIPT_NAME
        try:
            data = _json_from_file(path)
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                "existing release receipt is invalid",
                stage="PROMOTING",
                run_id=run_id,
            ) from exc
        try:
            receipt, key = _receipt_from_data(data)
        except ValueError as exc:
            raise ArtifactIntegrityError(
                "existing release receipt has an invalid schema",
                stage="PROMOTING",
                run_id=run_id,
            ) from exc
        if receipt.run_id != run_id:
            raise ArtifactIntegrityError(
                "existing release receipt belongs to a different run",
                stage="PROMOTING",
                run_id=run_id,
            )
        self._verify_published_release(receipt)
        return receipt, key

    def _pending_rollback(
        self,
        receipt: PromotionReceipt,
    ) -> RollbackReceipt | None:
        rollback_dir = receipt.release_path / "rollback_receipts"
        try:
            paths = sorted(rollback_dir.glob("rollback-*.json"), reverse=True)
        except OSError as exc:
            raise ArtifactIntegrityError(
                "rollback receipt directory cannot be inspected",
                stage="PROMOTING",
                run_id=receipt.run_id,
            ) from exc
        for path in paths:
            try:
                data = _json_from_file(path)
                rollback = _rollback_from_data(data)
            except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                raise ArtifactIntegrityError(
                    "rollback receipt is invalid",
                    stage="PROMOTING",
                    run_id=receipt.run_id,
                ) from exc
            if (
                rollback.source_run_id == receipt.run_id
                and rollback.previous_model == receipt.current_model
                and rollback.restored_model == receipt.previous_model
            ):
                return rollback
        return None

    @staticmethod
    def _copy_artifact(
        artifact: ArtifactRef,
        destination: Path,
        *,
        run_id: str,
    ) -> ArtifactRef:
        source = Path(artifact.path)
        try:
            source_metadata = source.lstat()
        except OSError as exc:
            raise ArtifactIntegrityError(
                "declared release artifact cannot be inspected",
                stage="PROMOTING",
                run_id=run_id,
                details={"artifact_type": artifact.artifact_type},
            ) from exc
        if not stat.S_ISREG(source_metadata.st_mode):
            raise ArtifactIntegrityError(
                "declared release artifact is not a regular file",
                stage="PROMOTING",
                run_id=run_id,
                details={"artifact_type": artifact.artifact_type},
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_descriptor = -1
        destination_descriptor = -1
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            source_descriptor = os.open(
                source,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            current_metadata = os.fstat(source_descriptor)
            if not stat.S_ISREG(current_metadata.st_mode):
                raise ValueError("artifact is not a regular file")
            destination_descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            while True:
                chunk = os.read(source_descriptor, _COPY_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                size_bytes += len(chunk)
                os.write(destination_descriptor, chunk)
            os.fsync(destination_descriptor)
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "declared release artifact cannot be copied",
                stage="PROMOTING",
                run_id=run_id,
                details={"artifact_type": artifact.artifact_type},
            ) from exc
        finally:
            if source_descriptor >= 0:
                os.close(source_descriptor)
            if destination_descriptor >= 0:
                os.close(destination_descriptor)
        actual_hash = digest.hexdigest()
        if actual_hash != artifact.content_hash or size_bytes != artifact.size_bytes:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
            raise ArtifactIntegrityError(
                "declared release artifact changed while being copied",
                stage="PROMOTING",
                run_id=run_id,
                details={"artifact_type": artifact.artifact_type},
            )
        return ArtifactRef(
            artifact_type=artifact.artifact_type,
            path=destination,
            content_hash=actual_hash,
            size_bytes=size_bytes,
        )

    def _copy_release_artifacts(
        self,
        snapshot: ReleaseSnapshot,
        staging: Path,
    ) -> tuple[ArtifactRef, ...]:
        types = [artifact.artifact_type for artifact in snapshot.artifacts]
        if len(types) != len(set(types)):
            raise ArtifactIntegrityError(
                "release snapshot contains duplicate artifact types",
                stage="PROMOTING",
                run_id=snapshot.run_id,
            )
        copied = []
        for artifact in sorted(snapshot.artifacts, key=lambda value: value.artifact_type):
            artifact_type = _safe_component(artifact.artifact_type, "artifact type")
            filename = _safe_component(Path(artifact.path).name, "artifact filename")
            destination = staging / "artifacts" / artifact_type / filename
            copied.append(
                self._copy_artifact(
                    artifact,
                    destination,
                    run_id=snapshot.run_id,
                )
            )
        return tuple(copied)

    @staticmethod
    def _published_artifacts(
        staging: Path,
        release_dir: Path,
        artifacts: Sequence[ArtifactRef],
        *,
        run_id: str,
    ) -> tuple[ArtifactRef, ...]:
        published = []
        for artifact in artifacts:
            try:
                relative = artifact.path.relative_to(staging)
            except ValueError as exc:
                raise ArtifactIntegrityError(
                    "staged release artifact escapes its staging directory",
                    stage="PROMOTING",
                    run_id=run_id,
                    details={"artifact_type": artifact.artifact_type},
                ) from exc
            published.append(
                ArtifactRef(
                    artifact_type=artifact.artifact_type,
                    path=release_dir / relative,
                    content_hash=artifact.content_hash,
                    size_bytes=artifact.size_bytes,
                )
            )
        return tuple(published)

    @staticmethod
    def _verify_artifacts(artifacts: Sequence[ArtifactRef], *, run_id: str) -> None:
        for artifact in artifacts:
            try:
                raw = _read_regular_file(artifact.path)
            except (OSError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    "release artifact cannot be verified",
                    stage="PROMOTING",
                    run_id=run_id,
                    details={"artifact_type": artifact.artifact_type},
                ) from exc
            if _hash_bytes(raw) != artifact.content_hash or len(raw) != artifact.size_bytes:
                raise ArtifactIntegrityError(
                    "release artifact hash does not match",
                    stage="PROMOTING",
                    run_id=run_id,
                    details={"artifact_type": artifact.artifact_type},
                )

    def _release_manifest(
        self,
        *,
        run: RunRecord,
        approval: PromotionApproval,
        snapshot: ReleaseSnapshot,
        copied: Sequence[ArtifactRef],
        prepared: PreparedActivation,
        activation: ActivationResult,
        report_hash: str,
        decision_hash: str,
        model_hash: str,
        schema_hash: str,
    ) -> dict[str, object]:
        return {
            "schema_version": _JSON_SCHEMA_VERSION,
            "run_id": run.run_id,
            "previous_model": snapshot.previous_model,
            "input_hash": run.input_hash,
            "policy_hash": run.policy_hash,
            "report_hash": report_hash,
            "decision_hash": decision_hash,
            "model_hash": model_hash,
            "schema_hash": schema_hash,
            "snapshot_manifest_hash": snapshot.manifest_hash,
            "artifacts": [to_plain_data(artifact) for artifact in copied],
            "approval": to_plain_data(approval),
            "target": {
                "name": getattr(prepared, "target_name", None),
                "pointer_payload": to_plain_data(prepared.pointer_payload),
                "prepared_metadata": to_plain_data(prepared.metadata),
                "activation_metadata": to_plain_data(activation.metadata),
            },
        }

    def _promotion_receipt_data(
        self,
        receipt: PromotionReceipt,
        *,
        idempotency_key: str,
        report_hash: str,
        decision_hash: str,
        model_hash: str,
        schema_hash: str,
    ) -> dict[str, object]:
        return {
            "schema_version": _JSON_SCHEMA_VERSION,
            "receipt_type": "promotion",
            "idempotency_key": idempotency_key,
            "run_id": receipt.run_id,
            "release_path": str(receipt.release_path),
            "promoted_at": receipt.promoted_at,
            "approved_by": receipt.approved_by,
            "previous_model": receipt.previous_model,
            "current_model": receipt.current_model,
            "release_manifest_hash": receipt.release_manifest_hash,
            "receipt_json_path": str(receipt.receipt_json_path),
            "receipt_html_path": str(receipt.receipt_html_path),
            "report_hash": report_hash,
            "decision_hash": decision_hash,
            "model_hash": model_hash,
            "schema_hash": schema_hash,
        }

    @staticmethod
    def _bootstrap_manifest(
        snapshot: BootstrapReleaseSnapshot,
        *,
        previous_model: str | None,
        source: str,
        artifacts: Sequence[ArtifactRef],
    ) -> dict[str, object]:
        return {
            "schema_version": _JSON_SCHEMA_VERSION,
            "release_type": "legacy-bootstrap",
            "run_id": snapshot.run_id,
            "previous_model": previous_model,
            "migration_source": source,
            "legacy_snapshot_hash": snapshot.content_hash,
            "artifacts": [to_plain_data(artifact) for artifact in artifacts],
        }

    @staticmethod
    def _bootstrap_artifact_hash(
        artifacts: Sequence[ArtifactRef],
        artifact_type: str,
        fallback: str,
    ) -> str:
        for artifact in artifacts:
            if artifact.artifact_type == artifact_type:
                return artifact.content_hash
        return fallback

    def _recover_bootstrap_locked(
        self,
        snapshot: BootstrapReleaseSnapshot,
        source: str,
        marker: Mapping[str, object],
    ) -> PromotionReceipt:
        if (
            marker["source"] != source
            or marker["snapshot_hash"] != snapshot.content_hash
            or marker["run_id"] != snapshot.run_id
        ):
            raise ArtifactIntegrityError(
                "legacy state differs from the immutable migration already recorded",
                stage="MIGRATION",
                run_id=snapshot.run_id,
                operator_action="Do not overwrite legacy history; create a separate migration workspace.",
            )
        durable = self.read_promotion_receipt(snapshot.run_id)
        if durable is None:
            raise ArtifactIntegrityError(
                "legacy migration marker has no bootstrap receipt",
                stage="MIGRATION",
                run_id=snapshot.run_id,
            )
        receipt, _ = durable
        current = self.read_current_model()
        if current is None:
            self._write_current_model(
                receipt.current_model,
                source_run_id=receipt.run_id,
                metadata={
                    "migration_source": source,
                    "legacy_snapshot_hash": snapshot.content_hash,
                    "release_manifest_hash": receipt.release_manifest_hash,
                },
            )
        elif current != receipt.current_model:
            raise PromotionConflictError(
                "current model changed after the legacy bootstrap was published",
                stage="MIGRATION",
                run_id=snapshot.run_id,
                details={"current": current, "bootstrap": receipt.current_model},
            )
        return receipt

    def import_bootstrap_release(
        self,
        snapshot: BootstrapReleaseSnapshot,
        *,
        source: str,
    ) -> PromotionReceipt:
        """Publish a verified read-only legacy snapshot as the initial release."""

        if not isinstance(snapshot, BootstrapReleaseSnapshot):
            raise TypeError("snapshot must be a BootstrapReleaseSnapshot")
        safe_source = _safe_component(source, "migration source")
        with self.workspace_lock():
            marker = self._read_legacy_marker()
            if marker is not None:
                return self._recover_bootstrap_locked(snapshot, safe_source, marker)

            current = self.read_current_model()
            if current is not None:
                raise PromotionConflictError(
                    "cannot import legacy state over an existing current model",
                    stage="MIGRATION",
                    run_id=snapshot.run_id,
                )
            release_dir = self._release_dir(snapshot.run_id)
            if release_dir.exists():
                durable = self.read_promotion_receipt(snapshot.run_id)
                if durable is None:
                    raise ArtifactIntegrityError(
                        "bootstrap release directory exists without a valid receipt",
                        stage="MIGRATION",
                        run_id=snapshot.run_id,
                    )
                receipt, _ = durable
                marker_data = self._legacy_marker_data(snapshot, safe_source)
                try:
                    _atomic_create_bytes(
                        self.legacy_migration_marker_path,
                        _json_bytes(marker_data),
                    )
                except FileExistsError:
                    marker = self._read_legacy_marker()
                    if marker is None:  # pragma: no cover - defensive filesystem race guard.
                        raise
                    return self._recover_bootstrap_locked(snapshot, safe_source, marker)
                return self._recover_bootstrap_locked(snapshot, safe_source, marker_data)

            staging = self._releases_dir / f".staging-{snapshot.run_id}-{uuid.uuid4().hex}"
            published = False
            try:
                staging.mkdir(mode=0o700)
                copied = self._copy_release_artifacts(
                    ReleaseSnapshot(
                        run_id=snapshot.run_id,
                        artifacts=snapshot.artifacts,
                        manifest_hash=snapshot.content_hash,
                        previous_model=None,
                    ),
                    staging,
                )
                self._verify_artifacts(copied, run_id=snapshot.run_id)
                published_artifacts = self._published_artifacts(
                    staging,
                    release_dir,
                    copied,
                    run_id=snapshot.run_id,
                )
                manifest = self._bootstrap_manifest(
                    snapshot,
                    previous_model=None,
                    source=safe_source,
                    artifacts=published_artifacts,
                )
                manifest_bytes = _json_bytes(manifest)
                manifest_hash = _hash_bytes(manifest_bytes)
                _atomic_create_bytes(staging / _RELEASE_MANIFEST_NAME, manifest_bytes)
                receipt = PromotionReceipt(
                    run_id=snapshot.run_id,
                    release_path=release_dir,
                    promoted_at=_utc_now(),
                    approved_by=safe_source,
                    previous_model=None,
                    current_model=snapshot.run_id,
                    release_manifest_hash=manifest_hash,
                    receipt_json_path=release_dir / _PROMOTION_RECEIPT_NAME,
                    receipt_html_path=release_dir / _PROMOTION_RECEIPT_HTML_NAME,
                )
                receipt_data = self._promotion_receipt_data(
                    receipt,
                    idempotency_key=f"{safe_source}:{snapshot.content_hash}",
                    report_hash=snapshot.content_hash,
                    decision_hash=snapshot.content_hash,
                    model_hash=self._bootstrap_artifact_hash(
                        published_artifacts,
                        "legacy-model",
                        snapshot.content_hash,
                    ),
                    schema_hash=self._bootstrap_artifact_hash(
                        published_artifacts,
                        "legacy-model-metadata",
                        snapshot.content_hash,
                    ),
                )
                _atomic_create_bytes(
                    staging / _PROMOTION_RECEIPT_NAME,
                    _json_bytes(receipt_data),
                )
                _atomic_create_bytes(
                    staging / _PROMOTION_RECEIPT_HTML_NAME,
                    _receipt_html("HeuriBoost Legacy Bootstrap Receipt", receipt_data),
                )
                os.replace(staging, release_dir)
                published = True
                _fsync_directory(self._releases_dir)
                marker_data = self._legacy_marker_data(snapshot, safe_source)
                _atomic_create_bytes(
                    self.legacy_migration_marker_path,
                    _json_bytes(marker_data),
                )
                if self._before_pointer_swap is not None:
                    self._before_pointer_swap()
                self._write_current_model(
                    snapshot.run_id,
                    source_run_id=snapshot.run_id,
                    metadata={
                        "migration_source": safe_source,
                        "legacy_snapshot_hash": snapshot.content_hash,
                        "release_manifest_hash": manifest_hash,
                    },
                )
                return receipt
            except BaseException:
                if not published and staging.exists():
                    shutil.rmtree(staging)
                raise

    def promote_locked(
        self,
        run: RunRecord,
        approval: PromotionApproval,
        target: Any,
        snapshot: ReleaseSnapshot,
        *,
        report_hash: str,
        decision_hash: str,
        model_hash: str,
        schema_hash: str,
    ) -> PromotionReceipt:
        """Publish one release while the caller holds ``workspace_lock``."""

        _safe_component(run.run_id, "run ID")
        if snapshot.run_id != run.run_id:
            raise ArtifactIntegrityError(
                "release snapshot belongs to a different run",
                stage="PROMOTING",
                run_id=run.run_id,
            )
        existing = self.read_promotion_receipt(run.run_id)
        if existing is not None:
            receipt, existing_key = existing
            if existing_key != approval.idempotency_key:
                raise PromotionConflictError(
                    "release already exists under a different idempotency key",
                    stage="PROMOTING",
                    run_id=run.run_id,
                )
            current = self.read_current_model()
            if current == receipt.current_model:
                return receipt
            if current != approval.expected_current_model:
                raise PromotionConflictError(
                    "existing release is not the active current model",
                    stage="PROMOTING",
                    run_id=run.run_id,
                )
            target_state = target.validate_target(approval.expected_current_model)
            if target_state.current_model != receipt.current_model:
                raise PromotionConflictError(
                    "existing release target is not already activated",
                    stage="PROMOTING",
                    run_id=run.run_id,
                )
            if self._before_pointer_swap is not None:
                self._before_pointer_swap()
            self._write_current_model(
                receipt.current_model,
                source_run_id=run.run_id,
                metadata={"release_manifest_hash": receipt.release_manifest_hash},
            )
            return receipt

        release_dir = self._release_dir(run.run_id)
        if release_dir.exists():
            raise ArtifactIntegrityError(
                "release directory exists without a valid receipt",
                stage="PROMOTING",
                run_id=run.run_id,
            )
        current = self.read_current_model()
        if current != snapshot.previous_model or current != approval.expected_current_model:
            raise PromotionConflictError(
                "current model changed before release publication",
                stage="PROMOTING",
                run_id=run.run_id,
                details={"current": current, "expected": approval.expected_current_model},
            )
        validation = target.validate_target(approval.expected_current_model)
        if not validation.valid or validation.current_model != current:
            raise PromotionConflictError(
                "promotion target no longer accepts the expected current model",
                stage="PROMOTING",
                run_id=run.run_id,
                details={"errors": list(validation.errors), "current": validation.current_model},
            )

        staging = self._releases_dir / f".staging-{run.run_id}-{uuid.uuid4().hex}"
        try:
            staging.mkdir(mode=0o700)
            copied = self._copy_release_artifacts(snapshot, staging)
            staged_snapshot = ReleaseSnapshot(
                run_id=run.run_id,
                artifacts=copied,
                manifest_hash=canonical_json_hash(
                    [to_plain_data(artifact) for artifact in copied]
                ),
                previous_model=current,
            )
            prepared = target.prepare_release(staged_snapshot)
            if prepared.run_id != run.run_id:
                raise ArtifactIntegrityError(
                    "promotion target prepared a release for a different run",
                    stage="PROMOTING",
                    run_id=run.run_id,
                )
            activation = target.activate(prepared)
            if activation.current_model != run.run_id:
                raise ArtifactIntegrityError(
                    "promotion target activated an unexpected current model",
                    stage="PROMOTING",
                    run_id=run.run_id,
                    details={"current_model": activation.current_model},
                )
            self._verify_artifacts(copied, run_id=run.run_id)
            published = self._published_artifacts(
                staging,
                release_dir,
                copied,
                run_id=run.run_id,
            )
            published_snapshot = ReleaseSnapshot(
                run_id=run.run_id,
                artifacts=published,
                manifest_hash=canonical_json_hash(
                    [to_plain_data(artifact) for artifact in published]
                ),
                previous_model=current,
            )
            manifest = self._release_manifest(
                run=run,
                approval=approval,
                snapshot=published_snapshot,
                copied=published,
                prepared=prepared,
                activation=activation,
                report_hash=report_hash,
                decision_hash=decision_hash,
                model_hash=model_hash,
                schema_hash=schema_hash,
            )
            manifest_bytes = _json_bytes(manifest)
            manifest_hash = _hash_bytes(manifest_bytes)
            _atomic_create_bytes(staging / _RELEASE_MANIFEST_NAME, manifest_bytes)
            receipt = PromotionReceipt(
                run_id=run.run_id,
                release_path=release_dir,
                promoted_at=_utc_now(),
                approved_by=approval.approved_by,
                previous_model=current,
                current_model=run.run_id,
                release_manifest_hash=manifest_hash,
                receipt_json_path=release_dir / _PROMOTION_RECEIPT_NAME,
                receipt_html_path=release_dir / _PROMOTION_RECEIPT_HTML_NAME,
            )
            receipt_data = self._promotion_receipt_data(
                receipt,
                idempotency_key=approval.idempotency_key,
                report_hash=report_hash,
                decision_hash=decision_hash,
                model_hash=model_hash,
                schema_hash=schema_hash,
            )
            _atomic_create_bytes(staging / _PROMOTION_RECEIPT_NAME, _json_bytes(receipt_data))
            _atomic_create_bytes(
                staging / _PROMOTION_RECEIPT_HTML_NAME,
                _receipt_html("HeuriBoost Promotion Receipt", receipt_data),
            )
            os.replace(staging, release_dir)
            _fsync_directory(self._releases_dir)
            if self._before_pointer_swap is not None:
                self._before_pointer_swap()
            self._write_current_model(
                run.run_id,
                source_run_id=run.run_id,
                metadata={
                    "release_manifest_hash": manifest_hash,
                    "target": to_plain_data(activation.metadata),
                },
            )
            return receipt
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def promote(
        self,
        run: RunRecord,
        approval: PromotionApproval,
        target: Any,
        snapshot: ReleaseSnapshot,
        *,
        report_hash: str,
        decision_hash: str,
        model_hash: str,
        schema_hash: str,
    ) -> PromotionReceipt:
        with self.workspace_lock():
            return self.promote_locked(
                run,
                approval,
                target,
                snapshot,
                report_hash=report_hash,
                decision_hash=decision_hash,
                model_hash=model_hash,
                schema_hash=schema_hash,
            )

    def rollback_locked(
        self,
        receipt: PromotionReceipt,
        target: Any,
        *,
        approved_by: str,
    ) -> RollbackReceipt:
        durable = self.read_promotion_receipt(receipt.run_id)
        if durable is None or durable[0] != receipt:
            raise ArtifactIntegrityError(
                "rollback receipt does not match the immutable promoted release",
                stage="PROMOTING",
                run_id=receipt.run_id,
            )
        receipt = durable[0]
        if not receipt.previous_model:
            raise PromotionConflictError(
                "the promoted release has no previous model to restore",
                stage="PROMOTING",
                run_id=receipt.run_id,
            )
        current = self.read_current_model()
        pending = self._pending_rollback(receipt)
        if pending is not None:
            if current == pending.restored_model:
                return pending
            if current != pending.previous_model:
                raise PromotionConflictError(
                    "current model changed after the rollback receipt was issued",
                    stage="PROMOTING",
                    run_id=receipt.run_id,
                )
            if self._before_pointer_swap is not None:
                self._before_pointer_swap()
            self._write_current_model(
                pending.restored_model,
                source_run_id=receipt.run_id,
                metadata={"rollback_receipt": str(pending.receipt_json_path)},
            )
            return pending
        if current != receipt.current_model:
            raise PromotionConflictError(
                "current model changed after the promotion receipt was issued",
                stage="PROMOTING",
                run_id=receipt.run_id,
            )
        activation = target.rollback(receipt)
        if activation.current_model != receipt.previous_model:
            raise ArtifactIntegrityError(
                "promotion target restored an unexpected model",
                stage="PROMOTING",
                run_id=receipt.run_id,
            )
        rollback_dir = receipt.release_path / "rollback_receipts"
        rollback_id = uuid.uuid4().hex
        json_path = rollback_dir / f"rollback-{rollback_id}.json"
        html_path = rollback_dir / f"rollback-{rollback_id}.html"
        rollback = RollbackReceipt(
            source_run_id=receipt.run_id,
            rolled_back_at=_utc_now(),
            approved_by=approved_by,
            previous_model=receipt.current_model,
            restored_model=receipt.previous_model,
            receipt_json_path=json_path,
            receipt_html_path=html_path,
        )
        rollback_data = {
            "schema_version": _JSON_SCHEMA_VERSION,
            "receipt_type": "rollback",
            **to_plain_data(rollback),
            "source_release_manifest_hash": receipt.release_manifest_hash,
            "target_metadata": to_plain_data(activation.metadata),
        }
        _atomic_create_bytes(json_path, _json_bytes(rollback_data))
        _atomic_create_bytes(
            html_path,
            _receipt_html("HeuriBoost Rollback Receipt", rollback_data),
        )
        if self._before_pointer_swap is not None:
            self._before_pointer_swap()
        self._write_current_model(
            receipt.previous_model,
            source_run_id=receipt.run_id,
            metadata={"rollback_receipt": str(json_path)},
        )
        return rollback

    def rollback(
        self,
        receipt: PromotionReceipt,
        target: Any,
        *,
        approved_by: str,
    ) -> RollbackReceipt:
        with self.workspace_lock():
            return self.rollback_locked(receipt, target, approved_by=approved_by)
