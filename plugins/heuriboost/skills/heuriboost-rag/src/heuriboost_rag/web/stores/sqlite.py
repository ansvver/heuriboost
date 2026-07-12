"""Versioned SQLite state for the local HeuriBoost Web Console."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.resources
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid

from ...reckless.contracts import DatasetRef, PromotionReceipt, RepairRequest, RunRecord, to_plain_data
from ...reckless.errors import HeuriBoostError, PromotionConflictError
from ...reckless.hashing import sha256_file
from ...reckless.state import RunState, assert_transition


_MIGRATION_PACKAGE = "heuriboost_rag.web.migrations"
_JSON_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_data(value: object) -> str:
    return json.dumps(
        to_plain_data(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_object(value: str, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _request_from_data(value: Mapping[str, object]) -> RepairRequest:
    expected = {
        "workspace_id",
        "base_dataset_id",
        "production_cases_id",
        "policy_version",
        "backend_name",
        "requested_by",
        "run_options",
    }
    if set(value) != expected or not isinstance(value["run_options"], Mapping):
        raise ValueError("run request has an invalid schema")
    return RepairRequest(
        workspace_id=_require_string(value["workspace_id"], "workspace_id"),
        base_dataset_id=_require_string(value["base_dataset_id"], "base_dataset_id"),
        production_cases_id=_require_string(value["production_cases_id"], "production_cases_id"),
        policy_version=_require_string(value["policy_version"], "policy_version"),
        backend_name=_require_string(value["backend_name"], "backend_name"),
        requested_by=_require_string(value["requested_by"], "requested_by"),
        run_options=dict(value["run_options"]),
    )


def _record_from_row(row: sqlite3.Row) -> RunRecord:
    request = _request_from_data(_json_object(row["request_json"], "run request"))
    metadata = _json_object(row["metadata_json"], "run metadata")
    error_text = row["error_json"]
    error = None if error_text is None else _json_object(error_text, "run error")
    return RunRecord(
        run_id=_require_string(row["id"], "run ID"),
        state=_require_string(row["state"], "run state"),
        version=int(row["version"]),
        request=request,
        policy_hash=_require_string(row["policy_hash"], "policy hash"),
        input_hash=_require_string(row["input_hash"], "input hash"),
        metadata=metadata,
        error=error,
    )


def _receipt_from_data(value: Mapping[str, object]) -> PromotionReceipt:
    required = {
        "run_id",
        "release_path",
        "promoted_at",
        "approved_by",
        "previous_model",
        "current_model",
        "release_manifest_hash",
        "receipt_json_path",
        "receipt_html_path",
    }
    if set(value) != required:
        raise ValueError("promotion receipt has an invalid schema")
    previous_model = value["previous_model"]
    if previous_model is not None and not isinstance(previous_model, str):
        raise ValueError("promotion receipt previous_model is invalid")
    for key in required - {"previous_model"}:
        _require_string(value[key], f"promotion receipt {key}")
    return PromotionReceipt(
        run_id=value["run_id"],
        release_path=Path(value["release_path"]),
        promoted_at=value["promoted_at"],
        approved_by=value["approved_by"],
        previous_model=previous_model,
        current_model=value["current_model"],
        release_manifest_hash=value["release_manifest_hash"],
        receipt_json_path=Path(value["receipt_json_path"]),
        receipt_html_path=Path(value["receipt_html_path"]),
    )


class SQLiteRunRepository:
    """Core-compatible run records with version-guarded writes."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    @staticmethod
    def _assert_immutable(stored: RunRecord, proposed: RunRecord) -> None:
        changed = [
            name
            for name in ("run_id", "request", "policy_hash", "input_hash")
            if getattr(stored, name) != getattr(proposed, name)
        ]
        if changed:
            raise ValueError("Run immutable fields cannot change: " + ", ".join(changed))

    def create(self, request: RepairRequest, policy_hash: str, input_hash: str) -> RunRecord:
        if not isinstance(request, RepairRequest):
            raise TypeError("request must be a RepairRequest")
        _require_string(policy_hash, "policy_hash")
        _require_string(input_hash, "input_hash")
        record = RunRecord(
            run_id=f"run-{uuid.uuid4().hex}",
            state=RunState.RECEIVED.value,
            version=1,
            request=request,
            policy_hash=policy_hash,
            input_hash=input_hash,
        )
        now = _utc_now()
        with self._store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, workspace_id, request_json, policy_hash, input_hash, state,
                    version, metadata_json, error_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    record.run_id,
                    request.workspace_id,
                    _json_data(request),
                    policy_hash,
                    input_hash,
                    record.state,
                    record.version,
                    _json_data(record.metadata),
                    now,
                    now,
                ),
            )
        return record

    def get(self, run_id: str) -> RunRecord:
        _require_string(run_id, "run_id")
        with self._store.connection() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"Unknown run record: {run_id!r}")
        return _record_from_row(row)

    def save(self, record: RunRecord) -> RunRecord:
        if not isinstance(record, RunRecord):
            raise TypeError("record must be a RunRecord")
        with self._store.transaction() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (record.run_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown run record: {record.run_id!r}")
            stored = _record_from_row(row)
            self._assert_immutable(stored, record)
            if record.version != stored.version:
                raise ValueError(
                    f"Rejecting stale run version {record.version}; stored version is {stored.version}"
                )
            if record == stored:
                raise ValueError("Run save contains no changes")
            if record.state != stored.state:
                assert_transition(stored.state, record.state)
            saved = replace(record, version=stored.version + 1)
            result = connection.execute(
                """
                UPDATE runs
                SET state = ?, version = ?, metadata_json = ?, error_json = ?, updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (
                    saved.state,
                    saved.version,
                    _json_data(saved.metadata),
                    None if saved.error is None else _json_data(saved.error),
                    _utc_now(),
                    saved.run_id,
                    stored.version,
                ),
            )
            if result.rowcount != 1:
                raise ValueError(f"Rejecting stale run version {record.version}")
        return saved

    def transition(
        self,
        run_id: str,
        state: RunState,
        metadata: Mapping[str, object] | None = None,
    ) -> RunRecord:
        target = state if isinstance(state, RunState) else RunState(state)
        update = {} if metadata is None else dict(metadata)
        stored = self.get(run_id)
        return self.save(
            replace(
                stored,
                state=target.value,
                metadata={**to_plain_data(stored.metadata), **to_plain_data(update)},
            )
        )

    def fail(self, run_id: str, state: RunState, error: HeuriBoostError) -> RunRecord:
        if not isinstance(error, HeuriBoostError):
            raise TypeError("error must be a HeuriBoostError")
        target = state if isinstance(state, RunState) else RunState(state)
        if not (target.value.startswith("BLOCKED_") or target.value.endswith("FAILED")):
            raise ValueError(f"fail target must be BLOCKED or FAILED: {target.value}")
        if error.code != target.value:
            raise ValueError(
                f"Error code {error.code!r} does not match failure state {target.value!r}"
            )
        stored = self.get(run_id)
        return self.save(replace(stored, state=target.value, error=error.to_dict()))


class SQLiteDatasetRepository:
    """Core-compatible dataset repository backed by Web Console datasets."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def get(self, dataset_id: str) -> DatasetRef:
        _require_string(dataset_id, "dataset_id")
        with self._store.connection() as connection:
            row = connection.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"Unknown dataset record: {dataset_id!r}")
        if row["status"] != "READY":
            raise ValueError(f"dataset {dataset_id!r} is not READY")
        metadata = _json_object(row["metadata_json"], "dataset metadata")
        core_input = metadata.get("core_input_path") or row["normalized_path"]
        if not isinstance(core_input, str) or not core_input:
            raise ValueError("dataset core_input_path must be a non-empty string")
        path = Path(core_input).expanduser().resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"dataset input must be a regular file: {path}")
        content_hash = metadata.get("core_input_hash")
        if content_hash is None:
            content_hash = sha256_file(path)
        if not isinstance(content_hash, str) or not content_hash:
            raise ValueError("dataset core_input_hash must be a non-empty string")
        return DatasetRef(
            dataset_id=row["id"],
            role=row["role"],
            path=path,
            content_hash=content_hash,
            schema_hash=row["schema_hash"],
            metadata=metadata,
        )


class SQLitePromotionRepository:
    """SQLite-backed promotion idempotency receipt store."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def find_by_idempotency_key(self, key: str) -> PromotionReceipt | None:
        _require_string(key, "idempotency_key")
        with self._store.connection() as connection:
            row = connection.execute(
                "SELECT receipt_json FROM promotions WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return None if row is None else _receipt_from_data(_json_object(row["receipt_json"], "promotion receipt"))

    def save(self, receipt: PromotionReceipt, idempotency_key: str) -> None:
        if not isinstance(receipt, PromotionReceipt):
            raise TypeError("receipt must be a PromotionReceipt")
        _require_string(idempotency_key, "idempotency_key")
        with self._store.transaction() as connection:
            row = connection.execute(
                "SELECT receipt_json FROM promotions WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if row is not None:
                existing = _receipt_from_data(_json_object(row["receipt_json"], "promotion receipt"))
                if existing != receipt:
                    raise PromotionConflictError(
                        "idempotency key is already bound to another promotion receipt",
                        stage="PROMOTING",
                        run_id=receipt.run_id,
                    )
                return
            connection.execute(
                """
                INSERT INTO promotions (idempotency_key, run_id, receipt_json, current_model, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (idempotency_key, receipt.run_id, _json_data(receipt), receipt.current_model, _utc_now()),
            )


class SQLiteStore:
    """SQLite connection, migration, Core adapters, and append-only audit access."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.datasets = SQLiteDatasetRepository(self)
        self.runs = SQLiteRunRepository(self)
        self.promotions = SQLitePromotionRepository(self)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _migrations() -> Sequence[tuple[int, str, str]]:
        root = importlib.resources.files(_MIGRATION_PACKAGE)
        migrations: list[tuple[int, str, str]] = []
        for item in root.iterdir():
            if item.name.endswith(".sql") and item.name[:4].isdigit():
                migrations.append((int(item.name[:4]), item.name, item.read_text(encoding="utf-8")))
        return tuple(sorted(migrations))

    def migrate(self) -> None:
        migrations = self._migrations()
        if not migrations:
            raise RuntimeError("no SQLite migrations are packaged")
        with self.connection() as connection:
            has_migration_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
            ).fetchone()
            applied: dict[int, sqlite3.Row] = {}
            if has_migration_table:
                for row in connection.execute("SELECT version, name, checksum FROM schema_migrations"):
                    applied[int(row["version"])] = row
            known_versions = {version for version, _, _ in migrations}
            if any(version not in known_versions for version in applied):
                raise RuntimeError("database schema is newer than this application")
            for version, name, sql in migrations:
                checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                existing = applied.get(version)
                if existing is not None:
                    if existing["name"] != name or existing["checksum"] != checksum:
                        raise RuntimeError(f"migration {name} checksum does not match the database")
                    continue
                script = (
                    "BEGIN IMMEDIATE;\n"
                    + sql
                    + "\nINSERT INTO schema_migrations (version, name, checksum, applied_at) VALUES ("
                    + f"{version}, {json.dumps(name)}, {json.dumps(checksum)}, {json.dumps(_utc_now())}"
                    + ");\nCOMMIT;"
                )
                connection.executescript(script)

    def table_names(self) -> set[str]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        return {str(row["name"]) for row in rows}

    def append_audit_event(self, event_type: str, payload: Mapping[str, object]) -> int:
        _require_string(event_type, "event_type")
        with self.transaction() as connection:
            result = connection.execute(
                "INSERT INTO audit_events (event_type, payload_json, occurred_at) VALUES (?, ?, ?)",
                (event_type, _json_data(payload), _utc_now()),
            )
        return int(result.lastrowid)

    def audit_events_after(self, event_id: int = 0, *, limit: int = 100) -> tuple[dict[str, object], ...]:
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
            raise ValueError("event_id must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, event_type, payload_json, occurred_at
                FROM audit_events WHERE id > ? ORDER BY id ASC LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return tuple(
            {
                "event_id": int(row["id"]),
                "event_type": row["event_type"],
                "payload": _json_object(row["payload_json"], "audit payload"),
                "occurred_at": row["occurred_at"],
            }
            for row in rows
        )

    def find_idempotency_result(self, operation: str, key: str) -> dict[str, object] | None:
        _require_string(operation, "operation")
        _require_string(key, "idempotency_key")
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM audit_events
                WHERE event_type = 'idempotency.result'
                ORDER BY id DESC
                """
            ).fetchall()
        for row in rows:
            payload = _json_object(row["payload_json"], "idempotency payload")
            if payload.get("operation") == operation and payload.get("key") == key:
                result = payload.get("result")
                if not isinstance(result, Mapping):
                    raise ValueError("idempotency result is invalid")
                return dict(result)
        return None

    def save_idempotency_result(self, operation: str, key: str, result: Mapping[str, object]) -> None:
        _require_string(operation, "operation")
        _require_string(key, "idempotency_key")
        self.append_audit_event(
            "idempotency.result",
            {"operation": operation, "key": key, "result": dict(result)},
        )

    def update_audit_event(self, event_id: int, payload: Mapping[str, object]) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE audit_events SET payload_json = ? WHERE id = ?",
                (_json_data(payload), event_id),
            )

    def delete_audit_event(self, event_id: int) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM audit_events WHERE id = ?", (event_id,))


__all__ = ["SQLiteStore"]
