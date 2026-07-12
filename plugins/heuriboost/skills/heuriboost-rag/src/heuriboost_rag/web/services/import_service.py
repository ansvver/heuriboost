"""Durable upload inspection and immutable dataset normalization."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

from fastapi import UploadFile
import pyarrow.parquet as pq

from ...importers.base import FieldMapping, ImportOptions, StoredUpload
from ...importers.csv_importer import CsvImporter
from ...importers.jsonl_importer import JsonlImporter
from ...importers.xlsx_importer import XlsxImporter
from ...reckless.hashing import sha256_file
from ..config import WebConfig
from ..stores.sqlite import SQLiteStore


_SUPPORTED_SUFFIXES = {
    ".csv": "csv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".xlsx": "xlsx",
}
_DATASET_ROLES = frozenset({"base", "production_cases"})


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _inspection_data(inspection: object) -> dict[str, object]:
    data = asdict(inspection)
    return dict(data)


def _materialize_core_jsonl(parquet_path: Path, destination: Path) -> str:
    if destination.exists():
        return sha256_file(destination)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            reader = pq.ParquetFile(parquet_path)
            for batch in reader.iter_batches():
                for row in batch.to_pylist():
                    handle.write(_json(row))
                    handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(destination)


class ImportService:
    def __init__(self, config: WebConfig, store: SQLiteStore) -> None:
        self.config = config
        self.store = store
        self.uploads_dir = config.data_dir / "uploads"
        self.datasets_dir = config.data_dir / "datasets"
        self.normalization_dir = self.datasets_dir / ".normalizing"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    async def create_upload(self, upload: UploadFile) -> dict[str, object]:
        filename = upload.filename or ""
        suffix = Path(filename).suffix.lower()
        format_name = _SUPPORTED_SUFFIXES.get(suffix)
        if format_name is None:
            raise ValueError("supported upload formats are CSV, JSONL, NDJSON, and XLSX")
        upload_id = f"upload-{uuid.uuid4().hex}"
        upload_dir = self.uploads_dir / upload_id
        upload_dir.mkdir(mode=0o700)
        stored_path = upload_dir / f"source{suffix}"
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with stored_path.open("xb") as handle:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > self.config.max_upload_bytes:
                        raise ValueError("upload exceeds the configured size limit")
                    digest.update(chunk)
                    handle.write(chunk)
            stored = StoredUpload(upload_id, stored_path, filename)
            inspection = self._importer(format_name).inspect(stored)
        except BaseException:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise
        finally:
            await upload.close()
        record = {
            "id": upload_id,
            "original_filename": filename,
            "stored_path": str(stored_path),
            "format_name": format_name,
            "size_bytes": size_bytes,
            "content_hash": digest.hexdigest(),
            "inspection": _inspection_data(inspection),
        }
        with self.store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO uploads (
                    id, original_filename, stored_path, format_name, size_bytes,
                    content_hash, inspection_json, schema_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, datetime('now'))
                """,
                (
                    upload_id,
                    filename,
                    str(stored_path),
                    format_name,
                    size_bytes,
                    digest.hexdigest(),
                    _json(record["inspection"]),
                ),
            )
        return record

    def get_upload(self, upload_id: str) -> dict[str, object]:
        with self.store.connection() as connection:
            row = connection.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"unknown upload: {upload_id}")
        return {
            "id": row["id"],
            "original_filename": row["original_filename"],
            "stored_path": row["stored_path"],
            "format_name": row["format_name"],
            "size_bytes": row["size_bytes"],
            "content_hash": row["content_hash"],
            "inspection": json.loads(row["inspection_json"]),
        }

    def sheets(self, upload_id: str) -> list[dict[str, object]]:
        upload = self._stored_upload(upload_id)
        if self.get_upload(upload_id)["format_name"] != "xlsx":
            raise ValueError("worksheets are available only for XLSX uploads")
        inspection = self._importer("xlsx").inspect(upload)
        return [asdict(sheet) for sheet in inspection.sheets]

    def preview(self, upload_id: str, options: ImportOptions) -> dict[str, object]:
        record = self.get_upload(upload_id)
        preview = self._importer(str(record["format_name"])).preview(
            self._stored_upload(upload_id), options
        )
        return asdict(preview)

    def normalize(
        self,
        upload_id: str,
        *,
        role: str,
        mapping: Mapping[str, str],
        options: ImportOptions | None = None,
    ) -> dict[str, object]:
        if role not in _DATASET_ROLES:
            raise ValueError("dataset role must be base or production_cases")
        record = self.get_upload(upload_id)
        normalized = self._importer(str(record["format_name"])).normalize(
            self._stored_upload(upload_id), FieldMapping(mapping), options
        )
        dataset_id = f"dataset-{normalized.semantic_hash[:24]}"
        destination_dir = self.datasets_dir / dataset_id
        destination = destination_dir / "normalized.parquet"
        core_input_path = destination_dir / "normalized.jsonl"
        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM datasets WHERE role = ? AND semantic_hash = ?",
                (role, normalized.semantic_hash),
            ).fetchone()
            if existing is not None:
                return self._dataset_data(existing)
            destination_dir.mkdir(mode=0o700)
            if not destination.exists():
                os.replace(normalized.parquet_path, destination)
            core_input_hash = _materialize_core_jsonl(destination, core_input_path)
            metadata = {
                "source_hash": normalized.source_hash,
                "core_input_path": str(core_input_path),
                "core_input_hash": core_input_hash,
                "mapping": dict(mapping),
                "warnings": list(normalized.warnings),
                "rows": normalized.rows,
                "columns": list(normalized.columns),
            }
            connection.execute(
                """
                INSERT INTO datasets (
                    id, workspace_id, upload_id, role, semantic_hash, schema_hash,
                    normalized_path, metadata_json, schema_version, status, created_at
                ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, 1, 'READY', datetime('now'))
                """,
                (
                    dataset_id,
                    upload_id,
                    role,
                    normalized.semantic_hash,
                    normalized.schema_hash,
                    str(destination),
                    _json(metadata),
                ),
            )
            row = connection.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:  # pragma: no cover - protected by the preceding INSERT.
            raise RuntimeError("normalized dataset was not persisted")
        return self._dataset_data(row)

    def list_datasets(self) -> list[dict[str, object]]:
        with self.store.connection() as connection:
            rows = connection.execute("SELECT * FROM datasets ORDER BY created_at DESC, id DESC").fetchall()
        return [self._dataset_data(row) for row in rows]

    def recent_runs(self) -> list[dict[str, object]]:
        with self.store.connection() as connection:
            rows = connection.execute(
                "SELECT id, state, updated_at FROM runs ORDER BY updated_at DESC, id DESC LIMIT 20"
            ).fetchall()
        return [{"id": row["id"], "state": row["state"], "updated_at": row["updated_at"]} for row in rows]

    def _importer(self, format_name: str):
        if format_name == "csv":
            return CsvImporter(self.normalization_dir)
        if format_name == "jsonl":
            return JsonlImporter(self.normalization_dir)
        if format_name == "xlsx":
            return XlsxImporter(
                self.normalization_dir,
                max_uncompressed_bytes=self.config.max_xlsx_uncompressed_bytes,
                max_sheets=self.config.max_xlsx_sheets,
                max_rows=self.config.max_xlsx_rows,
                max_columns=self.config.max_xlsx_columns,
            )
        raise ValueError(f"unsupported import format: {format_name}")

    def _stored_upload(self, upload_id: str) -> StoredUpload:
        record = self.get_upload(upload_id)
        return StoredUpload(upload_id, Path(str(record["stored_path"])), str(record["original_filename"]))

    @staticmethod
    def _dataset_data(row: Any) -> dict[str, object]:
        return {
            "id": row["id"],
            "upload_id": row["upload_id"],
            "role": row["role"],
            "semantic_hash": row["semantic_hash"],
            "schema_hash": row["schema_hash"],
            "normalized_path": row["normalized_path"],
            "status": row["status"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }


__all__ = ["ImportService"]
