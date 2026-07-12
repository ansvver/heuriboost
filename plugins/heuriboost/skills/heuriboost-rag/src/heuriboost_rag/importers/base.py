"""Shared contracts and deterministic normalization for tabular imports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.parquet as pq

from ..reckless.hashing import canonical_json_hash, sha256_file


REQUIRED_COLUMNS = (
    "domain",
    "query_id",
    "query",
    "doc_id",
    "text",
    "relevance",
    "split",
    "rank",
    "score",
    "case_id",
    "shown_doc_id",
    "shown_doc_text",
    "user_verdict",
)
_NUMERIC_COLUMNS = frozenset({"rank", "score", "relevance"})


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class StoredUpload:
    upload_id: str
    source_path: Path
    original_filename: str

    def __post_init__(self) -> None:
        _nonempty_string(self.upload_id, "upload_id")
        _nonempty_string(self.original_filename, "original_filename")
        source = Path(self.source_path).expanduser()
        try:
            metadata = source.lstat()
        except OSError as exc:
            raise ValueError(f"upload source is unavailable: {source}") from exc
        if not source.is_file() or source.is_symlink() or not metadata:
            raise ValueError(f"upload source must be a regular file: {source}")
        object.__setattr__(self, "source_path", source.resolve(strict=True))


@dataclass(frozen=True)
class ImportOptions:
    delimiter: str | None = None
    sheet_name: str | None = None
    header_row: int = 1

    def __post_init__(self) -> None:
        if self.delimiter is not None and (
            not isinstance(self.delimiter, str) or len(self.delimiter) != 1
        ):
            raise ValueError("delimiter must be one character or null")
        if self.sheet_name is not None:
            _nonempty_string(self.sheet_name, "sheet_name")
        if isinstance(self.header_row, bool) or not isinstance(self.header_row, int):
            raise ValueError("header_row must be an integer")
        if self.header_row < 1:
            raise ValueError("header_row must be at least one")


@dataclass(frozen=True)
class FieldMapping:
    source_to_target: Mapping[str, str]

    def __post_init__(self) -> None:
        raw = dict(self.source_to_target)
        if not raw:
            raise ValueError("field mapping must contain at least one field")
        targets: set[str] = set()
        normalized: dict[str, str] = {}
        for source, target in raw.items():
            safe_source = _nonempty_string(source, "mapping source")
            safe_target = _nonempty_string(target, "mapping target")
            if safe_target not in REQUIRED_COLUMNS:
                raise ValueError(f"unsupported mapping target: {safe_target}")
            if safe_target in targets:
                raise ValueError(f"multiple source fields map to {safe_target}")
            targets.add(safe_target)
            normalized[safe_source] = safe_target
        object.__setattr__(self, "source_to_target", MappingProxyType(normalized))


@dataclass(frozen=True)
class ImportInspection:
    format_name: str
    columns: tuple[str, ...]
    rows: int
    warnings: tuple[str, ...] = ()
    sheets: tuple[SheetInfo, ...] = ()


@dataclass(frozen=True)
class SheetInfo:
    name: str
    visible: bool
    rows: int
    columns: int


@dataclass(frozen=True)
class PreviewPage:
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, object], ...]
    total_rows: int


@dataclass(frozen=True)
class NormalizedDataset:
    parquet_path: Path
    source_hash: str
    semantic_hash: str
    schema_hash: str
    columns: tuple[str, ...]
    rows: int
    warnings: tuple[str, ...]


@runtime_checkable
class DatasetImporter(Protocol):
    formats: tuple[str, ...]

    def inspect(self, upload: StoredUpload) -> ImportInspection: ...

    def preview(self, upload: StoredUpload, options: ImportOptions) -> PreviewPage: ...

    def normalize(
        self,
        upload: StoredUpload,
        mapping: FieldMapping,
        options: ImportOptions | None = None,
    ) -> NormalizedDataset: ...


def normalize_column_name(value: object) -> str:
    return _nonempty_string(value, "column name")


def normalize_records(
    records: Sequence[Mapping[str, object]],
    mapping: FieldMapping,
) -> list[dict[str, str | None]]:
    normalized: list[dict[str, str | None]] = []
    for row_number, record in enumerate(records, start=1):
        canonical = {column: None for column in REQUIRED_COLUMNS}
        for source, target in mapping.source_to_target.items():
            if source in record:
                canonical[target] = _normalize_value(target, record[source], row_number)
        normalized.append(canonical)
    if not normalized:
        raise ValueError("dataset must contain at least one row")
    return normalized


def write_normalized_dataset(
    output_dir: Path,
    upload: StoredUpload,
    records: Sequence[Mapping[str, object]],
    mapping: FieldMapping,
    *,
    warnings: Sequence[str] = (),
) -> NormalizedDataset:
    normalized_rows = normalize_records(records, mapping)
    schema_hash = canonical_json_hash(
        [{"name": column, "type": "string"} for column in REQUIRED_COLUMNS]
    )
    semantic_hash = canonical_json_hash(
        {"columns": list(REQUIRED_COLUMNS), "rows": normalized_rows}
    )
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"normalized-{semantic_hash}.parquet"
    if not destination.exists():
        schema = pa.schema([pa.field(column, pa.string(), nullable=True) for column in REQUIRED_COLUMNS])
        table = pa.Table.from_pylist(normalized_rows, schema=schema)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination_dir,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            pq.write_table(table, temporary, compression="zstd")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return NormalizedDataset(
        parquet_path=destination,
        source_hash=sha256_file(upload.source_path),
        semantic_hash=semantic_hash,
        schema_hash=schema_hash,
        columns=REQUIRED_COLUMNS,
        rows=len(normalized_rows),
        warnings=tuple(warnings),
    )


def inspection_from_records(
    format_name: str,
    columns: Sequence[str],
    records: Sequence[Mapping[str, object]],
    *,
    warnings: Sequence[str] = (),
) -> ImportInspection:
    return ImportInspection(
        format_name=format_name,
        columns=tuple(columns),
        rows=len(records),
        warnings=tuple(warnings),
    )


def preview_from_records(
    columns: Sequence[str],
    records: Sequence[Mapping[str, object]],
    *,
    limit: int = 50,
) -> PreviewPage:
    return PreviewPage(
        columns=tuple(columns),
        rows=tuple(dict(row) for row in records[:limit]),
        total_rows=len(records),
    )


def _normalize_value(column: str, value: object, row_number: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"row {row_number} has a non-finite {column} value")
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError(f"row {row_number} has an unsupported {column} value")
    text = " ".join(str(value).strip().split())
    if not text:
        return None
    if column in _NUMERIC_COLUMNS:
        try:
            number = Decimal(text)
        except InvalidOperation:
            return text
        if not number.is_finite():
            raise ValueError(f"row {row_number} has a non-finite {column} value")
        normalized = format(number.normalize(), "f")
        return "0" if normalized in {"-0", ""} else normalized
    return text


__all__ = [
    "DatasetImporter",
    "FieldMapping",
    "ImportInspection",
    "ImportOptions",
    "NormalizedDataset",
    "PreviewPage",
    "REQUIRED_COLUMNS",
    "SheetInfo",
    "StoredUpload",
    "inspection_from_records",
    "normalize_column_name",
    "preview_from_records",
    "write_normalized_dataset",
]
