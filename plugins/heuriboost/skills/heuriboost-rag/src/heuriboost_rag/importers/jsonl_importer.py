"""Strict line-oriented JSON object import support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .base import (
    DatasetImporter,
    FieldMapping,
    ImportInspection,
    ImportOptions,
    NormalizedDataset,
    PreviewPage,
    StoredUpload,
    inspection_from_records,
    normalize_column_name,
    preview_from_records,
    write_normalized_dataset,
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON value {value}")


class JsonlImporter:
    formats = ("jsonl", "ndjson")

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()

    def inspect(self, upload: StoredUpload) -> ImportInspection:
        columns, records = self._read(upload)
        return inspection_from_records("jsonl", columns, records)

    def preview(self, upload: StoredUpload, options: ImportOptions) -> PreviewPage:
        del options
        columns, records = self._read(upload)
        return preview_from_records(columns, records)

    def normalize(
        self,
        upload: StoredUpload,
        mapping: FieldMapping,
        options: ImportOptions | None = None,
    ) -> NormalizedDataset:
        del options
        _, records = self._read(upload)
        return write_normalized_dataset(self.output_dir, upload, records, mapping)

    @staticmethod
    def _read(upload: StoredUpload) -> tuple[tuple[str, ...], list[Mapping[str, object]]]:
        try:
            lines = upload.source_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError("JSONL must be valid UTF-8") from exc
        except OSError as exc:
            raise ValueError(f"cannot read JSONL upload: {upload.source_path}") from exc
        records: list[Mapping[str, object]] = []
        columns: set[str] = set()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_constant,
                )
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            record: dict[str, object] = {}
            try:
                for key, value in parsed.items():
                    record[normalize_column_name(key)] = value
            except ValueError as exc:
                raise ValueError(f"invalid JSONL object at line {line_number}") from exc
            records.append(record)
            columns.update(record)
        if not records:
            raise ValueError("JSONL must contain at least one object")
        return tuple(sorted(columns)), records


assert isinstance(JsonlImporter(Path(".")), DatasetImporter)


__all__ = ["JsonlImporter"]
