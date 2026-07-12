"""Strict UTF-8 comma-separated import support."""

from __future__ import annotations

import csv
import io
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


class CsvImporter:
    formats = ("csv",)

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()

    def inspect(self, upload: StoredUpload) -> ImportInspection:
        columns, records = self._read(upload, ImportOptions())
        return inspection_from_records("csv", columns, records)

    def preview(self, upload: StoredUpload, options: ImportOptions) -> PreviewPage:
        columns, records = self._read(upload, options)
        return preview_from_records(columns, records)

    def normalize(
        self,
        upload: StoredUpload,
        mapping: FieldMapping,
        options: ImportOptions | None = None,
    ) -> NormalizedDataset:
        _, records = self._read(upload, options or ImportOptions())
        return write_normalized_dataset(self.output_dir, upload, records, mapping)

    @staticmethod
    def _read(upload: StoredUpload, options: ImportOptions) -> tuple[tuple[str, ...], list[Mapping[str, object]]]:
        try:
            text = upload.source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV must be valid UTF-8") from exc
        except OSError as exc:
            raise ValueError(f"cannot read CSV upload: {upload.source_path}") from exc
        delimiter = options.delimiter or ","
        if options.delimiter is None:
            first_line = text.splitlines()[0] if text.splitlines() else ""
            if any(marker in first_line for marker in (";", "\t", "|")) and "," not in first_line:
                raise ValueError("CSV delimiter must be explicitly selected")
        try:
            reader = csv.DictReader(io.StringIO(text), delimiter=delimiter, strict=True)
            raw_headers = reader.fieldnames
            if not raw_headers:
                raise ValueError("CSV requires a header row")
            headers = tuple(normalize_column_name(value) for value in raw_headers)
            if len(set(headers)) != len(headers):
                raise ValueError("CSV header contains duplicate fields")
            records: list[Mapping[str, object]] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"CSV row {row_number} has more fields than its header")
                record = {headers[index]: row.get(raw_headers[index]) for index in range(len(headers))}
                if any(value not in {None, ""} for value in record.values()):
                    records.append(record)
        except csv.Error as exc:
            raise ValueError(f"invalid CSV: {exc}") from exc
        if not records:
            raise ValueError("CSV must contain at least one data row")
        return headers, records


assert isinstance(CsvImporter(Path(".")), DatasetImporter)


__all__ = ["CsvImporter"]
