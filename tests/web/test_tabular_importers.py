from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from heuriboost_rag.importers.base import (
    FieldMapping,
    REQUIRED_COLUMNS,
    StoredUpload,
)
from heuriboost_rag.importers.csv_importer import CsvImporter
from heuriboost_rag.importers.jsonl_importer import JsonlImporter


class TabularImporterTests(unittest.TestCase):
    def _mapping(self) -> FieldMapping:
        return FieldMapping(
            {
                "domain": "domain",
                "query": "query",
                "text": "text",
                "relevance": "relevance",
                "rank": "rank",
            }
        )

    def test_csv_and_jsonl_normalize_equivalent_records_to_same_semantic_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "source.csv"
            csv_path.write_text(
                "domain,query,text,relevance,rank\n"
                "tax, What is a wash sale? ,A wash sale has a 30-day rule.,good,1\n",
                encoding="utf-8",
            )
            jsonl_path = root / "source.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "domain": "tax",
                        "query": "What is a wash sale?",
                        "text": "A wash sale has a 30-day rule.",
                        "relevance": "good",
                        "rank": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            csv_result = CsvImporter(root / "csv").normalize(
                StoredUpload("upload-csv", csv_path, "source.csv"), self._mapping()
            )
            jsonl_result = JsonlImporter(root / "jsonl").normalize(
                StoredUpload("upload-jsonl", jsonl_path, "source.jsonl"),
                self._mapping(),
            )

            self.assertEqual(csv_result.semantic_hash, jsonl_result.semantic_hash)
            self.assertEqual(csv_result.columns, REQUIRED_COLUMNS)
            self.assertEqual(
                pq.read_table(csv_result.parquet_path).to_pylist(),
                pq.read_table(jsonl_result.parquet_path).to_pylist(),
            )

    def test_csv_rejects_invalid_utf8_and_unselected_delimiter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = root / "invalid.csv"
            invalid.write_bytes(b"query,text,relevance\n\xff,answer,good\n")
            importer = CsvImporter(root / "datasets")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                importer.inspect(StoredUpload("invalid", invalid, "invalid.csv"))

            delimited = root / "semicolon.csv"
            delimited.write_text(
                "query;text;relevance\nquestion;answer;good\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "delimiter"):
                importer.inspect(StoredUpload("semicolon", delimited, "semicolon.csv"))

    def test_jsonl_rejects_arrays_and_reports_the_exact_malformed_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            array_path = root / "array.jsonl"
            array_path.write_text('[{"query":"q"}]\n', encoding="utf-8")
            importer = JsonlImporter(root / "datasets")
            with self.assertRaisesRegex(ValueError, "line 1"):
                importer.inspect(StoredUpload("array", array_path, "array.jsonl"))

            broken_path = root / "broken.jsonl"
            broken_path.write_text(
                '{"query":"q"}\n{"text":\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "line 2"):
                importer.inspect(StoredUpload("broken", broken_path, "broken.jsonl"))


if __name__ == "__main__":
    unittest.main()
