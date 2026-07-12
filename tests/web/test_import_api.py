from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from openpyxl import Workbook

from heuriboost_rag.web.app import create_app
from heuriboost_rag.web.config import WebConfig


class ImportApiTests(unittest.TestCase):
    def test_upload_normalize_and_workbench_list_an_immutable_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TestClient(create_app(WebConfig.for_test(Path(tmp)))) as client:
                uploaded = client.post(
                    "/api/imports",
                    files={
                        "file": (
                            "../../production.csv",
                            b"query,shown_doc_text,user_verdict\nWhat is a wash sale?,A wash sale has a 30-day rule.,good\n",
                            "text/csv",
                        )
                    },
                )

                self.assertEqual(uploaded.status_code, 201, uploaded.text)
                upload = uploaded.json()
                self.assertNotIn("..", upload["stored_path"])
                self.assertNotIn("production.csv", upload["stored_path"])
                self.assertEqual(upload["inspection"]["format_name"], "csv")

                normalized = client.post(
                    f"/api/imports/{upload['id']}/normalize",
                    json={
                        "role": "production_cases",
                        "mapping": {
                            "query": "query",
                            "shown_doc_text": "shown_doc_text",
                            "user_verdict": "user_verdict",
                        },
                    },
                )
                self.assertEqual(normalized.status_code, 201, normalized.text)
                dataset = normalized.json()
                self.assertTrue(dataset["normalized_path"].endswith("normalized.parquet"))
                self.assertTrue(dataset["metadata"]["core_input_path"].endswith("normalized.jsonl"))
                self.assertTrue(Path(dataset["metadata"]["core_input_path"]).is_file())
                self.assertRegex(dataset["metadata"]["core_input_hash"], r"^[0-9a-f]{64}$")

                datasets = client.get("/api/datasets")
                self.assertEqual(datasets.status_code, 200)
                self.assertEqual(datasets.json()[0]["id"], dataset["id"])

                workbench = client.get("/")
                self.assertEqual(workbench.status_code, 200)
                self.assertIn("运行工作台", workbench.text)
                self.assertIn(dataset["id"], workbench.text)

    def test_xlsx_exposes_sheets_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cases.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Cases"
            sheet.append(["query", "shown_doc_text", "user_verdict"])
            sheet.append(["q", "d", "good"])
            workbook.save(source)
            workbook.close()

            with TestClient(create_app(WebConfig.for_test(Path(tmp) / "data"))) as client:
                uploaded = client.post(
                    "/api/imports",
                    files={"file": ("cases.xlsx", source.read_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                )
                self.assertEqual(uploaded.status_code, 201, uploaded.text)
                upload_id = uploaded.json()["id"]

                sheets = client.get(f"/api/imports/{upload_id}/sheets")
                self.assertEqual(sheets.status_code, 200)
                self.assertEqual(sheets.json()[0]["name"], "Cases")
                preview = client.post(
                    f"/api/imports/{upload_id}/preview",
                    json={"sheet_name": "Cases", "header_row": 1},
                )
                self.assertEqual(preview.status_code, 200, preview.text)
                self.assertEqual(preview.json()["rows"][0]["query"], "q")

    def test_invalid_upload_uses_a_structured_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TestClient(create_app(WebConfig.for_test(Path(tmp)))) as client:
                response = client.post(
                    "/api/imports",
                    files={"file": ("unsupported.txt", b"not a dataset", "text/plain")},
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["code"], "INVALID_INPUT")
            self.assertIn("supported", response.json()["message"])


if __name__ == "__main__":
    unittest.main()
