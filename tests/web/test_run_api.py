from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from heuriboost_rag.web.app import create_app
from heuriboost_rag.web.config import WebConfig


class RunApiTests(unittest.TestCase):
    def _dataset(self, client: TestClient, role: str, content: bytes, mapping: dict[str, str]) -> str:
        uploaded = client.post("/api/imports", files={"file": (f"{role}.csv", content, "text/csv")})
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        normalized = client.post(
            f"/api/imports/{uploaded.json()['id']}/normalize",
            json={"role": role, "mapping": mapping},
        )
        self.assertEqual(normalized.status_code, 201, normalized.text)
        return normalized.json()["id"]

    def test_create_run_freezes_dataset_references_and_enqueues_one_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(WebConfig.for_test(Path(tmp)))
            with TestClient(app) as client:
                base_id = self._dataset(
                    client,
                    "base",
                    b"query,text,relevance\nq,document,good\n",
                    {"query": "query", "text": "text", "relevance": "relevance"},
                )
                cases_id = self._dataset(
                    client,
                    "production_cases",
                    b"query,shown_doc_text,user_verdict\nq,document,good\n",
                    {"query": "query", "shown_doc_text": "shown_doc_text", "user_verdict": "user_verdict"},
                )
                response = client.post(
                    "/api/runs",
                    headers={"Idempotency-Key": "create-run-1"},
                    json={"base_dataset_id": base_id, "production_cases_id": cases_id},
                )

                self.assertEqual(response.status_code, 201, response.text)
                payload = response.json()
                self.assertEqual(payload["state"], "RECEIVED")
                self.assertEqual(payload["job_status"], "QUEUED")
                fetched = client.get(f"/api/runs/{payload['id']}")
                self.assertEqual(fetched.json()["input_hash"], payload["input_hash"])

                cancelled = client.post(
                    f"/api/runs/{payload['id']}/cancel",
                    headers={"Idempotency-Key": "cancel-run-1"},
                )
                self.assertEqual(cancelled.status_code, 200, cancelled.text)
                self.assertEqual(cancelled.json()["job_status"], "CANCELLED")
                retried = client.post(
                    f"/api/runs/{payload['id']}/retry",
                    headers={"Idempotency-Key": "retry-run-1"},
                )
                self.assertEqual(retried.status_code, 201, retried.text)
                self.assertEqual(retried.json()["job_status"], "QUEUED")

    def test_create_run_idempotency_key_returns_original_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TestClient(create_app(WebConfig.for_test(Path(tmp)))) as client:
                base_id = self._dataset(
                    client,
                    "base",
                    b"query,text,relevance\nq,document,good\n",
                    {"query": "query", "text": "text", "relevance": "relevance"},
                )
                cases_id = self._dataset(
                    client,
                    "production_cases",
                    b"query,shown_doc_text,user_verdict\nq,document,good\n",
                    {"query": "query", "shown_doc_text": "shown_doc_text", "user_verdict": "user_verdict"},
                )
                first = client.post(
                    "/api/runs",
                    headers={"Idempotency-Key": "create-run-idempotent"},
                    json={"base_dataset_id": base_id, "production_cases_id": cases_id},
                )
                second = client.post(
                    "/api/runs",
                    headers={"Idempotency-Key": "create-run-idempotent"},
                    json={"base_dataset_id": base_id, "production_cases_id": cases_id},
                )

                self.assertEqual(first.status_code, 201, first.text)
                self.assertEqual(second.status_code, 201, second.text)
                self.assertEqual(second.json()["id"], first.json()["id"])

    def test_queued_run_can_execute_through_configured_core_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(WebConfig.for_test(Path(tmp)))
            with TestClient(app) as client:
                base_id = self._dataset(
                    client,
                    "base",
                    b"query_id,query,doc_id,text,relevance\nq1,query,d1,document,3\n",
                    {"query_id": "query_id", "query": "query", "doc_id": "doc_id", "text": "text", "relevance": "relevance"},
                )
                cases_id = self._dataset(
                    client,
                    "production_cases",
                    b"case_id,query,shown_doc_id,shown_doc_text,user_verdict\ncase-1,query,d1,document,good\n",
                    {"case_id": "case_id", "query": "query", "shown_doc_id": "shown_doc_id", "shown_doc_text": "shown_doc_text", "user_verdict": "user_verdict"},
                )
                created = client.post(
                    "/api/runs",
                    headers={"Idempotency-Key": "execute-run-1"},
                    json={"base_dataset_id": base_id, "production_cases_id": cases_id},
                )
                self.assertEqual(created.status_code, 201, created.text)
                run_id = created.json()["id"]

                job = app.state.job_executor.execute_next(app.state.runtime.run_existing)

                self.assertIsNotNone(job)
                fetched = client.get(f"/api/runs/{run_id}")
                self.assertEqual(fetched.status_code, 200, fetched.text)
                self.assertEqual(fetched.json()["state"], "READY_FOR_PROMOTION")
                self.assertEqual(fetched.json()["job_status"], "SUCCEEDED")


if __name__ == "__main__":
    unittest.main()
