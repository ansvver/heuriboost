from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from heuriboost_rag.web.app import create_app
from heuriboost_rag.web.config import WebConfig


class PromotionApiTests(unittest.TestCase):
    def _dataset(self, client: TestClient, role: str, content: bytes, mapping: dict[str, str]) -> str:
        uploaded = client.post("/api/imports", files={"file": (f"{role}.csv", content, "text/csv")})
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        normalized = client.post(
            f"/api/imports/{uploaded.json()['id']}/normalize",
            json={"role": role, "mapping": mapping},
        )
        self.assertEqual(normalized.status_code, 201, normalized.text)
        return normalized.json()["id"]

    def _ready_run(self, app, client: TestClient, key: str = "create-ready-run") -> str:
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
            headers={"Idempotency-Key": key},
            json={"base_dataset_id": base_id, "production_cases_id": cases_id},
        )
        self.assertEqual(created.status_code, 201, created.text)
        app.state.job_executor.execute_next(app.state.runtime.run_existing)
        return created.json()["id"]

    def _promote(self, client: TestClient, run_id: str, key: str) -> dict[str, object]:
        promoted = client.post(
            f"/api/promotions/{run_id}",
            headers={"Idempotency-Key": key},
            json={"approved_by": "local-operator"},
        )
        self.assertEqual(promoted.status_code, 201, promoted.text)
        return promoted.json()["receipt"]

    def test_report_page_and_promotion_use_core_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(WebConfig.for_test(Path(tmp)))
            with TestClient(app) as client:
                run_id = self._ready_run(app, client)

                report = client.get(f"/api/reports/{run_id}")
                self.assertEqual(report.status_code, 200, report.text)
                self.assertIn("<html", report.text.lower())
                self.assertEqual(report.text.lower().count("<!doctype"), 1)
                self.assertEqual(report.text.lower().count("<html"), 1)
                self.assertEqual(report.text.lower().count("<body"), 1)
                self.assertIn("sha256-", report.headers["content-security-policy"])
                live = client.get(f"/runs/{run_id}/report")
                self.assertEqual(live.status_code, 200, live.text)
                self.assertIn("批准并 Promote", live.text)
                self.assertIn("data-report-hash", live.text)
                self.assertEqual(live.text.lower().count("<!doctype"), 1)
                self.assertEqual(live.text.lower().count("<html"), 1)
                self.assertEqual(live.text.lower().count("<body"), 1)
                self.assertIn("sha256-", live.headers["content-security-policy"])
                self.assertIn("features", live.text)
                self.assertIn("explainability", live.text)
                self.assertIn("模型特征", live.text)
                self.assertIn("可解释性", live.text)
                self.assertIn("结论", live.text)
                self.assertIn("用了什么数据", live.text)
                self.assertIn("走了什么流程", live.text)
                self.assertIn("指标变化", live.text)
                self.assertIn("沉淀产物", live.text)
                self.assertIn("甘特图", live.text)
                self.assertIn('class="timeline"', live.text)
                self.assertIn("feature_synthesis", live.text)
                self.assertIn("feature_details", live.text)
                self.assertIn("data-feature-tooltip", live.text)
                self.assertNotIn("feature-table", live.text)
                self.assertNotIn("特征说明", live.text)

                payload = live.json() if live.headers.get("content-type") == "application/json" else {}
                promoted = client.post(
                    f"/api/promotions/{run_id}",
                    headers={"Idempotency-Key": "promote-ready-run"},
                    json={"approved_by": "local-operator"},
                )

                self.assertEqual(payload, {})
                self.assertEqual(promoted.status_code, 201, promoted.text)
                receipt = promoted.json()["receipt"]
                self.assertEqual(receipt["run_id"], run_id)
                self.assertEqual(receipt["current_model"], run_id)
                self.assertEqual(client.get(f"/api/runs/{run_id}").json()["state"], "PROMOTED")

    def test_rollback_restores_previous_core_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(WebConfig.for_test(Path(tmp)))
            with TestClient(app) as client:
                first = self._ready_run(app, client, "create-first-run")
                self._promote(client, first, "promote-first")
                second = self._ready_run(app, client, "create-second-run")
                self._promote(client, second, "promote-second")

                rolled_back = client.post(
                    f"/api/releases/{second}/rollback",
                    headers={"Idempotency-Key": "rollback-second"},
                    json={"approved_by": "local-operator"},
                )

                self.assertEqual(rolled_back.status_code, 201, rolled_back.text)
                receipt = rolled_back.json()["receipt"]
                self.assertEqual(receipt["source_run_id"], second)
                self.assertEqual(receipt["restored_model"], first)


if __name__ == "__main__":
    unittest.main()
