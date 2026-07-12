from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from heuriboost_rag.web.app import create_app
from heuriboost_rag.web.config import WebConfig


class SecurityTests(unittest.TestCase):
    def test_launch_token_sets_http_only_cookie_and_writes_require_csrf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = WebConfig(
                data_dir=Path(tmp),
                session_token="launch-token",
                csrf_token="csrf-token",
                security_enabled=True,
            )
            with TestClient(create_app(config)) as client:
                blocked = client.get("/")
                self.assertEqual(blocked.status_code, 401)

                launched = client.get("/?token=launch-token")
                self.assertEqual(launched.status_code, 200)
                self.assertIn("httponly", launched.headers["set-cookie"].lower())
                self.assertEqual(launched.headers["x-frame-options"], "DENY")
                self.assertIn(
                    '<meta name="csrf-token" content="csrf-token">',
                    launched.text,
                )
                self.assertIn("/static/console.js?v=", launched.text)

                missing_csrf = client.post("/api/runs", headers={"Idempotency-Key": "k"}, json={})
                self.assertEqual(missing_csrf.status_code, 403)
                invalid_payload = client.post(
                    "/api/runs",
                    headers={"Idempotency-Key": "k", "X-CSRF-Token": "csrf-token"},
                    json={},
                )
                self.assertEqual(invalid_payload.status_code, 422)

    def test_console_javascript_sends_csrf_header_from_page_meta(self) -> None:
        static_js = (
            Path(__file__).resolve().parents[2]
            / "plugins"
            / "heuriboost"
            / "skills"
            / "heuriboost-rag"
            / "src"
            / "heuriboost_rag"
            / "web"
            / "static"
            / "console.js"
        ).read_text(encoding="utf-8")

        self.assertIn('meta[name="csrf-token"]', static_js)
        self.assertIn('"X-CSRF-Token"', static_js)

    def test_workbench_javascript_persists_uploads_and_can_start_runs(self) -> None:
        static_js = (
            Path(__file__).resolve().parents[2]
            / "plugins"
            / "heuriboost"
            / "skills"
            / "heuriboost-rag"
            / "src"
            / "heuriboost_rag"
            / "web"
            / "static"
            / "console.js"
        ).read_text(encoding="utf-8")

        self.assertIn('`/api/imports/${upload.id}/normalize`', static_js)
        self.assertIn('requestJson("/api/datasets"', static_js)
        self.assertIn('requestJson("/api/runs"', static_js)
        self.assertIn('"Idempotency-Key"', static_js)
        self.assertIn("query_text", static_js)
        self.assertIn("raw_final_score", static_js)


if __name__ == "__main__":
    unittest.main()
