from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

from heuriboost_rag.reckless.contracts import RepairRequest
from heuriboost_rag.reckless.state import RunState
from heuriboost_rag.web.app import create_app
from heuriboost_rag.web.config import WebConfig


class AppBootstrapTests(unittest.TestCase):
    def test_health_endpoint_and_local_session_cookie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = WebConfig.for_test(Path(tmp))
            with TestClient(create_app(config)) as client:
                response = client.get("/health")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})

    def test_non_loopback_bind_requires_explicit_shared_mode(self) -> None:
        with self.assertRaises(ValueError):
            WebConfig(data_dir=Path("/tmp/hb"), host="0.0.0.0", shared_mode=False)

    def test_launch_url_omits_token_when_security_is_disabled(self) -> None:
        config = WebConfig(
            data_dir=Path("/tmp/hb-no-security"),
            security_enabled=False,
            session_token="hidden-token",
        )

        self.assertEqual(config.launch_url, "http://127.0.0.1:8787/")

    def test_security_disabled_allows_local_api_without_token_or_csrf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = WebConfig(
                data_dir=Path(tmp),
                security_enabled=False,
                job_worker_enabled=False,
            )
            with TestClient(create_app(config)) as client:
                response = client.post("/api/runs", json={})

            self.assertNotIn(response.status_code, {401, 403})

    def test_background_worker_executes_queued_run_on_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = WebConfig.for_test(Path(tmp)).with_job_worker(enabled=True, poll_seconds=0.01)
            app = create_app(config)
            run = app.state.store.runs.create(
                RepairRequest("workspace", "base", "cases", "1", "test", "tester"),
                "policy",
                "input",
            )
            app.state.job_executor.enqueue(run.run_id)

            def run_existing(run_id: str):
                record = app.state.store.runs.transition(run_id, RunState.VALIDATING)
                return app.state.store.runs.transition(record.run_id, RunState.BLOCKED_INPUT)

            object.__setattr__(app.state.runtime, "run_existing", run_existing)
            with TestClient(app) as client:
                for _ in range(100):
                    response = client.get(f"/api/runs/{run.run_id}")
                    if response.json()["job_status"] == "BLOCKED":
                        break
                    time.sleep(0.01)

            self.assertEqual(response.json()["state"], "BLOCKED_INPUT")
            self.assertEqual(response.json()["job_status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
