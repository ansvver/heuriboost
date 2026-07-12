from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from heuriboost_rag.web.app import create_app
from heuriboost_rag.web.config import WebConfig


class SseTests(unittest.TestCase):
    def test_replays_events_after_last_event_id_from_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(WebConfig.for_test(Path(tmp)))
            store = app.state.store
            store.append_audit_event("run.created", {"run_id": "run-1", "status": "QUEUED"})
            store.append_audit_event("stage.progress", {"run_id": "run-1", "stage": "TRAINING", "status": "RUNNING"})
            store.append_audit_event("run.ready", {"run_id": "run-1", "status": "READY_FOR_PROMOTION"})

            with TestClient(app) as client:
                response = client.get("/api/runs/run-1/events", headers={"Last-Event-ID": "1"})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
            self.assertIn("id: 2", response.text)
            self.assertIn("id: 3", response.text)
            self.assertNotIn("id: 1", response.text)
            self.assertLess(response.text.index("id: 2"), response.text.index("id: 3"))


if __name__ == "__main__":
    unittest.main()
