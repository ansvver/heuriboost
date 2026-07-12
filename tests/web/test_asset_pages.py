from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from heuriboost_rag.web.app import create_app
from heuriboost_rag.web.config import WebConfig


class AssetPageTests(unittest.TestCase):
    def test_read_only_asset_audit_and_settings_pages_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(WebConfig.for_test(Path(tmp)))
            app.state.store.append_audit_event("asset.fixture", {"run_id": "run-1"})
            with TestClient(app) as client:
                for path, marker in (
                    ("/datasets", "数据集"),
                    ("/models", "模型版本"),
                    ("/gates", "回归门禁"),
                    ("/audit", "asset.fixture"),
                    ("/settings", "default-rag"),
                ):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertIn(marker, response.text)


if __name__ == "__main__":
    unittest.main()
