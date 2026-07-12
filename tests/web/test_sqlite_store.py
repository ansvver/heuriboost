from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest

from heuriboost_rag.reckless.contracts import (
    DatasetRef,
    PromotionReceipt,
    RepairRequest,
)
from heuriboost_rag.reckless.hashing import sha256_file
from heuriboost_rag.reckless.errors import PromotionConflictError
from heuriboost_rag.reckless.state import RunState
from heuriboost_rag.web.stores.sqlite import SQLiteStore


_TABLES = {
    "workspaces",
    "uploads",
    "datasets",
    "import_profiles",
    "runs",
    "run_stages",
    "jobs",
    "artifacts",
    "approvals",
    "promotions",
    "audit_events",
    "schema_migrations",
}


class SQLiteStoreTests(unittest.TestCase):
    def _request(self) -> RepairRequest:
        return RepairRequest(
            workspace_id="workspace-1",
            base_dataset_id="base-1",
            production_cases_id="production-1",
            policy_version="policy-v1",
            backend_name="xgboost",
            requested_by="operator",
        )

    def _receipt(self, *, current_model: str = "model-v2") -> PromotionReceipt:
        return PromotionReceipt(
            run_id="run-1",
            release_path=Path("releases/run-1"),
            promoted_at="2026-07-10T00:00:00Z",
            approved_by="operator",
            previous_model="model-v1",
            current_model=current_model,
            release_manifest_hash="manifest-hash",
            receipt_json_path=Path("releases/run-1/promotion_receipt.json"),
            receipt_html_path=Path("releases/run-1/promotion_receipt.html"),
        )

    def test_migration_creates_exactly_required_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "heuriboost.db")
            store.migrate()

            self.assertEqual(store.table_names(), _TABLES)

    def test_audit_events_are_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "heuriboost.db")
            store.migrate()
            event_id = store.append_audit_event("run.created", {"run_id": "run-1"})

            with self.assertRaises(sqlite3.IntegrityError):
                store.update_audit_event(event_id, {"changed": True})
            with self.assertRaises(sqlite3.IntegrityError):
                store.delete_audit_event(event_id)

    def test_migration_rejects_a_database_newer_than_known_migrations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heuriboost.db"
            store = SQLiteStore(path)
            store.migrate()
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    """
                    INSERT INTO schema_migrations (version, name, checksum, applied_at)
                    VALUES (2, '0002_future.sql', 'future', '2026-07-10T00:00:00Z')
                    """
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(RuntimeError, "newer than this application"):
                SQLiteStore(path).migrate()

    def test_run_repository_uses_versions_to_reject_stale_writers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "heuriboost.db")
            store.migrate()
            record = store.runs.create(self._request(), "policy-hash", "input-hash")

            updated = store.runs.transition(
                record.run_id,
                RunState.VALIDATING,
                {"stage": "validation"},
            )

            self.assertEqual(updated.version, record.version + 1)
            self.assertEqual(updated.state, RunState.VALIDATING.value)
            self.assertEqual(updated.metadata["stage"], "validation")
            with self.assertRaisesRegex(ValueError, "stale run version"):
                store.runs.save(replace(record, metadata={"stale": True}))

    def test_dataset_repository_returns_core_dataset_ref_from_core_input_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteStore(root / "heuriboost.db")
            store.migrate()
            parquet_path = root / "normalized.parquet"
            parquet_path.write_bytes(b"parquet placeholder")
            core_input_path = root / "normalized.jsonl"
            core_input_path.write_text('{"query_id":"q1"}\n', encoding="utf-8")
            content_hash = sha256_file(core_input_path)
            with store.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO datasets (
                        id, workspace_id, upload_id, role, semantic_hash, schema_hash,
                        normalized_path, metadata_json, schema_version, status, created_at
                    ) VALUES (
                        'dataset-1', NULL, NULL, 'production_cases', 'semantic-hash',
                        'schema-hash', ?, ?, 1, 'READY', datetime('now')
                    )
                    """,
                    (
                        str(parquet_path),
                        '{"core_input_path":"' + str(core_input_path) + '","core_input_hash":"' + content_hash + '"}',
                    ),
                )

            dataset = store.datasets.get("dataset-1")

            self.assertIsInstance(dataset, DatasetRef)
            self.assertEqual(dataset.dataset_id, "dataset-1")
            self.assertEqual(dataset.role, "production_cases")
            self.assertEqual(dataset.path, core_input_path.resolve())
            self.assertEqual(dataset.content_hash, content_hash)

    def test_promotion_repository_is_idempotent_and_rejects_key_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "heuriboost.db")
            store.migrate()
            receipt = self._receipt()

            store.promotions.save(receipt, "promotion-key")
            store.promotions.save(receipt, "promotion-key")

            self.assertEqual(
                store.promotions.find_by_idempotency_key("promotion-key"),
                receipt,
            )
            with self.assertRaises(PromotionConflictError):
                store.promotions.save(
                    self._receipt(current_model="other-model"),
                    "promotion-key",
                )


if __name__ == "__main__":
    unittest.main()
