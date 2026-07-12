from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from heuriboost_rag.reckless.contracts import RepairRequest
from heuriboost_rag.web.jobs.executor import JobStatus, LocalJobExecutor
from heuriboost_rag.web.stores.sqlite import SQLiteStore


class JobExecutorTests(unittest.TestCase):
    def _run_id(self, store: SQLiteStore) -> str:
        return store.runs.create(
            RepairRequest(
                workspace_id="workspace",
                base_dataset_id="base",
                production_cases_id="cases",
                policy_version="1",
                backend_name="test",
                requested_by="tester",
            ),
            "policy",
            "input",
        ).run_id

    def test_claims_one_job_and_keeps_other_jobs_queued_after_store_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heuriboost.db"
            store = SQLiteStore(path)
            store.migrate()
            executor = LocalJobExecutor(store)
            first = executor.enqueue(self._run_id(store))
            second = executor.enqueue(self._run_id(store))

            claimed = executor.claim_next()
            self.assertEqual(claimed.job_id, first)
            self.assertIsNone(executor.claim_next())
            self.assertEqual(LocalJobExecutor(SQLiteStore(path)).get(second).status, JobStatus.QUEUED)

    def test_stale_heartbeats_become_interrupted_without_auto_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "heuriboost.db")
            store.migrate()
            executor = LocalJobExecutor(store)
            job_id = executor.enqueue(self._run_id(store))
            executor.claim_next()
            executor.start(job_id, pid=4242)
            stale = datetime.now(timezone.utc) - timedelta(minutes=10)
            executor.heartbeat(job_id, occurred_at=stale)

            recovered = executor.recover_stale(
                heartbeat_before=datetime.now(timezone.utc) - timedelta(minutes=1),
                is_pid_alive=lambda _: False,
            )

            self.assertEqual(recovered, (job_id,))
            self.assertEqual(executor.get(job_id).status, JobStatus.INTERRUPTED)

    def test_cancel_and_retry_create_new_history_without_mutating_original_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "heuriboost.db")
            store.migrate()
            executor = LocalJobExecutor(store)
            job_id = executor.enqueue(self._run_id(store))
            executor.request_cancel(job_id)
            self.assertEqual(executor.get(job_id).status, JobStatus.CANCELLED)

            retry_id = executor.retry(job_id)
            self.assertNotEqual(retry_id, job_id)
            self.assertEqual(executor.get(job_id).status, JobStatus.CANCELLED)
            retry = executor.get(retry_id)
            self.assertEqual(retry.status, JobStatus.QUEUED)
            self.assertEqual(retry.parent_job_id, job_id)


if __name__ == "__main__":
    unittest.main()
