"""SQLite-backed single-worker queue state and recovery operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import os
from typing import Protocol
import uuid

from ...reckless.contracts import RunRecord
from ...reckless.state import RunState
from ..stores.sqlite import SQLiteStore


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    INTERRUPTED = "INTERRUPTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


_ACTIVE = frozenset({JobStatus.CLAIMED, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED})
_TERMINAL = frozenset({JobStatus.SUCCEEDED, JobStatus.BLOCKED, JobStatus.INTERRUPTED, JobStatus.CANCELLED, JobStatus.FAILED})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value: datetime | None) -> str:
    if value is None:
        return _utc_now()
    if value.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    run_id: str
    parent_job_id: str | None
    status: JobStatus
    version: int
    pid: int | None
    heartbeat_at: str | None
    cancel_requested_at: str | None
    attempt: int


class JobExecutor(Protocol):
    def enqueue(self, run_id: str) -> str: ...
    def request_cancel(self, job_id: str) -> None: ...
    def retry(self, job_id: str) -> str: ...


class LocalJobExecutor:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    @staticmethod
    def _record(row) -> JobRecord:
        return JobRecord(
            job_id=row["id"],
            run_id=row["run_id"],
            parent_job_id=row["parent_job_id"],
            status=JobStatus(row["status"]),
            version=int(row["version"]),
            pid=row["pid"],
            heartbeat_at=row["heartbeat_at"],
            cancel_requested_at=row["cancel_requested_at"],
            attempt=int(row["attempt"]),
        )

    def get(self, job_id: str) -> JobRecord:
        with self.store.connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"unknown job: {job_id}")
        return self._record(row)

    def enqueue(self, run_id: str, *, parent_job_id: str | None = None, attempt: int = 1) -> str:
        job_id = f"job-{uuid.uuid4().hex}"
        now = _utc_now()
        with self.store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, run_id, parent_job_id, status, version, pid, heartbeat_at,
                    cancel_requested_at, attempt, result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 1, NULL, ?, NULL, ?, NULL, ?, ?)
                """,
                (job_id, run_id, parent_job_id, JobStatus.QUEUED.value, now, attempt, now, now),
            )
        return job_id

    def claim_next(self) -> JobRecord | None:
        with self.store.transaction() as connection:
            active = connection.execute(
                "SELECT 1 FROM jobs WHERE status IN (?, ?, ?) LIMIT 1",
                tuple(status.value for status in _ACTIVE),
            ).fetchone()
            if active is not None:
                return None
            row = connection.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC, id ASC LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            now = _utc_now()
            result = connection.execute(
                """
                UPDATE jobs SET status = ?, version = version + 1, heartbeat_at = ?, updated_at = ?
                WHERE id = ? AND version = ? AND status = ?
                """,
                (JobStatus.CLAIMED.value, now, now, row["id"], row["version"], JobStatus.QUEUED.value),
            )
            if result.rowcount != 1:
                return None
            claimed = connection.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
        return None if claimed is None else self._record(claimed)

    def start(self, job_id: str, *, pid: int) -> JobRecord:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("pid must be a positive integer")
        job = self.get(job_id)
        if job.status is JobStatus.CANCEL_REQUESTED:
            return self._update(job, JobStatus.CANCELLED, pid=None)
        if job.status is not JobStatus.CLAIMED:
            raise ValueError("only CLAIMED jobs can start")
        return self._update(job, JobStatus.RUNNING, pid=pid)

    def heartbeat(self, job_id: str, *, occurred_at: datetime | None = None) -> JobRecord:
        job = self.get(job_id)
        if job.status not in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}:
            raise ValueError("only active jobs can send a heartbeat")
        with self.store.transaction() as connection:
            result = connection.execute(
                "UPDATE jobs SET heartbeat_at = ?, updated_at = ? WHERE id = ? AND version = ?",
                (_timestamp(occurred_at), _utc_now(), job.job_id, job.version),
            )
            if result.rowcount != 1:
                raise ValueError("job changed before its heartbeat was recorded")
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job.job_id,)).fetchone()
        return self._record(row)

    def request_cancel(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.status in _TERMINAL:
            return
        target = JobStatus.CANCELLED if job.status is JobStatus.QUEUED else JobStatus.CANCEL_REQUESTED
        self._update(job, target, cancel_requested_at=_utc_now(), pid=job.pid)

    def finish(self, job_id: str, status: JobStatus) -> JobRecord:
        if status not in {JobStatus.SUCCEEDED, JobStatus.BLOCKED, JobStatus.FAILED}:
            raise ValueError("finish status must be SUCCEEDED, BLOCKED, or FAILED")
        job = self.get(job_id)
        if job.status not in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}:
            raise ValueError("only active jobs can finish")
        return self._update(job, JobStatus.CANCELLED if job.status is JobStatus.CANCEL_REQUESTED else status, pid=None)

    def retry(self, job_id: str) -> str:
        job = self.get(job_id)
        if job.status not in _TERMINAL:
            raise ValueError("only terminal jobs can be retried")
        return self.enqueue(job.run_id, parent_job_id=job.job_id, attempt=job.attempt + 1)

    def execute_next(self, runner: Callable[[str], RunRecord]) -> JobRecord | None:
        claimed = self.claim_next()
        if claimed is None:
            return None
        running = self.start(claimed.job_id, pid=os.getpid())
        self.store.append_audit_event(
            "job.running",
            {"run_id": running.run_id, "job_id": running.job_id, "attempt": running.attempt},
        )
        try:
            run = runner(running.run_id)
        except InterruptedError:
            interrupted = self._update(running, JobStatus.INTERRUPTED, pid=None)
            self.store.append_audit_event(
                "job.interrupted",
                {"run_id": running.run_id, "job_id": running.job_id},
            )
            return interrupted
        except BaseException as exc:
            failed = self._update(running, JobStatus.FAILED, pid=None)
            self.store.append_audit_event(
                "job.failed",
                {
                    "run_id": running.run_id,
                    "job_id": running.job_id,
                    "exception_type": type(exc).__name__,
                },
            )
            raise
        status = JobStatus.SUCCEEDED
        if run.state in {
            RunState.BLOCKED_INPUT.value,
            RunState.BLOCKED_EVALUATION.value,
            RunState.BLOCKED_NOT_ELIGIBLE.value,
        }:
            status = JobStatus.BLOCKED
        elif run.state in {RunState.INTERRUPTED.value}:
            status = JobStatus.INTERRUPTED
        elif run.state.endswith("FAILED"):
            status = JobStatus.FAILED
        finished = self.finish(running.job_id, status if status is not JobStatus.INTERRUPTED else JobStatus.FAILED)
        if status is JobStatus.INTERRUPTED:
            finished = self._update(finished, JobStatus.INTERRUPTED, pid=None)
        self.store.append_audit_event(
            "job.finished",
            {"run_id": run.run_id, "job_id": running.job_id, "job_status": finished.status.value, "run_state": run.state},
        )
        return finished

    def recover_stale(
        self,
        *,
        heartbeat_before: datetime,
        is_pid_alive: Callable[[int], bool],
    ) -> tuple[str, ...]:
        cutoff = _timestamp(heartbeat_before)
        recovered: list[str] = []
        with self.store.transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status IN (?, ?, ?) AND (heartbeat_at IS NULL OR heartbeat_at < ?)
                ORDER BY created_at ASC, id ASC
                """,
                (JobStatus.CLAIMED.value, JobStatus.RUNNING.value, JobStatus.CANCEL_REQUESTED.value, cutoff),
            ).fetchall()
            for row in rows:
                pid = row["pid"]
                if isinstance(pid, int) and is_pid_alive(pid):
                    continue
                target = JobStatus.CANCELLED if row["status"] == JobStatus.CANCEL_REQUESTED.value else JobStatus.INTERRUPTED
                result = connection.execute(
                    """
                    UPDATE jobs SET status = ?, version = version + 1, pid = NULL, updated_at = ?
                    WHERE id = ? AND version = ?
                    """,
                    (target.value, _utc_now(), row["id"], row["version"]),
                )
                if result.rowcount == 1:
                    recovered.append(row["id"])
        return tuple(recovered)

    def _update(
        self,
        job: JobRecord,
        status: JobStatus,
        *,
        pid: int | None,
        cancel_requested_at: str | None = None,
    ) -> JobRecord:
        with self.store.transaction() as connection:
            result = connection.execute(
                """
                UPDATE jobs
                SET status = ?, version = version + 1, pid = ?, heartbeat_at = ?,
                    cancel_requested_at = COALESCE(?, cancel_requested_at), updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (status.value, pid, _utc_now(), cancel_requested_at, _utc_now(), job.job_id, job.version),
            )
            if result.rowcount != 1:
                raise ValueError("job changed before its state could be updated")
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job.job_id,)).fetchone()
        return self._record(row)


__all__ = ["JobExecutor", "JobRecord", "JobStatus", "LocalJobExecutor"]
