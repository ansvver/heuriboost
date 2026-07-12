"""Worker-facing helpers kept separate from HTTP request handling."""

from __future__ import annotations

from collections.abc import Callable

from .executor import JobStatus, LocalJobExecutor


def complete_claimed_job(executor: LocalJobExecutor, job_id: str, pid: int, run: Callable[[], JobStatus]) -> None:
    executor.start(job_id, pid=pid)
    executor.finish(job_id, run())
