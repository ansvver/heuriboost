"""Startup recovery for interrupted local worker jobs."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from .executor import LocalJobExecutor


class JobSupervisor:
    def __init__(self, executor: LocalJobExecutor, is_pid_alive: Callable[[int], bool]) -> None:
        self.executor = executor
        self.is_pid_alive = is_pid_alive

    def recover(self, heartbeat_before: datetime) -> tuple[str, ...]:
        return self.executor.recover_stale(
            heartbeat_before=heartbeat_before,
            is_pid_alive=self.is_pid_alive,
        )
