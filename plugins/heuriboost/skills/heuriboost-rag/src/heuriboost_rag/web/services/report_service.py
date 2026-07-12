"""Report access for immutable Core Pre Promote artifacts."""

from __future__ import annotations

from ..runtime import WebRuntime
from ...reckless.contracts import ReportArtifact
from ...reckless.report import render_run_pre_promote_report, render_run_pre_promote_report_html


class ReportService:
    def __init__(self, runtime: WebRuntime) -> None:
        self.runtime = runtime

    def render(self, run_id: str, *, locale: str | None = None) -> ReportArtifact:
        run = self.runtime.stores.runs.get(run_id)
        return render_run_pre_promote_report(
            run,
            self.runtime.artifacts,
            locale=locale or self.runtime.config.default_locale,
        )

    def html(self, run_id: str, *, locale: str | None = None) -> str:
        run = self.runtime.stores.runs.get(run_id)
        return render_run_pre_promote_report_html(
            run,
            self.runtime.artifacts,
            locale=locale or self.runtime.config.default_locale,
        )


__all__ = ["ReportService"]
