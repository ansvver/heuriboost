"""Promotion application service for Web-approved Core releases."""

from __future__ import annotations

from datetime import datetime, timezone

from ...reckless.contracts import PromotionApproval, to_plain_data
from ...reckless.promotion import promote_repair, rollback_release
from ..runtime import WebRuntime
from .report_service import ReportService


class PromotionService:
    def __init__(self, runtime: WebRuntime, reports: ReportService) -> None:
        self.runtime = runtime
        self.reports = reports

    def promote(self, run_id: str, *, approved_by: str, idempotency_key: str) -> dict[str, object]:
        report = self.reports.render(run_id)
        decision_hash = report.manifest.get("decision_hash")
        if not isinstance(decision_hash, str) or not decision_hash:
            raise ValueError("pre-promote report does not contain a decision hash")
        approval = PromotionApproval(
            run_id=run_id,
            approved_by=approved_by,
            approved_at=datetime.now(timezone.utc).isoformat(),
            report_hash=report.html_hash,
            decision_hash=decision_hash,
            expected_current_model=self.runtime.releases.read_current_model(),
            idempotency_key=idempotency_key,
        )
        receipt = promote_repair(
            run_id,
            approval,
            self.runtime.promotion_target,
            self.runtime.promotion_stores,
        )
        return {"receipt": to_plain_data(receipt)}

    def rollback(self, run_id: str, *, approved_by: str) -> dict[str, object]:
        existing = self.runtime.releases.read_promotion_receipt(run_id)
        if existing is None:
            raise FileNotFoundError(f"promotion receipt not found for run: {run_id}")
        receipt, _ = existing
        rolled_back = rollback_release(
            receipt,
            self.runtime.promotion_target,
            self.runtime.promotion_stores,
            approved_by=approved_by,
        )
        return {"receipt": to_plain_data(rolled_back)}


__all__ = ["PromotionService"]
