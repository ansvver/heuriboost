from __future__ import annotations

from typing import Mapping

from .contracts import _freeze_mapping, to_plain_data


def _reconstruct_heuriboost_error(
    error_type: type[HeuriBoostError],
    code: str,
    message: str,
    stage: str,
    run_id: str | None,
    retryable: bool,
    details: Mapping[str, object],
    operator_action: str,
) -> HeuriBoostError:
    error = error_type.__new__(error_type)
    HeuriBoostError.__init__(
        error,
        code,
        message,
        stage=stage,
        run_id=run_id,
        retryable=retryable,
        details=details,
        operator_action=operator_action,
    )
    return error


class HeuriBoostError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        run_id: str | None = None,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
        operator_action: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.run_id = run_id
        self.retryable = retryable
        self.details = _freeze_mapping({} if details is None else details)
        self.operator_action = operator_action

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "run_id": self.run_id,
            "retryable": self.retryable,
            "details": to_plain_data(self.details),
            "operator_action": self.operator_action,
        }

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            _reconstruct_heuriboost_error,
            (
                type(self),
                self.code,
                self.message,
                self.stage,
                self.run_id,
                self.retryable,
                self.details,
                self.operator_action,
            ),
        )


class InputBlockedError(HeuriBoostError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        run_id: str | None = None,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
        operator_action: str = "",
    ) -> None:
        super().__init__(
            "BLOCKED_INPUT",
            message,
            stage=stage,
            run_id=run_id,
            retryable=retryable,
            details=details,
            operator_action=operator_action,
        )


class NotEligibleError(HeuriBoostError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        run_id: str | None = None,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
        operator_action: str = "",
    ) -> None:
        super().__init__(
            "BLOCKED_NOT_ELIGIBLE",
            message,
            stage=stage,
            run_id=run_id,
            retryable=retryable,
            details=details,
            operator_action=operator_action,
        )


class EvaluationBlockedError(HeuriBoostError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        run_id: str | None = None,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
        operator_action: str = "",
    ) -> None:
        super().__init__(
            "BLOCKED_EVALUATION",
            message,
            stage=stage,
            run_id=run_id,
            retryable=retryable,
            details=details,
            operator_action=operator_action,
        )


class PromotionConflictError(HeuriBoostError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        run_id: str | None = None,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
        operator_action: str = "",
    ) -> None:
        super().__init__(
            "PROMOTION_CONFLICT",
            message,
            stage=stage,
            run_id=run_id,
            retryable=retryable,
            details=details,
            operator_action=operator_action,
        )


class ArtifactIntegrityError(HeuriBoostError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        run_id: str | None = None,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
        operator_action: str = "",
    ) -> None:
        super().__init__(
            "ARTIFACT_INTEGRITY",
            message,
            stage=stage,
            run_id=run_id,
            retryable=retryable,
            details=details,
            operator_action=operator_action,
        )
