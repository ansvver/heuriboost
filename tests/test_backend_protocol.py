from __future__ import annotations

import unittest

from heuriboost_rag.backends.base import PromotionTarget, RepairBackend
from heuriboost_rag.reckless.contracts import (
    ActivationResult,
    ArtifactVerification,
    CandidateModel,
    CompiledInputs,
    EvaluationResult,
    PreparedActivation,
    PromotionReceipt,
    ReleaseSnapshot,
    RepairRequest,
    RunContext,
    TargetValidation,
    ValidationResult,
)
from heuriboost_rag.reckless.hashing import ExecutionIdentity


class CompleteRepairBackend:
    name = "fake-repair"

    def execution_identity(self) -> ExecutionIdentity:
        raise AssertionError("test fake methods are not invoked")

    def validate(
        self, request: RepairRequest, context: RunContext
    ) -> ValidationResult:
        raise AssertionError("test fake methods are not invoked")

    def compile(self, request: RepairRequest, context: RunContext) -> CompiledInputs:
        raise AssertionError("test fake methods are not invoked")

    def train(self, inputs: CompiledInputs, context: RunContext) -> CandidateModel:
        raise AssertionError("test fake methods are not invoked")

    def evaluate(
        self, candidate: CandidateModel, context: RunContext
    ) -> EvaluationResult:
        raise AssertionError("test fake methods are not invoked")

    def verify_artifacts(
        self, candidate: CandidateModel, context: RunContext
    ) -> ArtifactVerification:
        raise AssertionError("test fake methods are not invoked")


class IncompleteRepairBackend:
    name = "incomplete-repair"

    def execution_identity(self) -> ExecutionIdentity:
        raise AssertionError("test fake methods are not invoked")

    def validate(
        self, request: RepairRequest, context: RunContext
    ) -> ValidationResult:
        raise AssertionError("test fake methods are not invoked")

    def compile(self, request: RepairRequest, context: RunContext) -> CompiledInputs:
        raise AssertionError("test fake methods are not invoked")

    def train(self, inputs: CompiledInputs, context: RunContext) -> CandidateModel:
        raise AssertionError("test fake methods are not invoked")

    def evaluate(
        self, candidate: CandidateModel, context: RunContext
    ) -> EvaluationResult:
        raise AssertionError("test fake methods are not invoked")


class CompletePromotionTarget:
    name = "fake-promotion"

    def validate_target(self, expected_current: str | None) -> TargetValidation:
        raise AssertionError("test fake methods are not invoked")

    def prepare_release(self, release: ReleaseSnapshot) -> PreparedActivation:
        raise AssertionError("test fake methods are not invoked")

    def activate(self, prepared: PreparedActivation) -> ActivationResult:
        raise AssertionError("test fake methods are not invoked")

    def rollback(self, receipt: PromotionReceipt) -> ActivationResult:
        raise AssertionError("test fake methods are not invoked")


class IncompletePromotionTarget:
    name = "incomplete-promotion"

    def validate_target(self, expected_current: str | None) -> TargetValidation:
        raise AssertionError("test fake methods are not invoked")

    def prepare_release(self, release: ReleaseSnapshot) -> PreparedActivation:
        raise AssertionError("test fake methods are not invoked")

    def activate(self, prepared: PreparedActivation) -> ActivationResult:
        raise AssertionError("test fake methods are not invoked")


class BackendProtocolTests(unittest.TestCase):
    def test_repair_backend_is_runtime_checkable(self) -> None:
        self.assertIsInstance(CompleteRepairBackend(), RepairBackend)
        self.assertNotIsInstance(IncompleteRepairBackend(), RepairBackend)

    def test_promotion_target_is_runtime_checkable(self) -> None:
        self.assertIsInstance(CompletePromotionTarget(), PromotionTarget)
        self.assertNotIsInstance(IncompletePromotionTarget(), PromotionTarget)


if __name__ == "__main__":
    unittest.main()
