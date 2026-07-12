"""Backend and promotion extension contracts for reckless-mode repairs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..reckless.contracts import (
    ActivationResult,
    ArtifactVerification,
    CandidateModel,
    CompiledInputs,
    Decision,
    EvaluationResult,
    PreparedActivation,
    PromotionReceipt,
    ReleaseSnapshot,
    RepairRequest,
    RunContext,
    SynthesizedFeatures,
    TargetValidation,
    ValidationResult,
)
from ..reckless.hashing import ExecutionIdentity


@runtime_checkable
class RepairBackend(Protocol):
    name: str

    def execution_identity(self) -> ExecutionIdentity: ...

    def validate(
        self, request: RepairRequest, context: RunContext
    ) -> ValidationResult: ...

    def compile(self, request: RepairRequest, context: RunContext) -> CompiledInputs: ...

    def train(self, inputs: CompiledInputs, context: RunContext) -> CandidateModel: ...

    def evaluate(
        self, candidate: CandidateModel, context: RunContext
    ) -> EvaluationResult: ...

    def verify_artifacts(
        self, candidate: CandidateModel, context: RunContext
    ) -> ArtifactVerification: ...


@runtime_checkable
class FeatureSynthesisBackend(Protocol):
    def synthesize_features(
        self,
        compiled: CompiledInputs,
        candidate: CandidateModel,
        evaluation: EvaluationResult,
        decision: Decision,
        context: RunContext,
    ) -> SynthesizedFeatures: ...


@runtime_checkable
class PromotionTarget(Protocol):
    name: str

    def validate_target(self, expected_current: str | None) -> TargetValidation: ...

    def prepare_release(self, release: ReleaseSnapshot) -> PreparedActivation: ...

    def activate(self, prepared: PreparedActivation) -> ActivationResult: ...

    def rollback(self, receipt: PromotionReceipt) -> ActivationResult: ...
