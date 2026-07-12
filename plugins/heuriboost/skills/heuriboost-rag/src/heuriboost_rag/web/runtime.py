"""Startup-time Core wiring for the Web Console."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

from ..reckless.contracts import (
    ArtifactRef,
    ArtifactVerification,
    CandidateModel,
    CompiledInputs,
    EvaluationResult,
    RepairRequest,
    RunContext,
    RunRecord,
    ValidationResult,
)
from ..reckless.hashing import ExecutionIdentity, sha256_file
from ..reckless.orchestrator import run_existing_reckless_repair
from ..reckless.policy import RecklessPolicy, load_policy
from ..reckless.promotion import PromotionStores
from ..reckless.release_store import FileReleaseStore
from ..reckless.storage import LocalArtifactStore, OrchestratorStores
from ..adapters.workspace import LocalFilePromotionTarget
from .config import WebConfig
from .stores.sqlite import SQLiteStore


def _artifact(artifact_type: str, path: Path) -> ArtifactRef:
    return ArtifactRef(
        artifact_type=artifact_type,
        path=path,
        content_hash=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _load_object(spec: str) -> object:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError("import path must use module:object form")
    module = importlib.import_module(module_name)
    value: object = module
    for part in attr.split("."):
        value = getattr(value, part)
    return value


def _instantiate(spec: str, options: dict[str, object]) -> object:
    target = _load_object(spec)
    factory = getattr(target, "from_web_options", None)
    if callable(factory):
        return factory(options)
    if callable(target):
        return target(**options)
    if options:
        raise ValueError(f"{spec} does not accept backend_options")
    return target


class DemoRepairBackend:
    """Tiny deterministic backend for local smoke tests and unconfigured demos."""

    name = "demo-rag"

    def __init__(self, feature_set_name: str = "demo-rag") -> None:
        self.feature_set_name = feature_set_name

    def execution_identity(self) -> ExecutionIdentity:
        return ExecutionIdentity(
            backend_version="demo-v1",
            feature_names=("demo_score",),
            feature_version=self.feature_set_name,
            code_commit="web-demo",
            training_params={"rounds": 1},
            random_seed=0,
        )

    def validate(self, request: RepairRequest, context: RunContext) -> ValidationResult:
        errors: list[str] = []
        for role, dataset in context.datasets.items():
            if dataset.role != role:
                errors.append(f"{role} dataset has role {dataset.role!r}")
            if not dataset.path.is_file():
                errors.append(f"{role} dataset path is missing")
        return ValidationResult(
            valid=not errors,
            metadata={"reason": "valid" if not errors else "invalid", "errors": tuple(errors)},
            warnings=(),
        )

    def compile(self, request: RepairRequest, context: RunContext) -> CompiledInputs:
        compiled_dir = context.run_dir / "demo-compiled"
        compiled_dir.mkdir(parents=True, exist_ok=True)
        payload = compiled_dir / "compiled-input.json"
        payload.write_text('{"compiled":true}\n', encoding="utf-8")
        return CompiledInputs(
            artifacts=(_artifact("compiled-input", payload),),
            metadata={"feature_set_name": self.feature_set_name, "touched_domains": ("default",)},
        )

    def train(self, inputs: CompiledInputs, context: RunContext) -> CandidateModel:
        model_dir = context.run_dir / "demo-trained"
        model_dir.mkdir(parents=True, exist_ok=True)
        model = model_dir / "reranker.json"
        model.write_text('{"model":"demo"}\n', encoding="utf-8")
        metadata = model_dir / "reranker_metadata.json"
        metadata.write_text('{"feature_schema":["demo_score"]}\n', encoding="utf-8")
        return CandidateModel(
            model_path=model,
            artifacts=(
                *inputs.artifacts,
                _artifact("xgboost-model", model),
                _artifact("xgboost-model-metadata", metadata),
            ),
            metadata={"feature_set_name": self.feature_set_name},
        )

    def verify_artifacts(self, candidate: CandidateModel, context: RunContext) -> ArtifactVerification:
        return ArtifactVerification(valid=candidate.model_path.is_file(), errors=())

    def evaluate(self, candidate: CandidateModel, context: RunContext) -> EvaluationResult:
        return EvaluationResult(
            acceptance_level="full",
            current_cases_passed=True,
            historical_gates_passed=True,
            global_metrics={"ndcg@10": 1.0, "mrr@10": 1.0},
            anchor_metrics={"ndcg@10": 0.0, "mrr@10": 0.0},
            touched_domains={
                "default": {
                    "ndcg@10": 1.0,
                    "mrr@10": 1.0,
                    "anchor_ndcg@10": 0.0,
                    "anchor_mrr@10": 0.0,
                }
            },
            artifacts_valid=True,
            details={"current_case_count": 1},
        )


@dataclass(frozen=True)
class WebRuntime:
    config: WebConfig
    policy: RecklessPolicy
    backend: Any
    promotion_target: Any
    artifacts: LocalArtifactStore
    releases: FileReleaseStore
    stores: OrchestratorStores
    promotion_stores: PromotionStores

    def run_existing(self, run_id: str) -> RunRecord:
        return run_existing_reckless_repair(run_id, self.backend, self.stores, self.policy)


def build_runtime(config: WebConfig, store: SQLiteStore) -> WebRuntime:
    policy = load_policy(config.policy_path) if config.policy_path is not None else RecklessPolicy.default()
    backend = _instantiate(config.backend, dict(config.backend_options or {}))
    artifact_root = config.data_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifacts = LocalArtifactStore(artifact_root)
    releases = FileReleaseStore(config.data_dir / "releases")
    if config.promotion_target is None:
        promotion_target = LocalFilePromotionTarget(releases)
    else:
        target_class = _load_object(config.promotion_target)
        try:
            promotion_target = target_class(releases=releases)
        except TypeError:
            promotion_target = target_class()
    stores = OrchestratorStores(
        datasets=store.datasets,
        runs=store.runs,
        artifacts=artifacts,
    )
    promotion_stores = PromotionStores(
        runs=store.runs,
        artifacts=artifacts,
        promotions=store.promotions,
        releases=releases,
    )
    return WebRuntime(
        config=config,
        policy=policy,
        backend=backend,
        promotion_target=promotion_target,
        artifacts=artifacts,
        releases=releases,
        stores=stores,
        promotion_stores=promotion_stores,
    )


__all__ = ["DemoRepairBackend", "WebRuntime", "build_runtime"]
