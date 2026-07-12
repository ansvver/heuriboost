"""Stage-by-stage execution for immutable, resumable Reckless repair runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .contracts import (
    ArtifactRef,
    ArtifactVerification,
    CandidateModel,
    CompiledInputs,
    DatasetRef,
    Decision,
    EvaluationResult,
    RepairRequest,
    RunContext,
    RunRecord,
    StageManifest,
    SynthesizedFeatures,
    ValidationResult,
    to_plain_data,
)
from .errors import (
    ArtifactIntegrityError,
    EvaluationBlockedError,
    HeuriBoostError,
    InputBlockedError,
    NotEligibleError,
)
from .hashing import (
    ExecutionIdentity,
    atomic_write_json,
    build_run_fingerprint,
    canonical_json_hash,
)
from .policy import RecklessPolicy, evaluate_promotion_eligibility
from .state import RunState
from .storage import ArtifactStore, OrchestratorStores


_COMPILED_STAGE = RunState.COMPILED.value
_TRAINED_STAGE = RunState.TRAINED.value
_FEATURE_SYNTHESIS_STAGE = "FEATURE_SYNTHESIS"
_TRAINED_RETRY_STAGE = "TRAINED_RETRY"
_REPORTING_STAGE = RunState.REPORTING.value
_STAGE_RESULT_ARTIFACT = "stage-result"
_REPORT_EVIDENCE_ARTIFACT = "report-evidence"
_FALLBACK_MODEL_ARTIFACT = "candidate-model"
_STAGE_RESULT_SCHEMA_VERSION = 1
_REPORT_EVIDENCE_SCHEMA_VERSION = 1
_MAX_RESULT_JSON_BYTES = 1024 * 1024


def _artifact_descriptor(artifact: ArtifactRef) -> dict[str, object]:
    return {
        "artifact_type": artifact.artifact_type,
        "content_hash": artifact.content_hash,
        "size_bytes": artifact.size_bytes,
    }


def _manifest_summary(manifest: StageManifest) -> dict[str, object]:
    artifacts = tuple(
        _artifact_descriptor(artifact)
        for artifact in sorted(manifest.artifacts, key=lambda item: item.artifact_type)
    )
    return {
        "stage": manifest.stage,
        "input_hash": manifest.input_hash,
        "artifacts": [to_plain_data(artifact) for artifact in manifest.artifacts],
        "artifact_set_hash": canonical_json_hash(artifacts),
    }


def _execution_identity_data(identity: ExecutionIdentity) -> dict[str, object]:
    return {
        "backend_version": identity.backend_version,
        "feature_names": list(identity.feature_names),
        "feature_version": identity.feature_version,
        "code_commit": identity.code_commit,
        "training_params": to_plain_data(identity.training_params),
        "random_seed": identity.random_seed,
    }


def _result_source_path(context: RunContext, stage: str, name: str) -> Path:
    return context.run_dir / "orchestrator" / stage.lower() / name


def _ensure_unique_artifacts(
    artifacts: Sequence[ArtifactRef],
    *,
    stage: str,
    run_id: str,
) -> tuple[ArtifactRef, ...]:
    if not all(isinstance(artifact, ArtifactRef) for artifact in artifacts):
        raise ArtifactIntegrityError(
            "backend returned an invalid artifact reference",
            stage=stage,
            run_id=run_id,
        )
    values = tuple(artifacts)
    types = [artifact.artifact_type for artifact in values]
    if len(types) != len(set(types)):
        raise ArtifactIntegrityError(
            "backend returned duplicate artifact types",
            stage=stage,
            run_id=run_id,
        )
    if _STAGE_RESULT_ARTIFACT in types or _REPORT_EVIDENCE_ARTIFACT in types:
        raise ArtifactIntegrityError(
            "backend returned an orchestrator-owned artifact type",
            stage=stage,
            run_id=run_id,
        )
    return values


def _binding_references(artifacts: Sequence[ArtifactRef]) -> list[dict[str, object]]:
    return [
        _artifact_descriptor(artifact)
        for artifact in artifacts
        if "binding" in artifact.artifact_type
    ]


def _compiled_stage_result(
    run: RunRecord,
    validation: ValidationResult,
    compiled: CompiledInputs,
) -> dict[str, object]:
    artifacts = _ensure_unique_artifacts(
        compiled.artifacts,
        stage=_COMPILED_STAGE,
        run_id=run.run_id,
    )
    return {
        "schema_version": _STAGE_RESULT_SCHEMA_VERSION,
        "stage": _COMPILED_STAGE,
        "input_hash": run.input_hash,
        "metadata": to_plain_data(compiled.metadata),
        "artifact_types": [artifact.artifact_type for artifact in artifacts],
        "binding_references": _binding_references(artifacts),
        "validation": to_plain_data(validation),
    }


def _candidate_model_artifact_type(
    candidate: CandidateModel,
    artifacts: tuple[ArtifactRef, ...],
    *,
    stage: str,
    run_id: str,
) -> str:
    model_refs = [
        artifact for artifact in artifacts if artifact.artifact_type == "xgboost-model"
    ]
    if not model_refs:
        return _FALLBACK_MODEL_ARTIFACT
    if len(model_refs) != 1 or model_refs[0].path != candidate.model_path:
        raise ArtifactIntegrityError(
            "candidate model path does not match xgboost-model artifact",
            stage=stage,
            run_id=run_id,
        )
    return "xgboost-model"


def _trained_stage_result(
    run: RunRecord,
    candidate: CandidateModel,
    *,
    stage: str = _TRAINED_STAGE,
) -> tuple[dict[str, object], str]:
    artifacts = _ensure_unique_artifacts(
        candidate.artifacts,
        stage=stage,
        run_id=run.run_id,
    )
    model_artifact_type = _candidate_model_artifact_type(
        candidate,
        artifacts,
        stage=stage,
        run_id=run.run_id,
    )
    return (
        {
            "schema_version": _STAGE_RESULT_SCHEMA_VERSION,
            "stage": stage,
            "input_hash": run.input_hash,
            "metadata": to_plain_data(candidate.metadata),
            "artifact_types": [artifact.artifact_type for artifact in artifacts],
            "binding_references": _binding_references(artifacts),
            "model_artifact_type": model_artifact_type,
        },
        model_artifact_type,
    )


def _feature_synthesis_stage_result(
    run: RunRecord,
    synthesized: SynthesizedFeatures,
) -> dict[str, object]:
    synthesis_artifacts = _ensure_unique_artifacts(
        synthesized.artifacts,
        stage=_FEATURE_SYNTHESIS_STAGE,
        run_id=run.run_id,
    )
    compiled_artifacts = _ensure_unique_artifacts(
        synthesized.compiled_inputs.artifacts,
        stage=_FEATURE_SYNTHESIS_STAGE,
        run_id=run.run_id,
    )
    return {
        "schema_version": _STAGE_RESULT_SCHEMA_VERSION,
        "stage": _FEATURE_SYNTHESIS_STAGE,
        "input_hash": run.input_hash,
        "metadata": to_plain_data(synthesized.metadata),
        "artifact_types": [artifact.artifact_type for artifact in synthesis_artifacts],
        "binding_references": _binding_references(synthesis_artifacts),
        "compiled_inputs": {
            "metadata": to_plain_data(synthesized.compiled_inputs.metadata),
            "artifact_types": [
                artifact.artifact_type for artifact in compiled_artifacts
            ],
            "binding_references": _binding_references(compiled_artifacts),
        },
    }


def _artifact_sources(
    artifacts: Sequence[ArtifactRef],
    *,
    extra_type: str,
    extra_path: Path,
    stage: str,
    run_id: str,
) -> dict[str, Path]:
    values = _ensure_unique_artifacts(artifacts, stage=stage, run_id=run_id)
    sources = {artifact.artifact_type: artifact.path for artifact in values}
    if extra_type in sources:
        raise ArtifactIntegrityError(
            "stage artifact type conflicts with an orchestrator artifact",
            stage=stage,
            run_id=run_id,
            details={"artifact_type": extra_type},
        )
    sources[extra_type] = extra_path
    return sources


def _write_json_artifact(
    context: RunContext,
    stage: str,
    name: str,
    payload: Mapping[str, object],
) -> Path:
    path = _result_source_path(context, stage, name)
    atomic_write_json(path, payload)
    return path


def _rebase_artifact(
    store: ArtifactStore,
    artifact: ArtifactRef,
    *,
    stage: str,
    run_id: str,
) -> ArtifactRef:
    root = getattr(store, "root", None)
    if not isinstance(root, Path):
        raise ArtifactIntegrityError(
            "artifact store must expose a filesystem root for stage restoration",
            stage=stage,
            run_id=run_id,
        )
    relative = Path(artifact.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactIntegrityError(
            "completed stage contains an unsafe artifact path",
            stage=stage,
            run_id=run_id,
            details={"artifact_type": artifact.artifact_type},
        )
    return ArtifactRef(
        artifact_type=artifact.artifact_type,
        path=root / relative,
        content_hash=artifact.content_hash,
        size_bytes=artifact.size_bytes,
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_verified_json(
    store: ArtifactStore,
    artifact: ArtifactRef,
    *,
    stage: str,
    run_id: str,
) -> Mapping[str, object]:
    rebased = _rebase_artifact(store, artifact, stage=stage, run_id=run_id)
    try:
        raw = rebased.path.read_bytes()
    except OSError as exc:
        raise ArtifactIntegrityError(
            "completed stage result cannot be read",
            stage=stage,
            run_id=run_id,
            details={"artifact_type": artifact.artifact_type},
        ) from exc
    if len(raw) != artifact.size_bytes or hashlib.sha256(raw).hexdigest() != artifact.content_hash:
        raise ArtifactIntegrityError(
            "completed stage result no longer matches its manifest",
            stage=stage,
            run_id=run_id,
            details={"artifact_type": artifact.artifact_type},
        )
    if len(raw) > _MAX_RESULT_JSON_BYTES:
        raise ArtifactIntegrityError(
            "completed stage result exceeds the JSON size limit",
            stage=stage,
            run_id=run_id,
            details={"artifact_type": artifact.artifact_type},
        )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            "completed stage result is not strict JSON",
            stage=stage,
            run_id=run_id,
            details={"artifact_type": artifact.artifact_type},
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(
            "completed stage result must be a JSON object",
            stage=stage,
            run_id=run_id,
            details={"artifact_type": artifact.artifact_type},
        )
    return value


def _require_fields(
    payload: Mapping[str, object],
    fields: frozenset[str],
    *,
    stage: str,
    run_id: str,
) -> None:
    if set(payload) != fields:
        raise ArtifactIntegrityError(
            "completed stage result has an unexpected schema",
            stage=stage,
            run_id=run_id,
            details={"fields": sorted(payload)},
        )


def _require_nonempty_string(
    value: object,
    label: str,
    *,
    stage: str,
    run_id: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactIntegrityError(
            f"completed stage result has an invalid {label}",
            stage=stage,
            run_id=run_id,
        )
    return value


def _result_artifacts(
    manifest: StageManifest,
    payload: Mapping[str, object],
    store: ArtifactStore,
    *,
    stage: str,
    run_id: str,
    core_only_types: frozenset[str] = frozenset(),
) -> tuple[ArtifactRef, ...]:
    raw_types = payload.get("artifact_types")
    if not isinstance(raw_types, list) or not all(
        isinstance(item, str) and item for item in raw_types
    ):
        raise ArtifactIntegrityError(
            "completed stage result has an invalid artifact type order",
            stage=stage,
            run_id=run_id,
        )
    types = tuple(raw_types)
    if len(types) != len(set(types)) or _STAGE_RESULT_ARTIFACT in types:
        raise ArtifactIntegrityError(
            "completed stage result has duplicate or reserved artifact types",
            stage=stage,
            run_id=run_id,
        )
    manifest_by_type = {artifact.artifact_type: artifact for artifact in manifest.artifacts}
    expected_manifest_types = {
        artifact.artifact_type
        for artifact in manifest.artifacts
        if artifact.artifact_type not in {_STAGE_RESULT_ARTIFACT, *core_only_types}
    }
    if set(types) != expected_manifest_types:
        raise ArtifactIntegrityError(
            "completed stage result does not bind the manifest artifact types",
            stage=stage,
            run_id=run_id,
        )
    refs = tuple(
        _rebase_artifact(store, manifest_by_type[artifact_type], stage=stage, run_id=run_id)
        for artifact_type in types
    )
    raw_bindings = payload.get("binding_references")
    expected_bindings = _binding_references(refs)
    if raw_bindings != expected_bindings:
        raise ArtifactIntegrityError(
            "completed stage result binding references do not match artifacts",
            stage=stage,
            run_id=run_id,
        )
    return refs


def _validation_from_data(
    value: object,
    *,
    stage: str,
    run_id: str,
) -> ValidationResult:
    if not isinstance(value, Mapping) or set(value) != {"valid", "metadata", "warnings"}:
        raise ArtifactIntegrityError(
            "completed stage result has invalid validation data",
            stage=stage,
            run_id=run_id,
        )
    valid = value["valid"]
    metadata = value["metadata"]
    warnings = value["warnings"]
    if type(valid) is not bool or not isinstance(metadata, Mapping):
        raise ArtifactIntegrityError(
            "completed stage result has invalid validation fields",
            stage=stage,
            run_id=run_id,
        )
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise ArtifactIntegrityError(
            "completed stage result has invalid validation warnings",
            stage=stage,
            run_id=run_id,
        )
    try:
        return ValidationResult(
            valid=valid,
            metadata=dict(metadata),
            warnings=tuple(warnings),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "completed stage result has unsafe validation data",
            stage=stage,
            run_id=run_id,
        ) from exc


def _restore_compiled_inputs(
    manifest: StageManifest,
    store: ArtifactStore,
    *,
    run_id: str,
    input_hash: str,
) -> tuple[CompiledInputs, ValidationResult]:
    result_refs = [
        artifact
        for artifact in manifest.artifacts
        if artifact.artifact_type == _STAGE_RESULT_ARTIFACT
    ]
    if len(result_refs) != 1:
        raise ArtifactIntegrityError(
            "compiled stage must contain exactly one stage-result artifact",
            stage=_COMPILED_STAGE,
            run_id=run_id,
        )
    payload = _read_verified_json(
        store,
        result_refs[0],
        stage=_COMPILED_STAGE,
        run_id=run_id,
    )
    _require_fields(
        payload,
        frozenset(
            {
                "schema_version",
                "stage",
                "input_hash",
                "metadata",
                "artifact_types",
                "binding_references",
                "validation",
            }
        ),
        stage=_COMPILED_STAGE,
        run_id=run_id,
    )
    if payload["schema_version"] != _STAGE_RESULT_SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            "compiled stage result schema version is unsupported",
            stage=_COMPILED_STAGE,
            run_id=run_id,
        )
    if payload["stage"] != _COMPILED_STAGE or payload["input_hash"] != input_hash:
        raise ArtifactIntegrityError(
            "compiled stage result does not match the requested run inputs",
            stage=_COMPILED_STAGE,
            run_id=run_id,
        )
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ArtifactIntegrityError(
            "compiled stage result metadata must be an object",
            stage=_COMPILED_STAGE,
            run_id=run_id,
        )
    artifacts = _result_artifacts(
        manifest,
        payload,
        store,
        stage=_COMPILED_STAGE,
        run_id=run_id,
    )
    try:
        compiled = CompiledInputs(artifacts=artifacts, metadata=dict(metadata))
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "compiled stage result metadata is unsafe",
            stage=_COMPILED_STAGE,
            run_id=run_id,
        ) from exc
    return compiled, _validation_from_data(
        payload["validation"],
        stage=_COMPILED_STAGE,
        run_id=run_id,
    )


def _restore_candidate_model(
    manifest: StageManifest,
    store: ArtifactStore,
    *,
    run_id: str,
    input_hash: str,
    stage: str = _TRAINED_STAGE,
) -> CandidateModel:
    result_refs = [
        artifact
        for artifact in manifest.artifacts
        if artifact.artifact_type == _STAGE_RESULT_ARTIFACT
    ]
    if len(result_refs) != 1:
        raise ArtifactIntegrityError(
            "trained stage must contain exactly one stage-result artifact",
            stage=stage,
            run_id=run_id,
        )
    payload = _read_verified_json(
        store,
        result_refs[0],
        stage=stage,
        run_id=run_id,
    )
    _require_fields(
        payload,
        frozenset(
            {
                "schema_version",
                "stage",
                "input_hash",
                "metadata",
                "artifact_types",
                "binding_references",
                "model_artifact_type",
            }
        ),
        stage=stage,
        run_id=run_id,
    )
    if payload["schema_version"] != _STAGE_RESULT_SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            "trained stage result schema version is unsupported",
            stage=stage,
            run_id=run_id,
        )
    if payload["stage"] != stage or payload["input_hash"] != input_hash:
        raise ArtifactIntegrityError(
            "trained stage result does not match the requested run inputs",
            stage=stage,
            run_id=run_id,
        )
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ArtifactIntegrityError(
            "trained stage result metadata must be an object",
            stage=stage,
            run_id=run_id,
        )
    model_artifact_type = _require_nonempty_string(
        payload["model_artifact_type"],
        "model_artifact_type",
        stage=stage,
        run_id=run_id,
    )
    artifacts = _result_artifacts(
        manifest,
        payload,
        store,
        stage=stage,
        run_id=run_id,
        core_only_types=(
            frozenset({_FALLBACK_MODEL_ARTIFACT})
            if model_artifact_type == _FALLBACK_MODEL_ARTIFACT
            else frozenset()
        ),
    )
    model_ref = next(
        (
            artifact
            for artifact in manifest.artifacts
            if artifact.artifact_type == model_artifact_type
        ),
        None,
    )
    if model_ref is None:
        raise ArtifactIntegrityError(
            "trained stage result model artifact is missing",
            stage=stage,
            run_id=run_id,
        )
    rebased_model = _rebase_artifact(
        store,
        model_ref,
        stage=stage,
        run_id=run_id,
    )
    candidate_artifacts = artifacts
    if model_artifact_type == _FALLBACK_MODEL_ARTIFACT:
        candidate_artifacts = (*artifacts, rebased_model)
    try:
        return CandidateModel(
            model_path=rebased_model.path,
            artifacts=candidate_artifacts,
            metadata=dict(metadata),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "trained stage result metadata is unsafe",
            stage=stage,
            run_id=run_id,
        ) from exc


def _seal_compiled_stage(
    run: RunRecord,
    context: RunContext,
    validation: ValidationResult,
    compiled: CompiledInputs,
    stores: OrchestratorStores,
) -> tuple[StageManifest, CompiledInputs, ValidationResult]:
    result = _compiled_stage_result(run, validation, compiled)
    result_path = _write_json_artifact(
        context,
        _COMPILED_STAGE,
        "stage-result.json",
        result,
    )
    manifest = stores.artifacts.complete_stage(
        run.run_id,
        _COMPILED_STAGE,
        run.input_hash,
        _artifact_sources(
            compiled.artifacts,
            extra_type=_STAGE_RESULT_ARTIFACT,
            extra_path=result_path,
            stage=_COMPILED_STAGE,
            run_id=run.run_id,
        ),
    )
    restored, restored_validation = _restore_compiled_inputs(
        manifest,
        stores.artifacts,
        run_id=run.run_id,
        input_hash=run.input_hash,
    )
    return manifest, restored, restored_validation


def _seal_trained_stage(
    run: RunRecord,
    context: RunContext,
    candidate: CandidateModel,
    stores: OrchestratorStores,
    *,
    stage: str = _TRAINED_STAGE,
) -> tuple[StageManifest, CandidateModel]:
    result, model_artifact_type = _trained_stage_result(run, candidate, stage=stage)
    result_path = _write_json_artifact(
        context,
        stage,
        "stage-result.json",
        result,
    )
    sources = _artifact_sources(
        candidate.artifacts,
        extra_type=_STAGE_RESULT_ARTIFACT,
        extra_path=result_path,
        stage=stage,
        run_id=run.run_id,
    )
    if model_artifact_type == _FALLBACK_MODEL_ARTIFACT:
        if model_artifact_type in sources:
            raise ArtifactIntegrityError(
                "candidate fallback model artifact conflicts with backend output",
                stage=stage,
                run_id=run.run_id,
            )
        sources[model_artifact_type] = candidate.model_path
    manifest = stores.artifacts.complete_stage(
        run.run_id,
        stage,
        run.input_hash,
        sources,
    )
    return manifest, _restore_candidate_model(
        manifest,
        stores.artifacts,
        run_id=run.run_id,
        input_hash=run.input_hash,
        stage=stage,
    )


def _seal_feature_synthesis_stage(
    run: RunRecord,
    context: RunContext,
    synthesized: SynthesizedFeatures,
    stores: OrchestratorStores,
) -> StageManifest:
    result = _feature_synthesis_stage_result(run, synthesized)
    result_path = _write_json_artifact(
        context,
        _FEATURE_SYNTHESIS_STAGE,
        "stage-result.json",
        result,
    )
    return stores.artifacts.complete_stage(
        run.run_id,
        _FEATURE_SYNTHESIS_STAGE,
        run.input_hash,
        _artifact_sources(
            synthesized.artifacts,
            extra_type=_STAGE_RESULT_ARTIFACT,
            extra_path=result_path,
            stage=_FEATURE_SYNTHESIS_STAGE,
            run_id=run.run_id,
        ),
    )


def _deduplicated_warnings(*warning_groups: Sequence[str]) -> list[str]:
    return sorted({warning for group in warning_groups for warning in group})


def _report_evidence(
    run: RunRecord,
    context: RunContext,
    policy: RecklessPolicy,
    identity: ExecutionIdentity,
    validation: ValidationResult,
    compiled: CompiledInputs,
    candidate: CandidateModel,
    evaluation: EvaluationResult,
    decision: Decision,
    manifests: Sequence[StageManifest],
    outcome: RunState,
    feature_synthesis: Mapping[str, object] | None = None,
) -> dict[str, object]:
    manifest_summaries = [_manifest_summary(manifest) for manifest in manifests]
    all_artifacts = [
        {
            "stage": manifest.stage,
            **to_plain_data(artifact),
        }
        for manifest in manifests
        for artifact in manifest.artifacts
    ]
    request = to_plain_data(run.request)
    datasets = {
        role: to_plain_data(dataset)
        for role, dataset in sorted(context.datasets.items())
    }
    validation_data = to_plain_data(validation)
    compilation_data = {
        "metadata": to_plain_data(compiled.metadata),
        "artifact_type_order": [artifact.artifact_type for artifact in compiled.artifacts],
        "artifact_refs": [
            _artifact_descriptor(artifact) for artifact in compiled.artifacts
        ],
    }
    candidate_refs = [_artifact_descriptor(artifact) for artifact in candidate.artifacts]
    model_ref = next(
        (
            _artifact_descriptor(artifact)
            for artifact in candidate.artifacts
            if artifact.path == candidate.model_path
        ),
        None,
    )
    training_data = {
        "metadata": to_plain_data(candidate.metadata),
        "model_artifact_type": (
            model_ref["artifact_type"]
            if model_ref is not None
            else _FALLBACK_MODEL_ARTIFACT
        ),
        "artifact_type_order": [artifact.artifact_type for artifact in candidate.artifacts],
        "candidate_refs": candidate_refs,
        "model_ref": model_ref,
    }
    evaluation_data = to_plain_data(evaluation)
    decision_data = to_plain_data(decision)
    identity_data = _execution_identity_data(identity)
    component_hashes = {
        "request": canonical_json_hash(request),
        "policy": run.policy_hash,
        "input": run.input_hash,
        "datasets": canonical_json_hash(datasets),
        "execution_identity": canonical_json_hash(identity_data),
        "validation": canonical_json_hash(validation_data),
        "compilation": canonical_json_hash(compilation_data),
        "training": canonical_json_hash(training_data),
        "evaluation": canonical_json_hash(evaluation_data),
        "decision": canonical_json_hash(decision_data),
        "completed_stages": canonical_json_hash(manifest_summaries),
    }
    evidence = {
        "schema_version": _REPORT_EVIDENCE_SCHEMA_VERSION,
        "run": {"run_id": run.run_id},
        "request": request,
        "policy": {
            "version": policy.version,
            "content_hash": run.policy_hash,
        },
        "input": {
            "input_hash": run.input_hash,
            "base_dataset_id": run.request.base_dataset_id,
            "production_cases_id": run.request.production_cases_id,
        },
        "outcome": {
            "state": outcome.value,
            "promotion_eligible": decision.promotion_eligible,
            "acceptance_level": decision.acceptance_level,
        },
        "datasets": datasets,
        "execution_identity": identity_data,
        "validation": validation_data,
        "compilation": compilation_data,
        "training": training_data,
        "evaluation": evaluation_data,
        "decision": decision_data,
        "warnings": _deduplicated_warnings(
            validation.warnings,
            evaluation.warnings,
            decision.warnings,
        ),
        "artifacts": all_artifacts,
        "completed_stage_manifests": manifest_summaries,
        "component_hashes": component_hashes,
    }
    if feature_synthesis is not None:
        synthesis_data = to_plain_data(feature_synthesis)
        evidence["feature_synthesis"] = synthesis_data
        component_hashes["feature_synthesis"] = canonical_json_hash(synthesis_data)
    return evidence


def _seal_report_evidence(
    run: RunRecord,
    context: RunContext,
    policy: RecklessPolicy,
    identity: ExecutionIdentity,
    validation: ValidationResult,
    compiled: CompiledInputs,
    candidate: CandidateModel,
    evaluation: EvaluationResult,
    decision: Decision,
    manifests: Sequence[StageManifest],
    outcome: RunState,
    stores: OrchestratorStores,
    feature_synthesis: Mapping[str, object] | None = None,
) -> StageManifest:
    evidence = _report_evidence(
        run,
        context,
        policy,
        identity,
        validation,
        compiled,
        candidate,
        evaluation,
        decision,
        manifests,
        outcome,
        feature_synthesis,
    )
    path = _write_json_artifact(
        context,
        _REPORTING_STAGE,
        "report-evidence.json",
        evidence,
    )
    return stores.artifacts.complete_stage(
        run.run_id,
        _REPORTING_STAGE,
        run.input_hash,
        {_REPORT_EVIDENCE_ARTIFACT: path},
    )


def _report_metadata(manifest: StageManifest) -> dict[str, object]:
    evidence_refs = [
        artifact
        for artifact in manifest.artifacts
        if artifact.artifact_type == _REPORT_EVIDENCE_ARTIFACT
    ]
    if len(evidence_refs) != 1:
        raise ArtifactIntegrityError(
            "reporting stage must contain exactly one report-evidence artifact",
            stage=_REPORTING_STAGE,
        )
    evidence = evidence_refs[0]
    return {
        "stage": manifest.stage,
        "artifact_type": evidence.artifact_type,
        "path": evidence.path.as_posix(),
        "content_hash": evidence.content_hash,
        "size_bytes": evidence.size_bytes,
    }


def _internal_failure(
    stores: OrchestratorStores,
    run: RunRecord,
    *,
    stage: str,
    exc: BaseException,
) -> RunRecord:
    error = HeuriBoostError(
        RunState.FAILED_INTERNAL.value,
        f"unexpected failure during {stage}",
        stage=stage,
        run_id=run.run_id,
        details={
            "exception_type": type(exc).__name__,
            "message": str(exc)[:500],
        },
        operator_action="Inspect the sealed artifacts and create a new run after correcting the cause.",
    )
    return stores.runs.fail(run.run_id, RunState.FAILED_INTERNAL, error)


def _blocked_input(
    stores: OrchestratorStores,
    run: RunRecord,
    error: InputBlockedError,
) -> RunRecord:
    return stores.runs.fail(run.run_id, RunState.BLOCKED_INPUT, error)


def _blocked_evaluation(
    stores: OrchestratorStores,
    run: RunRecord,
    error: EvaluationBlockedError,
) -> RunRecord:
    return stores.runs.fail(run.run_id, RunState.BLOCKED_EVALUATION, error)


def _blocked_not_eligible(
    stores: OrchestratorStores,
    run: RunRecord,
    error: NotEligibleError,
) -> RunRecord:
    return stores.runs.fail(run.run_id, RunState.BLOCKED_NOT_ELIGIBLE, error)


def _run_context(
    run: RunRecord,
    stores: OrchestratorStores,
) -> tuple[RunContext, DatasetRef, DatasetRef]:
    base = stores.datasets.get(run.request.base_dataset_id)
    cases = stores.datasets.get(run.request.production_cases_id)
    context = RunContext(
        run_id=run.run_id,
        run_dir=stores.artifacts.run_dir(run.run_id),
        datasets={"base": base, "production_cases": cases},
        options=run.request.run_options,
    )
    return context, base, cases


def _not_eligible_only(decision: Decision) -> bool:
    return bool(decision.blockers) and set(decision.blockers) == {"acceptance_level"}


def _report_after_evaluation(
    run: RunRecord,
    context: RunContext,
    backend: Any,
    stores: OrchestratorStores,
    policy: RecklessPolicy,
    identity: ExecutionIdentity,
    validation: ValidationResult,
    compiled: CompiledInputs,
    candidate: CandidateModel,
    evaluation: EvaluationResult,
    decision: Decision,
    manifests: Sequence[StageManifest],
    outcome: RunState,
    feature_synthesis: Mapping[str, object] | None = None,
) -> RunRecord:
    try:
        reporting = stores.runs.transition(run.run_id, RunState.REPORTING)
        report_manifest = _seal_report_evidence(
            reporting,
            context,
            policy,
            identity,
            validation,
            compiled,
            candidate,
            evaluation,
            decision,
            manifests,
            outcome,
            stores,
            feature_synthesis,
        )
        return stores.runs.transition(
            reporting.run_id,
            outcome,
            metadata={"report_evidence": _report_metadata(report_manifest)},
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _internal_failure(
            stores,
            stores.runs.get(run.run_id),
            stage=_REPORTING_STAGE,
            exc=exc,
        )


def _feature_synthesis_summary(
    synthesized: SynthesizedFeatures,
    synthesis_manifest: StageManifest,
    initial_evaluation: EvaluationResult,
    initial_decision: Decision,
) -> dict[str, object]:
    return {
        "attempted": True,
        "metadata": to_plain_data(synthesized.metadata),
        "compiled_inputs_metadata": to_plain_data(synthesized.compiled_inputs.metadata),
        "initial_evaluation": to_plain_data(initial_evaluation),
        "initial_decision": to_plain_data(initial_decision),
        "stage": _manifest_summary(synthesis_manifest),
    }


def _synthesize_features_once(
    run: RunRecord,
    context: RunContext,
    backend: Any,
    stores: OrchestratorStores,
    compiled: CompiledInputs,
    candidate: CandidateModel,
    evaluation: EvaluationResult,
    decision: Decision,
) -> tuple[RunRecord, SynthesizedFeatures, StageManifest] | RunRecord:
    synthesize = getattr(backend, "synthesize_features", None)
    if not callable(synthesize):
        return run
    synthesizing = stores.runs.transition(run.run_id, RunState.SYNTHESIZING_FEATURES)
    try:
        synthesized = synthesize(compiled, candidate, evaluation, decision, context)
        if not isinstance(synthesized, SynthesizedFeatures):
            raise TypeError("synthesize_features must return SynthesizedFeatures")
        if not isinstance(synthesized.compiled_inputs, CompiledInputs):
            raise TypeError("synthesized compiled_inputs must be CompiledInputs")
        synthesis_manifest = _seal_feature_synthesis_stage(
            synthesizing,
            context,
            synthesized,
            stores,
        )
    except EvaluationBlockedError as exc:
        return _blocked_evaluation(stores, synthesizing, exc)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _internal_failure(
            stores,
            synthesizing,
            stage=_FEATURE_SYNTHESIS_STAGE,
            exc=exc,
        )
    return synthesizing, synthesized, synthesis_manifest


def _train_and_evaluate_synthesized_retry(
    run: RunRecord,
    context: RunContext,
    backend: Any,
    stores: OrchestratorStores,
    policy: RecklessPolicy,
    identity: ExecutionIdentity,
    validation: ValidationResult,
    synthesized: SynthesizedFeatures,
    synthesis_manifest: StageManifest,
    initial_evaluation: EvaluationResult,
    initial_decision: Decision,
    manifests: Sequence[StageManifest],
) -> RunRecord:
    training = stores.runs.transition(run.run_id, RunState.TRAINING)
    try:
        retry_candidate: CandidateModel = backend.train(
            synthesized.compiled_inputs,
            context,
        )
    except InterruptedError as exc:
        return stores.runs.transition(
            training.run_id,
            RunState.INTERRUPTED,
            metadata={
                "interruption": {
                    "stage": RunState.TRAINING.value,
                    "reason": str(exc)[:500],
                }
            },
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _internal_failure(stores, training, stage=RunState.TRAINING.value, exc=exc)
    try:
        retry_manifest, sealed_retry_candidate = _seal_trained_stage(
            training,
            context,
            retry_candidate,
            stores,
            stage=_TRAINED_RETRY_STAGE,
        )
        trained = stores.runs.transition(
            training.run_id,
            RunState.TRAINED,
            metadata={
                "feature_synthesis_stage": _manifest_summary(synthesis_manifest),
                "trained_retry_stage": _manifest_summary(retry_manifest),
            },
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _internal_failure(stores, training, stage=_TRAINED_RETRY_STAGE, exc=exc)

    evaluating = stores.runs.transition(trained.run_id, RunState.EVALUATING)
    try:
        verification: ArtifactVerification = backend.verify_artifacts(
            sealed_retry_candidate,
            context,
        )
        if not verification.valid:
            raise EvaluationBlockedError(
                "candidate artifact verification failed after feature synthesis",
                stage=_TRAINED_RETRY_STAGE,
                run_id=evaluating.run_id,
                details={"errors": tuple(verification.errors)},
                operator_action="Inspect the sealed synthesized candidate artifacts and create a new run.",
            )
        retry_evaluation: EvaluationResult = backend.evaluate(
            sealed_retry_candidate,
            context,
        )
        retry_decision = evaluate_promotion_eligibility(policy, retry_evaluation)
    except NotEligibleError as exc:
        return _blocked_not_eligible(stores, evaluating, exc)
    except EvaluationBlockedError as exc:
        return _blocked_evaluation(stores, evaluating, exc)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _internal_failure(
            stores,
            evaluating,
            stage=RunState.EVALUATING.value,
            exc=exc,
        )

    synthesis_summary = _feature_synthesis_summary(
        synthesized,
        synthesis_manifest,
        initial_evaluation,
        initial_decision,
    )
    retry_manifests = (*manifests, synthesis_manifest, retry_manifest)
    if not retry_decision.promotion_eligible:
        if _not_eligible_only(retry_decision):
            return _report_after_evaluation(
                evaluating,
                context,
                backend,
                stores,
                policy,
                identity,
                validation,
                synthesized.compiled_inputs,
                sealed_retry_candidate,
                retry_evaluation,
                retry_decision,
                retry_manifests,
                RunState.BLOCKED_NOT_ELIGIBLE,
                synthesis_summary,
            )
        return _blocked_evaluation(
            stores,
            evaluating,
            EvaluationBlockedError(
                "promotion policy evaluation failed after feature synthesis",
                stage=RunState.EVALUATING.value,
                run_id=evaluating.run_id,
                details={
                    "blockers": retry_decision.blockers,
                    "feature_synthesis": to_plain_data(synthesis_summary),
                },
                operator_action="Inspect synthesis evidence and create a new run after correcting the regression.",
            ),
        )
    return _report_after_evaluation(
        evaluating,
        context,
        backend,
        stores,
        policy,
        identity,
        validation,
        synthesized.compiled_inputs,
        sealed_retry_candidate,
        retry_evaluation,
        retry_decision,
        retry_manifests,
        RunState.READY_FOR_PROMOTION,
        synthesis_summary,
    )


def _evaluate_after_training(
    run: RunRecord,
    context: RunContext,
    backend: Any,
    stores: OrchestratorStores,
    policy: RecklessPolicy,
    identity: ExecutionIdentity,
    validation: ValidationResult,
    compiled: CompiledInputs,
    candidate: CandidateModel,
    manifests: Sequence[StageManifest],
) -> RunRecord:
    evaluating = stores.runs.transition(run.run_id, RunState.EVALUATING)
    try:
        verification: ArtifactVerification = backend.verify_artifacts(candidate, context)
        if not verification.valid:
            raise EvaluationBlockedError(
                "candidate artifact verification failed",
                stage=_TRAINED_STAGE,
                run_id=evaluating.run_id,
                details={"errors": tuple(verification.errors)},
                operator_action="Inspect the sealed candidate artifacts and create a new run.",
            )
        evaluation: EvaluationResult = backend.evaluate(candidate, context)
        decision = evaluate_promotion_eligibility(policy, evaluation)
    except NotEligibleError as exc:
        return _blocked_not_eligible(stores, evaluating, exc)
    except EvaluationBlockedError as exc:
        return _blocked_evaluation(stores, evaluating, exc)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _internal_failure(stores, evaluating, stage=RunState.EVALUATING.value, exc=exc)

    if not decision.promotion_eligible:
        if _not_eligible_only(decision):
            return _report_after_evaluation(
                evaluating,
                context,
                backend,
                stores,
                policy,
                identity,
                validation,
                compiled,
                candidate,
                evaluation,
                decision,
                manifests,
                RunState.BLOCKED_NOT_ELIGIBLE,
            )
        synthesized = _synthesize_features_once(
            evaluating,
            context,
            backend,
            stores,
            compiled,
            candidate,
            evaluation,
            decision,
        )
        if isinstance(synthesized, tuple):
            (
                synthesizing,
                synthesized_features,
                synthesis_manifest,
            ) = synthesized
            return _train_and_evaluate_synthesized_retry(
                synthesizing,
                context,
                backend,
                stores,
                policy,
                identity,
                validation,
                synthesized_features,
                synthesis_manifest,
                evaluation,
                decision,
                manifests,
            )
        if synthesized.state != evaluating.state:
            return synthesized
        return _blocked_evaluation(
            stores,
            evaluating,
            EvaluationBlockedError(
                "promotion policy evaluation failed",
                stage=RunState.EVALUATING.value,
                run_id=evaluating.run_id,
                details={"blockers": decision.blockers},
                operator_action="Inspect evaluation evidence and create a new run after correcting the regression.",
            ),
        )
    return _report_after_evaluation(
        evaluating,
        context,
        backend,
        stores,
        policy,
        identity,
        validation,
        compiled,
        candidate,
        evaluation,
        decision,
        manifests,
        RunState.READY_FOR_PROMOTION,
    )


def _train_after_compilation(
    run: RunRecord,
    context: RunContext,
    backend: Any,
    stores: OrchestratorStores,
    policy: RecklessPolicy,
    identity: ExecutionIdentity,
    validation: ValidationResult,
    compiled: CompiledInputs,
    compiled_manifest: StageManifest,
) -> RunRecord:
    training = stores.runs.transition(run.run_id, RunState.TRAINING)
    try:
        candidate: CandidateModel = backend.train(compiled, context)
    except InterruptedError as exc:
        return stores.runs.transition(
            training.run_id,
            RunState.INTERRUPTED,
            metadata={
                "interruption": {
                    "stage": RunState.TRAINING.value,
                    "reason": str(exc)[:500],
                }
            },
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _internal_failure(stores, training, stage=RunState.TRAINING.value, exc=exc)
    try:
        trained_manifest, sealed_candidate = _seal_trained_stage(
            training,
            context,
            candidate,
            stores,
        )
        trained = stores.runs.transition(
            training.run_id,
            RunState.TRAINED,
            metadata={"trained_stage": _manifest_summary(trained_manifest)},
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _internal_failure(stores, training, stage=_TRAINED_STAGE, exc=exc)
    return _evaluate_after_training(
        trained,
        context,
        backend,
        stores,
        policy,
        identity,
        validation,
        compiled,
        sealed_candidate,
        (compiled_manifest, trained_manifest),
    )


def _compile_after_validation(
    run: RunRecord,
    context: RunContext,
    backend: Any,
    stores: OrchestratorStores,
    policy: RecklessPolicy,
    identity: ExecutionIdentity,
    validation: ValidationResult,
) -> RunRecord:
    try:
        compiled: CompiledInputs = backend.compile(run.request, context)
    except InputBlockedError as exc:
        return _blocked_input(stores, run, exc)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _internal_failure(stores, run, stage=_COMPILED_STAGE, exc=exc)
    try:
        manifest, sealed_compiled, sealed_validation = _seal_compiled_stage(
            run,
            context,
            validation,
            compiled,
            stores,
        )
        completed = stores.runs.transition(
            run.run_id,
            RunState.COMPILED,
            metadata={
                "compiled_stage": _manifest_summary(manifest),
                "validation": to_plain_data(sealed_validation),
            },
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _internal_failure(stores, run, stage=_COMPILED_STAGE, exc=exc)
    return _train_after_compilation(
        completed,
        context,
        backend,
        stores,
        policy,
        identity,
        sealed_validation,
        sealed_compiled,
        manifest,
    )


def _validate_and_compile(
    run: RunRecord,
    context: RunContext,
    backend: Any,
    stores: OrchestratorStores,
    policy: RecklessPolicy,
    identity: ExecutionIdentity,
) -> RunRecord:
    validating = stores.runs.transition(run.run_id, RunState.VALIDATING)
    try:
        validation: ValidationResult = backend.validate(validating.request, context)
    except InputBlockedError as exc:
        return _blocked_input(stores, validating, exc)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _internal_failure(
            stores,
            validating,
            stage=RunState.VALIDATING.value,
            exc=exc,
        )
    if not validation.valid:
        return _blocked_input(
            stores,
            validating,
            InputBlockedError(
                "backend validation rejected the repair input",
                stage=RunState.VALIDATING.value,
                run_id=validating.run_id,
                details={
                    "metadata": to_plain_data(validation.metadata),
                    "warnings": list(validation.warnings),
                },
                operator_action="Correct the input data or backend configuration, then create a new run.",
            ),
        )
    return _compile_after_validation(
        validating,
        context,
        backend,
        stores,
        policy,
        identity,
        validation,
    )


def run_reckless_repair(
    request: RepairRequest,
    backend: Any,
    stores: OrchestratorStores,
    policy: RecklessPolicy,
) -> RunRecord:
    """Create and execute a new Reckless repair run through report evidence."""

    base = stores.datasets.get(request.base_dataset_id)
    cases = stores.datasets.get(request.production_cases_id)
    identity: ExecutionIdentity = backend.execution_identity()
    input_hash = build_run_fingerprint(request, policy, base, cases, identity)
    created = stores.runs.create(request, policy.content_hash, input_hash)
    context = RunContext(
        run_id=created.run_id,
        run_dir=stores.artifacts.run_dir(created.run_id),
        datasets={"base": base, "production_cases": cases},
        options=request.run_options,
    )
    return _validate_and_compile(created, context, backend, stores, policy, identity)


def run_existing_reckless_repair(
    run_id: str,
    backend: Any,
    stores: OrchestratorStores,
    policy: RecklessPolicy,
) -> RunRecord:
    """Execute a pre-created RECEIVED run without allocating a second run ID."""

    run = stores.runs.get(run_id)
    if run.state != RunState.RECEIVED.value:
        raise ValueError("only RECEIVED runs can be executed")
    if run.policy_hash != policy.content_hash:
        raise ArtifactIntegrityError(
            "pre-created run policy hash does not match the effective policy",
            stage=RunState.RECEIVED.value,
            run_id=run.run_id,
        )
    base = stores.datasets.get(run.request.base_dataset_id)
    cases = stores.datasets.get(run.request.production_cases_id)
    identity: ExecutionIdentity = backend.execution_identity()
    input_hash = build_run_fingerprint(run.request, policy, base, cases, identity)
    if run.input_hash != input_hash:
        raise ArtifactIntegrityError(
            "pre-created run input hash does not match current execution inputs",
            stage=RunState.RECEIVED.value,
            run_id=run.run_id,
            details={"expected": run.input_hash, "actual": input_hash},
        )
    context = RunContext(
        run_id=run.run_id,
        run_dir=stores.artifacts.run_dir(run.run_id),
        datasets={"base": base, "production_cases": cases},
        options=run.request.run_options,
    )
    return _validate_and_compile(run, context, backend, stores, policy, identity)


def resume_reckless_repair(
    run_id: str,
    backend: Any,
    stores: OrchestratorStores,
    policy: RecklessPolicy,
) -> RunRecord:
    """Resume only an intentionally interrupted run from sealed compile output."""

    run = stores.runs.get(run_id)
    if run.state != RunState.INTERRUPTED.value:
        raise ValueError("only INTERRUPTED runs can resume")
    try:
        context, base, cases = _run_context(run, stores)
        identity: ExecutionIdentity = backend.execution_identity()
        current_fingerprint = build_run_fingerprint(
            run.request,
            policy,
            base,
            cases,
            identity,
        )
        if run.policy_hash != policy.content_hash or current_fingerprint != run.input_hash:
            raise ArtifactIntegrityError(
                "resume inputs or execution identity no longer match the interrupted run",
                stage=_COMPILED_STAGE,
                run_id=run.run_id,
                operator_action="Create a new run with the changed inputs or backend configuration.",
            )
        manifest = stores.artifacts.load_completed_stage(
            run.run_id,
            _COMPILED_STAGE,
            run.input_hash,
        )
        compiled, validation = _restore_compiled_inputs(
            manifest,
            stores.artifacts,
            run_id=run.run_id,
            input_hash=run.input_hash,
        )
    except ArtifactIntegrityError:
        raise
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ArtifactIntegrityError(
            "unable to restore the interrupted run from sealed compile output",
            stage=_COMPILED_STAGE,
            run_id=run.run_id,
            operator_action="Inspect the interrupted run and create a new run if its inputs are unavailable.",
        ) from exc
    return _train_after_compilation(
        run,
        context,
        backend,
        stores,
        policy,
        identity,
        validation,
        compiled,
        manifest,
    )
