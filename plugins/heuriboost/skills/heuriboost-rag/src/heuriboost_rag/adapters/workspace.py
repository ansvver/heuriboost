"""Local filesystem adapter for the Reckless package APIs.

The adapter deliberately owns only local registration and configuration.  It
does not recreate the legacy mutable ledger, gate, or current-model writes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping

from ..backends.legacy_runtime import LegacyRuntime, resolve_legacy_runtime
from ..backends.xgboost_rag import (
    CompileSettings,
    PinnedFile,
    TrainingSettings,
    XGBoostRagBackend,
    XGBoostRagConfig,
)
from ..reckless.contracts import (
    ActivationResult,
    DatasetRef,
    PreparedActivation,
    PromotionApproval,
    PromotionReceipt,
    ReleaseSnapshot,
    RepairRequest,
    ReportArtifact,
    TargetValidation,
)
from ..reckless.errors import ArtifactIntegrityError, PromotionConflictError
from ..reckless.hashing import canonical_json_hash, sha256_file
from ..reckless.orchestrator import resume_reckless_repair, run_reckless_repair
from ..reckless.policy import (
    EvaluationPolicy,
    InputPolicy,
    PromotionPolicy,
    RecklessPolicy,
)
from ..reckless.promotion import (
    JsonPromotionRepository,
    PromotionStores,
    promote_repair as promote_repair_run,
)
from ..reckless.release_store import FileReleaseStore
from ..reckless.report import render_run_pre_promote_report
from ..reckless.storage import (
    JsonDatasetRepository,
    JsonRunRepository,
    LocalArtifactStore,
    OrchestratorStores,
)


_WORKSPACE_DIR_NAME = ".reckless"
_WORKSPACE_CONFIG_NAME = "workspace.json"
_WORKSPACE_SCHEMA_VERSION = 1
_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUPPORTED_DATASET_SUFFIXES = frozenset({".csv", ".jsonl", ".ndjson"})
_LEGACY_STATE_NAMES = (
    "ledger.json",
    "gates.jsonl",
    "promoted_repair_samples.csv",
)


def _require_workspace_id(value: str) -> str:
    if not isinstance(value, str) or _WORKSPACE_ID.fullmatch(value) is None:
        raise ValueError(f"unsafe workspace ID: {value!r}")
    return value


def default_promotion_idempotency_key(run_id: str) -> str:
    """Return the stable local retry key for one immutable promotion run."""

    return f"local-promote-{_require_workspace_id(run_id)}"


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect {label}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return loaded


def _regular_source(path: Path, label: str) -> Path:
    source = Path(path).expanduser()
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {source}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file: {source}")
    if source.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {source}")
    try:
        return source.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"cannot resolve {label}: {source}") from exc


def _schema_from_csv(path: Path) -> tuple[str, tuple[str, ...]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"cannot read CSV header: {path}") from exc
    if not header or any(not isinstance(field, str) or not field for field in header):
        raise ValueError(f"CSV requires a non-empty header: {path}")
    if len(set(header)) != len(header):
        raise ValueError(f"CSV header contains duplicate fields: {path}")
    return "csv", tuple(header)


def _schema_from_jsonl(path: Path) -> tuple[str, tuple[str, ...]]:
    fields: set[str] = set()
    rows = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_strict_object,
                        parse_constant=_reject_json_constant,
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid JSONL row {line_number}: {path}"
                    ) from exc
                if not isinstance(value, dict) or any(
                    not isinstance(field, str) or not field for field in value
                ):
                    raise ValueError(
                        f"JSONL row {line_number} must be an object with string fields"
                    )
                fields.update(value)
                rows += 1
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read JSONL data: {path}") from exc
    if rows == 0 or not fields:
        raise ValueError(f"JSONL dataset must contain at least one object: {path}")
    return "jsonl", tuple(sorted(fields))


def _dataset_schema(path: Path) -> tuple[str, tuple[str, ...]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _schema_from_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return _schema_from_jsonl(path)
    supported = ", ".join(sorted(_SUPPORTED_DATASET_SUFFIXES))
    raise ValueError(f"unsupported dataset type {path.suffix!r}; expected one of {supported}")


def _pinned_file(path: Path, label: str) -> PinnedFile:
    source = _regular_source(path, label)
    return PinnedFile(path=source, content_hash=sha256_file(source))


def _pinned_data(pinned: PinnedFile | None) -> dict[str, str] | None:
    if pinned is None:
        return None
    return {
        "path": str(pinned.path),
        "content_hash": pinned.content_hash,
    }


def _pinned_from_data(value: object, label: str) -> PinnedFile | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object or null")
    if set(value) != {"path", "content_hash"}:
        raise ValueError(f"{label} has an invalid schema")
    path = Path(_require_string(value["path"], f"{label}.path"))
    content_hash = _require_string(value["content_hash"], f"{label}.content_hash")
    return PinnedFile(path=path, content_hash=content_hash)


def _compile_settings_data(settings: CompileSettings) -> dict[str, object]:
    return {
        "resplit": settings.resplit,
        "split_ratios": list(settings.split_ratios),
        "split_seed": settings.split_seed,
        "case_top_k": settings.case_top_k,
        "strict": settings.strict,
        "acceptance_level": settings.acceptance_level,
        "min_global_test_queries": settings.min_global_test_queries,
        "min_domain_test_queries": settings.min_domain_test_queries,
        "min_docs_per_query": settings.min_docs_per_query,
    }


def _compile_settings_from_data(value: object) -> CompileSettings:
    if not isinstance(value, Mapping):
        raise ValueError("compile_settings must be an object")
    expected = {
        "resplit",
        "split_ratios",
        "split_seed",
        "case_top_k",
        "strict",
        "acceptance_level",
        "min_global_test_queries",
        "min_domain_test_queries",
        "min_docs_per_query",
    }
    if set(value) != expected:
        raise ValueError("compile_settings has an invalid schema")
    ratios = value["split_ratios"]
    if not isinstance(ratios, list):
        raise ValueError("compile_settings.split_ratios must be a list")
    return CompileSettings(
        resplit=value["resplit"],
        split_ratios=tuple(ratios),
        split_seed=value["split_seed"],
        case_top_k=value["case_top_k"],
        strict=value["strict"],
        acceptance_level=value["acceptance_level"],
        min_global_test_queries=value["min_global_test_queries"],
        min_domain_test_queries=value["min_domain_test_queries"],
        min_docs_per_query=value["min_docs_per_query"],
    )


def _training_settings_data(settings: TrainingSettings) -> dict[str, object]:
    return {
        "rounds": settings.rounds,
        "random_seed": settings.random_seed,
    }


def _training_settings_from_data(value: object) -> TrainingSettings:
    if not isinstance(value, Mapping) or set(value) != {"rounds", "random_seed"}:
        raise ValueError("training_settings has an invalid schema")
    return TrainingSettings(
        rounds=value["rounds"],
        random_seed=value["random_seed"],
    )


def _policy_data(policy: RecklessPolicy) -> dict[str, object]:
    return {
        "version": policy.version,
        "acceptance_level": policy.acceptance_level,
        "input": {
            "min_global_test_queries": policy.input.min_global_test_queries,
            "min_domain_test_queries": policy.input.min_domain_test_queries,
            "min_docs_per_query": policy.input.min_docs_per_query,
            "require_authoritative_labels": policy.input.require_authoritative_labels,
        },
        "evaluation": {
            "require_all_current_cases": policy.evaluation.require_all_current_cases,
            "require_all_historical_gates": policy.evaluation.require_all_historical_gates,
            "require_global_ndcg_improvement": policy.evaluation.require_global_ndcg_improvement,
            "require_global_mrr_improvement": policy.evaluation.require_global_mrr_improvement,
            "allow_touched_domain_regression": policy.evaluation.allow_touched_domain_regression,
        },
        "promotion": {
            "allow_weak": policy.promotion.allow_weak,
            "require_explicit_human_approval": policy.promotion.require_explicit_human_approval,
            "allow_anchor_reset": policy.promotion.allow_anchor_reset,
            "allow_gate_retirement": policy.promotion.allow_gate_retirement,
        },
        "content_hash": policy.content_hash,
    }


def _policy_from_data(value: object) -> RecklessPolicy:
    if not isinstance(value, Mapping):
        raise ValueError("policy must be an object")
    expected = {"version", "acceptance_level", "input", "evaluation", "promotion", "content_hash"}
    if set(value) != expected:
        raise ValueError("policy has an invalid schema")
    input_data = value["input"]
    evaluation_data = value["evaluation"]
    promotion_data = value["promotion"]
    if not isinstance(input_data, Mapping):
        raise ValueError("policy.input must be an object")
    if not isinstance(evaluation_data, Mapping):
        raise ValueError("policy.evaluation must be an object")
    if not isinstance(promotion_data, Mapping):
        raise ValueError("policy.promotion must be an object")
    policy = RecklessPolicy(
        version=value["version"],
        acceptance_level=value["acceptance_level"],
        input=InputPolicy(**dict(input_data)),
        evaluation=EvaluationPolicy(**dict(evaluation_data)),
        promotion=PromotionPolicy(**dict(promotion_data)),
    )
    if policy.content_hash != _require_string(value["content_hash"], "policy.content_hash"):
        raise ArtifactIntegrityError(
            "workspace policy hash does not match its stored configuration",
            stage="CONFIGURATION",
            operator_action="Create a new workspace from a verified policy file.",
        )
    return policy


def policy_for_compile_settings(
    policy: RecklessPolicy,
    compile_settings: CompileSettings,
) -> RecklessPolicy:
    """Freeze legacy CLI acceptance and minimums into the effective policy.

    The legacy command line historically controlled these input thresholds.  The
    effective policy is reconstructed rather than smuggled through
    ``RepairRequest.run_options`` so the chosen values participate in the run
    policy hash and immutable workspace configuration.
    """

    if not isinstance(policy, RecklessPolicy):
        raise TypeError("policy must be a RecklessPolicy")
    if not isinstance(compile_settings, CompileSettings):
        raise TypeError("compile_settings must be CompileSettings")
    return RecklessPolicy(
        version=policy.version,
        acceptance_level=compile_settings.acceptance_level,
        input=InputPolicy(
            min_global_test_queries=compile_settings.min_global_test_queries,
            min_domain_test_queries=compile_settings.min_domain_test_queries,
            min_docs_per_query=compile_settings.min_docs_per_query,
            require_authoritative_labels=policy.input.require_authoritative_labels,
        ),
        evaluation=policy.evaluation,
        promotion=policy.promotion,
    )


def _workspace_config(
    *,
    workspace_id: str,
    policy: RecklessPolicy,
    runtime: LegacyRuntime,
    config: XGBoostRagConfig,
) -> dict[str, object]:
    return {
        "schema_version": _WORKSPACE_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "policy": _policy_data(policy),
        "backend": {
            "name": XGBoostRagBackend.name,
            "code_revision": config.code_revision,
            "legacy_code_manifest_hash": config.legacy_code_manifest_hash,
            "feature_recipes": _pinned_data(config.feature_recipes),
            "historical_gates": _pinned_data(config.historical_gates),
            "anchor_ledger": _pinned_data(config.anchor_ledger),
            "promoted_samples": _pinned_data(config.promoted_samples),
            "include_promoted_samples": config.include_promoted_samples,
            "compile_settings": _compile_settings_data(config.compile_settings),
            "training_settings": _training_settings_data(config.training_settings),
            "runtime_manifest_hash": runtime.code_manifest_hash,
        },
    }


def _backend_from_config(value: object) -> XGBoostRagConfig:
    if not isinstance(value, Mapping):
        raise ValueError("backend must be an object")
    expected = {
        "name",
        "code_revision",
        "legacy_code_manifest_hash",
        "feature_recipes",
        "historical_gates",
        "anchor_ledger",
        "promoted_samples",
        "include_promoted_samples",
        "compile_settings",
        "training_settings",
        "runtime_manifest_hash",
    }
    if set(value) != expected:
        raise ValueError("backend has an invalid schema")
    if value["name"] != XGBoostRagBackend.name:
        raise ValueError("unsupported local workspace backend")
    if type(value["include_promoted_samples"]) is not bool:
        raise ValueError("backend.include_promoted_samples must be a boolean")
    feature_recipes = _pinned_from_data(value["feature_recipes"], "feature_recipes")
    historical_gates = _pinned_from_data(value["historical_gates"], "historical_gates")
    anchor_ledger = _pinned_from_data(value["anchor_ledger"], "anchor_ledger")
    promoted_samples = _pinned_from_data(value["promoted_samples"], "promoted_samples")
    if feature_recipes is None or historical_gates is None or anchor_ledger is None:
        raise ValueError("workspace backend requires feature recipes, gates, and anchor")
    return XGBoostRagConfig(
        code_revision=_require_string(value["code_revision"], "backend.code_revision"),
        legacy_code_manifest_hash=_require_string(
            value["legacy_code_manifest_hash"],
            "backend.legacy_code_manifest_hash",
        ),
        feature_recipes=feature_recipes,
        historical_gates=historical_gates,
        anchor_ledger=anchor_ledger,
        promoted_samples=promoted_samples,
        include_promoted_samples=value["include_promoted_samples"],
        compile_settings=_compile_settings_from_data(value["compile_settings"]),
        training_settings=_training_settings_from_data(value["training_settings"]),
    )


def _read_workspace_config(path: Path) -> tuple[str, RecklessPolicy, XGBoostRagConfig]:
    data = _json_mapping(path, "workspace configuration")
    if set(data) != {"schema_version", "workspace_id", "policy", "backend"}:
        raise ValueError("workspace configuration has an invalid schema")
    if data["schema_version"] != _WORKSPACE_SCHEMA_VERSION:
        raise ValueError("unsupported workspace configuration version")
    workspace_id = _require_workspace_id(data["workspace_id"])
    policy = _policy_from_data(data["policy"])
    config = _backend_from_config(data["backend"])
    return workspace_id, policy, config


def _write_workspace_config(path: Path, payload: Mapping[str, object]) -> None:
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True)
class LocalFilePromotionTarget:
    """A local target that leaves pointer ownership to ``FileReleaseStore``."""

    releases: FileReleaseStore

    name = "local-file"

    def validate_target(self, expected_current: str | None) -> TargetValidation:
        current = self.releases.read_current_model()
        valid = current == expected_current
        return TargetValidation(
            valid=valid,
            current_model=current,
            errors=() if valid else ("current model changed",),
        )

    def prepare_release(self, release: ReleaseSnapshot) -> PreparedActivation:
        if not release.artifacts:
            raise ArtifactIntegrityError(
                "local release contains no artifacts",
                stage="PROMOTING",
                run_id=release.run_id,
            )
        return PreparedActivation(
            run_id=release.run_id,
            pointer_payload={"target": self.name, "run_id": release.run_id},
            metadata={
                "target": self.name,
                "artifact_count": len(release.artifacts),
                "release_snapshot_hash": release.manifest_hash,
            },
        )

    def activate(self, prepared: PreparedActivation) -> ActivationResult:
        return ActivationResult(
            current_model=prepared.run_id,
            metadata={"target": self.name, "activation": "pointer-owned-by-release-store"},
        )

    def rollback(self, receipt: PromotionReceipt) -> ActivationResult:
        if receipt.previous_model is None:
            raise PromotionConflictError(
                "release has no previous model to restore",
                stage="PROMOTING",
                run_id=receipt.run_id,
            )
        return ActivationResult(
            current_model=receipt.previous_model,
            metadata={"target": self.name, "rollback": "pointer-owned-by-release-store"},
        )


@dataclass(frozen=True)
class LocalWorkspace:
    """A complete local wiring of the package contracts for CLI adapters."""

    output_dir: Path
    root: Path
    workspace_id: str
    policy: RecklessPolicy
    backend: XGBoostRagBackend
    stores: OrchestratorStores
    promotion_stores: PromotionStores
    promotion_target: LocalFilePromotionTarget

    def register_local_dataset(self, role: str, path: Path) -> DatasetRef:
        if role not in {"base", "production_cases"}:
            raise ValueError("dataset role must be 'base' or 'production_cases'")
        source = _regular_source(path, f"{role} dataset")
        data_format, fields = _dataset_schema(source)
        content_hash = sha256_file(source)
        schema_hash = canonical_json_hash(
            {"format": data_format, "fields": list(fields)}
        )
        dataset = DatasetRef(
            dataset_id=f"{role}-{content_hash[:24]}",
            role=role,
            path=source,
            content_hash=content_hash,
            schema_hash=schema_hash,
            metadata={
                "format": data_format,
                "fields": list(fields),
                "source_name": source.name,
            },
        )
        repository = self.stores.datasets
        try:
            save = getattr(repository, "save")
        except AttributeError as exc:  # pragma: no cover - constructor wiring guards it.
            raise TypeError("local workspace dataset repository cannot save") from exc
        try:
            return save(dataset)
        except FileExistsError:
            existing = repository.get(dataset.dataset_id)
            if existing != dataset:
                raise ArtifactIntegrityError(
                    "dataset ID already exists with different immutable contents",
                    stage="DATASET_REGISTRATION",
                    details={"dataset_id": dataset.dataset_id},
                )
            return existing

    def create_repair_request(
        self,
        base_dataset: DatasetRef,
        production_cases: DatasetRef,
        *,
        requested_by: str,
    ) -> RepairRequest:
        if base_dataset.role != "base":
            raise ValueError("base_dataset must have role 'base'")
        if production_cases.role != "production_cases":
            raise ValueError("production_cases must have role 'production_cases'")
        return RepairRequest(
            workspace_id=self.workspace_id,
            base_dataset_id=base_dataset.dataset_id,
            production_cases_id=production_cases.dataset_id,
            policy_version=str(self.policy.version),
            backend_name=self.backend.name,
            requested_by=_require_string(requested_by, "requested_by"),
            run_options={},
        )

    def run(self, request: RepairRequest):
        if request.workspace_id != self.workspace_id:
            raise ValueError("request belongs to a different workspace")
        return run_reckless_repair(request, self.backend, self.stores, self.policy)

    def resume(self, run_id: str):
        return resume_reckless_repair(run_id, self.backend, self.stores, self.policy)

    def render_report(self, run_id: str, *, locale: str = "zh-CN") -> ReportArtifact:
        return render_run_pre_promote_report(
            self.stores.runs.get(run_id),
            self.stores.artifacts,
            locale=locale,
        )

    def _legacy_state_paths(self) -> tuple[Path, ...]:
        legacy_dir = self.output_dir / ".heuriboost"
        return tuple(legacy_dir / name for name in _LEGACY_STATE_NAMES)

    def assert_promotion_allowed(self) -> None:
        legacy_paths = tuple(path for path in self._legacy_state_paths() if path.exists())
        if legacy_paths and not self.promotion_stores.releases.legacy_migration_complete():
            raise PromotionConflictError(
                "legacy repair state must be migrated before promotion",
                stage="PROMOTING",
                details={"legacy_paths": [str(path) for path in legacy_paths]},
                operator_action="Run migrate_reckless_state.py before promoting a new release.",
            )

    def build_local_approval(
        self,
        run_id: str,
        *,
        approved_by: str,
        idempotency_key: str | None = None,
    ) -> PromotionApproval:
        self.assert_promotion_allowed()
        report = self.render_report(run_id)
        decision_hash = report.manifest.get("decision_hash")
        if not isinstance(decision_hash, str) or not decision_hash:
            raise ArtifactIntegrityError(
                "pre-promote report has no decision hash",
                stage="PROMOTING",
                run_id=run_id,
            )
        return PromotionApproval(
            run_id=run_id,
            approved_by=_require_string(approved_by, "approved_by"),
            approved_at=datetime.now(timezone.utc).isoformat(),
            report_hash=report.html_hash,
            decision_hash=decision_hash,
            expected_current_model=self.promotion_stores.releases.read_current_model(),
            idempotency_key=idempotency_key or default_promotion_idempotency_key(run_id),
        )

    def promote(
        self,
        run_id: str,
        approval: PromotionApproval | None = None,
        *,
        approved_by: str | None = None,
        idempotency_key: str | None = None,
    ) -> PromotionReceipt:
        self.assert_promotion_allowed()
        if approval is None:
            if approved_by is None:
                raise ValueError("approved_by is required when approval is omitted")
            approval = self.build_local_approval(
                run_id,
                approved_by=approved_by,
                idempotency_key=idempotency_key,
            )
        return promote_repair_run(
            run_id,
            approval,
            self.promotion_target,
            self.promotion_stores,
        )

    def find_single_promotable_run(self) -> str:
        runs_dir = self.root / "runs"
        if not runs_dir.is_dir():
            raise PromotionConflictError(
                "workspace has no runnable repair records",
                stage="PROMOTING",
                operator_action="Pass --run-id after creating a ready Reckless run.",
            )
        candidates: list[str] = []
        for child in sorted(runs_dir.iterdir(), key=lambda item: item.name):
            if child.is_symlink() or not child.is_dir():
                continue
            try:
                record = self.stores.runs.get(child.name)
            except (FileNotFoundError, ValueError):
                continue
            if record.state in {"READY_FOR_PROMOTION", "PROMOTION_FAILED"}:
                candidates.append(record.run_id)
        if len(candidates) != 1:
            qualifier = "no" if not candidates else "multiple"
            raise PromotionConflictError(
                f"{qualifier} promotable runs found; pass --run-id explicitly",
                stage="PROMOTING",
                details={"run_ids": candidates},
                operator_action="Select exactly one READY_FOR_PROMOTION run.",
            )
        return candidates[0]


def _workspace_from_parts(
    output_dir: Path,
    workspace_id: str,
    policy: RecklessPolicy,
    config: XGBoostRagConfig,
    *,
    runtime: LegacyRuntime | None = None,
) -> LocalWorkspace:
    root = (Path(output_dir).expanduser().resolve() / _WORKSPACE_DIR_NAME)
    root.mkdir(parents=True, exist_ok=True)
    runtime = resolve_legacy_runtime() if runtime is None else runtime
    backend = XGBoostRagBackend(config, runtime=runtime)
    datasets = JsonDatasetRepository(root)
    artifacts = LocalArtifactStore(root)
    runs = JsonRunRepository(root)
    releases = FileReleaseStore(root)
    promotion_stores = PromotionStores(
        runs=runs,
        artifacts=artifacts,
        promotions=JsonPromotionRepository(root),
        releases=releases,
    )
    return LocalWorkspace(
        output_dir=Path(output_dir).expanduser().resolve(),
        root=root,
        workspace_id=workspace_id,
        policy=policy,
        backend=backend,
        stores=OrchestratorStores(datasets=datasets, runs=runs, artifacts=artifacts),
        promotion_stores=promotion_stores,
        promotion_target=LocalFilePromotionTarget(releases),
    )


def bootstrap_local_workspace(
    output_dir: Path,
    *,
    workspace_id: str,
    policy: RecklessPolicy,
    feature_recipes: Path,
    historical_gates: Path,
    anchor_ledger: Path,
    compile_settings: CompileSettings,
    training_settings: TrainingSettings,
    code_revision: str,
    promoted_samples: Path | None = None,
    include_promoted_samples: bool = False,
) -> LocalWorkspace:
    """Create one immutable local workspace configuration or reopen its match."""

    safe_workspace_id = _require_workspace_id(workspace_id)
    if not isinstance(policy, RecklessPolicy):
        raise TypeError("policy must be a RecklessPolicy")
    if not isinstance(compile_settings, CompileSettings):
        raise TypeError("compile_settings must be CompileSettings")
    if not isinstance(training_settings, TrainingSettings):
        raise TypeError("training_settings must be TrainingSettings")
    if not compile_settings.strict:
        raise ValueError("local Reckless workspace requires strict compile settings")
    runtime = resolve_legacy_runtime()
    recipes = _pinned_file(feature_recipes, "feature recipes")
    if recipes.path != runtime.feature_recipe_path.resolve():
        raise ValueError("feature recipes must be the trusted bundled recipe source")
    gates = _pinned_file(historical_gates, "historical gates")
    anchor = _pinned_file(anchor_ledger, "anchor ledger")
    promoted = (
        _pinned_file(promoted_samples, "promoted samples")
        if promoted_samples is not None
        else None
    )
    config = XGBoostRagConfig(
        code_revision=_require_string(code_revision, "code_revision"),
        legacy_code_manifest_hash=runtime.code_manifest_hash,
        feature_recipes=recipes,
        historical_gates=gates,
        anchor_ledger=anchor,
        promoted_samples=promoted,
        include_promoted_samples=include_promoted_samples,
        compile_settings=compile_settings,
        training_settings=training_settings,
    )
    workspace = _workspace_from_parts(
        output_dir,
        safe_workspace_id,
        policy,
        config,
        runtime=runtime,
    )
    payload = _workspace_config(
        workspace_id=safe_workspace_id,
        policy=policy,
        runtime=runtime,
        config=config,
    )
    config_path = workspace.root / _WORKSPACE_CONFIG_NAME
    with workspace.promotion_stores.releases.workspace_lock():
        try:
            _write_workspace_config(config_path, payload)
        except FileExistsError:
            existing = _json_mapping(config_path, "workspace configuration")
            if canonical_json_hash(existing) != canonical_json_hash(payload):
                raise ArtifactIntegrityError(
                    "workspace already exists with a different immutable configuration",
                    stage="CONFIGURATION",
                    operator_action="Use a new output directory for changed backend or policy settings.",
                )
    return workspace


def open_local_workspace(output_dir: Path) -> LocalWorkspace:
    """Open an existing workspace from its persisted configuration only."""

    normalized_output = Path(output_dir).expanduser().resolve()
    config_path = normalized_output / _WORKSPACE_DIR_NAME / _WORKSPACE_CONFIG_NAME
    workspace_id, policy, config = _read_workspace_config(config_path)
    runtime = resolve_legacy_runtime()
    if runtime.code_manifest_hash != config.legacy_code_manifest_hash:
        raise ArtifactIntegrityError(
            "trusted legacy runtime no longer matches the workspace configuration",
            stage="CONFIGURATION",
            operator_action="Create a new workspace after reviewing the changed legacy runtime.",
        )
    return _workspace_from_parts(
        normalized_output,
        workspace_id,
        policy,
        config,
        runtime=runtime,
    )


__all__ = [
    "LocalFilePromotionTarget",
    "LocalWorkspace",
    "bootstrap_local_workspace",
    "default_promotion_idempotency_key",
    "open_local_workspace",
    "policy_for_compile_settings",
]
