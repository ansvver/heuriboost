"""Immutable adapter around the legacy HeuriBoost XGBoost repair runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as distribution_version
import json
import math
from pathlib import Path
import platform
import re
import shutil
import tempfile
from typing import Any, Mapping

from ..reckless.contracts import (
    ArtifactRef,
    ArtifactVerification,
    CandidateModel,
    CompiledInputs,
    DatasetRef,
    EvaluationResult,
    RepairRequest,
    RunContext,
    ValidationResult,
)
from ..reckless.hashing import (
    ExecutionIdentity,
    atomic_write_json,
    canonical_json_hash,
    sha256_file,
)
from .legacy_runtime import (
    LegacyRuntime,
    LegacyRuntimeResolutionError,
    resolve_legacy_runtime,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BACKEND_VERSION = "xgboost-rag-v1"
_MODEL_ARTIFACT = "xgboost-model"
_MODEL_METADATA_ARTIFACT = "xgboost-model-metadata"
_BASE_SNAPSHOT_ARTIFACT = "base-dataset-snapshot"
_PRODUCTION_SNAPSHOT_ARTIFACT = "production-cases-snapshot"
_FEATURE_RECIPE_SNAPSHOT_ARTIFACT = "feature-recipes-snapshot"
_HISTORICAL_GATES_SNAPSHOT_ARTIFACT = "historical-gates-snapshot"
_ANCHOR_LEDGER_SNAPSHOT_ARTIFACT = "anchor-ledger-snapshot"
_PROMOTED_SAMPLES_SNAPSHOT_ARTIFACT = "promoted-samples-snapshot"
_COMPILED_BASE_ARTIFACT = "compiled-base-dataset"
_COMPILED_REGRESSION_ARTIFACT = "compiled-regression-cases"
_COMPILED_CASES_ARTIFACT = "compiled-production-cases"
_COMPILED_CURRENT_CASE_SET_ARTIFACT = "compiled-current-case-set"
_COMPILED_REPORT_ARTIFACT = "compiled-report"
_COMPILED_BINDING_ARTIFACT = "compiled-input-binding"
_CANDIDATE_BINDING_ARTIFACT = "candidate-binding"
_POLICY_METRIC_NAMES = ("ndcg@10", "mrr@10")
_BINDING_SCHEMA_VERSION = 1
_COMPILED_SEMANTIC_METADATA_KEYS = (
    "runtime_config_hash",
    "code_revision",
    "feature_names",
    "feature_version",
    "compile_settings",
    "training_settings",
    "pinned_input_hashes",
    "legacy_code_manifest_hash",
    "expected_legacy_code_manifest_hash",
    "touched_domains",
    "warnings",
    "promoted_samples_enabled",
)
_COMPILED_BINDING_METADATA_KEYS = frozenset(
    {
        "compiled_metadata_hash",
        "compiled_payload_artifact_set_hash",
        "compiled_binding_hash",
        "compiled_artifact_set_hash",
    }
)
_CANDIDATE_BINDING_METADATA_KEYS = frozenset(
    {
        "candidate_metadata_hash",
        "candidate_payload_artifact_set_hash",
        "candidate_binding_hash",
        "candidate_artifact_set_hash",
    }
)


def _runtime_dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name, distribution in (
        ("xgboost", "xgboost"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("pyyaml", "PyYAML"),
    ):
        try:
            versions[name] = distribution_version(distribution)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _settings_dict(settings: "CompileSettings") -> dict[str, object]:
    return {
        "resplit": settings.resplit,
        "split_ratios": settings.split_ratios,
        "split_seed": settings.split_seed,
        "case_top_k": settings.case_top_k,
        "strict": settings.strict,
        "acceptance_level": settings.acceptance_level,
        "min_global_test_queries": settings.min_global_test_queries,
        "min_domain_test_queries": settings.min_domain_test_queries,
        "min_docs_per_query": settings.min_docs_per_query,
    }


def _training_settings_dict(settings: "TrainingSettings") -> dict[str, object]:
    return {
        "rounds": settings.rounds,
        "random_seed": settings.random_seed,
    }


@dataclass(frozen=True)
class PinnedFile:
    """A startup-selected source file bound to a SHA-256 digest."""

    path: Path
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not isinstance(self.content_hash, str) or _SHA256_RE.fullmatch(
            self.content_hash
        ) is None:
            raise ValueError("content_hash must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CompileSettings:
    resplit: bool = False
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_seed: int = 42
    case_top_k: int = 3
    strict: bool = True
    acceptance_level: str = "full"
    min_global_test_queries: int = 50
    min_domain_test_queries: int = 10
    min_docs_per_query: int = 2

    def __post_init__(self) -> None:
        _require_bool(self.resplit, "resplit")
        ratios = tuple(float(value) for value in self.split_ratios)
        if len(ratios) != 3 or any(value < 0.0 for value in ratios):
            raise ValueError("split_ratios must contain three non-negative values")
        if sum(ratios) <= 0.0:
            raise ValueError("split_ratios must have a positive sum")
        object.__setattr__(self, "split_ratios", ratios)
        if type(self.split_seed) is not int:
            raise TypeError("split_seed must be an integer")
        _require_positive_int(self.case_top_k, "case_top_k")
        _require_bool(self.strict, "strict")
        if self.acceptance_level not in {"full", "weak"}:
            raise ValueError("acceptance_level must be 'full' or 'weak'")
        _require_positive_int(self.min_global_test_queries, "min_global_test_queries")
        _require_positive_int(self.min_domain_test_queries, "min_domain_test_queries")
        _require_positive_int(self.min_docs_per_query, "min_docs_per_query")


@dataclass(frozen=True)
class TrainingSettings:
    rounds: int = 40
    random_seed: int = 42

    def __post_init__(self) -> None:
        _require_positive_int(self.rounds, "rounds")
        if type(self.random_seed) is not int:
            raise TypeError("random_seed must be an integer")


@dataclass(frozen=True)
class XGBoostRagConfig:
    """All model-affecting startup configuration for the legacy adapter."""

    code_revision: str
    legacy_code_manifest_hash: str
    feature_recipes: PinnedFile
    historical_gates: PinnedFile
    anchor_ledger: PinnedFile
    promoted_samples: PinnedFile | None = None
    include_promoted_samples: bool = False
    compile_settings: CompileSettings = field(default_factory=CompileSettings)
    training_settings: TrainingSettings = field(default_factory=TrainingSettings)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.code_revision, "code_revision")
        if not isinstance(self.legacy_code_manifest_hash, str) or _SHA256_RE.fullmatch(
            self.legacy_code_manifest_hash
        ) is None:
            raise ValueError(
                "legacy_code_manifest_hash must be a lowercase SHA-256 digest"
            )
        for name in ("feature_recipes", "historical_gates", "anchor_ledger"):
            if not isinstance(getattr(self, name), PinnedFile):
                raise TypeError(f"{name} must be a PinnedFile")
        if self.promoted_samples is not None and not isinstance(
            self.promoted_samples, PinnedFile
        ):
            raise TypeError("promoted_samples must be a PinnedFile or None")
        _require_bool(self.include_promoted_samples, "include_promoted_samples")
        if self.include_promoted_samples and self.promoted_samples is None:
            raise ValueError(
                "promoted_samples is required when include_promoted_samples is true"
            )
        if not isinstance(self.compile_settings, CompileSettings):
            raise TypeError("compile_settings must be CompileSettings")
        if not isinstance(self.training_settings, TrainingSettings):
            raise TypeError("training_settings must be TrainingSettings")


def _artifact_ref(artifact_type: str, path: Path) -> ArtifactRef:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"required artifact is missing: {source}")
    return ArtifactRef(
        artifact_type=artifact_type,
        path=source,
        content_hash=sha256_file(source),
        size_bytes=source.stat().st_size,
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _copy_snapshot(source: Path, destination: Path, expected_hash: str) -> Path:
    if not source.is_file():
        raise ValueError(f"pinned source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != expected_hash:
        raise ValueError(f"pinned source changed while being snapshotted: {source}")
    return destination


def _plain_metrics(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} metrics must be a mapping")
    metrics: dict[str, float] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} contains an invalid metric name")
        if type(raw) not in (int, float) or not math.isfinite(float(raw)):
            raise ValueError(f"{label} contains a non-finite {name} metric")
        metrics[name] = float(raw)
    for name in _POLICY_METRIC_NAMES:
        if name not in metrics:
            raise ValueError(f"{label} is missing a finite {name} metric")
    return metrics


def _artifact_set_hash(artifacts: tuple[ArtifactRef, ...]) -> str:
    """Bind artifact content, not locations that Core 6 may relocate on resume."""

    return canonical_json_hash(
        tuple(
            {
                "artifact_type": artifact.artifact_type,
                "content_hash": artifact.content_hash,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in sorted(
                artifacts,
                key=lambda artifact: (
                    artifact.artifact_type,
                    artifact.content_hash,
                    artifact.size_bytes,
                ),
            )
        )
    )


def _artifact_descriptors(artifacts: tuple[ArtifactRef, ...]) -> tuple[dict[str, object], ...]:
    """Canonical path-independent descriptors used inside adapter bindings."""

    return tuple(
        {
            "artifact_type": artifact.artifact_type,
            "content_hash": artifact.content_hash,
            "size_bytes": artifact.size_bytes,
        }
        for artifact in sorted(
            artifacts,
            key=lambda artifact: (
                artifact.artifact_type,
                artifact.content_hash,
                artifact.size_bytes,
            ),
        )
    )


def _without_artifact_type(
    artifacts: tuple[ArtifactRef, ...], artifact_type: str
) -> tuple[ArtifactRef, ...]:
    return tuple(
        artifact for artifact in artifacts if artifact.artifact_type != artifact_type
    )


def _execution_identity_data(identity: ExecutionIdentity) -> dict[str, object]:
    return {
        "backend_version": identity.backend_version,
        "feature_names": identity.feature_names,
        "feature_version": identity.feature_version,
        "code_commit": identity.code_commit,
        "training_params": identity.training_params,
        "random_seed": identity.random_seed,
    }


def _dataset_binding_data(dataset: DatasetRef) -> dict[str, object]:
    """Bind semantic DatasetRef fields while permitting run-local relocation."""

    return {
        "dataset_id": dataset.dataset_id,
        "role": dataset.role,
        "content_hash": dataset.content_hash,
        "schema_hash": dataset.schema_hash,
        "metadata": dataset.metadata,
    }


def _compiled_semantic_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        name: metadata[name]
        for name in _COMPILED_SEMANTIC_METADATA_KEYS
        if name in metadata
    }


def _candidate_semantic_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        str(name): value
        for name, value in metadata.items()
        if name not in _CANDIDATE_BINDING_METADATA_KEYS
    }


def _read_binding_payload(artifact: ArtifactRef, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} cannot be loaded: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _plain_value(value: object) -> object:
    """Convert legacy scalar containers to contract-safe plain data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("legacy result contains a non-finite float")
        return value
    if isinstance(value, Path):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_plain_value(item) for item in value)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return _plain_value(item_method())
    raise TypeError(f"unsupported legacy result value: {type(value).__name__}")


def _artifact_by_type(
    artifacts: tuple[ArtifactRef, ...], artifact_type: str
) -> ArtifactRef:
    matches = [artifact for artifact in artifacts if artifact.artifact_type == artifact_type]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {artifact_type!r} artifact, found {len(matches)}"
        )
    return matches[0]


class XGBoostRagBackend:
    """Run the legacy repair flow from immutable startup configuration only."""

    __slots__ = ("_config", "_runtime")

    name = "xgboost-rag"

    def __init__(
        self,
        config: XGBoostRagConfig | None = None,
        *,
        runtime: LegacyRuntime | None = None,
    ) -> None:
        if config is not None and not isinstance(config, XGBoostRagConfig):
            raise TypeError("config must be XGBoostRagConfig or None")
        if runtime is not None and not isinstance(runtime, LegacyRuntime):
            raise TypeError("runtime must be LegacyRuntime or None")
        if config is not None and runtime is None:
            runtime = resolve_legacy_runtime()
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_runtime", runtime)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    @property
    def config(self) -> XGBoostRagConfig | None:
        return self._config

    @property
    def runtime(self) -> LegacyRuntime:
        if self._runtime is None:
            raise LegacyRuntimeResolutionError("unconfigured backend has no legacy runtime")
        return self._runtime

    def _config_material(self) -> dict[str, object]:
        if self.config is None:
            return {
                "configured": False,
                "compile_settings": _settings_dict(CompileSettings()),
                "training_settings": _training_settings_dict(TrainingSettings()),
                "pinned_input_hashes": {},
                "promoted_samples_enabled": False,
            }

        pinned_hashes = {
            "feature_recipes": self.config.feature_recipes.content_hash,
            "historical_gates": self.config.historical_gates.content_hash,
            "anchor_ledger": self.config.anchor_ledger.content_hash,
        }
        if self.config.promoted_samples is not None:
            pinned_hashes["promoted_samples"] = self.config.promoted_samples.content_hash
        return {
            "configured": True,
            "code_revision": self.config.code_revision,
            "expected_legacy_code_manifest_hash": self.config.legacy_code_manifest_hash,
            "compile_settings": _settings_dict(self.config.compile_settings),
            "training_settings": _training_settings_dict(self.config.training_settings),
            "pinned_input_hashes": pinned_hashes,
            "promoted_samples_enabled": self.config.include_promoted_samples,
        }

    def _identity_material(self) -> dict[str, object]:
        material = self._config_material()
        material["runtime_dependency_versions"] = _runtime_dependency_versions()
        if self.config is not None:
            material["legacy_code_manifest_hash"] = self.runtime.code_manifest_hash
            material["legacy_fixed_training_params"] = (
                self.runtime.fixed_training_params_mapping
            )
            material["captured_feature_recipe_hash"] = self.runtime.feature_recipe_hash
        return material

    def _runtime_config_hash(self) -> str:
        return canonical_json_hash(self._identity_material())

    def _validation_config_hash(self) -> str:
        try:
            return self._runtime_config_hash()
        except (LegacyRuntimeResolutionError, OSError):
            return canonical_json_hash(self._config_material())

    def execution_identity(self) -> ExecutionIdentity:
        material = self._identity_material()
        if self.config is None:
            return ExecutionIdentity(
                backend_version=_BACKEND_VERSION,
                feature_names=("unconfigured",),
                feature_version="unconfigured",
                code_commit="unconfigured",
                training_params={
                    "runtime_config_hash": self._runtime_config_hash(),
                    **material,
                },
                random_seed=TrainingSettings().random_seed,
            )

        registry = self.runtime.registry
        feature_names = tuple(str(name) for name in registry.feature_names)
        if not feature_names:
            raise ValueError("legacy runtime provided no feature names")
        return ExecutionIdentity(
            backend_version=_BACKEND_VERSION,
            feature_names=feature_names,
            feature_version=str(registry.feature_version),
            code_commit=self.config.code_revision,
            training_params={
                "runtime_config_hash": self._runtime_config_hash(),
                **material,
            },
            random_seed=self.config.training_settings.random_seed,
        )

    def _dataset(
        self, context: RunContext, key: str, expected_id: str
    ) -> DatasetRef | None:
        value = context.datasets.get(key)
        if value is None:
            value = context.datasets.get(expected_id)
        return value if isinstance(value, DatasetRef) else None

    def _pinned_file_errors(self, pinned: PinnedFile, label: str) -> list[str]:
        errors: list[str] = []
        path = pinned.path
        if not path.is_file():
            return [f"{label} source is missing: {path}"]
        if path.stat().st_size == 0:
            errors.append(f"{label} source is empty: {path}")
        try:
            actual_hash = sha256_file(path)
        except OSError as exc:
            errors.append(f"cannot read {label} source: {exc}")
        else:
            if actual_hash != pinned.content_hash:
                errors.append(f"{label} hash mismatch")
        return errors

    def _dataset_errors(self, dataset: DatasetRef | None, label: str) -> list[str]:
        if dataset is None:
            return [f"missing {label} DatasetRef"]
        errors: list[str] = []
        if dataset.role != label:
            errors.append(
                f"{label} DatasetRef role must be {label!r}, got {dataset.role!r}"
            )
        if not isinstance(dataset.schema_hash, str) or not dataset.schema_hash:
            errors.append(f"{label} DatasetRef has no schema_hash")
        if not dataset.path.is_file():
            return [*errors, f"{label} dataset file is missing: {dataset.path}"]
        try:
            actual_hash = sha256_file(dataset.path)
        except OSError as exc:
            errors.append(f"cannot read {label} dataset: {exc}")
        else:
            if actual_hash != dataset.content_hash:
                errors.append(f"{label} DatasetRef content hash mismatch")
        return errors

    def _load_anchor(self, path: Path) -> dict[str, object]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid anchor ledger: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError("anchor ledger must be a JSON object")
        anchor = raw.get("anchor")
        if not isinstance(anchor, Mapping):
            raise ValueError("anchor ledger must contain a non-empty anchor")
        global_metrics = _plain_metrics(anchor.get("global"), "anchor global")
        domains = anchor.get("domains")
        if not isinstance(domains, Mapping) or not domains:
            raise ValueError("anchor ledger must contain non-empty anchor domains")
        plain_domains = {
            str(domain): _plain_metrics(metrics, f"anchor domain {domain!r}")
            for domain, metrics in domains.items()
        }
        return {"global": global_metrics, "domains": plain_domains}

    def _legacy_options(self, output_dir: Path) -> object:
        if self.config is None:
            raise ValueError("backend is unconfigured")
        binding_errors = self._runtime_binding_errors()
        if binding_errors:
            raise ValueError("runtime binding failed: " + "; ".join(binding_errors))
        settings = self.config.compile_settings
        return self.runtime.compile_options_type(
            output_dir=output_dir,
            resplit=settings.resplit,
            split_ratios=settings.split_ratios,
            split_seed=settings.split_seed,
            case_top_k=settings.case_top_k,
            strict=settings.strict,
            acceptance_level=settings.acceptance_level,
            min_global_test_queries=settings.min_global_test_queries,
            min_domain_test_queries=settings.min_domain_test_queries,
            min_docs_per_query=settings.min_docs_per_query,
        )

    def _runtime_binding_errors(self) -> list[str]:
        if self.config is None:
            return ["backend is unconfigured; pinned runtime inputs are required"]
        errors: list[str] = []
        try:
            runtime = self.runtime
            code_manifest_hash = runtime.code_manifest_hash
            captured_recipe_hash = runtime.feature_recipe_hash
        except (LegacyRuntimeResolutionError, OSError, SystemExit) as exc:
            return [f"legacy runtime initialization failed: {exc}"]
        source_path = Path(runtime.feature_recipe_path)
        if source_path.resolve() != self.config.feature_recipes.path.resolve():
            errors.append("configured feature recipe source does not match legacy runtime")
        errors.extend(
            self._pinned_file_errors(self.config.feature_recipes, "feature_recipes")
        )
        if captured_recipe_hash != self.config.feature_recipes.content_hash:
            errors.append(
                "captured feature recipe hash does not match active config"
            )
        if not isinstance(code_manifest_hash, str) or _SHA256_RE.fullmatch(
            code_manifest_hash
        ) is None:
            errors.append("legacy_code_manifest_hash is invalid")
        elif code_manifest_hash != self.config.legacy_code_manifest_hash:
            errors.append(
                "legacy_code_manifest_hash does not match the startup-pinned legacy code"
            )
        registry = runtime.registry
        if not tuple(registry.feature_names):
            errors.append("legacy runtime registry has no feature names")
        if not isinstance(registry.feature_version, str) or not registry.feature_version:
            errors.append("legacy runtime registry has no feature version")
        if self.config.training_settings.random_seed != 42:
            errors.append("legacy runtime fixes random_seed at 42")
        if not self.config.compile_settings.strict:
            errors.append("configured execution requires strict compile settings")
        return errors

    def _artifact_root_errors(
        self, artifacts: tuple[ArtifactRef, ...], context: RunContext
    ) -> tuple[str, ...]:
        return tuple(
            f"{artifact.artifact_type}: artifact is outside context.run_dir"
            for artifact in artifacts
            if not _is_within(artifact.path, context.run_dir)
        )

    def _context_dataset(self, context: RunContext, role: str) -> DatasetRef | None:
        direct = context.datasets.get(role)
        if isinstance(direct, DatasetRef) and direct.role == role:
            return direct
        matches = [
            dataset
            for dataset in context.datasets.values()
            if isinstance(dataset, DatasetRef) and dataset.role == role
        ]
        return matches[0] if len(matches) == 1 else None

    def _context_snapshot_errors(
        self, artifacts: tuple[ArtifactRef, ...], context: RunContext
    ) -> tuple[str, ...]:
        expected_snapshots = (
            (_BASE_SNAPSHOT_ARTIFACT, "base", self._context_dataset(context, "base")),
            (
                _PRODUCTION_SNAPSHOT_ARTIFACT,
                "production_cases",
                self._context_dataset(context, "production_cases"),
            ),
        )
        errors: list[str] = []
        for artifact_type, label, dataset in expected_snapshots:
            if dataset is None:
                errors.append(f"missing {label} DatasetRef in RunContext")
                continue
            try:
                snapshot = _artifact_by_type(artifacts, artifact_type)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if snapshot.content_hash != dataset.content_hash:
                errors.append(
                    f"{label} dataset snapshot hash does not match RunContext"
                )
        return tuple(errors)

    def _pinned_snapshot_errors(
        self, artifacts: tuple[ArtifactRef, ...]
    ) -> tuple[str, ...]:
        if self.config is None:
            return ("backend is unconfigured",)
        pinned_snapshots = [
            (
                _FEATURE_RECIPE_SNAPSHOT_ARTIFACT,
                "feature_recipes",
                self.config.feature_recipes,
            ),
            (
                _HISTORICAL_GATES_SNAPSHOT_ARTIFACT,
                "historical_gates",
                self.config.historical_gates,
            ),
            (
                _ANCHOR_LEDGER_SNAPSHOT_ARTIFACT,
                "anchor_ledger",
                self.config.anchor_ledger,
            ),
        ]
        if self.config.include_promoted_samples:
            if self.config.promoted_samples is None:
                return ("configured promoted samples are missing",)
            pinned_snapshots.append(
                (
                    _PROMOTED_SAMPLES_SNAPSHOT_ARTIFACT,
                    "promoted_samples",
                    self.config.promoted_samples,
                )
            )

        errors: list[str] = []
        for artifact_type, label, pinned in pinned_snapshots:
            try:
                snapshot = _artifact_by_type(artifacts, artifact_type)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if snapshot.content_hash != pinned.content_hash:
                errors.append(f"{label} snapshot hash does not match active config")
        return tuple(errors)

    def _expected_artifact_types(self, *, include_model: bool) -> tuple[str, ...]:
        required_types = [
            _BASE_SNAPSHOT_ARTIFACT,
            _PRODUCTION_SNAPSHOT_ARTIFACT,
            _FEATURE_RECIPE_SNAPSHOT_ARTIFACT,
            _HISTORICAL_GATES_SNAPSHOT_ARTIFACT,
            _ANCHOR_LEDGER_SNAPSHOT_ARTIFACT,
            _COMPILED_BASE_ARTIFACT,
            _COMPILED_REGRESSION_ARTIFACT,
            _COMPILED_CASES_ARTIFACT,
            _COMPILED_CURRENT_CASE_SET_ARTIFACT,
            _COMPILED_REPORT_ARTIFACT,
            _COMPILED_BINDING_ARTIFACT,
        ]
        if self.config is not None and self.config.include_promoted_samples:
            required_types.append(_PROMOTED_SAMPLES_SNAPSHOT_ARTIFACT)
        if include_model:
            required_types.extend(
                (
                    _MODEL_ARTIFACT,
                    _MODEL_METADATA_ARTIFACT,
                    _CANDIDATE_BINDING_ARTIFACT,
                )
            )
        return tuple(required_types)

    def _required_artifact_errors(
        self,
        artifacts: tuple[ArtifactRef, ...],
        *,
        include_model: bool,
    ) -> tuple[str, ...]:
        required_types = self._expected_artifact_types(include_model=include_model)
        errors = [
            f"expected exactly one {artifact_type!r} artifact, found "
            f"{sum(artifact.artifact_type == artifact_type for artifact in artifacts)}"
            for artifact_type in required_types
            if sum(artifact.artifact_type == artifact_type for artifact in artifacts) != 1
        ]
        expected = set(required_types)
        errors.extend(
            f"unexpected artifact type: {artifact_type!r}"
            for artifact_type in sorted(
                {artifact.artifact_type for artifact in artifacts} - expected
            )
        )
        return tuple(errors)

    def _compiled_artifact_set_hash_errors(
        self,
        metadata: Mapping[str, object],
        artifacts: tuple[ArtifactRef, ...],
        *,
        include_model: bool,
    ) -> tuple[str, ...]:
        compiled_types = set(self._expected_artifact_types(include_model=False))
        compiled_artifacts = tuple(
            artifact for artifact in artifacts if artifact.artifact_type in compiled_types
        )
        errors: list[str] = []
        if metadata.get("compiled_artifact_set_hash") != _artifact_set_hash(
            compiled_artifacts
        ):
            errors.append("compiled_artifact_set_hash does not match artifacts")
        if include_model and metadata.get("candidate_artifact_set_hash") != _artifact_set_hash(
            artifacts
        ):
            errors.append("candidate_artifact_set_hash does not match artifacts")
        return tuple(errors)

    def _compiled_payload_artifacts(
        self, artifacts: tuple[ArtifactRef, ...]
    ) -> tuple[ArtifactRef, ...]:
        payload_types = set(self._expected_artifact_types(include_model=False))
        payload_types.discard(_COMPILED_BINDING_ARTIFACT)
        return tuple(
            artifact for artifact in artifacts if artifact.artifact_type in payload_types
        )

    def _compiled_binding_payload(
        self,
        context: RunContext,
        semantic_metadata: Mapping[str, object],
        payload_artifacts: tuple[ArtifactRef, ...],
    ) -> dict[str, object]:
        base = self._context_dataset(context, "base")
        production_cases = self._context_dataset(context, "production_cases")
        if base is None or production_cases is None:
            raise ValueError("compiled binding requires both input DatasetRefs")
        identity = self.execution_identity()
        return {
            "binding_schema_version": _BINDING_SCHEMA_VERSION,
            "binding_kind": "compiled-inputs",
            "run_id": context.run_id,
            "input_dataset_refs": {
                "base": _dataset_binding_data(base),
                "production_cases": _dataset_binding_data(production_cases),
            },
            "execution_identity": _execution_identity_data(identity),
            "runtime_config_hash": identity.training_params["runtime_config_hash"],
            "compiled_metadata": dict(semantic_metadata),
            "compiled_metadata_hash": canonical_json_hash(semantic_metadata),
            "compiled_payload_artifacts": _artifact_descriptors(payload_artifacts),
            "compiled_payload_artifact_set_hash": _artifact_set_hash(
                payload_artifacts
            ),
        }

    def _candidate_binding_payload(
        self,
        metadata: Mapping[str, object],
        payload_artifacts: tuple[ArtifactRef, ...],
    ) -> dict[str, object]:
        model_artifact = _artifact_by_type(payload_artifacts, _MODEL_ARTIFACT)
        model_metadata_artifact = _artifact_by_type(
            payload_artifacts, _MODEL_METADATA_ARTIFACT
        )
        semantic_metadata = _candidate_semantic_metadata(metadata)
        return {
            "binding_schema_version": _BINDING_SCHEMA_VERSION,
            "binding_kind": "candidate-model",
            "execution_identity": _execution_identity_data(self.execution_identity()),
            "compiled_binding_hash": metadata.get("compiled_binding_hash"),
            "compiled_metadata_hash": metadata.get("compiled_metadata_hash"),
            "candidate_metadata": semantic_metadata,
            "candidate_metadata_hash": canonical_json_hash(semantic_metadata),
            "candidate_payload_artifacts": _artifact_descriptors(payload_artifacts),
            "candidate_payload_artifact_set_hash": _artifact_set_hash(
                payload_artifacts
            ),
            "model_artifact": _artifact_descriptors((model_artifact,))[0],
            "model_metadata_artifact": _artifact_descriptors(
                (model_metadata_artifact,)
            )[0],
        }

    def _write_binding_artifact(
        self,
        run_dir: Path,
        artifact_type: str,
        filename: str,
        payload: Mapping[str, object],
    ) -> ArtifactRef:
        destination = run_dir / "bindings" / filename
        atomic_write_json(destination, payload)
        return _artifact_ref(artifact_type, destination)

    def _compiled_binding_errors(
        self,
        metadata: Mapping[str, object],
        artifacts: tuple[ArtifactRef, ...],
        context: RunContext,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        try:
            binding_artifact = _artifact_by_type(
                artifacts, _COMPILED_BINDING_ARTIFACT
            )
        except ValueError as exc:
            return (str(exc),)
        payload_artifacts = self._compiled_payload_artifacts(artifacts)
        semantic_metadata = _compiled_semantic_metadata(metadata)
        missing_metadata = [
            key
            for key in _COMPILED_SEMANTIC_METADATA_KEYS
            if key not in semantic_metadata
        ]
        if missing_metadata:
            errors.append(
                "compiled binding metadata is missing: " + ", ".join(missing_metadata)
            )
        semantic_hash = canonical_json_hash(semantic_metadata)
        if metadata.get("compiled_metadata_hash") != semantic_hash:
            errors.append("compiled binding metadata hash does not match compiled metadata")
        payload_set_hash = _artifact_set_hash(payload_artifacts)
        if metadata.get("compiled_payload_artifact_set_hash") != payload_set_hash:
            errors.append(
                "compiled binding payload artifact set hash does not match artifacts"
            )
        if metadata.get("compiled_binding_hash") != binding_artifact.content_hash:
            errors.append("compiled binding hash does not match binding artifact")
        try:
            payload = _read_binding_payload(binding_artifact, "compiled binding")
            expected = self._compiled_binding_payload(
                context,
                semantic_metadata,
                payload_artifacts,
            )
            if canonical_json_hash(payload) != canonical_json_hash(expected):
                errors.append(
                    "compiled binding does not match active inputs, identity, metadata, or artifacts"
                )
        except (TypeError, ValueError, OSError) as exc:
            errors.append(f"compiled binding cannot be verified: {exc}")
        return tuple(errors)

    def _candidate_binding_errors(
        self,
        metadata: Mapping[str, object],
        artifacts: tuple[ArtifactRef, ...],
    ) -> tuple[str, ...]:
        errors: list[str] = []
        try:
            binding_artifact = _artifact_by_type(
                artifacts, _CANDIDATE_BINDING_ARTIFACT
            )
        except ValueError as exc:
            return (str(exc),)
        payload_artifacts = _without_artifact_type(
            artifacts, _CANDIDATE_BINDING_ARTIFACT
        )
        semantic_metadata = _candidate_semantic_metadata(metadata)
        semantic_hash = canonical_json_hash(semantic_metadata)
        if metadata.get("candidate_metadata_hash") != semantic_hash:
            errors.append("candidate binding metadata hash does not match candidate metadata")
        payload_set_hash = _artifact_set_hash(payload_artifacts)
        if metadata.get("candidate_payload_artifact_set_hash") != payload_set_hash:
            errors.append(
                "candidate binding payload artifact set hash does not match artifacts"
            )
        if metadata.get("candidate_binding_hash") != binding_artifact.content_hash:
            errors.append("candidate binding hash does not match binding artifact")
        try:
            payload = _read_binding_payload(binding_artifact, "candidate binding")
            expected = self._candidate_binding_payload(
                metadata,
                payload_artifacts,
            )
            if canonical_json_hash(payload) != canonical_json_hash(expected):
                errors.append(
                    "candidate binding does not match identity, metadata, model, or artifacts"
                )
        except (TypeError, ValueError, OSError) as exc:
            errors.append(f"candidate binding cannot be verified: {exc}")
        return tuple(errors)

    def _identity_errors(self, metadata: Mapping[str, object]) -> tuple[str, ...]:
        try:
            active_code_manifest_hash = self.runtime.code_manifest_hash
            active_hash = self._runtime_config_hash()
        except (LegacyRuntimeResolutionError, OSError, SystemExit) as exc:
            return (f"active runtime_config_hash is unavailable: {exc}",)
        errors: list[str] = []
        if self.config is not None and metadata.get(
            "expected_legacy_code_manifest_hash"
        ) != self.config.legacy_code_manifest_hash:
            errors.append(
                "expected_legacy_code_manifest_hash does not match active config"
            )
        if metadata.get("legacy_code_manifest_hash") != active_code_manifest_hash:
            errors.append(
                "legacy_code_manifest_hash does not match active execution identity"
            )
        if metadata.get("runtime_config_hash") != active_hash:
            errors.append("runtime_config_hash does not match active execution identity")
        return tuple(errors)

    def _operational_errors(
        self,
        metadata: Mapping[str, object],
        artifacts: tuple[ArtifactRef, ...],
        context: RunContext,
        *,
        include_model: bool,
    ) -> tuple[str, ...]:
        errors = list(self._runtime_binding_errors())
        if errors:
            return tuple(errors)
        errors.extend(self._identity_errors(metadata))
        errors.extend(self._artifact_root_errors(artifacts, context))
        errors.extend(self._verify_refs(artifacts))
        errors.extend(self._context_snapshot_errors(artifacts, context))
        errors.extend(self._pinned_snapshot_errors(artifacts))
        errors.extend(self._required_artifact_errors(artifacts, include_model=include_model))
        errors.extend(
            self._compiled_artifact_set_hash_errors(
                metadata,
                artifacts,
                include_model=include_model,
            )
        )
        errors.extend(self._compiled_binding_errors(metadata, artifacts, context))
        if include_model:
            errors.extend(self._candidate_binding_errors(metadata, artifacts))
        return tuple(errors)

    def validate(self, request: RepairRequest, context: RunContext) -> ValidationResult:
        if self.config is None:
            return ValidationResult(
                valid=False,
                metadata={"reason": "backend is unconfigured"},
            )

        errors = self._runtime_binding_errors()
        base = self._dataset(context, "base", request.base_dataset_id)
        production_cases = self._dataset(
            context, "production_cases", request.production_cases_id
        )
        if request.backend_name != self.name:
            errors.append(
                f"request backend_name {request.backend_name!r} does not match {self.name!r}"
            )
        if base is not None and base.dataset_id != request.base_dataset_id:
            errors.append("base DatasetRef id does not match request")
        if (
            production_cases is not None
            and production_cases.dataset_id != request.production_cases_id
        ):
            errors.append("production_cases DatasetRef id does not match request")
        errors.extend(self._dataset_errors(base, "base"))
        errors.extend(self._dataset_errors(production_cases, "production_cases"))
        errors.extend(
            self._pinned_file_errors(self.config.historical_gates, "historical_gates")
        )
        errors.extend(self._pinned_file_errors(self.config.anchor_ledger, "anchor_ledger"))
        if self.config.promoted_samples is not None:
            errors.extend(
                self._pinned_file_errors(
                    self.config.promoted_samples, "promoted_samples"
                )
            )

        anchor: dict[str, object] | None = None
        if not errors:
            try:
                anchor = self._load_anchor(self.config.anchor_ledger.path)
                gates = self.runtime.load_gates(self.config.historical_gates.path)
                if not isinstance(gates, list) or not gates:
                    errors.append("historical_gates must contain at least one gate")
            except (OSError, ValueError, json.JSONDecodeError, SystemExit) as exc:
                errors.append(f"pinned gate or anchor input is invalid: {exc}")

        warnings: tuple[str, ...] = ()
        if not errors and base is not None and production_cases is not None:
            context.run_dir.mkdir(parents=True, exist_ok=True)
            try:
                with tempfile.TemporaryDirectory(
                    prefix=".legacy-compile-", dir=context.run_dir
                ) as scratch:
                    result = self.runtime.compile_repair_inputs(
                        base.path,
                        production_cases.path,
                        self._legacy_options(Path(scratch)),
                    )
                touched_domains = tuple(
                    str(value) for value in getattr(result, "touched_domains", ())
                )
                if not touched_domains:
                    errors.append("legacy compiler produced no touched domains")
                elif anchor is not None:
                    anchor_domains = anchor["domains"]
                    for domain in touched_domains:
                        if domain not in anchor_domains:
                            errors.append(
                                f"anchor ledger has no metrics for touched domain {domain!r}"
                            )
                raw_warnings = getattr(result, "warnings", ())
                warnings = tuple(str(value) for value in raw_warnings)
            except (OSError, ValueError, json.JSONDecodeError, SystemExit) as exc:
                errors.append(f"legacy compile validation failed: {exc}")

        return ValidationResult(
            valid=not errors,
            metadata={
                "reason": "valid" if not errors else "validation failed",
                "errors": tuple(errors),
                "runtime_config_hash": self._validation_config_hash(),
            },
            warnings=warnings,
        )

    def _compile_artifacts(
        self, result: object, run_dir: Path
    ) -> tuple[ArtifactRef, ...]:
        case_sets_dir = Path(getattr(result, "case_sets_dir"))
        candidates = (
            (
                _COMPILED_BASE_ARTIFACT,
                Path(getattr(result, "base_dataset_path")),
                "base-dataset.csv",
            ),
            (
                _COMPILED_REGRESSION_ARTIFACT,
                Path(getattr(result, "regression_cases_path")),
                "regression-cases.yaml",
            ),
            (
                _COMPILED_CASES_ARTIFACT,
                Path(getattr(result, "production_cases_json_path")),
                "production-cases.json",
            ),
            (
                _COMPILED_CURRENT_CASE_SET_ARTIFACT,
                case_sets_dir / "current_production_cases.csv",
                "current-case-set.csv",
            ),
            (
                _COMPILED_REPORT_ARTIFACT,
                Path(getattr(result, "compile_report_path")),
                "compile-report.md",
            ),
        )
        artifacts = []
        normalized_dir = run_dir / "compiled-artifacts"
        for artifact_type, path, filename in candidates:
            if not _is_within(path, run_dir):
                raise ValueError(f"legacy compiler returned artifact outside run dir: {path}")
            normalized_path = _copy_snapshot(
                path,
                normalized_dir / filename,
                sha256_file(path),
            )
            artifacts.append(_artifact_ref(artifact_type, normalized_path))
        return tuple(artifacts)

    def compile(self, request: RepairRequest, context: RunContext) -> CompiledInputs:
        validation = self.validate(request, context)
        if not validation.valid:
            raise ValueError("cannot compile invalid request: " + "; ".join(
                str(value) for value in validation.metadata["errors"]
            ))
        if self.config is None:
            raise ValueError("backend is unconfigured")
        binding_errors = self._runtime_binding_errors()
        if binding_errors:
            raise ValueError(
                "runtime binding failed before compilation snapshots: "
                + "; ".join(binding_errors)
            )

        base = self._dataset(context, "base", request.base_dataset_id)
        production_cases = self._dataset(
            context, "production_cases", request.production_cases_id
        )
        if base is None or production_cases is None:
            raise ValueError("validated datasets are missing")

        run_dir = context.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        input_dir = run_dir / "inputs"
        base_snapshot = _copy_snapshot(
            base.path,
            input_dir / f"base_dataset{base.path.suffix or '.data'}",
            base.content_hash,
        )
        production_snapshot = _copy_snapshot(
            production_cases.path,
            input_dir / f"production_cases{production_cases.path.suffix or '.data'}",
            production_cases.content_hash,
        )
        feature_snapshot = _copy_snapshot(
            self.config.feature_recipes.path,
            input_dir / f"feature_recipes{self.config.feature_recipes.path.suffix or '.yaml'}",
            self.config.feature_recipes.content_hash,
        )
        gates_snapshot = _copy_snapshot(
            self.config.historical_gates.path,
            input_dir / f"historical_gates{self.config.historical_gates.path.suffix or '.jsonl'}",
            self.config.historical_gates.content_hash,
        )
        ledger_snapshot = _copy_snapshot(
            self.config.anchor_ledger.path,
            input_dir / f"anchor_ledger{self.config.anchor_ledger.path.suffix or '.json'}",
            self.config.anchor_ledger.content_hash,
        )
        artifacts = [
            _artifact_ref(_BASE_SNAPSHOT_ARTIFACT, base_snapshot),
            _artifact_ref(_PRODUCTION_SNAPSHOT_ARTIFACT, production_snapshot),
            _artifact_ref(_FEATURE_RECIPE_SNAPSHOT_ARTIFACT, feature_snapshot),
            _artifact_ref(_HISTORICAL_GATES_SNAPSHOT_ARTIFACT, gates_snapshot),
            _artifact_ref(_ANCHOR_LEDGER_SNAPSHOT_ARTIFACT, ledger_snapshot),
        ]
        if self.config.include_promoted_samples:
            promoted = self.config.promoted_samples
            if promoted is None:
                raise ValueError("configured promoted samples are missing")
            promoted_snapshot = _copy_snapshot(
                promoted.path,
                input_dir / f"promoted_samples{promoted.path.suffix or '.csv'}",
                promoted.content_hash,
            )
            artifacts.append(
                _artifact_ref(_PROMOTED_SAMPLES_SNAPSHOT_ARTIFACT, promoted_snapshot)
            )

        compile_root = run_dir / "compile"
        result = self.runtime.compile_repair_inputs(
            base_snapshot,
            production_snapshot,
            self._legacy_options(compile_root),
        )
        artifacts.extend(self._compile_artifacts(result, run_dir))
        identity = self.execution_identity()
        compiled_payload_artifacts = tuple(artifacts)
        semantic_metadata = {
            "runtime_config_hash": identity.training_params["runtime_config_hash"],
            "code_revision": self.config.code_revision,
            "feature_names": identity.feature_names,
            "feature_version": identity.feature_version,
            "compile_settings": _settings_dict(self.config.compile_settings),
            "training_settings": _training_settings_dict(self.config.training_settings),
            "pinned_input_hashes": identity.training_params["pinned_input_hashes"],
            "legacy_code_manifest_hash": identity.training_params[
                "legacy_code_manifest_hash"
            ],
            "expected_legacy_code_manifest_hash": identity.training_params[
                "expected_legacy_code_manifest_hash"
            ],
            "touched_domains": tuple(
                str(value) for value in getattr(result, "touched_domains", ())
            ),
            "warnings": tuple(str(value) for value in getattr(result, "warnings", ())),
            "promoted_samples_enabled": self.config.include_promoted_samples,
        }
        compiled_binding = self._write_binding_artifact(
            run_dir,
            _COMPILED_BINDING_ARTIFACT,
            "compiled-input-binding.json",
            self._compiled_binding_payload(
                context,
                semantic_metadata,
                compiled_payload_artifacts,
            ),
        )
        compiled_artifacts = (*compiled_payload_artifacts, compiled_binding)
        return CompiledInputs(
            artifacts=compiled_artifacts,
            metadata={
                **semantic_metadata,
                "compiled_metadata_hash": canonical_json_hash(semantic_metadata),
                "compiled_payload_artifact_set_hash": _artifact_set_hash(
                    compiled_payload_artifacts
                ),
                "compiled_binding_hash": compiled_binding.content_hash,
                "compiled_artifact_set_hash": _artifact_set_hash(compiled_artifacts),
            },
        )

    def _verify_refs(self, artifacts: tuple[ArtifactRef, ...]) -> tuple[str, ...]:
        errors: list[str] = []
        for artifact in artifacts:
            path = artifact.path
            if not path.is_file():
                errors.append(f"{artifact.artifact_type}: missing artifact")
                continue
            try:
                actual_hash = sha256_file(path)
                actual_size = path.stat().st_size
            except OSError as exc:
                errors.append(f"{artifact.artifact_type}: cannot read artifact: {exc}")
                continue
            if actual_hash != artifact.content_hash:
                errors.append(f"{artifact.artifact_type}: hash mismatch")
            if actual_size != artifact.size_bytes:
                errors.append(f"{artifact.artifact_type}: size mismatch")
        return tuple(errors)

    def train(self, inputs: CompiledInputs, context: RunContext) -> CandidateModel:
        if self.config is None:
            raise ValueError("backend is unconfigured")
        errors = self._operational_errors(
            inputs.metadata,
            inputs.artifacts,
            context,
            include_model=False,
        )
        if errors:
            raise ValueError("compiled artifact verification failed: " + "; ".join(errors))
        base = _artifact_by_type(inputs.artifacts, _COMPILED_BASE_ARTIFACT)
        current_cases = _artifact_by_type(
            inputs.artifacts, _COMPILED_CURRENT_CASE_SET_ARTIFACT
        )
        promoted_path: Path | None = None
        if self.config.include_promoted_samples:
            promoted_path = _artifact_by_type(
                inputs.artifacts, _PROMOTED_SAMPLES_SNAPSHOT_ARTIFACT
            ).path

        base_frame = self.runtime.load_dataset(base.path)
        repair_samples = self.runtime.load_dataset(current_cases.path)
        training_frame = self.runtime.merge_training_frames(
            base_frame,
            repair_samples,
            promoted_samples_path=promoted_path,
        )
        training_root = context.run_dir / "training"
        self.runtime.train_model_from_frame(
            training_frame,
            training_root,
            rounds=self.config.training_settings.rounds,
        )
        model_path = training_root / "models" / "reranker.json"
        metadata_path = training_root / "models" / "reranker_metadata.json"
        model_artifact = _artifact_ref(_MODEL_ARTIFACT, model_path)
        metadata_artifact = _artifact_ref(_MODEL_METADATA_ARTIFACT, metadata_path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"candidate metadata is invalid: {exc}") from exc
        if not isinstance(metadata, Mapping):
            raise ValueError("candidate metadata must be a JSON object")

        candidate_payload_artifacts = (
            *inputs.artifacts,
            model_artifact,
            metadata_artifact,
        )
        identity = self.execution_identity()
        candidate_semantic_metadata = {
            **dict(inputs.metadata),
            "runtime_config_hash": identity.training_params["runtime_config_hash"],
            "legacy_code_manifest_hash": identity.training_params[
                "legacy_code_manifest_hash"
            ],
            "expected_legacy_code_manifest_hash": identity.training_params[
                "expected_legacy_code_manifest_hash"
            ],
            "code_revision": self.config.code_revision,
            "feature_names": tuple(identity.feature_names),
            "feature_version": identity.feature_version,
            "training_settings": _training_settings_dict(
                self.config.training_settings
            ),
            "touched_domains": inputs.metadata.get("touched_domains", ()),
            "compile_warnings": inputs.metadata.get("warnings", ()),
            "model_metadata": _plain_value(metadata),
        }
        candidate_binding = self._write_binding_artifact(
            context.run_dir,
            _CANDIDATE_BINDING_ARTIFACT,
            "candidate-binding.json",
            self._candidate_binding_payload(
                candidate_semantic_metadata,
                candidate_payload_artifacts,
            ),
        )
        candidate_artifacts = (*candidate_payload_artifacts, candidate_binding)
        candidate = CandidateModel(
            model_path=model_path,
            artifacts=candidate_artifacts,
            metadata={
                **candidate_semantic_metadata,
                "candidate_metadata_hash": canonical_json_hash(
                    candidate_semantic_metadata
                ),
                "candidate_payload_artifact_set_hash": _artifact_set_hash(
                    candidate_payload_artifacts
                ),
                "candidate_binding_hash": candidate_binding.content_hash,
                "candidate_artifact_set_hash": _artifact_set_hash(candidate_artifacts),
            },
        )
        verification = self.verify_artifacts(candidate, context)
        if not verification.valid:
            raise ValueError(
                "candidate artifact verification failed: " + "; ".join(verification.errors)
            )
        return candidate

    def _invalid_evaluation(
        self, errors: tuple[str, ...], warnings: tuple[str, ...] = ()
    ) -> EvaluationResult:
        return EvaluationResult(
            acceptance_level=(
                self.config.compile_settings.acceptance_level
                if self.config is not None
                else "full"
            ),
            current_cases_passed=False,
            historical_gates_passed=False,
            global_metrics={},
            anchor_metrics={},
            touched_domains={},
            artifacts_valid=False,
            details={"artifact_errors": errors},
            warnings=warnings,
        )

    def evaluate(self, candidate: CandidateModel, context: RunContext) -> EvaluationResult:
        if self.config is None:
            return self._invalid_evaluation(("backend is unconfigured",))
        operational_errors = self._operational_errors(
            candidate.metadata,
            candidate.artifacts,
            context,
            include_model=True,
        )
        if operational_errors:
            return self._invalid_evaluation(operational_errors)
        verification = self.verify_artifacts(candidate, context)
        if not verification.valid:
            return self._invalid_evaluation(verification.errors)
        try:
            base = _artifact_by_type(candidate.artifacts, _COMPILED_BASE_ARTIFACT)
            production_cases = _artifact_by_type(candidate.artifacts, _COMPILED_CASES_ARTIFACT)
            gates = _artifact_by_type(candidate.artifacts, _HISTORICAL_GATES_SNAPSHOT_ARTIFACT)
            ledger = _artifact_by_type(candidate.artifacts, _ANCHOR_LEDGER_SNAPSHOT_ARTIFACT)
            base_frame = self.runtime.load_dataset(base.path)
            cases = self.runtime.load_compiled_production_cases(production_cases.path)
            model = self.runtime.load_model(candidate.model_path)
            global_metrics, _ = self.runtime.evaluate_model_on_split(
                model, base_frame, "test"
            )
            domain_metrics = self.runtime.evaluate_model_by_domain(
                model, base_frame, "test"
            )
            current_case_results = self.runtime.evaluate_cases(
                cases,
                model,
                acceptance_level=self.config.compile_settings.acceptance_level,
            )
            historical_gates = self.runtime.load_gates(gates.path)
            gate_results = self.runtime.evaluate_cases(
                historical_gates,
                model,
                acceptance_level="full",
            )
            anchor = self._load_anchor(ledger.path)
            current_metrics = _plain_metrics(global_metrics, "global")
            anchor_metrics = _plain_metrics(anchor["global"], "anchor global")
            if not isinstance(domain_metrics, Mapping):
                raise ValueError("legacy domain metrics must be a mapping")
            touched_domains = tuple(
                str(value) for value in candidate.metadata.get("touched_domains", ())
            )
            if not touched_domains:
                touched_domains = tuple(
                    str(value) for value in domain_metrics.keys()
                )
            anchor_domains = anchor["domains"]
            touched: dict[str, dict[str, float]] = {}
            raw_domain_metrics: dict[str, dict[str, float]] = {}
            for domain in touched_domains:
                current = _plain_metrics(domain_metrics.get(domain), f"domain {domain!r}")
                baseline = _plain_metrics(
                    anchor_domains.get(domain), f"anchor domain {domain!r}"
                )
                raw_domain_metrics[domain] = current
                touched[domain] = {
                    **{name: current[name] for name in _POLICY_METRIC_NAMES},
                    **{
                        f"anchor_{name}": baseline[name]
                        for name in _POLICY_METRIC_NAMES
                    },
                }
            plain_current_cases = _plain_value(current_case_results)
            plain_gate_results = _plain_value(gate_results)
            if not isinstance(plain_current_cases, tuple) or not isinstance(
                plain_gate_results, tuple
            ):
                raise ValueError("legacy case evaluation must return a sequence")
            current_passed = bool(plain_current_cases) and all(
                isinstance(result, Mapping) and bool(result.get("passed"))
                for result in plain_current_cases
            )
            gates_passed = bool(plain_gate_results) and all(
                isinstance(result, Mapping) and bool(result.get("passed"))
                for result in plain_gate_results
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError, SystemExit) as exc:
            return self._invalid_evaluation(
                (f"legacy evaluation failed: {exc}",),
                tuple(str(value) for value in candidate.metadata.get("compile_warnings", ())),
            )

        return EvaluationResult(
            acceptance_level=self.config.compile_settings.acceptance_level,
            current_cases_passed=current_passed,
            historical_gates_passed=gates_passed,
            global_metrics=current_metrics,
            anchor_metrics=anchor_metrics,
            touched_domains=touched,
            artifacts_valid=True,
            details={
                "current_case_results": plain_current_cases,
                "historical_gate_results": plain_gate_results,
                "runtime_config_hash": self.execution_identity().training_params[
                    "runtime_config_hash"
                ],
                "raw_global_metrics": current_metrics,
                "raw_anchor_metrics": anchor_metrics,
                "raw_domain_metrics": raw_domain_metrics,
            },
            warnings=tuple(
                str(value) for value in candidate.metadata.get("compile_warnings", ())
            ),
        )

    def _model_metadata_errors(self, metadata: Mapping[str, object]) -> tuple[str, ...]:
        if self.config is None:
            return ("backend is unconfigured",)
        registry = self.runtime.registry
        expected_versions = registry.feature_versions_mapping
        expected_params = self.runtime.fixed_training_params_mapping
        errors: list[str] = []
        if metadata.get("feature_names") != list(registry.feature_names):
            errors.append("candidate model metadata feature_names do not match registry")
        if metadata.get("feature_set_name") != registry.feature_set_name:
            errors.append("candidate model metadata feature_set_name does not match registry")
        if metadata.get("feature_set_version") != registry.feature_set_version:
            errors.append(
                "candidate model metadata feature_set_version does not match registry"
            )
        if metadata.get("feature_versions") != expected_versions:
            errors.append("candidate model metadata feature_versions do not match registry")
        if metadata.get("params") != expected_params:
            errors.append(
                "candidate model metadata params do not match legacy fixed params"
            )
        if metadata.get("rounds") != self.config.training_settings.rounds:
            errors.append("candidate model metadata rounds do not match training settings")
        return tuple(errors)

    def verify_artifacts(
        self,
        candidate: CandidateModel,
        context: RunContext,
    ) -> ArtifactVerification:
        errors = list(
            self._operational_errors(
                candidate.metadata,
                candidate.artifacts,
                context,
                include_model=True,
            )
        )
        if errors:
            return ArtifactVerification(valid=False, errors=tuple(errors))
        try:
            model_artifact = _artifact_by_type(candidate.artifacts, _MODEL_ARTIFACT)
            metadata_artifact = _artifact_by_type(
                candidate.artifacts, _MODEL_METADATA_ARTIFACT
            )
        except ValueError as exc:
            errors.append(str(exc))
            return ArtifactVerification(valid=False, errors=tuple(errors))
        if model_artifact.path != candidate.model_path:
            errors.append("candidate model_path does not match model artifact")
        try:
            parsed_metadata = json.loads(metadata_artifact.path.read_text(encoding="utf-8"))
            if not isinstance(parsed_metadata, Mapping):
                errors.append("candidate model metadata is not a JSON object")
            else:
                errors.extend(self._model_metadata_errors(parsed_metadata))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"candidate model metadata cannot be loaded: {exc}")
        if not errors:
            try:
                self.runtime.load_model(candidate.model_path)
            except (OSError, ValueError, SystemExit) as exc:
                errors.append(f"candidate model cannot be loaded: {exc}")
        return ArtifactVerification(valid=not errors, errors=tuple(errors))
