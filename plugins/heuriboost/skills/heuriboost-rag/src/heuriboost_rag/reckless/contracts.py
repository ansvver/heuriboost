from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class _FrozenMapping(MappingABC[str, object]):
    __slots__ = ("__data",)

    def __init__(self, items: tuple[tuple[str, object], ...]) -> None:
        ordered = dict(sorted(items, key=lambda item: item[0]))
        object.__setattr__(self, "_FrozenMapping__data", MappingProxyType(ordered))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __getitem__(self, key: str) -> object:
        return self.__data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self.__data)!r})"

    def __copy__(self) -> _FrozenMapping:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenMapping:
        copied = type(self)(
            tuple((key, deepcopy(value, memo)) for key, value in self.__data.items())
        )
        memo[id(self)] = copied
        return copied

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), (tuple(self.__data.items()),)


def _is_public_contract(value: object) -> bool:
    return (
        not isinstance(value, type)
        and is_dataclass(value)
        and type(value).__module__ == __name__
    )


def _enum_value_is_immutable(value: object) -> bool:
    if isinstance(value, Enum):
        return _enum_value_is_immutable(value.value)
    if isinstance(value, (tuple, frozenset)):
        return all(_enum_value_is_immutable(item) for item in value)
    return value is None or isinstance(value, (bool, int, float, str, Path))


def _freeze_value(value: object) -> object:
    if _is_public_contract(value):
        return value
    if isinstance(value, MappingABC):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, Path):
        return value
    if isinstance(value, Enum):
        if not _enum_value_is_immutable(value.value):
            raise TypeError(
                f"Unsupported mutable Enum value for {type(value).__name__}"
            )
        return value
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported contract value type: {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen_items = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Contract mapping keys must be str, got {type(key).__name__}"
            )
        frozen_items.append((key, _freeze_value(item)))
    return _FrozenMapping(tuple(frozen_items))


def _freeze_mapping_fields(instance: object, *field_names: str) -> None:
    for field_name in field_names:
        value = getattr(instance, field_name)
        if value is not None:
            object.__setattr__(instance, field_name, _freeze_mapping(value))


def _deterministic_set_key(value: object) -> tuple[str, str, str]:
    plain = to_plain_data(value)
    return (
        type(value).__module__,
        type(value).__qualname__,
        json.dumps(
            plain,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def to_plain_data(value: object) -> object:
    if _is_public_contract(value):
        return {
            field.name: to_plain_data(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, MappingABC):
        for key in value:
            if not isinstance(key, str):
                raise TypeError(
                    f"Plain-data mapping keys must be str, got {type(key).__name__}"
                )
        return {key: to_plain_data(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            to_plain_data(item)
            for item in sorted(value, key=_deterministic_set_key)
        ]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return to_plain_data(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported plain-data value type: {type(value).__name__}")


@dataclass(frozen=True)
class RepairRequest:
    workspace_id: str
    base_dataset_id: str
    production_cases_id: str
    policy_version: str
    backend_name: str
    requested_by: str
    run_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_mapping_fields(self, "run_options")


@dataclass(frozen=True)
class PromotionApproval:
    run_id: str
    approved_by: str
    approved_at: str
    report_hash: str
    decision_hash: str
    expected_current_model: str | None
    idempotency_key: str


@dataclass(frozen=True)
class GateCheck:
    check_id: str
    label: str
    passed: bool
    observed: object
    required: object
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed", _freeze_value(self.observed))
        object.__setattr__(self, "required", _freeze_value(self.required))


@dataclass(frozen=True)
class Decision:
    promotion_eligible: bool
    acceptance_level: str
    checks: tuple[GateCheck, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    role: str
    path: Path
    content_hash: str
    schema_hash: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_mapping_fields(self, "metadata")


@dataclass(frozen=True)
class ArtifactRef:
    artifact_type: str
    path: Path
    content_hash: str
    size_bytes: int


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    metadata: Mapping[str, object]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _freeze_mapping_fields(self, "metadata")


@dataclass(frozen=True)
class CompiledInputs:
    artifacts: tuple[ArtifactRef, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        _freeze_mapping_fields(self, "metadata")


@dataclass(frozen=True)
class SynthesizedFeatures:
    artifacts: tuple[ArtifactRef, ...]
    compiled_inputs: CompiledInputs
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        _freeze_mapping_fields(self, "metadata")


@dataclass(frozen=True)
class CandidateModel:
    model_path: Path
    artifacts: tuple[ArtifactRef, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        _freeze_mapping_fields(self, "metadata")


@dataclass(frozen=True)
class ArtifactVerification:
    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class TargetValidation:
    valid: bool
    current_model: str | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class PreparedActivation:
    run_id: str
    pointer_payload: Mapping[str, object]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        _freeze_mapping_fields(self, "pointer_payload", "metadata")


@dataclass(frozen=True)
class ActivationResult:
    current_model: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        _freeze_mapping_fields(self, "metadata")


@dataclass(frozen=True)
class EvaluationResult:
    acceptance_level: str
    current_cases_passed: bool
    historical_gates_passed: bool
    global_metrics: Mapping[str, float]
    anchor_metrics: Mapping[str, float]
    touched_domains: Mapping[str, Mapping[str, float]]
    artifacts_valid: bool
    details: Mapping[str, object]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _freeze_mapping_fields(
            self,
            "global_metrics",
            "anchor_metrics",
            "touched_domains",
            "details",
        )


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    datasets: Mapping[str, DatasetRef]
    options: Mapping[str, object]

    def __post_init__(self) -> None:
        _freeze_mapping_fields(self, "datasets", "options")


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    state: str
    version: int
    request: RepairRequest
    policy_hash: str
    input_hash: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    error: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _freeze_mapping_fields(self, "metadata", "error")


@dataclass(frozen=True)
class StageManifest:
    stage: str
    input_hash: str
    artifacts: tuple[ArtifactRef, ...]
    started_at: str
    completed_at: str


@dataclass(frozen=True)
class ReportArtifact:
    html_path: Path
    data_path: Path
    manifest_path: Path
    data_hash: str
    html_hash: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        _freeze_mapping_fields(self, "manifest")


@dataclass(frozen=True)
class ReleaseSnapshot:
    run_id: str
    artifacts: tuple[ArtifactRef, ...]
    manifest_hash: str
    previous_model: str | None


@dataclass(frozen=True)
class PromotionReceipt:
    run_id: str
    release_path: Path
    promoted_at: str
    approved_by: str
    previous_model: str | None
    current_model: str
    release_manifest_hash: str
    receipt_json_path: Path
    receipt_html_path: Path


@dataclass(frozen=True)
class RollbackReceipt:
    source_run_id: str
    rolled_back_at: str
    approved_by: str
    previous_model: str
    restored_model: str
    receipt_json_path: Path
    receipt_html_path: Path
