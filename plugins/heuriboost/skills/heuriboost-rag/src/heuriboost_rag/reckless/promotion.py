"""Server-side promotion revalidation, idempotency, and rollback."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Protocol
import uuid

from .contracts import (
    ArtifactRef,
    PromotionApproval,
    PromotionReceipt,
    ReleaseSnapshot,
    RollbackReceipt,
    RunRecord,
    to_plain_data,
)
from .errors import ArtifactIntegrityError, HeuriBoostError, PromotionConflictError
from .hashing import canonical_json_hash
from .release_store import FileReleaseStore
from .report import build_report_data, load_sealed_report_evidence
from .state import RunState
from .storage import ArtifactStore, RunRepository


_JSON_SCHEMA_VERSION = 1
_REPORTING_STAGE = "REPORTING"
_TRAINED_STAGE = "TRAINED"
_TRAINED_RETRY_STAGE = "TRAINED_RETRY"
_PROMOTION_RECORD_FIELDS = frozenset(
    {"schema_version", "idempotency_key", "receipt"}
)
_REQUIRED_REPORT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "report_type",
        "run_id",
        "locale",
        "data_path",
        "html_path",
        "data_hash",
        "html_hash",
        "decision_hash",
        "model_hash",
        "policy_hash",
    }
)
_MODEL_TYPES = frozenset({"xgboost-model", "candidate-model", "model"})
_SCHEMA_TYPES = frozenset({"xgboost-model-metadata", "model-schema", "schema"})


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not (float("-inf") < parsed < float("inf")):
        raise ValueError("non-finite JSON number")
    return parsed


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            to_plain_data(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        .replace("<", "\\u003c")
        .encode("utf-8")
    )


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"not a regular file: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode):
            raise ValueError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> Mapping[str, object]:
    raw = _read_regular(path)
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
        parse_float=_strict_float,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON file must be an object: {path}")
    return value


def _rebase(artifacts: ArtifactStore, artifact: ArtifactRef, *, run_id: str) -> ArtifactRef:
    root = getattr(artifacts, "root", None)
    if not isinstance(root, Path):
        raise ArtifactIntegrityError(
            "artifact store must expose a filesystem root for promotion",
            stage="PROMOTING",
            run_id=run_id,
        )
    relative = Path(artifact.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactIntegrityError(
            "release artifact path is unsafe",
            stage="PROMOTING",
            run_id=run_id,
        )
    return ArtifactRef(
        artifact_type=artifact.artifact_type,
        path=root / relative,
        content_hash=artifact.content_hash,
        size_bytes=artifact.size_bytes,
    )


def _artifact_from_file(artifact_type: str, path: Path) -> ArtifactRef:
    raw = _read_regular(path)
    return ArtifactRef(
        artifact_type=artifact_type,
        path=path,
        content_hash=_hash_bytes(raw),
        size_bytes=len(raw),
    )


def _mapping(value: object, label: str, *, run_id: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(
            f"{label} must be a JSON object",
            stage="PROMOTING",
            run_id=run_id,
        )
    return value


def _string(value: object, label: str, *, run_id: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactIntegrityError(
            f"{label} must be a non-empty string",
            stage="PROMOTING",
            run_id=run_id,
        )
    return value


def _receipt_from_data(value: Mapping[str, object]) -> PromotionReceipt:
    required = {
        "run_id",
        "release_path",
        "promoted_at",
        "approved_by",
        "previous_model",
        "current_model",
        "release_manifest_hash",
        "receipt_json_path",
        "receipt_html_path",
    }
    if set(value) != required:
        raise ValueError("promotion receipt has an invalid schema")
    previous_model = value["previous_model"]
    if previous_model is not None and not isinstance(previous_model, str):
        raise ValueError("promotion receipt previous_model is invalid")
    strings = required - {"previous_model"}
    if any(not isinstance(value[key], str) or not value[key] for key in strings):
        raise ValueError("promotion receipt has an invalid string field")
    return PromotionReceipt(
        run_id=value["run_id"],
        release_path=Path(value["release_path"]),
        promoted_at=value["promoted_at"],
        approved_by=value["approved_by"],
        previous_model=previous_model,
        current_model=value["current_model"],
        release_manifest_hash=value["release_manifest_hash"],
        receipt_json_path=Path(value["receipt_json_path"]),
        receipt_html_path=Path(value["receipt_html_path"]),
    )


class PromotionRepository(Protocol):
    def find_by_idempotency_key(self, key: str) -> PromotionReceipt | None: ...

    def save(self, receipt: PromotionReceipt, idempotency_key: str) -> None: ...


class JsonPromotionRepository:
    """A small durable idempotency map for local-only promotion calls."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve() / "promotions"
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        descriptor = os.open(
            self.root / ".repository.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or not key:
            raise ValueError("idempotency key must be a non-empty string")
        return self.root / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    def _read(self, key: str) -> PromotionReceipt | None:
        path = self._path(key)
        try:
            record = _read_json(path)
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                "promotion idempotency record is invalid",
                stage="PROMOTING",
            ) from exc
        if set(record) != _PROMOTION_RECORD_FIELDS:
            raise ArtifactIntegrityError(
                "promotion idempotency record has an invalid schema",
                stage="PROMOTING",
            )
        if record["schema_version"] != _JSON_SCHEMA_VERSION or record["idempotency_key"] != key:
            raise ArtifactIntegrityError(
                "promotion idempotency record does not match its key",
                stage="PROMOTING",
            )
        try:
            return _receipt_from_data(_mapping(record["receipt"], "receipt", run_id="unknown"))
        except ValueError as exc:
            raise ArtifactIntegrityError(
                "promotion idempotency record has an invalid receipt",
                stage="PROMOTING",
            ) from exc

    def find_by_idempotency_key(self, key: str) -> PromotionReceipt | None:
        with self._lock():
            return self._read(key)

    def save(self, receipt: PromotionReceipt, idempotency_key: str) -> None:
        if not isinstance(receipt, PromotionReceipt):
            raise TypeError("receipt must be a PromotionReceipt")
        with self._lock():
            existing = self._read(idempotency_key)
            if existing is not None:
                if existing != receipt:
                    raise PromotionConflictError(
                        "idempotency key is already bound to another promotion receipt",
                        stage="PROMOTING",
                        run_id=receipt.run_id,
                    )
                return
            path = self._path(idempotency_key)
            payload = {
                "schema_version": _JSON_SCHEMA_VERSION,
                "idempotency_key": idempotency_key,
                "receipt": to_plain_data(receipt),
            }
            temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
            try:
                temporary.write_bytes(_json_bytes(payload))
                os.replace(temporary, path)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


@dataclass(frozen=True)
class PromotionStores:
    runs: RunRepository
    artifacts: ArtifactStore
    promotions: PromotionRepository
    releases: FileReleaseStore


@dataclass(frozen=True)
class _PromotionEvidence:
    snapshot: ReleaseSnapshot
    report_hash: str
    decision_hash: str
    model_hash: str
    schema_hash: str


def _report_manifest_and_data(
    run: RunRecord,
    artifacts: ArtifactStore,
) -> tuple[Mapping[str, object], Mapping[str, object], tuple[ArtifactRef, ...]]:
    evidence = load_sealed_report_evidence(run, artifacts)
    expected_data = build_report_data(evidence)
    report_dir = artifacts.run_dir(run.run_id) / "reports"
    manifest_path = report_dir / "pre_promote_report_manifest.json"
    data_path = report_dir / "pre_promote_report_data.json"
    html_path = report_dir / "pre_promote_report.html"
    try:
        manifest = _read_json(manifest_path)
        data_raw = _read_regular(data_path)
        html_raw = _read_regular(html_path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            "immutable Pre Promote report is unavailable or invalid",
            stage="PROMOTING",
            run_id=run.run_id,
        ) from exc
    if set(manifest) != _REQUIRED_REPORT_MANIFEST_FIELDS:
        raise ArtifactIntegrityError(
            "Pre Promote report manifest has an invalid schema",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    if (
        manifest["schema_version"] != _JSON_SCHEMA_VERSION
        or manifest["report_type"] != "pre-promote"
        or manifest["run_id"] != run.run_id
        or manifest["data_path"] != data_path.name
        or manifest["html_path"] != html_path.name
    ):
        raise ArtifactIntegrityError(
            "Pre Promote report manifest does not match the requested run",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    if (
        _hash_bytes(data_raw) != manifest["data_hash"]
        or _hash_bytes(html_raw) != manifest["html_hash"]
        or data_raw != _json_bytes(expected_data)
    ):
        raise ArtifactIntegrityError(
            "Pre Promote report no longer matches sealed run evidence",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    expected_decision_hash = _mapping(
        evidence["component_hashes"],
        "report evidence component hashes",
        run_id=run.run_id,
    ).get("decision")
    if (
        manifest["decision_hash"] != expected_decision_hash
        or manifest["policy_hash"] != run.policy_hash
    ):
        raise ArtifactIntegrityError(
            "Pre Promote report manifest has stale decision or policy evidence",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    return (
        manifest,
        expected_data,
        (
            _artifact_from_file("pre-promote-report-data", data_path),
            _artifact_from_file("pre-promote-report-html", html_path),
            _artifact_from_file("pre-promote-report-manifest", manifest_path),
        ),
    )


def _promotion_evidence(run: RunRecord, artifacts: ArtifactStore) -> _PromotionEvidence:
    manifest, report_data, report_artifacts = _report_manifest_and_data(run, artifacts)
    decision = _mapping(report_data["decision"], "report decision", run_id=run.run_id)
    if (
        run.state
        not in {
            RunState.READY_FOR_PROMOTION.value,
            RunState.PROMOTION_FAILED.value,
        }
        or decision.get("status") != RunState.READY_FOR_PROMOTION.value
        or decision.get("promotion_eligible") is not True
        or decision.get("acceptance_level") != "full"
        or decision.get("blockers") != []
    ):
        raise PromotionConflictError(
            "run is not eligible for promotion",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    checks = report_data["gate_checks"]
    if not isinstance(checks, list) or any(
        not isinstance(check, Mapping) or check.get("passed") is not True
        for check in checks
    ):
        raise PromotionConflictError(
            "one or more hard gate checks are not passing",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    expected_model_hash = _string(
        manifest["model_hash"],
        "report model hash",
        run_id=run.run_id,
    )
    reported_model_stage = None
    report_artifact_rows = report_data.get("artifacts")
    if isinstance(report_artifact_rows, list):
        for artifact in report_artifact_rows:
            if not isinstance(artifact, Mapping):
                continue
            if (
                artifact.get("artifact_type") in _MODEL_TYPES
                and artifact.get("content_hash") == expected_model_hash
                and isinstance(artifact.get("stage"), str)
            ):
                reported_model_stage = artifact["stage"]
                break
    candidate_stages = []
    for stage in (reported_model_stage, _TRAINED_RETRY_STAGE, _TRAINED_STAGE):
        if isinstance(stage, str) and stage and stage not in candidate_stages:
            candidate_stages.append(stage)
    trained = None
    trained_refs: tuple[ArtifactRef, ...] = ()
    model_refs: list[ArtifactRef] = []
    schema_refs: list[ArtifactRef] = []
    for stage in candidate_stages:
        try:
            candidate_trained = artifacts.load_completed_stage(run.run_id, stage, run.input_hash)
        except FileNotFoundError:
            continue
        candidate_refs = tuple(
            _rebase(artifacts, artifact, run_id=run.run_id)
            for artifact in candidate_trained.artifacts
        )
        candidate_model_refs = [
            artifact for artifact in candidate_refs if artifact.artifact_type in _MODEL_TYPES
        ]
        candidate_schema_refs = [
            artifact for artifact in candidate_refs if artifact.artifact_type in _SCHEMA_TYPES
        ]
        if (
            len(candidate_model_refs) == 1
            and len(candidate_schema_refs) == 1
            and candidate_model_refs[0].content_hash == expected_model_hash
        ):
            trained = candidate_trained
            trained_refs = candidate_refs
            model_refs = candidate_model_refs
            schema_refs = candidate_schema_refs
            break
    if trained is None:
        raise ArtifactIntegrityError(
            "no completed training stage matches the Pre Promote report model hash",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    reporting = artifacts.load_completed_stage(run.run_id, _REPORTING_STAGE, run.input_hash)
    report_evidence_refs = tuple(
        _rebase(artifacts, artifact, run_id=run.run_id)
        for artifact in reporting.artifacts
    )
    if len(model_refs) != 1 or len(schema_refs) != 1:
        raise ArtifactIntegrityError(
            "trained stage does not expose exactly one model and schema artifact",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    if expected_model_hash != model_refs[0].content_hash:
        raise ArtifactIntegrityError(
            "Pre Promote report model hash does not match the trained stage",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    all_refs = (*trained_refs, *report_evidence_refs, *report_artifacts)
    types = [artifact.artifact_type for artifact in all_refs]
    if len(types) != len(set(types)):
        raise ArtifactIntegrityError(
            "release input contains duplicate artifact types",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    snapshot = ReleaseSnapshot(
        run_id=run.run_id,
        artifacts=all_refs,
        manifest_hash=canonical_json_hash([to_plain_data(artifact) for artifact in all_refs]),
        previous_model=None,
    )
    return _PromotionEvidence(
        snapshot=snapshot,
        report_hash=_string(manifest["html_hash"], "report html hash", run_id=run.run_id),
        decision_hash=_string(manifest["decision_hash"], "report decision hash", run_id=run.run_id),
        model_hash=model_refs[0].content_hash,
        schema_hash=schema_refs[0].content_hash,
    )


def _validate_approval(approval: PromotionApproval, *, run_id: str) -> None:
    if approval.run_id != run_id:
        raise PromotionConflictError(
            "approval belongs to a different run",
            stage="PROMOTING",
            run_id=run_id,
        )
    for label, value in (
        ("approved_by", approval.approved_by),
        ("approved_at", approval.approved_at),
        ("report_hash", approval.report_hash),
        ("decision_hash", approval.decision_hash),
        ("idempotency_key", approval.idempotency_key),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"approval.{label} must be a non-empty string")


def assert_ready_and_unchanged(
    run: RunRecord,
    approval: PromotionApproval,
    artifacts: ArtifactStore,
    releases: FileReleaseStore,
) -> _PromotionEvidence:
    """Revalidate every approved promotion input against sealed evidence."""

    _validate_approval(approval, run_id=run.run_id)
    evidence = _promotion_evidence(run, artifacts)
    if approval.report_hash != evidence.report_hash:
        raise ArtifactIntegrityError(
            "approval report hash does not match immutable report",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    if approval.decision_hash != evidence.decision_hash:
        raise ArtifactIntegrityError(
            "approval decision hash does not match immutable evidence",
            stage="PROMOTING",
            run_id=run.run_id,
        )
    current = releases.read_current_model()
    if current != approval.expected_current_model:
        raise PromotionConflictError(
            "current model no longer matches the approval",
            stage="PROMOTING",
            run_id=run.run_id,
            details={"current": current, "expected": approval.expected_current_model},
        )
    return evidence


def _promotion_failure(stores: PromotionStores, run_id: str, exc: BaseException) -> None:
    error = HeuriBoostError(
        RunState.PROMOTION_FAILED.value,
        "promotion failed before current-model activation completed",
        stage="PROMOTING",
        run_id=run_id,
        retryable=True,
        details={"exception_type": type(exc).__name__, "message": str(exc)[:500]},
        operator_action="Inspect the immutable release and retry the same promotion only after resolving the target failure.",
    )
    stores.runs.fail(run_id, RunState.PROMOTION_FAILED, error)


def promote_repair(
    run_id: str,
    approval: PromotionApproval,
    target: Any,
    stores: PromotionStores,
) -> PromotionReceipt:
    """Promote exactly one ready run under a workspace-wide lock."""

    with stores.releases.workspace_lock():
        _validate_approval(approval, run_id=run_id)
        existing = stores.promotions.find_by_idempotency_key(approval.idempotency_key)
        if existing is not None:
            if existing.run_id != run_id:
                raise PromotionConflictError(
                    "idempotency key belongs to a different promotion run",
                    stage="PROMOTING",
                    run_id=run_id,
                )
            durable = stores.releases.read_promotion_receipt(run_id)
            if durable is None or durable[0] != existing:
                raise ArtifactIntegrityError(
                    "idempotency record does not match the immutable release receipt",
                    stage="PROMOTING",
                    run_id=run_id,
                )
            persisted = stores.runs.get(run_id)
            if persisted.state == RunState.PROMOTING.value:
                if stores.releases.read_current_model() != existing.current_model:
                    raise PromotionConflictError(
                        "idempotency record points to a release that is not current",
                        stage="PROMOTING",
                        run_id=run_id,
                    )
                stores.runs.transition(
                    run_id,
                    RunState.PROMOTED,
                    metadata={
                        "promotion": {
                            "release_manifest_hash": existing.release_manifest_hash,
                            "current_model": existing.current_model,
                        }
                    },
                )
            return existing
        run = stores.runs.get(run_id)
        if run.state == RunState.PROMOTING.value:
            durable = stores.releases.read_promotion_receipt(run_id)
            if durable is None:
                raise PromotionConflictError(
                    "promotion is still in progress and has no durable receipt",
                    stage="PROMOTING",
                    run_id=run_id,
                )
            receipt, durable_key = durable
            if durable_key != approval.idempotency_key:
                raise PromotionConflictError(
                    "durable release belongs to a different idempotency key",
                    stage="PROMOTING",
                    run_id=run_id,
                )
            if stores.releases.read_current_model() != receipt.current_model:
                raise PromotionConflictError(
                    "durable release has not activated the current model",
                    stage="PROMOTING",
                    run_id=run_id,
                )
            stores.promotions.save(receipt, approval.idempotency_key)
            stores.runs.transition(
                run_id,
                RunState.PROMOTED,
                metadata={
                    "promotion": {
                        "release_manifest_hash": receipt.release_manifest_hash,
                        "current_model": receipt.current_model,
                    }
                },
            )
            return receipt
        evidence = assert_ready_and_unchanged(run, approval, stores.artifacts, stores.releases)
        if run.state not in {
            RunState.READY_FOR_PROMOTION.value,
            RunState.PROMOTION_FAILED.value,
        }:
            raise PromotionConflictError(
                "run is not available for promotion",
                stage="PROMOTING",
                run_id=run_id,
            )
        promoting = stores.runs.transition(run_id, RunState.PROMOTING)
        snapshot = ReleaseSnapshot(
            run_id=evidence.snapshot.run_id,
            artifacts=evidence.snapshot.artifacts,
            manifest_hash=evidence.snapshot.manifest_hash,
            previous_model=approval.expected_current_model,
        )
        try:
            receipt = stores.releases.promote_locked(
                promoting,
                approval,
                target,
                snapshot,
                report_hash=evidence.report_hash,
                decision_hash=evidence.decision_hash,
                model_hash=evidence.model_hash,
                schema_hash=evidence.schema_hash,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            _promotion_failure(stores, run_id, exc)
            raise
        stores.promotions.save(receipt, approval.idempotency_key)
        stores.runs.transition(
            run_id,
            RunState.PROMOTED,
            metadata={
                "promotion": {
                    "release_manifest_hash": receipt.release_manifest_hash,
                    "current_model": receipt.current_model,
                }
            },
        )
        return receipt


def rollback_release(
    receipt: PromotionReceipt,
    target: Any,
    stores: PromotionStores,
    *,
    approved_by: str,
) -> RollbackReceipt:
    if not isinstance(approved_by, str) or not approved_by:
        raise ValueError("approved_by must be a non-empty string")
    with stores.releases.workspace_lock():
        return stores.releases.rollback_locked(
            receipt,
            target,
            approved_by=approved_by,
        )
