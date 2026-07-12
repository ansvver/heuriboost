"""Create durable Reckless runs from immutable Web datasets."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from ...reckless.contracts import RepairRequest, RunRecord, to_plain_data
from ...reckless.hashing import build_run_fingerprint
from ..jobs.executor import LocalJobExecutor
from ..runtime import WebRuntime
from ..stores.sqlite import SQLiteStore


class RunService:
    def __init__(self, store: SQLiteStore, executor: LocalJobExecutor, runtime: WebRuntime) -> None:
        self.store = store
        self.executor = executor
        self.runtime = runtime

    def create_run(
        self,
        base_dataset_id: str,
        production_cases_id: str,
        *,
        requested_by: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        existing = self.store.find_idempotency_result("run.create", idempotency_key)
        if existing is not None:
            return existing
        base = self._dataset(base_dataset_id, "base")
        cases = self._dataset(production_cases_id, "production_cases")
        request = RepairRequest(
            workspace_id=self.runtime.config.workspace_id,
            base_dataset_id=base_dataset_id,
            production_cases_id=production_cases_id,
            policy_version=str(self.runtime.policy.version),
            backend_name=self.runtime.backend.name,
            requested_by=requested_by,
        )
        input_hash = build_run_fingerprint(
            request,
            self.runtime.policy,
            self.store.datasets.get(base_dataset_id),
            self.store.datasets.get(production_cases_id),
            self.runtime.backend.execution_identity(),
        )
        record = self.store.runs.create(request, self.runtime.policy.content_hash, input_hash)
        job_id = self.executor.enqueue(record.run_id)
        self.store.append_audit_event("run.created", {"run_id": record.run_id, "job_id": job_id})
        result = self._data(record, job_id=job_id, job_status="QUEUED")
        self.store.save_idempotency_result("run.create", idempotency_key, result)
        return result

    def get_run(self, run_id: str) -> dict[str, object]:
        record = self.store.runs.get(run_id)
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT id, status FROM jobs WHERE run_id = ? ORDER BY created_at DESC, id DESC LIMIT 1", (run_id,)
            ).fetchone()
        return self._data(record, job_id=None if row is None else row["id"], job_status=None if row is None else row["status"])

    def cancel(self, run_id: str) -> dict[str, object]:
        job_id = self._latest_job_id(run_id)
        self.executor.request_cancel(job_id)
        self.store.append_audit_event("run.cancel_requested", {"run_id": run_id, "job_id": job_id})
        return self.get_run(run_id)

    def retry(self, run_id: str) -> dict[str, object]:
        job_id = self._latest_job_id(run_id)
        retry_id = self.executor.retry(job_id)
        self.store.append_audit_event("run.retry_queued", {"run_id": run_id, "job_id": retry_id, "parent_job_id": job_id})
        return self.get_run(run_id)

    def _latest_job_id(self, run_id: str) -> str:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT id FROM jobs WHERE run_id = ? ORDER BY created_at DESC, id DESC LIMIT 1", (run_id,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"run has no job: {run_id}")
        return str(row["id"])

    def _dataset(self, dataset_id: str, expected_role: str):
        with self.store.connection() as connection:
            row = connection.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"unknown dataset: {dataset_id}")
        if row["role"] != expected_role or row["status"] != "READY":
            raise ValueError(f"dataset {dataset_id} is not a READY {expected_role} dataset")
        return row

    def _data(self, record: RunRecord, *, job_id: str | None, job_status: str | None) -> dict[str, object]:
        metadata = to_plain_data(record.metadata)
        artifact_root = getattr(getattr(self.runtime, "artifacts", None), "root", None)
        return {
            "id": record.run_id,
            "state": record.state,
            "version": record.version,
            "input_hash": record.input_hash,
            "policy_hash": record.policy_hash,
            "request": to_plain_data(record.request),
            "metadata": metadata,
            "stage_sections": _stage_sections(record.run_id, record.state, metadata, artifact_root=artifact_root),
            "job_id": job_id,
            "job_status": job_status,
            "error": None if record.error is None else to_plain_data(record.error),
        }


def _stage_sections(run_id: str, state: str, metadata: object, *, artifact_root: object = None) -> list[dict[str, object]]:
    if not isinstance(metadata, Mapping):
        metadata = {}
    report_evidence = _mapping(metadata.get("report_evidence"))
    promotion = _mapping(metadata.get("promotion"))
    llm_feature_names = _synthesized_feature_names(run_id, metadata, artifact_root)
    definitions = (
        {
            "id": "stage-validation",
            "title": "校验",
            "metadata_key": "validation",
            "active_states": {"VALIDATING"},
            "passed_states": {
                "COMPILED",
                "TRAINING",
                "TRAINED",
                "EVALUATING",
                "SYNTHESIZING_FEATURES",
                "REPORTING",
                "READY_FOR_PROMOTION",
                "PROMOTING",
                "PROMOTED",
            },
            "description": "输入数据、字段映射、基础规则和 backend 预检结果。",
        },
        {
            "id": "stage-compile",
            "title": "编译",
            "metadata_key": "compiled_stage",
            "active_states": {"COMPILED"},
            "passed_states": {
                "TRAINING",
                "TRAINED",
                "EVALUATING",
                "SYNTHESIZING_FEATURES",
                "REPORTING",
                "READY_FOR_PROMOTION",
                "PROMOTING",
                "PROMOTED",
            },
            "description": "将 base dataset 与 production cases 固化成本次训练输入和 stage artifact。",
        },
        {
            "id": "stage-training",
            "title": "训练",
            "metadata_key": "trained_stage",
            "active_states": {"TRAINING"},
            "passed_states": {
                "TRAINED",
                "EVALUATING",
                "SYNTHESIZING_FEATURES",
                "REPORTING",
                "READY_FOR_PROMOTION",
                "PROMOTING",
                "PROMOTED",
            },
            "description": "第一轮模型训练产物、训练 metadata 和模型 artifact。",
        },
        {
            "id": "stage-evaluation",
            "title": "评测",
            "metadata_key": None,
            "active_states": {"EVALUATING"},
            "passed_states": {
                "SYNTHESIZING_FEATURES",
                "REPORTING",
                "READY_FOR_PROMOTION",
                "PROMOTING",
                "PROMOTED",
            },
            "description": "门禁评测阶段；详细指标会进入 Pre Promote 报告或阻断错误详情。",
        },
        {
            "id": "stage-feature-synthesis",
            "title": "特征合成",
            "metadata_key": "feature_synthesis_stage",
            "active_states": {"SYNTHESIZING_FEATURES"},
            "passed_states": {"TRAINING", "TRAINED", "EVALUATING", "REPORTING", "READY_FOR_PROMOTION", "PROMOTING", "PROMOTED"},
            "description": "当有限训练轮次未过 gate 时，生成可审计的新特征候选并固化为 retry 输入。",
            "extra_data": {"llm_feature_names": llm_feature_names} if llm_feature_names else None,
        },
        {
            "id": "stage-training-retry",
            "title": "重训",
            "metadata_key": "trained_retry_stage",
            "active_states": {"TRAINING"},
            "passed_states": {"TRAINED", "EVALUATING", "REPORTING", "READY_FOR_PROMOTION", "PROMOTING", "PROMOTED"},
            "description": "使用合成特征后的 retry 训练产物；没有触发 feature synthesis 时这里会保持未开始。",
            "extra_data": {"llm_feature_names": llm_feature_names} if llm_feature_names else None,
        },
        {
            "id": "stage-report",
            "title": "报告",
            "metadata_key": "report_evidence",
            "active_states": {"REPORTING"},
            "passed_states": {"READY_FOR_PROMOTION", "PROMOTING", "PROMOTED"},
            "description": "Pre Promote 报告证据、报告 hash 和可打开的报告页面。",
            "link": {"href": f"/runs/{run_id}/report", "label": "打开 Pre Promote 报告"} if report_evidence else None,
        },
        {
            "id": "stage-approval",
            "title": "批准",
            "metadata_key": None,
            "active_states": {"READY_FOR_PROMOTION"},
            "passed_states": {"PROMOTING", "PROMOTED"},
            "description": "报告通过后等待人工批准；已 promote 的 run 表示批准流程已完成。",
            "data": _approval_data(state, bool(report_evidence), bool(promotion)),
        },
        {
            "id": "stage-promote",
            "title": "Promote",
            "metadata_key": "promotion",
            "active_states": {"PROMOTING"},
            "passed_states": {"PROMOTED"},
            "description": "不可变 release 指针切换结果和 release manifest hash。",
        },
    )
    return [_stage_section(definition, state, metadata) for definition in definitions]


def _stage_section(definition: Mapping[str, object], state: str, metadata: Mapping[str, object]) -> dict[str, object]:
    metadata_key = definition.get("metadata_key")
    data = _mapping(metadata.get(metadata_key)) if isinstance(metadata_key, str) else _mapping(definition.get("data"))
    extra_data = _mapping(definition.get("extra_data"))
    if extra_data:
        data = {**data, **extra_data}
    artifacts = _artifacts(data)
    summary_items = _summary_items(data, artifacts)
    return {
        "id": definition["id"],
        "title": definition["title"],
        "status": _stage_status(definition, state, bool(data)),
        "description": definition["description"],
        "summary_items": summary_items,
        "artifacts": artifacts,
        "data_json": "" if not data else json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        "link": definition.get("link"),
    }


def _stage_status(definition: Mapping[str, object], state: str, has_data: bool) -> str:
    if definition["id"] == "stage-approval":
        if state == "READY_FOR_PROMOTION":
            return "待批准"
        if state in {"PROMOTING", "PROMOTED"}:
            return "已完成"
    if has_data:
        return "已完成"
    active_states = definition.get("active_states")
    if isinstance(active_states, set) and state in active_states:
        if definition["id"] == "stage-approval":
            return "待批准"
        return "进行中"
    passed_states = definition.get("passed_states")
    if isinstance(passed_states, set) and state in passed_states:
        return "已通过"
    if state.startswith("BLOCKED"):
        return "已阻断"
    return "未开始"


def _approval_data(state: str, report_ready: bool, promoted: bool) -> dict[str, object]:
    if not report_ready and not promoted:
        return {}
    return {
        "report_ready": report_ready,
        "approval_state": "approved" if promoted or state in {"PROMOTING", "PROMOTED"} else "waiting_for_approval",
    }


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _artifacts(data: Mapping[str, object]) -> list[dict[str, object]]:
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    return [artifact for artifact in artifacts if isinstance(artifact, dict)]


def _summary_items(data: Mapping[str, object], artifacts: list[dict[str, object]]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if artifacts:
        items.append({"label": "artifact 数量", "value": str(len(artifacts))})
    for key, label in (
        ("stage", "阶段"),
        ("valid", "校验结果"),
        ("artifact_type", "artifact 类型"),
        ("artifact_set_hash", "artifact set hash"),
        ("content_hash", "content hash"),
        ("release_manifest_hash", "release manifest hash"),
        ("current_model", "当前模型"),
        ("llm_feature_names", "LLM 特征"),
        ("features_added", "新增特征"),
        ("path", "路径"),
        ("size_bytes", "大小"),
    ):
        if key in data:
            items.append({"label": label, "value": _format_summary_value(data[key])})
    nested_metadata = data.get("metadata")
    if isinstance(nested_metadata, Mapping):
        for key, label in (("feature_set_name", "特征集"), ("errors", "错误数"), ("warnings", "警告数")):
            if key in nested_metadata:
                items.append({"label": label, "value": _format_summary_value(nested_metadata[key])})
    return items


def _format_summary_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return "0"
        if all(isinstance(item, (str, int, float, bool)) for item in value) and len(value) <= 5:
            return ", ".join(str(item) for item in value)
        return f"{len(value)} 项"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


_MAX_FEATURE_ARTIFACT_BYTES = 25 * 1024 * 1024


def _synthesized_feature_names(run_id: str, metadata: Mapping[str, object], artifact_root: object) -> list[str]:
    if not isinstance(artifact_root, Path):
        return []
    names: set[str] = set()
    for stage_key in ("feature_synthesis_stage", "trained_retry_stage"):
        stage_data = _mapping(metadata.get(stage_key))
        for artifact in _artifacts(stage_data):
            if artifact.get("artifact_type") not in {"feature-synthesis-specs", "xgboost-model-metadata", "training-metadata"}:
                continue
            path = _safe_artifact_path(artifact_root, run_id, artifact.get("path"))
            if path is None:
                continue
            size_bytes = artifact.get("size_bytes")
            if isinstance(size_bytes, int) and size_bytes > _MAX_FEATURE_ARTIFACT_BYTES:
                continue
            try:
                if path.stat().st_size > _MAX_FEATURE_ARTIFACT_BYTES:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            names.update(_extract_llm_feature_names(payload))
    return sorted(names)


def _safe_artifact_path(root: Path, run_id: str, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    relative = Path(value)
    if relative.is_absolute() or not relative.parts:
        return None
    if any(part in {"", ".", ".."} for part in relative.parts):
        return None
    if len(relative.parts) < 3 or relative.parts[:2] != ("runs", run_id):
        return None
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def _extract_llm_feature_names(payload: object) -> set[str]:
    if not isinstance(payload, Mapping):
        return set()
    names: set[str] = set()
    for key in ("llm_feature_names", "feature_names"):
        values = payload.get(key)
        if isinstance(values, list):
            names.update(value for value in values if _is_llm_feature_name(value))
    for key in ("features", "llm_feature_specs", "feature_specs", "specs"):
        values = payload.get(key)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, Mapping):
                    name = value.get("name")
                    if _is_llm_feature_name(name):
                        names.add(name)
    return names


def _is_llm_feature_name(value: object) -> bool:
    return isinstance(value, str) and value.startswith("llm_")
