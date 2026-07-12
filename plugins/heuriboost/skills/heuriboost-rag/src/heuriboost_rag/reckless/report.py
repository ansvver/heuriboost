"""Immutable bilingual Pre Promote report rendering."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from importlib import resources
import json
import math
import os
from pathlib import Path
import stat
import uuid

from .contracts import ArtifactRef, ReportArtifact, RunRecord, to_plain_data
from .errors import ArtifactIntegrityError
from .storage import ArtifactStore


_REPORT_SCHEMA_VERSION = 1
_REPORTING_STAGE = "REPORTING"
_REPORT_EVIDENCE_ARTIFACT = "report-evidence"
_DATA_FILENAME = "pre_promote_report_data.json"
_HTML_FILENAME = "pre_promote_report.html"
_MANIFEST_FILENAME = "pre_promote_report_manifest.json"
_SUPPORTED_LOCALES = frozenset({"zh-CN", "en"})
_REQUIRED_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "run",
        "decision",
        "data_lineage",
        "validation",
        "compilation",
        "training",
        "evaluation",
        "features",
        "explainability",
        "operator_summary",
        "data_overview",
        "process_timeline",
        "metric_summary",
        "deposits",
        "gate_checks",
        "warnings",
        "artifacts",
        "reproducibility",
    }
)
_REQUIRED_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "run",
        "request",
        "policy",
        "input",
        "outcome",
        "datasets",
        "execution_identity",
        "validation",
        "compilation",
        "training",
        "evaluation",
        "decision",
        "warnings",
        "artifacts",
        "completed_stage_manifests",
        "component_hashes",
    }
)
_MODEL_ARTIFACT_TYPES = frozenset({"model", "xgboost-model", "candidate-model"})


def _error(
    message: str,
    *,
    run_id: str | None = None,
    details: Mapping[str, object] | None = None,
) -> ArtifactIntegrityError:
    return ArtifactIntegrityError(
        message,
        stage=_REPORTING_STAGE,
        run_id=run_id,
        details=details,
        operator_action="Inspect the sealed report evidence and render into a new run directory.",
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _plain_mapping(
    value: object,
    *,
    label: str,
    run_id: str | None = None,
) -> dict[str, object]:
    try:
        plain = to_plain_data(value)
        json.dumps(plain, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise _error(f"{label} contains unsupported data", run_id=run_id) from exc
    if not isinstance(plain, Mapping):
        raise _error(f"{label} must be a JSON object", run_id=run_id)
    return dict(plain)


def _mapping(
    value: object,
    *,
    label: str,
    run_id: str | None = None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be a JSON object", run_id=run_id)
    return value


def _string(
    value: object,
    *,
    label: str,
    run_id: str | None = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise _error(f"{label} must be a non-empty string", run_id=run_id)
    return value


def _json_text(value: object, *, run_id: str | None = None) -> str:
    try:
        text = json.dumps(
            to_plain_data(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise _error("report data cannot be encoded as strict JSON", run_id=run_id) from exc
    return text.replace("<", "\\u003c")


def _json_bytes(value: object, *, run_id: str | None = None) -> bytes:
    return _json_text(value, run_id=run_id).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_section(
    report_data: Mapping[str, object],
    key: str,
    *,
    run_id: str,
) -> Mapping[str, object]:
    if key not in report_data:
        raise _error(
            "report data is missing a mandatory section",
            run_id=run_id,
            details={"key": key},
        )
    return _mapping(report_data[key], label=f"report_data.{key}", run_id=run_id)


def _validate_report_data(value: object) -> dict[str, object]:
    report_data = _plain_mapping(value, label="report_data")
    raw_run = report_data.get("run")
    run = raw_run if isinstance(raw_run, Mapping) else {}
    run_id = run.get("run_id") if isinstance(run.get("run_id"), str) else None
    missing = sorted(_REQUIRED_REPORT_KEYS - set(report_data))
    if missing:
        raise _error(
            "report data is missing mandatory fields",
            run_id=run_id,
            details={"missing": missing},
        )
    if report_data["schema_version"] != _REPORT_SCHEMA_VERSION:
        raise _error("unsupported report data schema version", run_id=run_id)

    run_section = _required_section(report_data, "run", run_id=run_id or "unknown")
    resolved_run_id = _string(run_section.get("run_id"), label="run.run_id")
    _string(run_section.get("state"), label="run.state", run_id=resolved_run_id)

    decision = _required_section(report_data, "decision", run_id=resolved_run_id)
    _string(decision.get("status"), label="decision.status", run_id=resolved_run_id)
    if type(decision.get("promotion_eligible")) is not bool:
        raise _error("decision.promotion_eligible must be a boolean", run_id=resolved_run_id)
    for key in (
        "data_lineage",
        "validation",
        "compilation",
        "training",
        "evaluation",
        "features",
        "explainability",
        "operator_summary",
        "data_overview",
        "process_timeline",
        "metric_summary",
    ):
        _required_section(report_data, key, run_id=resolved_run_id)
    for key in ("gate_checks", "warnings", "artifacts", "deposits"):
        if not isinstance(report_data[key], list):
            raise _error(f"report_data.{key} must be a list", run_id=resolved_run_id)
    reproducibility = _required_section(
        report_data,
        "reproducibility",
        run_id=resolved_run_id,
    )
    _string(
        reproducibility.get("policy_hash"),
        label="reproducibility.policy_hash",
        run_id=resolved_run_id,
    )
    _string(
        reproducibility.get("code_revision"),
        label="reproducibility.code_revision",
        run_id=resolved_run_id,
    )
    return json.loads(
        _json_text(report_data, run_id=resolved_run_id),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
        parse_float=_strict_float,
    )


def build_report_data(report_evidence: Mapping[str, object]) -> dict[str, object]:
    """Normalize an immutable Core 6 evidence document for report rendering."""

    evidence = _plain_mapping(report_evidence, label="report_evidence")
    missing = sorted(_REQUIRED_EVIDENCE_KEYS - set(evidence))
    if missing:
        raise _error(
            "sealed report evidence is missing mandatory fields",
            details={"missing": missing},
        )
    run = _mapping(evidence["run"], label="report_evidence.run")
    run_id = _string(run.get("run_id"), label="report_evidence.run.run_id")
    request = _mapping(evidence["request"], label="report_evidence.request", run_id=run_id)
    policy = _mapping(evidence["policy"], label="report_evidence.policy", run_id=run_id)
    inputs = _mapping(evidence["input"], label="report_evidence.input", run_id=run_id)
    outcome = _mapping(evidence["outcome"], label="report_evidence.outcome", run_id=run_id)
    identity = _mapping(
        evidence["execution_identity"],
        label="report_evidence.execution_identity",
        run_id=run_id,
    )
    decision = _mapping(evidence["decision"], label="report_evidence.decision", run_id=run_id)
    hashes = _mapping(
        evidence["component_hashes"],
        label="report_evidence.component_hashes",
        run_id=run_id,
    )
    policy_hash = _string(policy.get("content_hash"), label="policy.content_hash", run_id=run_id)
    decision_hash = _string(
        hashes.get("decision"),
        label="component_hashes.decision",
        run_id=run_id,
    )
    state = _string(outcome.get("state"), label="outcome.state", run_id=run_id)
    promotion_eligible = outcome.get("promotion_eligible")
    if type(promotion_eligible) is not bool:
        raise _error("outcome.promotion_eligible must be a boolean", run_id=run_id)
    datasets = _mapping(evidence["datasets"], label="report_evidence.datasets", run_id=run_id)
    validation_evidence = _mapping(
        evidence["validation"],
        label="report_evidence.validation",
        run_id=run_id,
    )
    compilation = _mapping(evidence["compilation"], label="report_evidence.compilation", run_id=run_id)
    training = _mapping(evidence["training"], label="report_evidence.training", run_id=run_id)
    evaluation = _mapping(evidence["evaluation"], label="report_evidence.evaluation", run_id=run_id)
    feature_synthesis = evidence.get("feature_synthesis")
    feature_summary = _feature_summary(
        identity=identity,
        compilation=compilation,
        training=training,
        feature_synthesis=feature_synthesis,
    )
    explainability = _explainability_summary(
        identity=identity,
        training=training,
        features=feature_summary,
    )
    decision_section = {
        **dict(decision),
        "status": state,
        "promotion_eligible": promotion_eligible,
    }
    data_lineage = {
        "base_dataset_id": inputs.get("base_dataset_id"),
        "production_cases_id": inputs.get("production_cases_id"),
        "datasets": datasets,
    }
    validation_section = {
        "passed": validation_evidence.get("valid"),
        "metadata": validation_evidence.get("metadata"),
        "warnings": validation_evidence.get("warnings"),
    }
    artifacts = evidence["artifacts"]
    gate_checks = decision.get("checks")
    reproducibility = {
        "policy_hash": policy_hash,
        "policy_version": policy.get("version"),
        "input_hash": inputs.get("input_hash"),
        "backend_version": identity.get("backend_version"),
        "feature_names": identity.get("feature_names"),
        "feature_version": identity.get("feature_version"),
        "code_revision": identity.get("code_commit"),
        "decision_hash": decision_hash,
        "component_hashes": hashes,
    }
    operator_summary = _operator_summary(
        run_id=run_id,
        state=state,
        promotion_eligible=promotion_eligible,
        decision=decision,
        training=training,
        evaluation=evaluation,
        reproducibility=reproducibility,
        artifacts=artifacts,
    )
    data_overview = _data_overview(
        inputs=inputs,
        datasets=datasets,
        compilation=compilation,
        training=training,
    )
    metric_summary = _metric_summary(
        evaluation=evaluation,
        decision=decision,
        training=training,
        compilation=compilation,
        features=feature_summary,
        feature_synthesis=feature_synthesis,
    )
    process_timeline = _process_timeline(
        inputs=inputs,
        validation=validation_evidence,
        compilation=compilation,
        training=training,
        evaluation=evaluation,
        decision=decision,
        features=feature_summary,
        feature_synthesis=feature_synthesis,
        completed_stage_manifests=evidence.get("completed_stage_manifests"),
    )
    deposits = _deposits(
        inputs=inputs,
        compilation=compilation,
        training=training,
        features=feature_summary,
        feature_synthesis=feature_synthesis,
        artifacts=artifacts,
        reproducibility=reproducibility,
    )

    return _validate_report_data(
        {
            "schema_version": _REPORT_SCHEMA_VERSION,
            "run": {
                "run_id": run_id,
                "state": state,
                "workspace_id": request.get("workspace_id"),
                "backend_name": request.get("backend_name"),
                "requested_by": request.get("requested_by"),
            },
            "decision": decision_section,
            "operator_summary": operator_summary,
            "data_lineage": data_lineage,
            "data_overview": data_overview,
            "process_timeline": process_timeline,
            "validation": validation_section,
            "compilation": compilation,
            "training": training,
            "evaluation": evaluation,
            "features": feature_summary,
            "explainability": explainability,
            "metric_summary": metric_summary,
            "gate_checks": gate_checks,
            "warnings": evidence["warnings"],
            "artifacts": artifacts,
            "deposits": deposits,
            "reproducibility": reproducibility,
        }
    )


def _feature_summary(
    *,
    identity: Mapping[str, object],
    compilation: Mapping[str, object],
    training: Mapping[str, object],
    feature_synthesis: object,
) -> dict[str, object]:
    training_metadata = _optional_mapping(training.get("metadata"))
    compilation_metadata = _optional_mapping(compilation.get("metadata"))
    identity_features = _string_list(identity.get("feature_names"))
    trained_features = _string_list(training_metadata.get("feature_names"))
    llm_features = _unique_strings(
        _string_list(training_metadata.get("llm_feature_names")),
        _string_list(compilation_metadata.get("llm_feature_names")),
        [name for name in trained_features if name.startswith("llm_")],
        [name for name in identity_features if name.startswith("llm_")],
    )
    all_features = _unique_strings(
        trained_features,
        identity_features,
        llm_features,
    )
    base_features = [name for name in all_features if name not in set(llm_features)]
    synthesized_features = _synthesized_feature_summaries(
        compilation_metadata,
        training_metadata,
        feature_synthesis,
    )
    feature_details = _feature_details(
        feature_names=all_features,
        llm_feature_names=set(llm_features),
        synthesized_features=synthesized_features,
    )
    return {
        "total_count": len(all_features),
        "base_feature_count": len(base_features),
        "llm_feature_count": len(llm_features),
        "feature_names": all_features,
        "base_feature_names": base_features,
        "llm_feature_names": llm_features,
        "feature_details": feature_details,
        "feature_set_name": training_metadata.get("feature_set_name") or compilation_metadata.get("feature_set_name"),
        "feature_set_version": training_metadata.get("feature_set_version") or identity.get("feature_version"),
        "synthesized_features": synthesized_features,
        "training_params": _optional_mapping(training_metadata.get("params")) or _optional_mapping(identity.get("training_params")),
        "dataset_summary": _optional_mapping(training_metadata.get("dataset_summary")),
    }


def _explainability_summary(
    *,
    identity: Mapping[str, object],
    training: Mapping[str, object],
    features: Mapping[str, object],
) -> dict[str, object]:
    training_metadata = _optional_mapping(training.get("metadata"))
    params = _optional_mapping(features.get("training_params"))
    feature_names = _string_list(features.get("feature_names"))
    llm_feature_names = set(_string_list(features.get("llm_feature_names")))
    synthesized = features.get("synthesized_features")
    return {
        "model": {
            "backend_version": identity.get("backend_version"),
            "feature_version": identity.get("feature_version"),
            "objective": params.get("objective"),
            "eval_metric": params.get("eval_metric"),
            "rounds": training_metadata.get("rounds"),
            "seed": params.get("seed") or identity.get("random_seed"),
        },
        "feature_groups": _feature_groups(feature_names, llm_feature_names),
        "synthesized_feature_explanations": synthesized if isinstance(synthesized, list) else [],
        "notes": _explainability_notes(bool(llm_feature_names), bool(_importance_items(training_metadata))),
        "feature_importance": _importance_items(training_metadata),
    }


def _optional_mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _unique_strings(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _synthesized_feature_summaries(
    compilation_metadata: Mapping[str, object],
    training_metadata: Mapping[str, object],
    feature_synthesis: object,
) -> list[dict[str, object]]:
    specs = _feature_specs(compilation_metadata, training_metadata, feature_synthesis)
    summaries: list[dict[str, object]] = []
    for spec in specs:
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            continue
        entries = spec.get("entries")
        entry_counts = _entry_counts(entries if isinstance(entries, list) else [])
        summary: dict[str, object] = {
            "name": name,
            "kind": spec.get("kind"),
            "description": spec.get("description"),
            **entry_counts,
        }
        example = _representative_entry(entries if isinstance(entries, list) else [])
        if example:
            summary["representative_entry"] = example
        summaries.append(summary)
    return summaries


def _feature_specs(
    compilation_metadata: Mapping[str, object],
    training_metadata: Mapping[str, object],
    feature_synthesis: object,
) -> list[Mapping[str, object]]:
    for source in (
        compilation_metadata.get("llm_feature_specs"),
        training_metadata.get("llm_feature_specs"),
        _optional_mapping(feature_synthesis).get("features"),
        _optional_mapping(_optional_mapping(feature_synthesis).get("metadata")).get("features"),
    ):
        if isinstance(source, list):
            specs = [item for item in source if isinstance(item, Mapping)]
            if specs:
                return specs
    return []


def _entry_counts(entries: list[object]) -> dict[str, int]:
    positive = negative = neutral = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        score = entry.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            neutral += 1
        elif score > 0:
            positive += 1
        elif score < 0:
            negative += 1
        else:
            neutral += 1
    return {
        "entry_count": positive + negative + neutral,
        "positive_entries": positive,
        "negative_entries": negative,
        "neutral_entries": neutral,
    }


def _representative_entry(entries: list[object]) -> dict[str, object]:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        return {
            key: entry[key]
            for key in ("query_id", "query_text", "doc_id", "doc_text", "score")
            if key in entry and isinstance(entry[key], (str, int, float))
        }
    return {}


def _feature_groups(feature_names: list[str], llm_feature_names: set[str]) -> list[dict[str, object]]:
    grouped: dict[str, list[str]] = {}
    for name in feature_names:
        group = _feature_group(name, llm_feature_names)
        grouped.setdefault(group, []).append(name)
    return [
        {"group": group, "count": len(names), "features": names}
        for group, names in grouped.items()
    ]


def _feature_group(name: str, llm_feature_names: set[str]) -> str:
    if name in llm_feature_names or name.startswith("llm_"):
        return "LLM synthesized"
    if name.startswith("heuristic_"):
        return "Heuristic matching"
    if name.startswith("source_"):
        return "Source flags"
    if name.startswith("strategy_"):
        return "Retrieval strategy"
    if "sale_status" in name:
        return "Sale status"
    if "overlap" in name or "product" in name or "query" in name or "combo" in name or "entity" in name:
        return "Text/entity overlap"
    if "score" in name or "recall" in name:
        return "Ranking scores"
    return "Other"


def _importance_items(training_metadata: Mapping[str, object]) -> list[dict[str, object]]:
    for key in ("feature_importance", "feature_importances", "model_feature_importance"):
        value = training_metadata.get(key)
        if isinstance(value, Mapping):
            return [
                {"feature": feature, "importance": importance}
                for feature, importance in value.items()
                if isinstance(feature, str) and isinstance(importance, (int, float)) and not isinstance(importance, bool)
            ]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _explainability_notes(has_llm_features: bool, has_importance: bool) -> list[str]:
    notes = [
        "Explanations are derived only from sealed report evidence and model metadata.",
        "Feature groups are deterministic name-based groupings for operator review.",
    ]
    if has_llm_features:
        notes.append("LLM synthesized features are materialized as deterministic feature columns before retry training.")
    if not has_importance:
        notes.append("Per-feature importance or SHAP values were not sealed for this run.")
    return notes


def _feature_details(
    *,
    feature_names: list[str],
    llm_feature_names: set[str],
    synthesized_features: list[dict[str, object]],
) -> list[dict[str, object]]:
    synthesized_by_name = {
        feature["name"]: feature
        for feature in synthesized_features
        if isinstance(feature.get("name"), str)
    }
    details: list[dict[str, object]] = []
    for name in feature_names:
        synthesized = synthesized_by_name.get(name, {})
        group = _feature_group(name, llm_feature_names)
        source = "LLM synthesized" if name in llm_feature_names or name.startswith("llm_") else "built-in"
        detail: dict[str, object] = {
            "name": name,
            "group": group,
            "source": source,
            "description": _feature_description(name, group=group, synthesized=synthesized),
        }
        for key in (
            "kind",
            "entry_count",
            "positive_entries",
            "negative_entries",
            "neutral_entries",
            "representative_entry",
        ):
            if key in synthesized:
                detail[key] = synthesized[key]
        details.append(detail)
    return details


def _feature_description(
    name: str,
    *,
    group: str,
    synthesized: Mapping[str, object],
) -> str:
    synthesized_description = synthesized.get("description")
    if isinstance(synthesized_description, str) and synthesized_description:
        return synthesized_description
    descriptions = {
        "raw_recall_score": "召回阶段给出的原始相关性分数；用于保留上游检索对候选文档的第一判断。",
        "normalized_recall_score": "归一化后的召回分数；把不同来源的召回分数放到可比较尺度上。",
        "heuristic_score": "规则特征的综合得分；汇总名称、关键词、来源和售卖状态等启发式信号。",
        "heuristic_recall": "规则召回命中信号；表示候选是否被启发式规则召回。",
        "heuristic_name_hit": "名称命中信号；衡量 query 是否直接命中产品名或关键别名。",
        "heuristic_main_entity_bonus": "主实体命中奖励；当 query 的核心产品实体与候选文档一致时加分。",
        "heuristic_keyword_overlap": "关键词重叠信号；衡量 query 与候选文档之间共享关键词的强弱。",
        "heuristic_term_match": "分词匹配信号；衡量 query 分词后与候选文本的覆盖程度。",
        "heuristic_text_match": "文本匹配信号；衡量 query 文本和候选文档文本的直接相似程度。",
        "heuristic_source_bias": "来源偏置信号；对内部/外部来源做稳定的排序倾向建模。",
        "heuristic_sale_bonus": "在售奖励信号；候选产品处于可售状态时提供正向排序信号。",
        "source_internal": "内部来源标记；表示候选来自内部知识库或内部商品数据。",
        "source_external": "外部来源标记；表示候选来自外部搜索或外部商品数据。",
        "product_name_in_query": "产品名出现在 query 中的标记；用于识别用户直接点名产品的场景。",
        "query_in_product_name": "query 被产品名覆盖的标记；用于识别短 query 与长产品名之间的包含关系。",
        "combo_name_in_query": "组合/套餐名出现在 query 中的标记；用于识别用户查询具体产品组合的场景。",
        "main_entity_product_hit": "主实体产品命中标记；判断 query 的核心实体是否对应候选产品。",
        "name_char_overlap_ratio": "产品名字符重叠比例；衡量 query 与产品名在字面上的接近程度。",
        "combo_char_overlap_ratio": "组合名字符重叠比例；衡量 query 与产品组合名的字面接近程度。",
        "query_product_length_ratio": "query 与产品名长度比例；帮助模型区分完整点名和模糊短 query。",
        "query_combo_length_ratio": "query 与组合名长度比例；帮助模型判断查询是否足够覆盖组合名。",
        "brand_overlap_count": "品牌词重叠数量；统计 query 和候选文档共享的品牌实体。",
        "type_overlap_count": "品类词重叠数量；统计 query 和候选文档共享的产品类型词。",
        "company_overlap_count": "公司/承保方重叠数量；统计 query 与候选文档共享的公司实体。",
        "platform_overlap_count": "平台词重叠数量；统计 query 和候选文档共享的平台或渠道词。",
        "sale_status_on": "候选产品在售标记；表示候选当前可售。",
        "sale_status_off": "候选产品停售标记；表示候选当前不可售或下架。",
        "combo_sale_status_on": "候选组合在售标记；表示组合产品当前可售。",
        "combo_sale_status_off": "候选组合停售标记；表示组合产品当前不可售或下架。",
        "strategy_search_api": "搜索 API 策略标记；表示候选由搜索 API 路径召回。",
        "strategy_query_mention": "query mention 策略标记；表示候选由 query 中的实体提及召回。",
        "strategy_query_mention_virtual": "虚拟 mention 策略标记；表示系统通过扩展/归一化实体提及生成该候选。",
        "mention_order_inverse": "实体提及顺序倒数分；越早被 query 提及的候选通常获得更强信号。",
        "demo_score": "排序分数；演示后端提供的基础模型输入信号，用于验证报告和 Promote 流程。",
    }
    if name in descriptions:
        return descriptions[name]
    if name.startswith("llm_"):
        return "LLM 根据生产案例自动合成的特征；用于把人工反馈中的正负样本经验转成可复用排序信号。"
    if "score" in name or "recall" in name:
        return f"排序分数类特征：{name}；用于表达候选文档在召回或排序阶段的相关性强弱。"
    if "overlap" in name or "hit" in name or "match" in name:
        return f"文本/实体匹配特征：{name}；用于衡量 query 与候选文档之间的字面或实体一致性。"
    if "sale_status" in name:
        return f"售卖状态特征：{name}；用于让模型区分可售、停售或组合售卖状态。"
    if name.startswith("source_"):
        return f"来源标记特征：{name}；用于告诉模型候选来自内部知识库、外部搜索或其他来源。"
    if name.startswith("strategy_"):
        return f"召回策略特征：{name}；用于记录候选是通过哪类召回策略进入排序池。"
    return f"模型输入特征：{name}；用于给排序模型提供补充信号，具体含义由当前特征集定义。"


def _operator_summary(
    *,
    run_id: str,
    state: str,
    promotion_eligible: bool,
    decision: Mapping[str, object],
    training: Mapping[str, object],
    evaluation: Mapping[str, object],
    reproducibility: Mapping[str, object],
    artifacts: object,
) -> dict[str, object]:
    checks = _check_items(decision.get("checks"))
    passed = sum(1 for item in checks if item.get("passed") is True)
    failed = len(checks) - passed
    blockers = _string_list(decision.get("blockers"))
    metrics = _global_metrics(evaluation)
    metric_bits = _metric_bits(metrics)
    if promotion_eligible:
        headline = "本次运行可晋级"
        reason = f"硬门禁 {passed}/{len(checks)} 通过，候选模型可以进入人工批准 Promote。"
    else:
        headline = "本次运行暂不可晋级"
        if blockers:
            reason = "硬门禁未全部通过，主要阻断项：" + "、".join(blockers)
        elif failed:
            reason = f"硬门禁仍有 {failed} 项未通过，需要继续修复后重训。"
        else:
            reason = "运行结果未达到晋级条件，需要查看门禁详情。"
    if metric_bits:
        reason = f"{reason} 最终指标：{', '.join(metric_bits)}。"
    return {
        "headline": headline,
        "reason": reason,
        "run_id": run_id,
        "state": state,
        "promotion_eligible": promotion_eligible,
        "acceptance_level": decision.get("acceptance_level"),
        "model_hash": _model_hash_from(training, artifacts),
        "policy_hash": reproducibility.get("policy_hash"),
        "decision_hash": reproducibility.get("decision_hash"),
        "gate_total": len(checks),
        "gate_passed": passed,
        "gate_failed": failed,
    }


def _data_overview(
    *,
    inputs: Mapping[str, object],
    datasets: Mapping[str, object],
    compilation: Mapping[str, object],
    training: Mapping[str, object],
) -> dict[str, object]:
    return {
        "summary": "本次运行使用基础标注数据训练排序模型，并用生产案例验证当前线上问题是否被修复。",
        "datasets": [
            _dataset_overview(
                role="base",
                title="基础训练数据",
                dataset_id=inputs.get("base_dataset_id"),
                dataset=datasets.get("base"),
                purpose="提供 query-doc 相关性样本，用于训练和验证基础排序能力。",
            ),
            _dataset_overview(
                role="production_cases",
                title="生产案例数据",
                dataset_id=inputs.get("production_cases_id"),
                dataset=datasets.get("production_cases"),
                purpose="承载线上反馈的 good/bad 案例，用于校验当前坏例是否被修复。",
            ),
        ],
        "split": _split_summary(compilation=compilation, training=training),
    }


def _dataset_overview(
    *,
    role: str,
    title: str,
    dataset_id: object,
    dataset: object,
    purpose: str,
) -> dict[str, object]:
    item = _optional_mapping(dataset)
    metadata = _optional_mapping(item.get("metadata"))
    columns = _string_list(metadata.get("columns"))
    return {
        "role": role,
        "title": title,
        "dataset_id": item.get("dataset_id") or dataset_id,
        "rows": _number_or_none(metadata.get("rows")),
        "field_count": len(columns),
        "fields": columns,
        "content_hash": item.get("content_hash"),
        "schema_hash": item.get("schema_hash"),
        "source_hash": metadata.get("source_hash"),
        "purpose": purpose,
    }


def _split_summary(
    *,
    compilation: Mapping[str, object],
    training: Mapping[str, object],
) -> dict[str, object]:
    compilation_metadata = _optional_mapping(compilation.get("metadata"))
    training_metadata = _optional_mapping(training.get("metadata"))
    dataset_summary = _optional_mapping(training_metadata.get("dataset_summary"))
    return {
        "train_rows": _first_number(compilation_metadata.get("train_rows"), dataset_summary.get("train_rows")),
        "validation_rows": _first_number(
            compilation_metadata.get("validation_rows"),
            dataset_summary.get("validation_rows"),
        ),
        "test_rows": _first_number(compilation_metadata.get("test_rows"), dataset_summary.get("test_rows")),
        "base_rows": _first_number(compilation_metadata.get("base_rows")),
        "production_case_rows": _first_number(compilation_metadata.get("production_case_rows")),
        "compiled_rows": _first_number(
            compilation_metadata.get("rows"),
            compilation_metadata.get("compiled_rows"),
            compilation_metadata.get("base_rows"),
            dataset_summary.get("rows"),
        ),
        "query_groups": _first_number(dataset_summary.get("query_groups")),
    }


def _process_timeline(
    *,
    inputs: Mapping[str, object],
    validation: Mapping[str, object],
    compilation: Mapping[str, object],
    training: Mapping[str, object],
    evaluation: Mapping[str, object],
    decision: Mapping[str, object],
    features: Mapping[str, object],
    feature_synthesis: object,
    completed_stage_manifests: object,
) -> dict[str, object]:
    completed_stages = _completed_stage_names(completed_stage_manifests)
    llm_features = _string_list(features.get("llm_feature_names"))
    synthesis = _optional_mapping(feature_synthesis)
    synthesis_metadata = _optional_mapping(synthesis.get("metadata"))
    initial_blockers = _string_list(synthesis_metadata.get("initial_blockers")) or _string_list(
        _optional_mapping(synthesis.get("initial_decision")).get("blockers")
    )
    validation_passed = validation.get("valid") is True
    promotion_eligible = decision.get("promotion_eligible") is True
    steps = [
        _timeline_step(
            "upload",
            "锁定输入数据",
            "completed",
            "锁定 base 与 production cases 的不可变数据快照。",
            1,
        ),
        _timeline_step(
            "validate",
            "校验",
            "passed" if validation_passed else "blocked",
            "字段映射和必填列校验通过。" if validation_passed else "输入校验未通过，需要先修正数据。",
            2,
        ),
        _timeline_step(
            "compile",
            "编译",
            "completed" if "COMPILED" in completed_stages or compilation else "pending",
            "把基础数据和生产案例编译成训练/评测输入。",
            3,
        ),
        _timeline_step(
            "train",
            "首轮训练",
            "completed" if "TRAINED" in completed_stages or training else "pending",
            "使用当前特征集训练候选排序模型。",
            4,
        ),
        _timeline_step(
            "evaluate",
            "首轮门禁",
            "blocked" if initial_blockers else ("passed" if promotion_eligible else "checked"),
            "首轮未过门禁，进入特征补强。" if initial_blockers else "完成指标与门禁检查。",
            5,
        ),
        _timeline_step(
            "feature_synthesis",
            "LLM 特征挖掘",
            "used" if llm_features else ("attempted" if synthesis.get("attempted") is True else "skipped"),
            (
                "触发 feature synthesis retry，新增 "
                + "、".join(llm_features)
                + ("；首轮阻断项：" + "、".join(initial_blockers) if initial_blockers else "。")
            )
            if llm_features
            else "本次没有新增 LLM 特征。",
            6,
        ),
        _timeline_step(
            "retry_train",
            "带新特征重训",
            "completed" if "TRAINED_RETRY" in completed_stages else ("skipped" if not llm_features else "completed"),
            "将新增特征物化为列后重训候选模型。" if llm_features else "无新增特征，跳过重训。",
            7,
        ),
        _timeline_step(
            "retry_evaluate",
            "重训后门禁",
            "passed" if promotion_eligible else "blocked",
            "重训候选模型通过最终门禁。" if promotion_eligible else "重训后仍未达到晋级条件。",
            8,
        ),
        _timeline_step(
            "report",
            "生成报告",
            "completed",
            "沉淀晋级前报告、报告数据和可复现哈希。",
            9,
        ),
        _timeline_step(
            "promote",
            "Promote",
            "ready" if promotion_eligible else "blocked",
            "等待人工批准 Promote。" if promotion_eligible else "Promote 被门禁阻断。",
            10,
        ),
    ]
    return {"kind": "gantt", "steps": steps}


def _timeline_step(
    step_id: str,
    title: str,
    status: str,
    summary: str,
    order: int,
) -> dict[str, object]:
    return {
        "id": step_id,
        "title": title,
        "status": status,
        "summary": summary,
        "offset": order - 1,
        "duration": 1,
    }


def _metric_summary(
    *,
    evaluation: Mapping[str, object],
    decision: Mapping[str, object],
    training: Mapping[str, object],
    compilation: Mapping[str, object],
    features: Mapping[str, object],
    feature_synthesis: object,
) -> dict[str, object]:
    global_metrics = _global_metrics(evaluation)
    synthesis = _optional_mapping(feature_synthesis)
    initial_metrics = _global_metrics(_optional_mapping(synthesis.get("initial_evaluation")))
    checks = _check_items(decision.get("checks"))
    passed = sum(1 for item in checks if item.get("passed") is True)
    failed = len(checks) - passed
    training_metadata = _optional_mapping(training.get("metadata"))
    params = _optional_mapping(features.get("training_params"))
    split = _split_summary(compilation=compilation, training=training)
    return {
        "global_metrics": global_metrics,
        "initial_global_metrics": initial_metrics,
        "comparison": _metric_comparison(initial_metrics, global_metrics),
        "gate_summary": {"total": len(checks), "passed": passed, "failed": failed},
        "training": {
            "objective": params.get("objective") or training_metadata.get("objective"),
            "eval_metric": params.get("eval_metric") or training_metadata.get("eval_metric"),
            "rounds": training_metadata.get("rounds") or params.get("rounds"),
            "seed": params.get("seed") or training_metadata.get("seed"),
            **split,
        },
    }


def _deposits(
    *,
    inputs: Mapping[str, object],
    compilation: Mapping[str, object],
    training: Mapping[str, object],
    features: Mapping[str, object],
    feature_synthesis: object,
    artifacts: object,
    reproducibility: Mapping[str, object],
) -> list[dict[str, object]]:
    all_artifacts = [
        *_artifact_list(artifacts),
        *_artifact_list(compilation.get("artifact_refs")),
        *_artifact_list(training.get("candidate_refs")),
    ]
    model_ref = _optional_mapping(training.get("model_ref"))
    model_artifact = model_ref or _first_artifact(all_artifacts, _is_model_artifact)
    compiled_artifact = _first_artifact(all_artifacts, lambda item: str(item.get("artifact_type", "")).startswith("compiled"))
    metadata_artifact = _first_artifact(
        all_artifacts,
        lambda item: item.get("artifact_type") in {"training-metadata", "xgboost-model-metadata", "model-metadata"},
    )
    synthesis = _optional_mapping(feature_synthesis)
    synthesis_stage = _optional_mapping(synthesis.get("stage"))
    synthesis_artifact = _first_artifact(
        _artifact_list(synthesis_stage.get("artifacts")),
        lambda item: str(item.get("artifact_type", "")).startswith("feature-synthesis"),
    )
    deposits: list[dict[str, object]] = [
        {
            "kind": "data_snapshot",
            "title": "输入数据快照",
            "summary": "记录 base/prod cases 的输入哈希，保证后续可复现同一批数据。",
            "content_hash": inputs.get("input_hash"),
        }
    ]
    if compiled_artifact:
        deposits.append(
            {
                "kind": "compiled_dataset",
                "title": "编译后的训练数据",
                "summary": "训练前物化出的 CSV/中间数据，包含基础样本与生产案例。",
                **_artifact_fields(compiled_artifact),
            }
        )
    if model_artifact:
        deposits.append(
            {
                "kind": "model",
                "title": "候选模型",
                "summary": "最终用于 Promote 的候选排序模型。",
                **_artifact_fields(model_artifact),
            }
        )
    if metadata_artifact:
        deposits.append(
            {
                "kind": "model_metadata",
                "title": "模型元数据",
                "summary": "记录训练参数、特征集和模型 schema。",
                **_artifact_fields(metadata_artifact),
            }
        )
    if _string_list(features.get("llm_feature_names")) or synthesis.get("attempted") is True:
        deposits.append(
            {
                "kind": "feature_synthesis",
                "title": "LLM 特征方案",
                "summary": "记录自动挖掘的新特征、证据样本和重训输入。",
                **_artifact_fields(synthesis_artifact),
            }
        )
    deposits.append(
        {
            "kind": "report",
            "title": "晋级前报告",
            "summary": "记录决策、门禁结果和可复现哈希，供 Promote 前人工审阅。",
            "content_hash": reproducibility.get("decision_hash"),
        }
    )
    return deposits


def _check_items(value: object) -> list[Mapping[str, object]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _artifact_list(value: object) -> list[Mapping[str, object]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _first_artifact(
    artifacts: list[Mapping[str, object]],
    predicate: object,
) -> Mapping[str, object]:
    for artifact in artifacts:
        if callable(predicate) and predicate(artifact):
            return artifact
    return {}


def _is_model_artifact(artifact: Mapping[str, object]) -> bool:
    return artifact.get("artifact_type") in _MODEL_ARTIFACT_TYPES


def _artifact_fields(artifact: Mapping[str, object]) -> dict[str, object]:
    if not artifact:
        return {}
    return {
        key: artifact[key]
        for key in ("artifact_type", "content_hash", "size_bytes", "path")
        if key in artifact
    }


def _model_hash_from(training: Mapping[str, object], artifacts: object) -> object:
    model_ref = _optional_mapping(training.get("model_ref"))
    if isinstance(model_ref.get("content_hash"), str) and model_ref.get("content_hash"):
        return model_ref.get("content_hash")
    model_artifact = _first_artifact(_artifact_list(artifacts), _is_model_artifact)
    return model_artifact.get("content_hash")


def _global_metrics(value: Mapping[str, object]) -> dict[str, object]:
    metrics = value.get("global_metrics")
    if isinstance(metrics, Mapping):
        return dict(metrics)
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, (int, float)) and not isinstance(item, bool)
    }


def _metric_comparison(
    before_metrics: Mapping[str, object],
    after_metrics: Mapping[str, object],
) -> list[dict[str, object]]:
    preferred = ["ndcg@10", "mrr@10"]
    keys = [
        *[key for key in preferred if key in before_metrics or key in after_metrics],
        *sorted(
            key
            for key in set(before_metrics) | set(after_metrics)
            if key not in preferred
        ),
    ]
    result: list[dict[str, object]] = []
    for key in keys:
        before = _number_or_none(before_metrics.get(key))
        after = _number_or_none(after_metrics.get(key))
        if before is None or after is None:
            continue
        result.append({"metric": key, "before": before, "after": after, "delta": after - before})
    return result


def _metric_bits(metrics: Mapping[str, object]) -> list[str]:
    result: list[str] = []
    for key in ("ndcg@10", "mrr@10"):
        value = _number_or_none(metrics.get(key))
        if value is not None:
            result.append(f"{key}={value:g}")
    return result


def _number_or_none(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _first_number(*values: object) -> int | float | None:
    for value in values:
        number = _number_or_none(value)
        if number is not None:
            return number
    return None


def _completed_stage_names(value: object) -> set[str]:
    stages: set[str] = set()
    if not isinstance(value, list):
        return stages
    for item in value:
        if not isinstance(item, Mapping):
            continue
        stage = item.get("stage")
        if isinstance(stage, str) and stage:
            stages.add(stage)
    return stages


def _load_i18n(locale: str) -> dict[str, object]:
    if locale not in _SUPPORTED_LOCALES:
        raise ValueError(f"unsupported report locale: {locale!r}")
    resource = resources.files(__package__).joinpath("i18n", f"{locale}.json")
    try:
        raw = resource.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except (FileNotFoundError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid packaged report locale: {locale}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"invalid packaged report locale: {locale}")
    return dict(value)


def build_localized_view(
    report_data: Mapping[str, object],
    locale: str,
) -> dict[str, object]:
    """Return localized labels without modifying report evidence or metrics."""

    return {
        "locale": locale,
        "labels": _load_i18n(locale),
        "raw_data": _validate_report_data(report_data),
    }


def _model_hash(report_data: Mapping[str, object], *, run_id: str) -> str:
    training = _required_section(report_data, "training", run_id=run_id)
    model_ref = training.get("model_ref")
    if isinstance(model_ref, Mapping) and isinstance(model_ref.get("content_hash"), str):
        return model_ref["content_hash"]
    for artifact in report_data["artifacts"]:
        if not isinstance(artifact, Mapping):
            continue
        if (
            artifact.get("artifact_type") in _MODEL_ARTIFACT_TYPES
            and isinstance(artifact.get("content_hash"), str)
            and artifact["content_hash"]
        ):
            return artifact["content_hash"]
    raise _error("report data does not bind a candidate model hash", run_id=run_id)


def _manifest_data(
    report_data: Mapping[str, object],
    *,
    data_hash: str,
    html_hash: str,
    locale: str,
) -> dict[str, object]:
    run = _required_section(report_data, "run", run_id="unknown")
    run_id = _string(run.get("run_id"), label="run.run_id")
    reproducibility = _required_section(report_data, "reproducibility", run_id=run_id)
    decision_hash = reproducibility.get("decision_hash")
    if not isinstance(decision_hash, str) or not decision_hash:
        decision_hash = _sha256_bytes(_json_bytes(report_data["decision"], run_id=run_id))
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "report_type": "pre-promote",
        "run_id": run_id,
        "locale": locale,
        "data_path": _DATA_FILENAME,
        "html_path": _HTML_FILENAME,
        "data_hash": data_hash,
        "html_hash": html_hash,
        "decision_hash": decision_hash,
        "model_hash": _model_hash(report_data, run_id=run_id),
        "policy_hash": _string(
            reproducibility.get("policy_hash"),
            label="reproducibility.policy_hash",
            run_id=run_id,
        ),
    }


def _template_html(report_data: Mapping[str, object], *, locale: str, run_id: str) -> bytes:
    try:
        template = (
            resources.files(__package__)
            .joinpath("templates", "pre_promote.html")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, UnicodeDecodeError) as exc:
        raise RuntimeError("packaged Pre Promote report template is unavailable") from exc
    translations = {supported: _load_i18n(supported) for supported in sorted(_SUPPORTED_LOCALES)}
    rendered = (
        template.replace("__REPORT_JSON__", _json_text(report_data, run_id=run_id))
        .replace("__I18N_JSON__", _json_text(translations, run_id=run_id))
        .replace("__INITIAL_LOCALE__", _json_text(locale, run_id=run_id))
    )
    if "__REPORT_JSON__" in rendered or "__I18N_JSON__" in rendered:
        raise RuntimeError("packaged Pre Promote report template has unresolved placeholders")
    return rendered.encode("utf-8")


def _read_existing(path: Path, *, run_id: str) -> bytes | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise _error(
            "immutable report target is not a regular file",
            run_id=run_id,
            details={"path": str(path)},
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise _error(
            "immutable report target cannot be read",
            run_id=run_id,
            details={"path": str(path)},
        ) from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable(path: Path, content: bytes, *, run_id: str) -> None:
    existing = _read_existing(path, run_id=run_id)
    if existing is not None:
        if existing != content:
            raise _error(
                "immutable report target already contains different content",
                run_id=run_id,
                details={"path": str(path)},
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            pass
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    existing = _read_existing(path, run_id=run_id)
    if existing != content:
        raise _error(
            "immutable report target conflicts with a concurrent writer",
            run_id=run_id,
            details={"path": str(path)},
        )


def render_pre_promote_report(
    report_data: Mapping[str, object],
    output_dir: Path,
    *,
    locale: str = "zh-CN",
) -> ReportArtifact:
    """Create or verify the three immutable Pre Promote report artifacts."""

    if locale not in _SUPPORTED_LOCALES:
        raise ValueError(f"unsupported report locale: {locale!r}")
    normalized = _validate_report_data(report_data)
    run_id = _string(
        _mapping(normalized["run"], label="report_data.run").get("run_id"),
        label="run.run_id",
    )
    destination = Path(output_dir)
    if not destination.name and destination == Path("."):
        raise ValueError("report output directory must be explicit")
    data_bytes = _json_bytes(normalized, run_id=run_id)
    html_bytes = _template_html(normalized, locale=locale, run_id=run_id)
    data_hash = _sha256_bytes(data_bytes)
    html_hash = _sha256_bytes(html_bytes)
    manifest = _manifest_data(
        normalized,
        data_hash=data_hash,
        html_hash=html_hash,
        locale=locale,
    )
    manifest_bytes = _json_bytes(manifest, run_id=run_id)
    targets = (
        (destination / _DATA_FILENAME, data_bytes),
        (destination / _HTML_FILENAME, html_bytes),
        (destination / _MANIFEST_FILENAME, manifest_bytes),
    )
    for path, content in targets:
        existing = _read_existing(path, run_id=run_id)
        if existing is not None and existing != content:
            raise _error(
                "immutable report target already contains different content",
                run_id=run_id,
                details={"path": str(path)},
            )
    for path, content in targets:
        _write_immutable(path, content, run_id=run_id)
    return ReportArtifact(
        html_path=destination / _HTML_FILENAME,
        data_path=destination / _DATA_FILENAME,
        manifest_path=destination / _MANIFEST_FILENAME,
        data_hash=data_hash,
        html_hash=html_hash,
        manifest=manifest,
    )


def _rebase_artifact(
    store: ArtifactStore,
    artifact: ArtifactRef,
    *,
    run_id: str,
) -> Path:
    root = getattr(store, "root", None)
    if not isinstance(root, Path):
        raise _error(
            "artifact store must expose a filesystem root for report evidence",
            run_id=run_id,
        )
    relative = Path(artifact.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise _error("report evidence artifact path is unsafe", run_id=run_id)
    return root / relative


def load_sealed_report_evidence(
    run: RunRecord,
    artifacts: ArtifactStore,
) -> dict[str, object]:
    """Load only the verified evidence snapshot recorded by Core 6."""

    metadata = _mapping(
        run.metadata.get("report_evidence"),
        label="run.report_evidence",
        run_id=run.run_id,
    )
    manifest = artifacts.load_completed_stage(run.run_id, _REPORTING_STAGE, run.input_hash)
    evidence_refs = [
        artifact
        for artifact in manifest.artifacts
        if artifact.artifact_type == _REPORT_EVIDENCE_ARTIFACT
    ]
    if len(evidence_refs) != 1:
        raise _error(
            "reporting stage must contain exactly one evidence artifact",
            run_id=run.run_id,
        )
    evidence_ref = evidence_refs[0]
    expected = {
        "artifact_type": evidence_ref.artifact_type,
        "path": evidence_ref.path.as_posix(),
        "content_hash": evidence_ref.content_hash,
        "size_bytes": evidence_ref.size_bytes,
    }
    if dict(metadata) != {"stage": _REPORTING_STAGE, **expected}:
        raise _error(
            "run report evidence metadata does not match the sealed manifest",
            run_id=run.run_id,
        )
    path = _rebase_artifact(artifacts, evidence_ref, run_id=run.run_id)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise _error("sealed report evidence cannot be read", run_id=run.run_id) from exc
    if len(raw) != evidence_ref.size_bytes or _sha256_bytes(raw) != evidence_ref.content_hash:
        raise _error("sealed report evidence no longer matches its manifest", run_id=run.run_id)
    try:
        evidence = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise _error("sealed report evidence is not strict JSON", run_id=run.run_id) from exc
    if not isinstance(evidence, Mapping):
        raise _error("sealed report evidence must be a JSON object", run_id=run.run_id)
    evidence_run = _mapping(
        evidence.get("run"),
        label="sealed report evidence run",
        run_id=run.run_id,
    )
    if _string(
        evidence_run.get("run_id"),
        label="sealed report evidence run.run_id",
        run_id=run.run_id,
    ) != run.run_id:
        raise _error(
            "sealed report evidence belongs to a different run",
            run_id=run.run_id,
        )
    return dict(evidence)


def render_run_pre_promote_report(
    run: RunRecord,
    artifacts: ArtifactStore,
    *,
    locale: str = "zh-CN",
) -> ReportArtifact:
    """Render a report from Core 6's sealed evidence without changing the run."""

    evidence = load_sealed_report_evidence(run, artifacts)
    report_data = build_report_data(evidence)
    return render_pre_promote_report(
        report_data,
        artifacts.run_dir(run.run_id) / "reports",
        locale=locale,
    )


def render_run_pre_promote_report_html(
    run: RunRecord,
    artifacts: ArtifactStore,
    *,
    locale: str = "zh-CN",
) -> str:
    """Render a non-persistent live report view from sealed evidence."""

    if locale not in _SUPPORTED_LOCALES:
        raise ValueError(f"unsupported report locale: {locale!r}")
    evidence = load_sealed_report_evidence(run, artifacts)
    report_data = build_report_data(evidence)
    return _template_html(report_data, locale=locale, run_id=run.run_id).decode("utf-8")
