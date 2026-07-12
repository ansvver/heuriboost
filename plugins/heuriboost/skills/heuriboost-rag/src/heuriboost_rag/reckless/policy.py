from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

import yaml

from .contracts import Decision, EvaluationResult, GateCheck


_TOP_LEVEL_KEYS = frozenset(
    {"version", "acceptance_level", "input", "evaluation", "promotion"}
)
_INPUT_KEYS = frozenset(
    {
        "min_global_test_queries",
        "min_domain_test_queries",
        "min_docs_per_query",
        "require_authoritative_labels",
    }
)
_EVALUATION_KEYS = frozenset(
    {
        "require_all_current_cases",
        "require_all_historical_gates",
        "require_global_ndcg_improvement",
        "require_global_mrr_improvement",
        "allow_touched_domain_regression",
    }
)
_PROMOTION_KEYS = frozenset(
    {
        "allow_weak",
        "require_explicit_human_approval",
        "allow_anchor_reset",
        "allow_gate_retirement",
    }
)


class _StrictPolicyLoader(yaml.SafeLoader):
    yaml_implicit_resolvers = {
        initial: list(resolvers)
        for initial, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def construct_mapping(
        self,
        node: yaml.MappingNode,
        deep: bool = False,
    ) -> dict[object, object]:
        if not isinstance(node, yaml.MappingNode):
            return super().construct_mapping(node, deep=deep)
        self.flatten_mapping(node)
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from exc
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


for _initial, _resolvers in tuple(
    _StrictPolicyLoader.yaml_implicit_resolvers.items()
):
    _StrictPolicyLoader.yaml_implicit_resolvers[_initial] = [
        resolver
        for resolver in _resolvers
        if resolver[0]
        not in {"tag:yaml.org,2002:bool", "tag:yaml.org,2002:int"}
    ]

_StrictPolicyLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$"),
    list("tf"),
)
_StrictPolicyLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^-?(?:0|[1-9][0-9]*)$"),
    list("-0123456789"),
)


def _construct_canonical_bool(
    loader: _StrictPolicyLoader,
    node: yaml.ScalarNode,
) -> object:
    raw = loader.construct_scalar(node)
    if raw not in {"true", "false"}:
        return raw
    return raw == "true"


def _construct_canonical_int(
    loader: _StrictPolicyLoader,
    node: yaml.ScalarNode,
) -> object:
    raw = loader.construct_scalar(node)
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", raw) is None:
        return raw
    return int(raw, 10)


_StrictPolicyLoader.add_constructor(
    "tag:yaml.org,2002:bool",
    _construct_canonical_bool,
)
_StrictPolicyLoader.add_constructor(
    "tag:yaml.org,2002:int",
    _construct_canonical_int,
)


def _require_positive_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def _require_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True)
class InputPolicy:
    min_global_test_queries: int = 50
    min_domain_test_queries: int = 10
    min_docs_per_query: int = 2
    require_authoritative_labels: bool = True

    def __post_init__(self) -> None:
        _require_positive_int(
            "input.min_global_test_queries",
            self.min_global_test_queries,
        )
        _require_positive_int(
            "input.min_domain_test_queries",
            self.min_domain_test_queries,
        )
        _require_positive_int("input.min_docs_per_query", self.min_docs_per_query)
        _require_bool(
            "input.require_authoritative_labels",
            self.require_authoritative_labels,
        )


@dataclass(frozen=True)
class EvaluationPolicy:
    require_all_current_cases: bool = True
    require_all_historical_gates: bool = True
    require_global_ndcg_improvement: bool = True
    require_global_mrr_improvement: bool = True
    allow_touched_domain_regression: bool = False

    def __post_init__(self) -> None:
        _require_bool(
            "evaluation.require_all_current_cases",
            self.require_all_current_cases,
        )
        _require_bool(
            "evaluation.require_all_historical_gates",
            self.require_all_historical_gates,
        )
        _require_bool(
            "evaluation.require_global_ndcg_improvement",
            self.require_global_ndcg_improvement,
        )
        _require_bool(
            "evaluation.require_global_mrr_improvement",
            self.require_global_mrr_improvement,
        )
        _require_bool(
            "evaluation.allow_touched_domain_regression",
            self.allow_touched_domain_regression,
        )


@dataclass(frozen=True)
class PromotionPolicy:
    allow_weak: bool = False
    require_explicit_human_approval: bool = True
    allow_anchor_reset: bool = False
    allow_gate_retirement: bool = False

    def __post_init__(self) -> None:
        _require_bool("promotion.allow_weak", self.allow_weak)
        _require_bool(
            "promotion.require_explicit_human_approval",
            self.require_explicit_human_approval,
        )
        _require_bool("promotion.allow_anchor_reset", self.allow_anchor_reset)
        _require_bool(
            "promotion.allow_gate_retirement",
            self.allow_gate_retirement,
        )


@dataclass(frozen=True)
class RecklessPolicy:
    version: int = 1
    acceptance_level: str = "full"
    input: InputPolicy = field(default_factory=InputPolicy)
    evaluation: EvaluationPolicy = field(default_factory=EvaluationPolicy)
    promotion: PromotionPolicy = field(default_factory=PromotionPolicy)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.version) is not int:
            raise TypeError("version must be an integer")
        if self.version != 1:
            raise ValueError(f"unsupported policy version: {self.version}")
        if type(self.acceptance_level) is not str:
            raise TypeError("acceptance_level must be a string")
        if self.acceptance_level not in {"full", "weak"}:
            raise ValueError("acceptance_level must be 'full' or 'weak'")
        if not isinstance(self.input, InputPolicy):
            raise TypeError("input must be an InputPolicy")
        if not isinstance(self.evaluation, EvaluationPolicy):
            raise TypeError("evaluation must be an EvaluationPolicy")
        if not isinstance(self.promotion, PromotionPolicy):
            raise TypeError("promotion must be a PromotionPolicy")
        canonical = json.dumps(
            _policy_dict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        object.__setattr__(
            self,
            "content_hash",
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def default(cls) -> RecklessPolicy:
        return cls()


def _policy_dict(policy: RecklessPolicy) -> dict[str, object]:
    return {
        "version": policy.version,
        "acceptance_level": policy.acceptance_level,
        "input": {
            "min_global_test_queries": policy.input.min_global_test_queries,
            "min_domain_test_queries": policy.input.min_domain_test_queries,
            "min_docs_per_query": policy.input.min_docs_per_query,
            "require_authoritative_labels": (
                policy.input.require_authoritative_labels
            ),
        },
        "evaluation": {
            "require_all_current_cases": (
                policy.evaluation.require_all_current_cases
            ),
            "require_all_historical_gates": (
                policy.evaluation.require_all_historical_gates
            ),
            "require_global_ndcg_improvement": (
                policy.evaluation.require_global_ndcg_improvement
            ),
            "require_global_mrr_improvement": (
                policy.evaluation.require_global_mrr_improvement
            ),
            "allow_touched_domain_regression": (
                policy.evaluation.allow_touched_domain_regression
            ),
        },
        "promotion": {
            "allow_weak": policy.promotion.allow_weak,
            "require_explicit_human_approval": (
                policy.promotion.require_explicit_human_approval
            ),
            "allow_anchor_reset": policy.promotion.allow_anchor_reset,
            "allow_gate_retirement": policy.promotion.allow_gate_retirement,
        },
    }


def _validate_mapping(
    value: object,
    *,
    name: str,
    allowed_keys: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, MappingABC):
        raise TypeError(f"{name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
    unknown_keys = set(value) - allowed_keys
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unknown {name} keys: {unknown}")
    return value


def _policy_section(
    data: Mapping[str, object],
    name: str,
    allowed_keys: frozenset[str],
) -> Mapping[str, object]:
    if name not in data:
        return {}
    return _validate_mapping(
        data[name],
        name=f"{name} policy",
        allowed_keys=allowed_keys,
    )


def load_policy(path: Path) -> RecklessPolicy:
    loaded = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=_StrictPolicyLoader,
    )
    if loaded is None:
        loaded = {}
    data = _validate_mapping(
        loaded,
        name="policy",
        allowed_keys=_TOP_LEVEL_KEYS,
    )
    input_data = _policy_section(data, "input", _INPUT_KEYS)
    evaluation_data = _policy_section(data, "evaluation", _EVALUATION_KEYS)
    promotion_data = _policy_section(data, "promotion", _PROMOTION_KEYS)
    return RecklessPolicy(
        version=data.get("version", 1),
        acceptance_level=data.get("acceptance_level", "full"),
        input=InputPolicy(**dict(input_data)),
        evaluation=EvaluationPolicy(**dict(evaluation_data)),
        promotion=PromotionPolicy(**dict(promotion_data)),
    )


def _required_boolean_check(
    *,
    check_id: str,
    label: str,
    observed: bool,
    required: bool,
) -> GateCheck:
    passed = not required or observed is True
    if not required:
        reason = f"{label} is not enforced; observed={observed!r}."
    elif passed:
        reason = f"{label} passed."
    else:
        reason = f"{label} is required and did not pass."
    return GateCheck(
        check_id=check_id,
        label=label,
        passed=passed,
        observed=observed,
        required={"expected": True, "enforced": required},
        reason=reason,
    )


def _invalid_metric_value(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        bit_length = value.bit_length()
        if bit_length <= 53:
            return str(value)
        sign = "negative" if value < 0 else "positive"
        return f"{sign} integer with {bit_length} bits"
    if type(value) is float:
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return repr(value)
    if type(value) is str:
        if len(value) <= 120:
            return value
        return f"{value[:117]}..."
    return f"value of type {type(value).__name__}"


def _rank_metric_evidence(
    metrics: Mapping[str, object],
    metric: str,
) -> tuple[dict[str, object], int | float | None]:
    if metric not in metrics:
        return (
            {
                "valid": False,
                "status": "missing",
                "value": "missing",
                "received_type": "missing",
            },
            None,
        )

    value = metrics[metric]
    if type(value) not in {int, float}:
        return (
            {
                "valid": False,
                "status": "invalid_type",
                "value": _invalid_metric_value(value),
                "received_type": type(value).__name__,
            },
            None,
        )
    if type(value) is float and not math.isfinite(value):
        return (
            {
                "valid": False,
                "status": "non_finite",
                "value": _invalid_metric_value(value),
                "received_type": "float",
            },
            None,
        )
    if not 0 <= value <= 1:
        return (
            {
                "valid": False,
                "status": "out_of_range",
                "value": _invalid_metric_value(value),
                "received_type": type(value).__name__,
            },
            None,
        )
    return (
        {
            "valid": True,
            "status": "valid",
            "value": value,
            "received_type": type(value).__name__,
        },
        value,
    )


def _global_metric_check(
    evaluation: EvaluationResult,
    *,
    metric: str,
    check_id: str,
    label: str,
    required: bool,
) -> GateCheck:
    candidate_evidence, candidate = _rank_metric_evidence(
        evaluation.global_metrics,
        metric,
    )
    anchor_evidence, anchor = _rank_metric_evidence(
        evaluation.anchor_metrics,
        metric,
    )
    comparable = candidate is not None and anchor is not None
    improved = comparable and candidate > anchor
    comparison_status = (
        "passed" if improved else "failed" if comparable else "not_comparable"
    )
    passed = not required or improved
    if not required:
        reason = (
            f"{label} is not enforced; comparison status is "
            f"{comparison_status}."
        )
    elif not comparable:
        reason = (
            f"{label} requires candidate and anchor metrics within [0, 1]; "
            "the values are not comparable."
        )
    elif improved:
        reason = f"{label} passed."
    else:
        reason = f"{label} requires the candidate to be strictly greater."
    return GateCheck(
        check_id=check_id,
        label=label,
        passed=passed,
        observed={
            "candidate": candidate_evidence,
            "anchor": anchor_evidence,
            "comparison": {
                "operator": ">",
                "status": comparison_status,
                "passed": improved,
                "comparable": comparable,
            },
        },
        required={"metric": metric, "operator": ">", "enforced": required},
        reason=reason,
    )


def _touched_domain_check(
    policy: RecklessPolicy,
    evaluation: EvaluationResult,
) -> GateCheck:
    domains: dict[str, object] = {}
    all_evidence_valid = bool(evaluation.touched_domains)
    all_non_regressing = bool(evaluation.touched_domains)

    for domain in sorted(evaluation.touched_domains):
        metrics = evaluation.touched_domains[domain]
        name_valid = bool(domain.strip())
        if not isinstance(metrics, MappingABC):
            domains[domain] = {
                "name_valid": name_valid,
                "valid": False,
                "non_regressing": False,
                "status": "invalid_metrics_type",
                "metrics": {},
            }
            all_evidence_valid = False
            all_non_regressing = False
            continue

        candidate_names = {
            metric for metric in metrics if not metric.startswith("anchor_")
        }
        anchor_names = {
            metric[len("anchor_") :]
            for metric in metrics
            if metric.startswith("anchor_")
        }
        metric_names = sorted(candidate_names | anchor_names)
        metric_comparisons: dict[str, object] = {}
        domain_evidence_valid = name_valid and bool(metric_names)
        domain_non_regressing = name_valid and bool(metric_names)

        for metric in metric_names:
            candidate_present = metric in candidate_names
            anchor_present = metric in anchor_names
            if candidate_present and anchor_present:
                pairing_status = "paired"
            elif candidate_present:
                pairing_status = "missing_anchor"
            else:
                pairing_status = "orphan_anchor"

            anchor_key = f"anchor_{metric}"
            candidate_evidence, candidate = _rank_metric_evidence(metrics, metric)
            anchor_evidence, anchor = _rank_metric_evidence(metrics, anchor_key)
            metric_name_valid = bool(metric.strip())
            comparable = (
                metric_name_valid
                and pairing_status == "paired"
                and candidate is not None
                and anchor is not None
            )
            non_regressing = comparable and candidate >= anchor
            comparison_status = (
                "passed"
                if non_regressing
                else "failed"
                if comparable
                else "not_comparable"
            )
            metric_valid = (
                metric_name_valid
                and pairing_status == "paired"
                and candidate_evidence["valid"] is True
                and anchor_evidence["valid"] is True
            )
            metric_comparisons[metric] = {
                "name_valid": metric_name_valid,
                "valid": metric_valid,
                "pairing_status": pairing_status,
                "candidate_key": metric,
                "anchor_key": anchor_key,
                "candidate": candidate_evidence,
                "anchor": anchor_evidence,
                "comparison": {
                    "operator": ">=",
                    "status": comparison_status,
                    "passed": non_regressing,
                    "comparable": comparable,
                },
            }
            domain_evidence_valid = domain_evidence_valid and metric_valid
            domain_non_regressing = (
                domain_non_regressing and metric_valid and non_regressing
            )

        if not metric_names:
            domain_status = "empty_metrics"
        elif not domain_evidence_valid:
            domain_status = "invalid"
        elif domain_non_regressing:
            domain_status = "valid"
        else:
            domain_status = "regression"
        domains[domain] = {
            "name_valid": name_valid,
            "valid": domain_evidence_valid,
            "non_regressing": domain_non_regressing,
            "status": domain_status,
            "metrics": metric_comparisons,
        }
        all_evidence_valid = all_evidence_valid and domain_evidence_valid
        all_non_regressing = all_non_regressing and domain_non_regressing

    observed_status = (
        "empty"
        if not evaluation.touched_domains
        else "invalid"
        if not all_evidence_valid
        else "valid"
        if all_non_regressing
        else "regression"
    )
    regression_allowed = policy.evaluation.allow_touched_domain_regression
    passed = regression_allowed or all_non_regressing
    if regression_allowed:
        reason = (
            "Touched-domain non-regression is not enforced; validation status "
            f"is {observed_status}."
        )
    elif passed:
        reason = "Every touched-domain metric is at least its anchor."
    else:
        reason = (
            "Touched domains require non-empty names, exact candidate/anchor "
            "pairs, metrics within [0, 1], and candidate >= anchor."
        )
    return GateCheck(
        check_id="touched_domain_non_regression",
        label="Touched domains non-regression",
        passed=passed,
        observed={
            "valid": all_evidence_valid,
            "non_regressing": all_non_regressing,
            "status": observed_status,
            "domains": domains,
        },
        required={
            "operator": ">=",
            "anchor_key": "anchor_<metric>",
            "enforced": not regression_allowed,
        },
        reason=reason,
    )


def _artifact_check(evaluation: EvaluationResult) -> GateCheck:
    passed = evaluation.artifacts_valid is True
    return GateCheck(
        check_id="artifact_integrity",
        label="Artifact integrity",
        passed=passed,
        observed=evaluation.artifacts_valid,
        required=True,
        reason=(
            "Artifact integrity passed."
            if passed
            else "All required artifacts must exist and match their hashes."
        ),
    )


def _acceptance_check(
    policy: RecklessPolicy,
    evaluation: EvaluationResult,
) -> GateCheck:
    passed = (
        policy.acceptance_level == "full"
        and evaluation.acceptance_level == "full"
    )
    if passed:
        reason = "Policy and evaluation acceptance levels are both 'full'."
    else:
        reason = (
            "V1 promotion requires policy and evaluation acceptance levels "
            f"to both be 'full'; observed policy={policy.acceptance_level!r}, "
            f"evaluation={evaluation.acceptance_level!r}."
        )
    return GateCheck(
        check_id="acceptance_level",
        label="Acceptance level",
        passed=passed,
        observed={
            "policy": policy.acceptance_level,
            "evaluation": evaluation.acceptance_level,
            "allow_weak": policy.promotion.allow_weak,
        },
        required={"policy": "full", "evaluation": "full"},
        reason=reason,
    )


def evaluate_promotion_eligibility(
    policy: RecklessPolicy,
    evaluation: EvaluationResult,
) -> Decision:
    checks = (
        _required_boolean_check(
            check_id="current_production_cases",
            label="Current production cases",
            observed=evaluation.current_cases_passed,
            required=policy.evaluation.require_all_current_cases,
        ),
        _required_boolean_check(
            check_id="historical_gates",
            label="Historical gates",
            observed=evaluation.historical_gates_passed,
            required=policy.evaluation.require_all_historical_gates,
        ),
        _global_metric_check(
            evaluation,
            metric="ndcg@10",
            check_id="global_ndcg_at_10_improvement",
            label="Global nDCG@10 strict improvement",
            required=policy.evaluation.require_global_ndcg_improvement,
        ),
        _global_metric_check(
            evaluation,
            metric="mrr@10",
            check_id="global_mrr_at_10_improvement",
            label="Global MRR@10 strict improvement",
            required=policy.evaluation.require_global_mrr_improvement,
        ),
        _touched_domain_check(policy, evaluation),
        _artifact_check(evaluation),
        _acceptance_check(policy, evaluation),
    )
    blockers = tuple(check.check_id for check in checks if not check.passed)
    return Decision(
        promotion_eligible=not blockers,
        acceptance_level=evaluation.acceptance_level,
        checks=checks,
        blockers=blockers,
        warnings=tuple(evaluation.warnings),
    )
