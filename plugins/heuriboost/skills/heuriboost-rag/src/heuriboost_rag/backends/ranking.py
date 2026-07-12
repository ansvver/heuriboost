"""Reusable XGBoost ranking helpers for HeuriBoost integrations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LABEL_MAP = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4}


def train_xgboost_ranker(
    *,
    train_features: Sequence[Sequence[float]],
    train_labels: Sequence[int],
    train_groups: Sequence[int],
    validation_features: Sequence[Sequence[float]],
    validation_labels: Sequence[int],
    validation_groups: Sequence[int],
    output_dir: Any,
    feature_names: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    rounds: int = 40,
    params: Mapping[str, Any] | None = None,
    labels_are_mapped: bool = False,
) -> dict[str, str]:
    """Train an XGBoost LambdaMART ranker from prepared feature rows.

    Integrations own their domain-specific feature extraction. HeuriBoost owns
    the generic grouped ranking training loop and model artifact metadata.
    """

    _validate_payload(train_features, train_labels, train_groups, "train")
    _validate_payload(validation_features, validation_labels, validation_groups, "validation")
    require_xgboost()
    import xgboost as xgb

    model_params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "tree_method": "hist",
        "seed": 42,
    }
    if params:
        model_params.update(dict(params))

    dtrain = _dmatrix(
        train_features,
        train_labels,
        train_groups,
        feature_names=feature_names,
        labels_are_mapped=labels_are_mapped,
    )
    dvalid = _dmatrix(
        validation_features,
        validation_labels,
        validation_groups,
        feature_names=feature_names,
        labels_are_mapped=labels_are_mapped,
    )
    booster = xgb.train(
        model_params,
        dtrain,
        num_boost_round=int(rounds),
        evals=[(dtrain, "train"), (dvalid, "validation")],
        verbose_eval=False,
    )

    models_dir = Path(output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "reranker.json"
    metadata_path = models_dir / "reranker_metadata.json"
    booster.save_model(str(model_path))

    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        {
            "feature_names": list(feature_names or []),
            "params": model_params,
            "rounds": int(rounds),
            "train_rows": int(len(train_features)),
            "validation_rows": int(len(validation_features)),
            "train_groups": int(len(train_groups)),
            "validation_groups": int(len(validation_groups)),
            "labels_are_mapped": bool(labels_are_mapped),
        }
    )
    _write_json(metadata_path, artifact_metadata)
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }


def predict_xgboost_ranker(
    *,
    features: Sequence[Sequence[float]],
    model_path: Any,
    feature_names: Sequence[str] | None = None,
) -> list[float]:
    """Score feature rows with a trained XGBoost ranker."""

    require_xgboost()
    import xgboost as xgb

    resolved_feature_names = _resolve_feature_names(feature_names, model_path)
    dmatrix = xgb.DMatrix(
        features,
        feature_names=list(resolved_feature_names or []) or None,
    )
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    predictions = [float(score) for score in booster.predict(dmatrix)]
    if len(predictions) != len(features):
        raise RuntimeError(
            f"XGBoost prediction row count mismatch: {len(predictions)} != {len(features)}"
        )
    if not all(math.isfinite(score) for score in predictions):
        raise ValueError("XGBoost ranker predictions must be finite")
    return predictions


def evaluate_xgboost_ranker(
    *,
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    groups: Sequence[int],
    model_path: Any,
    split: str = "validation",
    query_group_order: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
    labels_are_mapped: bool = False,
) -> dict[str, Any]:
    """Evaluate a trained XGBoost ranker on one grouped split."""

    _validate_payload(features, labels, groups, split)
    require_xgboost()
    import xgboost as xgb

    resolved_feature_names = _resolve_feature_names(feature_names, model_path)
    dmatrix = _dmatrix(
        features,
        labels,
        groups,
        feature_names=resolved_feature_names,
        labels_are_mapped=labels_are_mapped,
    )
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    predictions = [float(score) for score in booster.predict(dmatrix)]
    mapped_labels = _coerce_labels(labels, labels_are_mapped=labels_are_mapped)
    return {
        "rows": int(len(features)),
        "query_groups": int(len(groups)),
        "split": split,
        "query_group_order": list(query_group_order or []),
        "group_sizes": [int(size) for size in groups],
        "ndcg@10": _ndcg_at_10(mapped_labels, groups, predictions),
        "mrr@10": _mrr_at_10(mapped_labels, groups, predictions),
    }


def map_relevance_labels(labels: Iterable[int]) -> list[int]:
    mapped = []
    for label in labels:
        value = int(label)
        if value not in LABEL_MAP:
            raise ValueError(f"unsupported relevance label: {value}")
        mapped.append(LABEL_MAP[value])
    return mapped


def require_xgboost() -> None:
    try:
        __import__("xgboost")
    except ImportError as exc:
        raise RuntimeError("Missing required dependency: xgboost") from exc


def _dmatrix(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    groups: Sequence[int],
    *,
    feature_names: Sequence[str] | None,
    labels_are_mapped: bool,
):
    import xgboost as xgb

    dmatrix = xgb.DMatrix(
        features,
        label=_coerce_labels(labels, labels_are_mapped=labels_are_mapped),
        feature_names=list(feature_names or []) or None,
    )
    dmatrix.set_group([int(size) for size in groups])
    return dmatrix


def _coerce_labels(labels: Sequence[int], *, labels_are_mapped: bool) -> list[int]:
    if labels_are_mapped:
        return [int(label) for label in labels]
    return map_relevance_labels(labels)


def _validate_payload(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    groups: Sequence[int],
    name: str,
) -> None:
    if not features:
        raise ValueError(f"{name} features are empty")
    if len(features) != len(labels):
        raise ValueError(
            f"{name} features/labels length mismatch: {len(features)} != {len(labels)}"
        )
    group_total = sum(int(size) for size in groups)
    if group_total != len(features):
        raise ValueError(f"{name} groups sum to {group_total}, expected {len(features)}")
    if any(int(size) <= 0 for size in groups):
        raise ValueError(f"{name} groups must all be positive")


def _ndcg_at_10(labels: Sequence[int], groups: Sequence[int], predictions: Sequence[float]) -> float:
    scores = []
    offset = 0
    for size in groups:
        group_labels = labels[offset : offset + size]
        group_predictions = predictions[offset : offset + size]
        scores.append(_group_ndcg(group_labels, group_predictions, 10))
        offset += size
    return float(sum(scores) / len(scores)) if scores else 0.0


def _mrr_at_10(labels: Sequence[int], groups: Sequence[int], predictions: Sequence[float]) -> float:
    scores = []
    offset = 0
    for size in groups:
        group_labels = labels[offset : offset + size]
        group_predictions = predictions[offset : offset + size]
        scores.append(_group_mrr(group_labels, group_predictions, 10))
        offset += size
    return float(sum(scores) / len(scores)) if scores else 0.0


def _group_ndcg(labels: Sequence[int], predictions: Sequence[float], k: int) -> float:
    ranked_labels = [
        label for _, label in sorted(zip(predictions, labels), key=lambda item: item[0], reverse=True)
    ][:k]
    ideal_labels = sorted(labels, reverse=True)[:k]
    ideal = _dcg(ideal_labels)
    if ideal <= 0:
        return 0.0
    return _dcg(ranked_labels) / ideal


def _group_mrr(labels: Sequence[int], predictions: Sequence[float], k: int) -> float:
    ranked_labels = [
        label for _, label in sorted(zip(predictions, labels), key=lambda item: item[0], reverse=True)
    ][:k]
    for index, label in enumerate(ranked_labels, start=1):
        if label > 0:
            return 1.0 / index
    return 0.0


def _dcg(labels: Sequence[int]) -> float:
    total = 0.0
    for index, label in enumerate(labels, start=1):
        total += (2**int(label) - 1) / math.log2(index + 1)
    return total


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_feature_names(
    feature_names: Sequence[str] | None,
    model_path: Any,
) -> Sequence[str] | None:
    if feature_names:
        return feature_names
    metadata_path = Path(model_path).with_name("reranker_metadata.json")
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    names = metadata.get("feature_names")
    if not isinstance(names, list):
        return None
    return [str(name) for name in names]
