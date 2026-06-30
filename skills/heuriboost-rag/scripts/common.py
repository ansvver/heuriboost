#!/usr/bin/env python3
"""Shared utilities for the HeuriBoost RAG V0 scripts."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Feature registry: importing `features` eagerly loads feature_recipes.yaml,
# registers the shared extract_all impl, and validates (impl binding, inputs
# allowlist, online_safe, required fields). Any failure raises SystemExit.
from features import REGISTRY, extract_features
from features.primitives import (
    entities,
    numbers,
    numeric_value,
    rank_inverse,
    tokenize,
)

FEATURE_NAMES = REGISTRY.names()

REQUIRED_COLUMNS = {
    "query_id",
    "query_text",
    "doc_id",
    "doc_text",
    "label",
    "split",
}

OPTIONAL_COLUMNS = {
    "chunk_id",
    "dense_rank",
    "dense_score",
    "sparse_rank",
    "sparse_score",
    "doc_text_ref",
}

VALID_SPLITS = {"train", "validation", "test"}
LABELS = {-1, 0, 1, 2, 3}


@dataclass(frozen=True)
class ValidationResult:
    rows: int
    query_groups: int
    splits: dict[str, int]
    warnings: list[str]


def require_dependencies(*names: str) -> None:
    missing = []
    broken = []
    for name in names:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
        except Exception as exc:
            broken.append((name, exc))
    if missing:
        deps = ", ".join(missing)
        print(f"Missing required Python dependencies: {deps}", file=sys.stderr)
        print(
            "Install them with: python -m pip install -r "
            "skills/heuriboost-rag/requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if broken:
        print("Required Python dependencies are installed but failed to import:", file=sys.stderr)
        for name, exc in broken:
            print(f"  - {name}: {exc}", file=sys.stderr)
        print(
            "If this is xgboost on macOS, install the OpenMP runtime with: "
            "brew install libomp",
            file=sys.stderr,
        )
        raise SystemExit(2)


def load_pandas():
    require_dependencies("pandas")
    import pandas as pd

    return pd


def load_dataset(path: str | Path):
    pd = load_pandas()
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    try:
        return pd.read_csv(dataset_path)
    except Exception as exc:
        raise SystemExit(f"Failed to read CSV {dataset_path}: {exc}") from exc


def validate_dataset_frame(df) -> ValidationResult:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {', '.join(missing)}")

    if df.empty:
        raise SystemExit("Dataset is empty.")

    warnings: list[str] = []

    for column in REQUIRED_COLUMNS:
        if column == "label":
            continue
        if df[column].isna().any():
            raise SystemExit(f"Required column has missing values: {column}")

    try:
        labels = df["label"].astype(int)
    except Exception as exc:
        raise SystemExit("Column label must contain integer values in {-1,0,1,2,3}.") from exc

    invalid_labels = sorted(set(labels) - LABELS)
    if invalid_labels:
        raise SystemExit(
            "Column label contains unsupported values: "
            + ", ".join(str(value) for value in invalid_labels)
        )

    split_values = set(df["split"].astype(str))
    invalid_splits = sorted(split_values - VALID_SPLITS)
    if invalid_splits:
        raise SystemExit(
            "Column split contains unsupported values: " + ", ".join(invalid_splits)
        )

    split_by_query = df.groupby("query_id")["split"].nunique()
    leaked = split_by_query[split_by_query > 1]
    if not leaked.empty:
        ids = ", ".join(str(idx) for idx in leaked.index[:10])
        raise SystemExit(f"query_id values cannot cross splits. Offenders: {ids}")

    group_sizes = df.groupby("query_id").size()
    singletons = group_sizes[group_sizes < 2]
    if not singletons.empty:
        warnings.append(
            f"{len(singletons)} query group(s) have fewer than 2 documents; ranking metrics may be weak."
        )

    if not (labels > 0).any():
        warnings.append("Dataset has no positive labels.")
    if not (labels == -1).any():
        warnings.append("Dataset has no hard-negative -1 labels.")

    required_splits = {"train", "validation"}
    missing_splits = sorted(required_splits - split_values)
    if missing_splits:
        warnings.append(
            "Training/evaluation is strongest with train and validation splits. Missing: "
            + ", ".join(missing_splits)
        )

    return ValidationResult(
        rows=len(df),
        query_groups=df["query_id"].nunique(),
        splits={key: int(value) for key, value in df["split"].value_counts().items()},
        warnings=warnings,
    )


def relevance_labels(df):
    labels = df["label"].astype(int)
    # XGBoost ranking labels must be non-negative. Keep hard negatives below
    # ordinary irrelevant documents so the model can learn to suppress them.
    return labels.map({-1: 0, 0: 1, 1: 2, 2: 3, 3: 4})


def sort_for_ranking(df):
    sort_columns = ["query_id"]
    if "dense_rank" in df.columns:
        sort_columns.append("dense_rank")
    elif "sparse_rank" in df.columns:
        sort_columns.append("sparse_rank")
    else:
        sort_columns.append("doc_id")
    return df.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def group_sizes(df) -> list[int]:
    return [int(size) for size in df.groupby("query_id", sort=False).size().tolist()]


def split_frame(df, split: str):
    return sort_for_ranking(df[df["split"].astype(str) == split].copy())


def ndcg_at_k(labels: Iterable[int], k: int) -> float:
    values = [max(int(label), 0) for label in labels]
    if not values:
        return 0.0
    cutoff = values[:k]
    dcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(cutoff))
    ideal = sorted(values, reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr_at_k(labels: Iterable[int], k: int) -> float:
    for idx, label in enumerate(list(labels)[:k], start=1):
        if int(label) > 0:
            return 1.0 / idx
    return 0.0


def recall_at_k(labels: Iterable[int], k: int) -> float:
    values = [int(label) for label in labels]
    positives = sum(1 for label in values if label > 0)
    if positives == 0:
        return 0.0
    found = sum(1 for label in values[:k] if label > 0)
    return found / positives


def hard_negative_at_k(labels: Iterable[int], k: int) -> int:
    return sum(1 for label in list(labels)[:k] if int(label) == -1)


def rank_by_baseline(df, baseline: str):
    frame = df.copy()
    if baseline == "dense":
        if "dense_rank" not in frame.columns:
            return None
        return frame.sort_values(["query_id", "dense_rank"], kind="stable")
    if baseline == "sparse":
        if "sparse_rank" not in frame.columns:
            return None
        return frame.sort_values(["query_id", "sparse_rank"], kind="stable")
    if baseline == "rrf":
        if not {"dense_rank", "sparse_rank"}.issubset(frame.columns):
            return None
        frame = frame.assign(
            _rrf=frame.apply(
                lambda row: (1.0 / (60.0 + numeric_value(row, "dense_rank", 10_000.0)))
                + (1.0 / (60.0 + numeric_value(row, "sparse_rank", 10_000.0))),
                axis=1,
            )
        )
        return frame.sort_values(["query_id", "_rrf"], ascending=[True, False], kind="stable")
    raise ValueError(f"Unknown baseline: {baseline}")


def rank_by_model(df, scores):
    frame = df.copy()
    frame["_heuriboost_score"] = scores
    return frame.sort_values(
        ["query_id", "_heuriboost_score"], ascending=[True, False], kind="stable"
    )


def evaluate_ranked_frame(ranked_df, k_values=(3, 5, 10)) -> dict[str, float]:
    per_query = []
    for query_id, group in ranked_df.groupby("query_id", sort=False):
        labels = group["label"].astype(int).tolist()
        row = {"query_id": query_id}
        for k in k_values:
            row[f"ndcg@{k}"] = ndcg_at_k(labels, k)
            row[f"mrr@{k}"] = mrr_at_k(labels, k)
            row[f"recall@{k}"] = recall_at_k(labels, k)
            row[f"hard_negative@{k}"] = hard_negative_at_k(labels, k)
        per_query.append(row)
    if not per_query:
        return {}
    metrics = {}
    for key in per_query[0]:
        if key == "query_id":
            continue
        metrics[key] = sum(float(row[key]) for row in per_query) / len(per_query)
    metrics["query_count"] = float(len(per_query))
    return metrics


def write_json(path: str | Path, data) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def ensure_output_dirs(output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir)
    models = root / "models"
    reports = root / "reports"
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return models, reports


def copy_regression_cases(source: str | Path | None, output_dir: str | Path) -> None:
    if source is None:
        return
    src = Path(source)
    if not src.exists():
        raise SystemExit(f"Regression cases file not found: {src}")
    dst = Path(output_dir) / "regression_cases.yaml"
    dst.write_text(src.read_text())


def load_case_sets(case_sets_path: str | Path, *, drop_source_case_id: bool = True):
    pd = load_pandas()
    path = Path(case_sets_path)
    if not path.exists():
        raise SystemExit(f"--case-sets path not found: {path}")

    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            return pd.DataFrame()
    else:
        csv_files = [path]

    frames = []
    for csv_file in csv_files:
        if csv_file.stat().st_size == 0:
            frame = pd.DataFrame()
        else:
            try:
                frame = pd.read_csv(csv_file)
            except Exception as exc:
                raise SystemExit(f"Failed to read case_set CSV {csv_file}: {exc}") from exc
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if drop_source_case_id and "source_case_id" in combined.columns:
        combined = combined.drop(columns=["source_case_id"])
    return combined
