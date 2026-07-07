#!/usr/bin/env python3
"""Ranking Snapshot construction for label-bearing ranking splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Snapshot:
    """A precomputed Ranking Snapshot for one split.

    `X` is the feature matrix, `y` the MAPPED relevance labels (0..4, for
    XGBoost training), `raw_labels` the ORIGINAL labels (-1..3, for nDCG
    scoring), and `groups` the query-group sizes.
    """

    X: Any
    y: Any
    raw_labels: list[int]
    groups: list[int]


FeatureExtractor = Callable[[Any], Any]


def snapshot_from_frame(
    df: Any,
    feature_extractor: FeatureExtractor | None = None,
) -> Snapshot:
    """Build a Ranking Snapshot from Query-Document Examples for one split."""
    from common import extract_features, group_sizes, relevance_labels

    if feature_extractor is None:
        feature_extractor = extract_features

    return Snapshot(
        X=feature_extractor(df),
        y=relevance_labels(df),
        raw_labels=[int(v) for v in df["label"].tolist()],
        groups=group_sizes(df),
    )
