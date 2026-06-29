#!/usr/bin/env python3
"""HPOEngine façade + Snapshot/TrialResult/Budget dataclasses (spec §12.5).

The engine is a thin façade over a pluggable backend (V0: OptunaBackend in
`optuna_backend.py`). It takes pre-computed feature snapshots and never touches
the raw CSV or regression cases — anti-leak by signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Snapshot:
    """A pre-computed feature snapshot for one split.

    `X` is the feature matrix (from REGISTRY.extract), `y` the MAPPED relevance
    labels (0..4, for xgboost training), `raw_labels` the ORIGINAL labels
    (-1..3, for nDCG scoring consistent with `evaluate_ranked_frame`/baseline
    0.853), `groups` the query-group sizes.
    """

    X: Any
    y: Any
    raw_labels: list[int]
    groups: list[int]


@dataclass(frozen=True)
class Budget:
    """Bounded HPO budget. `n_trials` is the primary; `timeout_sec` optional."""

    n_trials: int
    timeout_sec: int | None = None


@dataclass(frozen=True)
class TrialResult:
    best_params: dict
    best_score: float                       # validation nDCG@10 (search objective)
    best_iteration: int                     # xgboost early-stopping best round
    trials: list[dict]                      # {params, score, state, failure_reason?}
    feature_set_name: str
    feature_set_version: int
    objective: str                          # "rank:ndcg"
    eval_metric: str                        # "ndcg@10"
    early_stopping_rounds: int
    num_boost_round: int
    seed: int
    n_trials: int                           # requested budget
    timeout_sec: int | None
    test_score: float | None = None         # post-hoc honest test nDCG@10 (NOT a search objective)


class HPOEngine:
    """Backend-agnostic HPO façade. V0 backend: OptunaBackend."""

    def __init__(self, backend: Any = None) -> None:
        if backend is None:
            from hpo.optuna_backend import OptunaBackend

            backend = OptunaBackend()
        self._backend = backend

    def optimize(
        self,
        task_profile: str,
        feature_set_name: str,
        feature_set_version: int,
        train_snapshot: Snapshot,
        valid_snapshot: Snapshot,
        budget: Budget,
        seed: int,
    ) -> TrialResult:
        return self._backend.run(
            train_snapshot=train_snapshot,
            valid_snapshot=valid_snapshot,
            budget=budget,
            seed=seed,
            feature_set_name=feature_set_name,
            feature_set_version=feature_set_version,
            task_profile=task_profile,
        )
