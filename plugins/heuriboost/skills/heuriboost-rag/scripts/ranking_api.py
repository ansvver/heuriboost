"""Compatibility exports for the moved ranking API."""

from heuriboost_rag.backends.ranking import (
    evaluate_xgboost_ranker,
    predict_xgboost_ranker,
    train_xgboost_ranker,
)

__all__ = [
    "train_xgboost_ranker",
    "evaluate_xgboost_ranker",
    "predict_xgboost_ranker",
]
