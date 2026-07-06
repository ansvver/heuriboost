#!/usr/bin/env python3
"""Primitive text/numeric helpers shared by feature recipes.

Lifted verbatim from `common.py` so the feature registry can own its own
low-level building blocks without a circular import on `common`.
"""

from __future__ import annotations

import math
import re


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", str(text).lower()))


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", str(text)))


def entities(text: str) -> set[str]:
    # Deterministic, dependency-free proxy for proper nouns / acronyms:
    # capitalized words (e.g. "Roth", "IRA") and all-caps acronyms (e.g. "ETF").
    return set(re.findall(r"\b[A-Z][a-zA-Z]+\b|\b[A-Z]{2,}\b", str(text)))


def numeric_value(row, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def rank_inverse(row, column: str) -> float:
    rank = numeric_value(row, column, 0.0)
    if rank <= 0:
        return 0.0
    return 1.0 / rank
