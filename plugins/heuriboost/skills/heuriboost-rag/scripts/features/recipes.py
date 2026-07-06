#!/usr/bin/env python3
"""The single shared feature implementation for the V0 Q-D reranker.

`extract_all(row)` holds the verbatim body of the pre-refactor
`common.extract_features` per-row computation. The registry dispatches to it;
per-feature dispatch is deferred to the ablation task.
"""

from __future__ import annotations

import math

from features.primitives import (
    entities,
    numbers,
    numeric_value,
    rank_inverse,
    tokenize,
)


def extract_all(row) -> dict[str, float]:
    query_text = str(row["query_text"])
    doc_text = str(row["doc_text"])
    query_tokens = tokenize(query_text)
    doc_tokens = tokenize(doc_text)
    shared_tokens = query_tokens & doc_tokens

    dense_rank_inv = rank_inverse(row, "dense_rank")
    sparse_rank_inv = rank_inverse(row, "sparse_rank")
    rrf_score = 0.0
    dense_rank = numeric_value(row, "dense_rank", 0.0)
    sparse_rank = numeric_value(row, "sparse_rank", 0.0)
    if dense_rank > 0:
        rrf_score += 1.0 / (60.0 + dense_rank)
    if sparse_rank > 0:
        rrf_score += 1.0 / (60.0 + sparse_rank)

    # Overlap of rarer/important terms approximated by longer shared tokens.
    important_term_overlap = float(
        sum(1 for token in shared_tokens if len(token) >= 6)
    )

    # Low-information-density proxy: low unique-token ratio or very short doc.
    unique_ratio = len(set(doc_tokens)) / max(len(doc_tokens), 1)
    low_information_density = (
        1.0 if (unique_ratio < 0.5 or len(doc_tokens) < 5) else 0.0
    )

    return {
        "dense_score": numeric_value(row, "dense_score"),
        "dense_rank_inverse": dense_rank_inv,
        "sparse_score": numeric_value(row, "sparse_score"),
        "sparse_rank_inverse": sparse_rank_inv,
        "rrf_score": rrf_score,
        "term_overlap_ratio": len(shared_tokens) / max(len(query_tokens), 1),
        "number_overlap_count": float(len(numbers(query_text) & numbers(doc_text))),
        "entity_overlap_count": float(
            len(entities(query_text) & entities(doc_text))
        ),
        "important_term_overlap": important_term_overlap,
        "low_information_density_flag": low_information_density,
        "doc_length_log": math.log1p(len(doc_tokens)),
        "query_length_log": math.log1p(len(query_tokens)),
    }
