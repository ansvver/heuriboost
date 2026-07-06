#!/usr/bin/env python3
"""Compile user-facing repair inputs into HeuriBoost internal artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from repair_cases import (
    DEFAULT_CASE_TOP_K,
    DEFAULT_MIN_DOCS_PER_QUERY,
    DEFAULT_MIN_DOMAIN_TEST_QUERIES,
    DEFAULT_MIN_GLOBAL_TEST_QUERIES,
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_SPLIT_SEED,
    CompileOptions,
    compile_repair_inputs,
)


def parse_split_ratios(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("split ratios must be train,validation,test")
    try:
        ratios = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("split ratios must be numeric") from exc
    if any(part < 0 for part in ratios):
        raise argparse.ArgumentTypeError("split ratios must be non-negative")
    if sum(ratios) <= 0:
        raise argparse.ArgumentTypeError("split ratios must sum to a positive value")
    return ratios  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", required=True, help="User-facing base dataset CSV/JSONL")
    parser.add_argument(
        "--production-cases",
        required=True,
        help="User-facing production cases CSV/JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="heuriboost_output",
        help="Output directory for compiled artifacts",
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        help="Replace an existing split column with a deterministic auto split",
    )
    parser.add_argument(
        "--split-ratio",
        type=parse_split_ratios,
        default=DEFAULT_SPLIT_RATIOS,
        help="Auto split ratios train,validation,test (default: 0.7,0.15,0.15)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"Auto split seed (default: {DEFAULT_SPLIT_SEED})",
    )
    parser.add_argument(
        "--case-top-k",
        type=int,
        default=DEFAULT_CASE_TOP_K,
        help=f"Case acceptance top_k (default: {DEFAULT_CASE_TOP_K})",
    )
    parser.add_argument(
        "--acceptance-level",
        choices=["full", "weak"],
        default="full",
        help="Compile-time acceptance level for strict checks",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Run strict repair sufficiency checks during compilation",
    )
    parser.add_argument(
        "--min-global-test-queries",
        type=int,
        default=DEFAULT_MIN_GLOBAL_TEST_QUERIES,
        help=f"Strict global test query minimum (default: {DEFAULT_MIN_GLOBAL_TEST_QUERIES})",
    )
    parser.add_argument(
        "--min-domain-test-queries",
        type=int,
        default=DEFAULT_MIN_DOMAIN_TEST_QUERIES,
        help=f"Strict touched-domain test query minimum (default: {DEFAULT_MIN_DOMAIN_TEST_QUERIES})",
    )
    parser.add_argument(
        "--min-docs-per-query",
        type=int,
        default=DEFAULT_MIN_DOCS_PER_QUERY,
        help=f"Strict minimum docs per validation/test query (default: {DEFAULT_MIN_DOCS_PER_QUERY})",
    )
    args = parser.parse_args()

    options = CompileOptions(
        output_dir=Path(args.output_dir),
        resplit=args.resplit,
        split_ratios=args.split_ratio,
        split_seed=args.split_seed,
        case_top_k=args.case_top_k,
        strict=args.strict,
        acceptance_level=args.acceptance_level,
        min_global_test_queries=args.min_global_test_queries,
        min_domain_test_queries=args.min_domain_test_queries,
        min_docs_per_query=args.min_docs_per_query,
    )
    result = compile_repair_inputs(args.base_dataset, args.production_cases, options)

    print("Production-case inputs compiled")
    print(f"Compiled dataset: {result.base_dataset_path}")
    print(f"Compiled cases: {result.regression_cases_path}")
    print(f"Compiled case_sets: {result.case_sets_dir}")
    print(f"Compile report: {result.compile_report_path}")
    print(f"Base rows: {len(result.base_df)}")
    print(f"Production cases: {len(result.production_cases)}")
    print(f"Repair sample rows: {len(result.repair_samples_df)}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
