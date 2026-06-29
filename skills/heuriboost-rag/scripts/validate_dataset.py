#!/usr/bin/env python3
"""Validate a HeuriBoost query-document CSV."""

from __future__ import annotations

import argparse

from common import load_dataset, validate_dataset_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to query_doc_examples.csv")
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    result = validate_dataset_frame(df)

    print("HeuriBoost dataset validation passed")
    print(f"Rows: {result.rows}")
    print(f"Query groups: {result.query_groups}")
    print("Splits:")
    for split, count in sorted(result.splits.items()):
        print(f"  {split}: {count}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
