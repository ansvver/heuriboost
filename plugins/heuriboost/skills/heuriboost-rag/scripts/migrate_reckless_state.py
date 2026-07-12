#!/usr/bin/env python3
"""Import legacy mutable Reckless state into one immutable bootstrap release."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from heuriboost_rag.reckless.errors import HeuriBoostError
from heuriboost_rag.reckless.migration import migrate_legacy_state
from heuriboost_rag.reckless.release_store import FileReleaseStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="heuriboost_output",
        help="Legacy repair output directory containing .heuriboost",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    try:
        receipt = migrate_legacy_state(
            output_dir / ".heuriboost",
            FileReleaseStore(output_dir / ".reckless"),
        )
    except (HeuriBoostError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Migrated legacy state: {receipt.run_id}")
    print(f"Current model: {receipt.current_model}")
    print(f"Bootstrap receipt: {receipt.receipt_html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
